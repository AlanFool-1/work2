"""S0 ODE-GNN client with Fedrated's persistent-state lifecycle."""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from models.s0.model import build_s0_model
from modules.federated import ClientModule
from modules.initial_state_visualizer import save_initial_state_snapshot
from modules.dynamics_visualizer import save_dynamics_snapshot
from dynamics import build_trajectory_basis, fit_ridge_generator
from dynamics.trajectory_basis import shared_probe
from functional_map import (
    descriptor_procrustes,
    normalized_descriptor,
    procrustes_distance,
    solve_orthogonal_fm,
    solve_regularized_fm,
)
from models.dissipative_injector import minimal_dissipative_projection


BINARY_DATASETS = {'Minesweeper', 'Tolokers', 'Questions'}


class Client(ClientModule):
    def __init__(self, args, w_id, g_id, sd):
        super().__init__(args, w_id, g_id, sd)
        self.model = build_s0_model(args).cuda(g_id)
        self._build_optimizer()

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.args.base_lr),
            weight_decay=float(self.args.weight_decay),
        )

    def switch_state(self, client_id):
        super().switch_state(client_id)
        # A partition contains one graph. Reusing its CUDA batch changes no
        # sample ordering but avoids rebuilding the same PyG Batch for every
        # train/validation/test forward.
        self._active_batch = None
        for batch in self.loader.pa_loader:
            self._active_batch = batch.cuda(self.gpu_id)
            # The official A-DGN forward accesses ``data.edge_weight``
            # directly. Our unweighted partitions omit that optional field,
            # so adapt the data boundary while leaving official code intact.
            if getattr(self._active_batch, 'edge_weight', None) is None:
                self._active_batch.edge_weight = None
            break
        if self._active_batch is None:
            raise RuntimeError('S0 requires one non-empty client graph.')

    def init_state(self):
        self._build_optimizer()
        self.functional_cache = None
        self.functional_global = None
        self.functional_payload = None
        self.functional_metrics = {}
        self.best = {
            'round': 0,
            'val_metric': float('-inf'),
            'test_metric': 0.0,
            'test_f1': 0.0,
        }

    def _checkpoint_payload(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'best': self.best,
            'functional_cache': self.functional_cache,
        }

    def save_state(self):
        torch_save(
            self.args.checkpt_path,
            f'{self.client_id}_current_state.pt',
            self._checkpoint_payload(),
        )

    def _checkpoint_filename(self):
        current_name = f'{self.client_id}_current_state.pt'
        current_path = os.path.join(self.args.checkpt_path, current_name)
        return (
            current_name if os.path.exists(current_path)
            else f'{self.client_id}_state.pt'
        )

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, self._checkpoint_filename())
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        # This deliberately precedes the server broadcast.  It is the original
        # Fedrated behavior: client-specific Adam moments persist while shared
        # model parameters are subsequently replaced by the global state.
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.best = loaded['best']
        self.functional_cache = loaded.get('functional_cache')

    def _maybe_save_best_state(self, val_metric, test_metric, test_f1):
        if float(val_metric) <= float(self.best['val_metric']):
            return
        self.best = {
            'round': self.curr_rnd + 1,
            'val_metric': float(val_metric),
            'test_metric': float(test_metric),
            'test_f1': float(test_f1),
        }
        torch_save(
            self.args.checkpt_path,
            f'{self.client_id}_state.pt',
            self._checkpoint_payload(),
        )

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        set_state_dict(self.model, self.sd['global'], self.gpu_id)
        self.functional_global = self.sd.get('functional_global')

    def on_round_begin(self):
        self._maybe_collect_initial_state()
        self._prepare_functional_injection()
        self.train()
        injection_ratio = self.model.functional_injection_ratio()
        self._calibrate_functional_dynamics(injection_ratio)
        self.transfer_to_server()

    def _as_device_tensor(self, value):
        return torch.as_tensor(
            value, device=self._first_batch().x.device,
            dtype=self._first_batch().x.dtype,
        )

    def _prototype_tensors(self):
        if self.functional_global is None:
            return [], []
        a_value = self.functional_global.get('A_star')
        b_value = self.functional_global.get('B_star')
        if a_value is None or b_value is None:
            return [], []
        if int(self.functional_global.get('num_prototypes', 1)) == 1:
            a_value, b_value = [a_value], [b_value]
        return (
            [self._as_device_tensor(value) for value in a_value],
            [self._as_device_tensor(value) for value in b_value],
        )

    def _select_prototype(self, descriptor, b_stars, previous_cluster=None,
                          a_local=None, a_stars=None):
        distances, affinities = zip(*[
            procrustes_distance(descriptor, target) for target in b_stars
        ])
        values = torch.stack(distances)
        if (
            self.args.fd_map_type == 'regularized'
            and a_local is not None and a_stars is not None
        ):
            scores = []
            for a_star, b_star in zip(a_stars, b_stars):
                _, metrics = self._solve_functional_map(
                    a_local, descriptor, a_star, b_star
                )
                scores.append(descriptor.new_tensor(metrics['fm_final_objective']))
            selection_values = torch.stack(scores)
        else:
            selection_values = values
        selected = int(torch.argmin(selection_values).item())
        switched = False
        if previous_cluster is not None and 0 <= int(previous_cluster) < len(b_stars):
            previous_cluster = int(previous_cluster)
            best = float(selection_values[selected].item())
            previous = float(selection_values[previous_cluster].item())
            required = float(self.args.fd_cluster_switch_margin)
            if selected != previous_cluster and best > previous * (1.0 - required):
                selected = previous_cluster
            switched = selected != previous_cluster
        return selected, distances, affinities, switched, selection_values

    def _solve_functional_map(self, a_local, descriptor, a_star, b_star,
                              warm_start=None, descriptor_only=False):
        if self.args.fd_map_type == 'orthogonal':
            if descriptor_only:
                c_map = descriptor_procrustes(descriptor, b_star)
                identity = torch.eye(c_map.shape[0], device=c_map.device, dtype=c_map.dtype)
                return c_map, {
                    'fm_desc_error': float(torch.linalg.matrix_norm(c_map @ descriptor - b_star).item()),
                    'fm_dyn_error': float('nan'),
                    'fm_orth_error': float(torch.linalg.matrix_norm(c_map.T @ c_map - identity).item()),
                    'fm_condition': 1.0,
                }
            return solve_orthogonal_fm(
                a_local, descriptor, a_star, b_star,
                steps=self.args.fd_fm_steps, learning_rate=self.args.fd_fm_lr,
                lambda_desc=self.args.fd_lambda_desc,
                lambda_dyn=self.args.fd_lambda_dyn, warm_start=warm_start,
            )
        return solve_regularized_fm(
            a_local, descriptor, a_star, b_star,
            steps=self.args.fd_fm_steps, learning_rate=self.args.fd_fm_lr,
            lambda_desc=self.args.fd_lambda_desc,
            lambda_dyn=(0.0 if descriptor_only else self.args.fd_lambda_dyn),
            ridge=self.args.fd_map_ridge,
            orthogonal_regularization=self.args.fd_map_orth_reg,
            max_condition=self.args.fd_map_max_condition,
            warm_start=warm_start,
        )

    def _transport_generator(self, c_map, a_local):
        if self.args.fd_map_type == 'orthogonal':
            return c_map @ a_local @ c_map.T
        return torch.linalg.solve(c_map.T, (c_map @ a_local).T).T

    def _pull_back_generator(self, c_map, a_star):
        if self.args.fd_map_type == 'orthogonal':
            return c_map.T @ a_star @ c_map
        return torch.linalg.solve(c_map, a_star @ c_map)

    @torch.no_grad()
    def _prepare_functional_injection(self):
        self.functional_payload = None
        self.functional_metrics = {'injection_native_ratio': 0.0}
        self.model.set_functional_injection()
        if not self.args.enable_functional_dynamics:
            return
        if self.functional_cache is None or self.functional_global is None:
            return
        if not self.functional_cache.get('map_initialized', False):
            return
        a_stars, b_stars = self._prototype_tensors()
        if not a_stars:
            return
        q_basis = self._as_device_tensor(self.functional_cache['Q'])
        a_local = self._as_device_tensor(self.functional_cache['A'])
        a_map = self._as_device_tensor(
            self.functional_cache.get('A_map', self.functional_cache['A'])
        )
        a_scale = float(self.functional_cache.get('A_scale', 1.0))
        descriptor = self._as_device_tensor(self.functional_cache['B'])
        previous_cluster = self.functional_cache.get('cluster_id')
        cluster_id, distances, affinities, switched, scores = self._select_prototype(
            descriptor, b_stars, previous_cluster, a_map, a_stars
        )
        warm_start = (
            self._as_device_tensor(self.functional_cache['C'])
            if cluster_id == previous_cluster else None
        )
        c_map, _ = self._solve_functional_map(
            a_map, descriptor, a_stars[cluster_id], b_stars[cluster_id],
            warm_start=warm_start,
        )
        self.functional_cache['C'] = c_map.detach().cpu()
        self.functional_cache['cluster_id'] = cluster_id
        a_star = a_stars[cluster_id]
        pulled_back = self._pull_back_generator(c_map, a_star)
        delta_raw = (
            a_scale * (pulled_back - a_map)
            if self.args.fd_normalize_generator else pulled_back - a_local
        )
        delta_safe, metrics = minimal_dissipative_projection(
            delta_raw, self.args.fd_spectral_limit
        )
        self.functional_metrics.update(metrics)
        self.functional_metrics.update({
            'cluster_id': cluster_id,
            'cluster_distance': float(distances[cluster_id].item()),
            'cluster_affinity': float(affinities[cluster_id].item()),
            'cluster_switched': float(switched),
            'cluster_score': float(scores[cluster_id].item()),
        })
        self.model.set_functional_injection(
            q_basis, delta_safe, self.args.fd_beta
        )

    @torch.no_grad()
    def _calibrate_functional_dynamics(self, injection_ratio):
        if not self.args.enable_functional_dynamics:
            return
        batch = self._first_batch()
        trajectory = self.model.encode_native_dynamics_states(batch)
        probe = shared_probe(
            trajectory.shape[-1], self.args.fd_probe_dim,
            self.args.seed, trajectory.device, trajectory.dtype,
        )
        q_basis, snapshot, _ = build_trajectory_basis(
            trajectory, probe, self.args.fd_rank
        )
        a_local, generator_metrics = fit_ridge_generator(
            trajectory, q_basis, self.args.adgn_step_size,
            self.args.fd_ridge_lambda,
        )
        a_scale_tensor = torch.linalg.matrix_norm(a_local)
        a_map = (
            a_local / torch.clamp(a_scale_tensor, min=1e-12)
            if self.args.fd_normalize_generator else a_local
        )
        descriptor = normalized_descriptor(q_basis, snapshot)
        previous = self.functional_cache
        basis_staleness = float('nan')
        if previous is not None:
            previous_q = self._as_device_tensor(previous['Q'])
            overlap = torch.linalg.matrix_norm(previous_q.T @ q_basis) ** 2
            basis_staleness = float((overlap / self.args.fd_rank).item())
        a_stars, b_stars = self._prototype_tensors()
        if not b_stars:
            c_map = torch.eye(
                self.args.fd_rank, device=trajectory.device,
                dtype=trajectory.dtype,
            )
            fm_metrics = {
                'fm_desc_error': float('nan'),
                'fm_dyn_error': float('nan'),
                'fm_orth_error': 0.0,
            }
            map_initialized = False
            cluster_id = -1
            cluster_distance = float('nan')
            cluster_affinity = float('nan')
            cluster_switched = False
        elif previous is None or not previous.get('map_initialized', False):
            # Online bootstrap: the first aggregate establishes B_star.  The
            # following calibration initializes C with descriptor-only
            # Procrustes; the dynamics term starts only after that map exists.
            cluster_id, distances, affinities, cluster_switched, scores = self._select_prototype(
                descriptor, b_stars, None, a_local, a_stars
            )
            bootstrap = self.functional_global.get('bootstrap_assignments', {})
            if self.client_id in bootstrap:
                cluster_id = int(bootstrap[self.client_id])
            b_star = b_stars[cluster_id]
            c_map, fm_metrics = self._solve_functional_map(
                a_map, descriptor, a_stars[cluster_id], b_star,
                descriptor_only=True,
            )
            cluster_distance = float(distances[cluster_id].item())
            cluster_affinity = float(affinities[cluster_id].item())
            map_initialized = True
        else:
            cluster_id, distances, affinities, cluster_switched, scores = self._select_prototype(
                descriptor, b_stars, previous.get('cluster_id'), a_map, a_stars
            )
            a_star = a_stars[cluster_id]
            b_star = b_stars[cluster_id]
            cluster_distance = float(distances[cluster_id].item())
            cluster_affinity = float(affinities[cluster_id].item())
            warm_start = (
                self._as_device_tensor(previous['C'])
                if previous is not None and cluster_id == previous.get('cluster_id')
                else None
            )
            c_map, fm_metrics = self._solve_functional_map(
                a_map, descriptor, a_star, b_star,
                warm_start=warm_start,
            )
            map_initialized = True
        a_transport = self._transport_generator(c_map, a_map)
        b_transport = c_map @ descriptor
        self.functional_cache = {
            'Q': q_basis.detach().cpu(),
            'A': a_local.detach().cpu(),
            'A_map': a_map.detach().cpu(),
            'A_scale': float(a_scale_tensor.item()),
            'C': c_map.detach().cpu(),
            'B': descriptor.detach().cpu(),
            'map_initialized': map_initialized,
            'cluster_id': cluster_id,
        }
        self.functional_payload = {
            'A_transport': a_transport.detach().cpu().numpy(),
            'B_transport': b_transport.detach().cpu().numpy(),
        }
        self.functional_metrics.update(generator_metrics)
        self.functional_metrics['A_scale'] = float(a_scale_tensor.item())
        self.functional_metrics.update(fm_metrics)
        self.functional_metrics['basis_staleness'] = basis_staleness
        self.functional_metrics['injection_native_ratio'] = float(
            injection_ratio
        )
        self.functional_metrics.update({
            'cluster_id': cluster_id,
            'cluster_distance': cluster_distance,
            'cluster_affinity': cluster_affinity,
            'cluster_switched': float(cluster_switched),
            'cluster_score': (
                float('nan') if cluster_id < 0 else float(scores[cluster_id].item())
            ),
        })

    @torch.no_grad()
    def _maybe_collect_initial_state(self):
        if not self.args.plot_initial_state and not self.args.plot_dynamics_html:
            return
        if self.curr_rnd != int(self.args.initial_state_snapshot_round):
            return
        batch = self._first_batch()
        severity = getattr(batch, 'heterogeneity_severity', None)
        severity = (
            float(severity.reshape(-1)[0].item())
            if severity is not None
            else (
                self.client_id / max(1, self.args.n_clients - 1)
                if self.args.dataset == 'Synthetic' else None
            )
        )
        if self.args.dataset == 'Synthetic' and self.args.plot_initial_state:
            save_initial_state_snapshot(
                self.args.log_path,
                self.client_id,
                self.model.encode_initial_state(batch),
                batch.y,
                stage='before',
                severity=severity,
            )
        if self.args.plot_dynamics_html:
            save_dynamics_snapshot(
                self.args.log_path,
                self.client_id,
                # A worker can switch between client graphs with different
                # node counts.  Never let a stale previous-client injection
                # cache enter a diagnostic forward; the revision also defines
                # trajectory diagnostics on the native A-DGN dynamics.
                self.model.encode_native_dynamics_states(batch),
                batch.y,
                severity=severity,
            )

    def _first_batch(self):
        if self._active_batch is None:
            raise RuntimeError('Client batch has not been initialized.')
        return self._active_batch

    def _loss(self, logits, labels, mask):
        if self.args.dataset in BINARY_DATASETS:
            return F.binary_cross_entropy_with_logits(
                logits[mask].reshape(-1),
                labels[mask].float().reshape(-1),
            )
        return F.cross_entropy(logits[mask], labels[mask].long())

    @torch.no_grad()
    def validate(self, mode='test'):
        batch = self._first_batch()
        self.model.eval()
        logits = self.model(batch)
        mask = batch.test_mask if mode == 'test' else batch.val_mask
        if int(mask.sum().item()) == 0:
            return 0.0, 0.0, 0.0
        loss = self._loss(logits, batch.y, mask)
        metric = self.accuracy(logits[mask], batch.y[mask])
        f1 = self.f1(logits[mask], batch.y[mask])
        return float(metric), float(loss.item()), float(f1)

    def _train_step(self):
        batch = self._first_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch)
        self._last_train_lss = self._loss(
            logits, batch.y, batch.train_mask
        )
        self._last_train_lss.backward()
        self.optimizer.step()

    def train(self):
        started = time.time()
        for _ in range(self.args.n_eps):
            self._train_step()
        val_metric, val_loss, _ = self.validate(mode='valid')
        test_metric, test_loss, test_f1 = self.validate(mode='test')
        self._maybe_save_best_state(val_metric, test_metric, test_f1)
        self._round_result = {
            'train_loss': float(self._last_train_lss.item()),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_metric),
            'test_loss': float(test_loss),
            'test_accuracy': float(test_metric),
            'test_f1': float(test_f1),
            'elapsed_seconds': float(time.time() - started),
        }

    def transfer_to_server(self):
        batch = self._first_batch()
        message = {
            'client_id': int(self.client_id),
            'model': get_state_dict(self.model),
            'train_size': int(batch.train_mask.sum().item()),
            **self._round_result,
        }
        if self.functional_payload is not None:
            message.update({
                'fd_A_transport': self.functional_payload['A_transport'],
                'fd_B_transport': self.functional_payload['B_transport'],
                'fd_cluster_id': int(self.functional_cache['cluster_id']),
                'fd_metrics': self.functional_metrics,
            })
        self.sd[self.client_id] = message
