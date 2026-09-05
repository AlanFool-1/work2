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

    def on_round_begin(self):
        self._maybe_collect_initial_state()
        self.train()
        self.transfer_to_server()

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
                self.model.encode_dynamics_states(batch),
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
        self.sd[self.client_id] = {
            'client_id': int(self.client_id),
            'model': get_state_dict(self.model),
            'train_size': int(batch.train_mask.sum().item()),
            **self._round_result,
        }
