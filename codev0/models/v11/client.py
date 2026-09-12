"""Method 1.1 client.

Local-only by design. The reference package deliberately contains no federation
logic and its integration guide asks for the model layer to be swapped in while
the data split, lifecycle, evaluation and best-validation selection stay
untouched, so this client never consumes the broadcast.

The server still runs its normal aggregate/broadcast lifecycle; those messages
are simply ignored, exactly as `models/local/` does. A number from this entry is
therefore a backbone result and must not be reported as a federated one.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from models.v11.builder import build_v11_model
from models.v11.diagnostics import generator_diagnostics, trajectory_diagnostics
from models.v11.losses import method1_local_loss
from modules.federated import ClientModule


BINARY_DATASETS = {'Minesweeper', 'Tolokers', 'Questions'}


def task_loss(logits, labels, mask, dataset):
    if dataset in BINARY_DATASETS:
        return F.binary_cross_entropy_with_logits(
            logits[mask].reshape(-1), labels[mask].float().reshape(-1)
        )
    return F.cross_entropy(logits[mask], labels[mask].long())


class Client(ClientModule):
    def __init__(self, args, w_id, g_id, sd):
        super().__init__(args, w_id, g_id, sd)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.seed))
            self.model = build_v11_model(args).cuda(g_id)
        self._initial_model = get_state_dict(self.model)
        self._active_batch = None
        self._build_optimizers()

    def _build_optimizers(self):
        self.main_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.args.lr_main),
            weight_decay=float(self.args.weight_decay),
        )
        self.optimizers = {'main': self.main_optimizer}
        self.optimizer = self.main_optimizer

    def switch_state(self, client_id):
        super().switch_state(client_id)
        self._active_batch = None
        for batch in self.loader.pa_loader:
            self._active_batch = batch.cuda(self.gpu_id)
            break
        if self._active_batch is None:
            raise RuntimeError('Method 1.1 requires one non-empty client graph.')

    def init_state(self):
        set_state_dict(self.model, self._initial_model, self.gpu_id)
        self._build_optimizers()
        self.best = {
            'round': 0,
            'val_metric': float('-inf'),
            'test_metric': 0.0,
            'test_f1': 0.0,
        }

    def _checkpoint_payload(self):
        return {
            'main_optimizer': self.main_optimizer.state_dict(),
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
        current = f'{self.client_id}_current_state.pt'
        if os.path.exists(os.path.join(self.args.checkpt_path, current)):
            return current
        return f'{self.client_id}_state.pt'

    def load_state(self):
        loaded = torch_load(
            self.args.checkpt_path, self._checkpoint_filename()
        )
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.main_optimizer.load_state_dict(loaded['main_optimizer'])
        self.best = loaded['best']

    def on_receive_message(self, curr_rnd):
        # Local-only: the broadcast is intentionally not consumed.
        self.curr_rnd = int(curr_rnd)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def _batch(self):
        return self._active_batch

    def _forward(self, batch, return_details=False):
        return self.model(
            batch.x,
            batch.edge_index,
            getattr(batch, 'edge_weight', None),
            return_details=return_details,
        )

    @torch.no_grad()
    def validate(self, mode='test'):
        batch = self._batch()
        self.model.eval()
        logits = self._forward(batch)
        mask = batch.test_mask if mode == 'test' else batch.val_mask
        if int(mask.sum().item()) == 0:
            return 0.0, 0.0, 0.0
        loss = task_loss(logits, batch.y, mask, self.args.dataset)
        metric = self.accuracy(logits[mask], batch.y[mask])
        f1 = self.f1(logits[mask], batch.y[mask])
        return float(metric), float(loss.item()), float(f1)

    def _maybe_save_best(self, val_metric, test_metric, test_f1):
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

    def train(self):
        started = time.time()
        batch = self._batch()
        config = self.model.config
        parts = {}
        grad_norm = torch.zeros(())
        for _ in range(int(self.args.n_eps)):
            self.model.train()
            self.main_optimizer.zero_grad(set_to_none=True)
            out = self._forward(batch, return_details=True)
            loss, parts = method1_local_loss(
                logits=out['logits'],
                labels=batch.y,
                trajectory=out['trajectory'],
                observable=self.model.observable,
                manifold_weight=config.manifold_weight,
                train_mask=batch.train_mask,
                label_smoothing=config.label_smoothing,
                koopman_field=self.model.koopman_field,
                operator_speed_weight=config.operator_speed_weight,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.args.max_grad_norm)
            )
            self.main_optimizer.step()

        val_metric, val_loss, _ = self.validate(mode='valid')
        test_metric, test_loss, test_f1 = self.validate(mode='test')
        self._maybe_save_best(val_metric, test_metric, test_f1)

        # Diagnostics come from one eval-mode forward so dropout does not
        # perturb the quantities a PH stability claim is read off.
        self.model.eval()
        with torch.no_grad():
            details = self._forward(batch, return_details=True)
            # `trajectory_diagnostics` names its norm columns z0_norm/zT_norm,
            # which collide with the V0.4 columns of the same meaning in
            # `z0_norm` and differ only by case in `zT_norm` vs `zt_norm`.
            # Prefix them here so the two protocols never share a column.
            trajectory = trajectory_diagnostics(details)
            trajectory['ph_z0_norm'] = trajectory.pop('z0_norm')
            trajectory['ph_zT_norm'] = trajectory.pop('zT_norm')
            diagnostics = {
                'task_loss': float(parts['task_loss'].item()),
                'train_loss': float(parts['task_loss'].item()),
                'manifold_loss': float(parts['manifold_loss'].item()),
                'operator_speed_loss': float(parts['operator_speed_loss'].item()),
                'main_grad_norm': float(grad_norm.item()),
                # Train accuracy is the direct read on the memorization failure
                # mode this version exists to fix, so it is logged every round.
                'train_accuracy': self.accuracy(
                    details['logits'][batch.train_mask], batch.y[batch.train_mask]
                ),
                **trajectory,
            }
            for entry in generator_diagnostics(self.model):
                band = int(entry['band'])
                for name in (
                    'alpha', 'beta', 'gamma', 'J_fro', 'R_fro',
                    'sym_max_eig', 'spectral_norm',
                ):
                    diagnostics[f'band{band}_{name}'] = float(entry[name])

        self._round_result = {
            **diagnostics,
            'val_loss': val_loss,
            'val_accuracy': val_metric,
            'test_loss': test_loss,
            'test_accuracy': test_metric,
            'test_f1': test_f1,
            'elapsed_seconds': float(time.time() - started),
        }

    def transfer_to_server(self):
        weights = get_state_dict(self.model)
        self.sd[self.client_id] = {
            'client_id': int(self.client_id),
            'model': weights,
            'train_size': int(self._batch().train_mask.sum().item()),
            'upload_bytes': int(sum(value.nbytes for value in weights.values())),
            **self._round_result,
        }
