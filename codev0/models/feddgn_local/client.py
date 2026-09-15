"""Local-only lifecycle adapter for the FedDGN backbone.

Communication and server aggregation remain active, but the FedAvg-style
replacement of local parameters by the broadcast global model is deliberately
commented out for this stage.  Each client therefore keeps an independent
model and persistent Adam trajectory across rounds.
"""

from __future__ import annotations

import os
import time

import torch

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from models.feddgn_local.model import build_feddgn_model
from models.feddgn_local.training import task_loss, train_feddgn_step
from modules.federated import ClientModule


class Client(ClientModule):
    def __init__(self, args, w_id, g_id, sd):
        super().__init__(args, w_id, g_id, sd)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.seed))
            self.model = build_feddgn_model(args).cuda(g_id)
        self._initial_model = get_state_dict(self.model)
        self._active_batch = None
        self._build_optimizer()

    def _lr(self):
        if hasattr(self.args, 'feddgn_lr'):
            return float(self.args.feddgn_lr)
        if hasattr(self.args, 'lr_main'):
            return float(self.args.lr_main)
        if hasattr(self.args, 'base_lr'):
            return float(self.args.base_lr)
        if hasattr(self.args, 'lr'):
            return float(self.args.lr)
        return 1.5e-2

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._lr(),
            weight_decay=float(getattr(self.args, 'weight_decay', 1e-4)),
        )

    def switch_state(self, client_id):
        super().switch_state(client_id)
        self._active_batch = None
        for batch in self.loader.pa_loader:
            self._active_batch = batch.cuda(self.gpu_id)
            break
        if self._active_batch is None:
            raise RuntimeError('FedDGN requires one non-empty client graph.')

    def init_state(self):
        set_state_dict(self.model, self._initial_model, self.gpu_id)
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
        current = f'{self.client_id}_current_state.pt'
        if os.path.exists(os.path.join(self.args.checkpt_path, current)):
            return current
        return f'{self.client_id}_state.pt'

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, self._checkpoint_filename())
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.best = loaded['best']

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)

        # LOCAL-ONLY STAGE -------------------------------------------------
        # Keep communication + server aggregation intact, but DO NOT replace
        # the client's local dynamics with the FedAvg broadcast yet.
        #
        # set_state_dict(self.model, self.sd['global'], self.gpu_id)
        # ------------------------------------------------------------------
        # Adam is intentionally persistent because the local model itself is
        # persistent.  When FedAvg is re-enabled later, rebuild/reset optimizer
        # state after installing the global model.

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def _batch(self):
        if self._active_batch is None:
            raise RuntimeError('Client graph has not been loaded.')
        return self._active_batch

    @torch.no_grad()
    def validate(self, mode='test'):
        batch = self._batch()
        self.model.eval()
        logits = self.model(batch, external_force=None, force_scale=0.0)
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
        diagnostics = None
        for _ in range(int(self.args.n_eps)):
            diagnostics = train_feddgn_step(
                self.model, self._batch(), self.optimizer, self.args
            )
        val_metric, val_loss, _ = self.validate(mode='valid')
        test_metric, test_loss, test_f1 = self.validate(mode='test')
        self._maybe_save_best(val_metric, test_metric, test_f1)
        self._round_result = {
            **{
                name: float(value.detach().cpu().item())
                for name, value in diagnostics.items()
            },
            'feddgn_lr': float(self.optimizer.param_groups[0]['lr']),
            'val_loss': val_loss,
            'val_accuracy': val_metric,
            'test_loss': test_loss,
            'test_accuracy': test_metric,
            'test_f1': test_f1,
            'elapsed_seconds': float(time.time() - started),
        }

    def _upload_payload(self):
        """Build one complete message before publishing it to the server."""

        weights = get_state_dict(self.model)
        return {
            'client_id': int(self.client_id),
            'model': weights,
            'train_size': int(self._batch().train_mask.sum().item()),
            'upload_bytes': int(sum(value.nbytes for value in weights.values())),
            **self._round_result,
        }

    def transfer_to_server(self):
        # A Manager.dict assignment is the server's completion signal. Build
        # the entire payload first so subclasses can extend it without ever
        # exposing a partially populated message.
        self.sd[self.client_id] = self._upload_payload()


__all__ = ['Client']
