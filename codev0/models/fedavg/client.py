"""FedAvg client with the reference two-layer GCN and O4 block-v.

DELIBERATE DEVIATION FROM THE REFERENCE
---------------------------------------
The reference implementation (``Fedrated/models/fedavg/client.py``) trains with
plain ``torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)``,
created once in ``init_state()`` and never rebuilt, with per-element second
moments and no optimizer communication.

This client instead runs the O4 ``FederatedAdamW`` protocol: every broadcast
rebuilds the optimizer around the server's block second moments
(``_build_o4_optimizer``), so the first moment is round-local, all elements of a
parameter share one scalar second moment, weight decay is decoupled, the second
moment's bias correction uses the global step, and ``main_v_blocks`` are
uploaded. Runs of ``--model fedavg`` therefore measure "FedAvg under the
proposed optimizer" rather than the paper's FedAvg baseline. ``local`` overrides
both hooks and keeps the reference's plain persistent Adam.

The data pipeline, model backbone, loss, broadcast and aggregation paths are
otherwise faithful; see ``models/nets.py`` and ``modules/fed_server.py``.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from models.nets import GCN
from modules.fed_optimizer import (
    FederatedAdamW,
    export_block_second_moment,
    initialize_second_moment_blocks,
    mapping_mean,
    optimizer_state_diagnostics,
)
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
            self.model = self.build_model().cuda(g_id)
        self._initial_model = get_state_dict(self.model)
        self._active_batch = None
        self._build_local_optimizer()

    def build_model(self):
        return GCN(
            self.args.n_feat,
            self.args.n_dims,
            self.args.n_clss,
            self.args,
        )

    def _build_local_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.args.base_lr),
            weight_decay=float(self.args.weight_decay),
        )

    def _build_o4_optimizer(self, global_v_step, block_v):
        named_parameters = list(self.model.named_parameters())
        self.optimizer = FederatedAdamW(
            named_parameters,
            lr=float(self.args.base_lr),
            weight_decay=float(self.args.weight_decay),
            global_v_step=int(global_v_step),
        )
        initialize_second_moment_blocks(
            self.optimizer, named_parameters, block_v
        )
        self._global_optimizer_diagnostics = {
            'global_v_step': float(global_v_step),
            'global_v_block_mean': float(mapping_mean(block_v)),
        }

    def switch_state(self, client_id):
        super().switch_state(client_id)
        self._active_batch = None
        for batch in self.loader.pa_loader:
            self._active_batch = batch.cuda(self.gpu_id)
            break
        if self._active_batch is None:
            raise RuntimeError('Reference client requires a non-empty graph.')

    def init_state(self):
        set_state_dict(self.model, self._initial_model, self.gpu_id)
        self._build_local_optimizer()
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
        loaded = torch_load(
            self.args.checkpt_path, self._checkpoint_filename()
        )
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.best = loaded['best']

    def on_receive_message(self, curr_rnd):
        self._accept_model_state(curr_rnd, self.sd['global'])

    def _install_model_state(self, curr_rnd, model_state, *, preserve_masks=False):
        self.curr_rnd = int(curr_rnd)
        set_state_dict(
            self.model,
            model_state,
            self.gpu_id,
            skip_stat=True,
            skip_mask=preserve_masks,
        )

    def _accept_model_state(self, curr_rnd, model_state, *, preserve_masks=False):
        self._install_model_state(
            curr_rnd, model_state, preserve_masks=preserve_masks
        )
        optimizer_meta = dict(self.sd.get('global_optimizer', {}))
        self._build_o4_optimizer(
            optimizer_meta.get('global_v_step', 0),
            optimizer_meta.get('main_v_blocks', {}),
        )

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
        logits = self.model(batch)
        mask = batch.test_mask if mode == 'test' else batch.val_mask
        if int(mask.sum().item()) == 0:
            return 0.0, 0.0, 0.0
        loss = task_loss(logits, batch.y, mask, self.args.dataset)
        metric = self.accuracy(logits[mask], batch.y[mask])
        f1 = self.f1(logits[mask], batch.y[mask])
        return float(metric), float(loss.item()), float(f1)

    def regularization_loss(self):
        return 0.0

    def train(self):
        started = time.time()
        train_loss = 0.0
        for _ in range(int(self.args.n_eps)):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            batch = self._batch()
            logits = self.model(batch)
            loss = task_loss(
                logits, batch.y, batch.train_mask, self.args.dataset
            )
            loss = loss + self.regularization_loss()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite client training loss.')
            loss.backward()
            self.optimizer.step()
            train_loss = float(loss.detach().cpu().item())

        val_metric, val_loss, _ = self.validate(mode='valid')
        test_metric, test_loss, test_f1 = self.validate(mode='test')
        self._maybe_save_best(val_metric, test_metric, test_f1)
        diagnostics = optimizer_state_diagnostics(self.optimizer)
        diagnostics.update(getattr(self, '_global_optimizer_diagnostics', {}))
        self._round_result = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_metric,
            'test_loss': test_loss,
            'test_accuracy': test_metric,
            'test_f1': test_f1,
            'elapsed_seconds': float(time.time() - started),
            **diagnostics,
        }

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

    def optimizer_state_upload(self):
        block_v = export_block_second_moment(
            self.optimizer, list(self.model.named_parameters())
        )
        return {'main_v_blocks': block_v}, 8 * len(block_v)

    def extra_upload(self):
        return {}, 0

    def transfer_to_server(self):
        weights = get_state_dict(self.model)
        optimizer_payload, optimizer_bytes = self.optimizer_state_upload()
        extra_payload, extra_bytes = self.extra_upload()
        payload = {
            'client_id': int(self.client_id),
            'model': weights,
            'train_size': int(self._batch().train_mask.sum().item()),
            'upload_bytes': int(
                sum(value.nbytes for value in weights.values())
                + optimizer_bytes
                + extra_bytes
            ),
            **self._round_result,
            **optimizer_payload,
            **extra_payload,
        }
        self.sd[self.client_id] = payload
