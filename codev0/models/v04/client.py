"""Federated-lifecycle client for Method V0.4."""

from __future__ import annotations

import os
import time

import torch

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from modules.fed_optimizer import (
    FederatedAdamW,
    export_block_second_moment,
    initialize_second_moment_blocks,
    mapping_mean,
    optimizer_state_diagnostics,
)
from models.v04.model import build_v04_model
from models.v04.training import task_loss, train_v04_step
from modules.federated import ClientModule


class Client(ClientModule):
    def __init__(self, args, w_id, g_id, sd):
        super().__init__(args, w_id, g_id, sd)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.seed))
            self.model = build_v04_model(args).cuda(g_id)
        self._initial_model = get_state_dict(self.model)
        self._active_batch = None
        self._build_optimizers()


    def _build_optimizers(self):
        # This is also the persistent fallback used by the one-line local
        # reproduction. O4 replaces only the shared optimizer after a global
        # model and its block-v metadata are accepted.
        self.main_optimizer = torch.optim.Adam(
            self.model.main_parameters(),
            lr=float(self.args.lr_main),
            weight_decay=float(self.args.weight_decay),
        )
        self.field_optimizer = torch.optim.Adam(
            self.model.field_parameters(),
            lr=float(self.args.lr_field),
            weight_decay=float(self.args.weight_decay),
        )
        self.probe_optimizer = torch.optim.Adam(
            self.model.probe_parameters(),
            lr=float(self.args.lr_probe),
            weight_decay=float(self.args.weight_decay),
        )
        self._finish_optimizer_registry()


    def _finish_optimizer_registry(self):
        self.optimizers = {
            'main': self.main_optimizer,
            'field': self.field_optimizer,
            'probe': self.probe_optimizer,
        }
        # Shared runtime compatibility; optimizer moments remain client-local.
        self.optimizer = self.main_optimizer

    def _build_fedadamw_main_optimizer(self, global_v_step):
        self.main_optimizer = FederatedAdamW(
            list(self.model.named_main_parameters()),
            lr=float(self.args.lr_main),
            weight_decay=float(self.args.weight_decay),
            global_v_step=int(global_v_step),
        )

    def switch_state(self, client_id):
        super().switch_state(client_id)
        self._active_batch = None
        for batch in self.loader.pa_loader:
            self._active_batch = batch.cuda(self.gpu_id)
            break
        if self._active_batch is None:
            raise RuntimeError('V0.4 requires one non-empty client graph.')

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
            'field_optimizer': self.field_optimizer.state_dict(),
            'probe_optimizer': self.probe_optimizer.state_dict(),
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
        self.field_optimizer.load_state_dict(loaded['field_optimizer'])
        self.probe_optimizer.load_state_dict(loaded['probe_optimizer'])
        self.best = loaded['best']

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        if self.args.local_only:
            # --local-only reproduces the accepted full-lifecycle local mode
            # without editing source: the client never installs the broadcast
            # model or the O4 optimizer state, so both the local parameters and
            # the persistent Adam trajectory stay continuous. The server still
            # aggregates and broadcasts; those messages are simply ignored.
            return
        self._accept_global_state()

    def _accept_global_state(self, model_state=None, optimizer_meta=None):
        """Install a server model and the O4 block-v optimizer state.

        Personalized algorithms reuse the same O4 protocol while
        supplying a client-specific model state.  The default path remains
        the ordinary global-state broadcast used by FedAvg.
        """

        if model_state is None:
            model_state = self.sd['global']
        set_state_dict(self.model, model_state, self.gpu_id)
        self._prepare_block_v_optimizer_state(optimizer_meta=optimizer_meta)

    def _prepare_block_v_optimizer_state(self, optimizer_meta=None):
        """Rebuild the main optimizer according to ``--fed-optimizer``.

        ``block_v`` is the O4 default. The three plain variants exist to locate
        how much of the federated protocol is actually needed: ``reset_adamw``
        and ``reset_adam`` build a fresh plain optimizer on every receive, while
        ``persistent_adam`` keeps the one built at init, so its moments span the
        whole run and the global model is swapped underneath them.
        """

        if optimizer_meta is None:
            optimizer_meta = self.sd.get('global_optimizer', {})
        meta = dict(optimizer_meta)
        mode = str(getattr(self.args, 'fed_optimizer', 'block_v'))
        global_v_step = 0
        global_v_mean = 0.0

        if mode == 'block_v':
            global_v_step = int(meta.get('global_v_step', 0))
            self._build_fedadamw_main_optimizer(global_v_step=global_v_step)
            named_main = list(self.model.named_main_parameters())
            global_v = meta.get('main_v_blocks', {})
            initialize_second_moment_blocks(
                self.main_optimizer, named_main, global_v
            )
            global_v_mean = mapping_mean(global_v)
        elif mode == 'persistent_adam':
            pass  # keep the optimizer and moments created at init
        else:
            optimizer_class = (
                torch.optim.AdamW if mode == 'reset_adamw' else torch.optim.Adam
            )
            self.main_optimizer = optimizer_class(
                self.model.main_parameters(),
                lr=float(self.args.lr_main),
                weight_decay=float(self.args.weight_decay),
            )

        # The field and probe are auxiliary training objects. They receive no
        # stale moments and do not participate in FedAdamW-v aggregation.
        self.field_optimizer = torch.optim.Adam(
            self.model.field_parameters(),
            lr=float(self.args.lr_field),
            weight_decay=float(self.args.weight_decay),
        )
        self.probe_optimizer = torch.optim.Adam(
            self.model.probe_parameters(),
            lr=float(self.args.lr_probe),
            weight_decay=float(self.args.weight_decay),
        )
        self._finish_optimizer_registry()
        self._global_optimizer_diagnostics = {
            'global_v_step': float(global_v_step),
            'global_v_block_mean': float(global_v_mean),
        }

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

    def _train_step(self):
        diagnostics = train_v04_step(
            self.model,
            self._batch(),
            self.optimizers,
            self.args,
            self.curr_rnd,
        )
        self._last_diagnostics = {
            name: float(value.detach().cpu().item())
            for name, value in diagnostics.items()
        }
        self._last_diagnostics.update(
            optimizer_state_diagnostics(self.main_optimizer)
        )
        self._last_diagnostics.update(
            getattr(
                self,
                '_global_optimizer_diagnostics',
                {
                    'global_v_step': 0.0,
                    'global_v_block_mean': 0.0,
                },
            )
        )

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
        for _ in range(int(self.args.n_eps)):
            self._train_step()
        val_metric, val_loss, _ = self.validate(mode='valid')
        test_metric, test_loss, test_f1 = self.validate(mode='test')
        self._maybe_save_best(val_metric, test_metric, test_f1)
        self._round_result = {
            **self._last_diagnostics,
            'val_loss': val_loss,
            'val_accuracy': val_metric,
            'test_loss': test_loss,
            'test_accuracy': test_metric,
            'test_f1': test_f1,
            'elapsed_seconds': float(time.time() - started),
        }


    def transfer_to_server(self):
        weights = get_state_dict(self.model)
        payload = {
            'client_id': int(self.client_id),
            'model': weights,
            'train_size': int(self._batch().train_mask.sum().item()),
            'upload_bytes': int(sum(value.nbytes for value in weights.values())),
            **self._round_result,
        }
        # Only the O4 protocol exchanges second moments; the plain variants
        # upload nothing, and the server's block-v aggregation then sees an
        # empty mapping for every client.
        main_v_blocks = {}
        if str(getattr(self.args, 'fed_optimizer', 'block_v')) == 'block_v':
            main_v_blocks = export_block_second_moment(
                self.main_optimizer, list(self.model.named_main_parameters())
            )
        payload['main_v_blocks'] = main_v_blocks
        payload['upload_bytes'] += 8 * len(main_v_blocks)
        self.sd[self.client_id] = payload
