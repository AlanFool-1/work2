"""Local A-DGN backbone with a private ResKoop K fitted on its rollout."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from models.feddgn.reskoop import (
    KOOPMAN_DICTIONARY_PROTOCOL,
    build_passive_reskoop,
)
from models.feddgn_local.client import Client as LocalFedDGNClient
from models.feddgn_local.training import train_feddgn_step


def _state_relative_delta(left, right):
    numerator = 0.0
    denominator = 0.0
    for name in sorted(left):
        first = np.asarray(left[name], dtype=np.float64)
        second = np.asarray(right[name], dtype=np.float64)
        numerator += float(np.square(first - second).sum())
        denominator += float(np.square(second).sum())
    return float(
        np.sqrt(numerator)
        / max(np.sqrt(denominator), np.finfo(np.float64).eps)
    )


class Client(LocalFedDGNClient):
    """Only the observable dictionary is federated; task dynamics stay local.

    K is fitted on this client's own rollout and never leaves the client. The
    private observable snapshots behind it are not uploaded either.
    """

    def __init__(self, args, w_id, g_id, sd):
        super().__init__(args, w_id, g_id, sd)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.seed) + 314159)
            self.koopman = build_passive_reskoop(args).cuda(g_id)
        self._initial_koopman = get_state_dict(self.koopman.dictionary)
        self._build_koopman_optimizer()
        self._client_round_started = None
        self._reset_private_koopman_state()

    def switch_state(self, client_id):
        self._client_round_started = time.perf_counter()
        super().switch_state(client_id)

    @staticmethod
    def _empty_dictionary_receive_diagnostics():
        return {
            'koopman_dictionary_received': 0.0,
            'koopman_dictionary_optimizer_reset': 0.0,
            'koopman_dictionary_pre_receive_delta': 0.0,
            'koopman_dictionary_post_receive_delta': 0.0,
            'koopman_dictionary_source_round': 0.0,
        }

    def _reset_private_koopman_state(self):
        self.local_koopman_round = 0
        self.local_operator_col = None
        self.koopman_dictionary_source_round = None
        self._dictionary_receive_diagnostics = (
            self._empty_dictionary_receive_diagnostics()
        )
        self._dictionary_receive_seconds = 0.0

    def _build_koopman_optimizer(self):
        parameters = [
            parameter for parameter in self.koopman.dictionary.parameters()
            if parameter.requires_grad
        ]
        self.koopman_optimizer = (
            torch.optim.Adam(
                parameters,
                lr=float(self.args.feddgn_koopman_lr),
                weight_decay=float(self.args.feddgn_koopman_weight_decay),
            ) if parameters else None
        )

    def init_state(self):
        super().init_state()
        set_state_dict(self.koopman.dictionary, self._initial_koopman, self.gpu_id)
        self._build_koopman_optimizer()
        self._reset_private_koopman_state()

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload.update({
            'koopman_dictionary': get_state_dict(self.koopman.dictionary),
            'koopman_optimizer': (
                None if self.koopman_optimizer is None
                else self.koopman_optimizer.state_dict()
            ),
            'koopman_dictionary_protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'koopman_dictionary_source_round': self.koopman_dictionary_source_round,
            'local_koopman_round': int(self.local_koopman_round),
            'local_operator_col': self.local_operator_col,
        })
        return payload

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, self._checkpoint_filename())
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.best = loaded['best']
        set_state_dict(
            self.koopman.dictionary, loaded['koopman_dictionary'], self.gpu_id
        )
        self._build_koopman_optimizer()
        optimizer_state = loaded.get('koopman_optimizer')
        if self.koopman_optimizer is not None and optimizer_state is not None:
            self.koopman_optimizer.load_state_dict(optimizer_state)
        self.koopman_dictionary_source_round = loaded.get(
            'koopman_dictionary_source_round'
        )
        self.local_koopman_round = int(loaded.get('local_koopman_round', 0))
        self.local_operator_col = loaded.get('local_operator_col')
        self._dictionary_receive_diagnostics = (
            self._empty_dictionary_receive_diagnostics()
        )
        self._dictionary_receive_seconds = 0.0

    def _receive_global_dictionary(self, curr_rnd):
        self._dictionary_receive_diagnostics = (
            self._empty_dictionary_receive_diagnostics()
        )
        self._dictionary_receive_seconds = 0.0
        expected_round = int(curr_rnd)
        if expected_round == 0:
            return
        started = time.perf_counter()
        message = self.sd['global_koopman_dictionary']
        global_state = dict(message['state_dict'])
        pre_delta = _state_relative_delta(
            get_state_dict(self.koopman.dictionary), global_state
        )
        set_state_dict(self.koopman.dictionary, global_state, self.gpu_id)
        post_delta = _state_relative_delta(
            get_state_dict(self.koopman.dictionary), global_state
        )
        self._build_koopman_optimizer()  # fresh Adam after dictionary FedAvg
        self.koopman_dictionary_source_round = expected_round
        self._dictionary_receive_diagnostics = {
            'koopman_dictionary_received': 1.0,
            'koopman_dictionary_optimizer_reset': 1.0,
            'koopman_dictionary_pre_receive_delta': float(pre_delta),
            'koopman_dictionary_post_receive_delta': float(post_delta),
            'koopman_dictionary_source_round': float(expected_round),
        }
        self._dictionary_receive_seconds = time.perf_counter() - started

    def on_receive_message(self, curr_rnd):
        super().on_receive_message(curr_rnd)
        self._receive_global_dictionary(curr_rnd)

    def _fit_koopman(self):
        with torch.no_grad():
            trajectory = self.model.forward_with_aux(self._batch()).trajectory
        result = self.koopman.fit_trajectory(
            trajectory,
            self.koopman_optimizer,
            outer_steps=int(self.args.feddgn_koopman_outer_steps),
            inner_steps=int(self.args.feddgn_koopman_inner_steps),
            batch_size=int(self.args.feddgn_koopman_batch_size),
            grad_clip=float(self.args.feddgn_koopman_grad_clip),
            sample_seed=(
                int(self.args.seed) + int(self.curr_rnd) * 1000
                + int(self.client_id)
            ),
        )
        return result, trajectory

    def _save_koopman_numeric_snapshot(self, result):
        round_number = self.curr_rnd + 1
        if (
            round_number != int(self.args.n_rnds)
            and round_number % int(self.args.feddgn_koopman_save_every) != 0
        ):
            return
        root = os.path.join(
            self.args.log_path, 'koopman', f'client_{self.client_id}'
        )
        torch_save(root, 'latest_K.pt', {
            'round': int(round_number),
            'K_column': self.local_operator_col,
            'eigenvalues': result['eigenvalues'].detach().cpu(),
            'mode_residuals': result['mode_residuals'].detach().cpu(),
        })

    def train(self):
        task_started = time.perf_counter()
        diagnostics = None
        for _ in range(int(self.args.n_eps)):
            diagnostics = train_feddgn_step(
                self.model, self._batch(), self.optimizer, self.args
            )
        val_metric, val_loss, _ = self.validate(mode='valid')
        test_metric, test_loss, test_f1 = self.validate(mode='test')
        self._maybe_save_best(val_metric, test_metric, test_f1)
        task_seconds = time.perf_counter() - task_started
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
            'task_seconds': float(task_seconds),
            'dictionary_receive_seconds': float(
                self._dictionary_receive_seconds
            ),
        }

        reskoop_started = time.perf_counter()
        result, trajectory = self._fit_koopman()
        fit_seconds = time.perf_counter() - reskoop_started
        self.local_operator_col = (
            result['operator'].transpose(0, 1).detach().cpu().numpy().copy()
        )
        self.local_koopman_round = int(self.curr_rnd + 1)
        self._save_koopman_numeric_snapshot(result)
        # Refresh the paired best-validation checkpoint with this round's K.
        if int(self.best['round']) == self.local_koopman_round:
            torch_save(
                self.args.checkpt_path,
                f'{self.client_id}_state.pt',
                self._checkpoint_payload(),
            )
        self._round_result.update({
            name: float(value.detach().cpu().item())
            for name, value in result['diagnostics'].items()
        })
        self._round_result.update(self._dictionary_receive_diagnostics)
        self._round_result['koopman_seconds'] = float(fit_seconds)
        self._round_result['reskoop_seconds'] = float(
            time.perf_counter() - reskoop_started
        )

    def _upload_payload(self):
        # No task model and no task optimizer leave this client.
        dictionary = get_state_dict(self.koopman.dictionary)
        dictionary_bytes = int(sum(value.nbytes for value in dictionary.values()))
        return {
            'client_id': int(self.client_id),
            'train_size': int(self._batch().train_mask.sum().item()),
            'upload_bytes': dictionary_bytes,
            'koopman_dictionary_protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'koopman_dictionary': dictionary,
            'koopman_dictionary_bytes': dictionary_bytes,
            'client_total_seconds': float(
                time.perf_counter() - self._client_round_started
            ),
            **self._round_result,
        }


__all__ = ['Client', 'KOOPMAN_DICTIONARY_PROTOCOL']
