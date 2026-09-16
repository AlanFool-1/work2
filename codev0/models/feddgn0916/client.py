"""Local A-DGN, private ResKoop K, and passive personalized K-star."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from models.feddgn0916.candidate_pipeline import CANDIDATE_PROTOCOL, make_koopman_packet
from models.feddgn0916.personalized_koopman import (
    empty_personalized_diagnostics,
    fit_personalized_koopman,
)
from models.feddgn0916.reskoop import (
    KOOPMAN_DICTIONARY_PROTOCOL,
    build_passive_reskoop,
)
from models.feddgn_local.client import Client as LocalFedDGNClient
from models.feddgn_local.training import train_feddgn_step


def _state_relative_delta(left, right):
    if set(left) != set(right):
        raise ValueError('Koopman dictionary state keys do not match.')
    numerator = 0.0
    denominator = 0.0
    for name in sorted(left):
        first = np.asarray(left[name], dtype=np.float64)
        second = np.asarray(right[name], dtype=np.float64)
        if first.shape != second.shape:
            raise ValueError(f'Koopman dictionary tensor {name} has incompatible shapes.')
        numerator += float(np.square(first - second).sum())
        denominator += float(np.square(second).sum())
    return float(
        np.sqrt(numerator)
        / max(np.sqrt(denominator), np.finfo(np.float64).eps)
    )


class Client(LocalFedDGNClient):
    """Only the observable dictionary is federated; task dynamics stay local.

    K-star is computed in the previous private K/Z/Z+ coordinate before the
    round-r global dictionary is received. It is saved for diagnosis only;
    task training and ResKoop rollout never consume it.
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
        self.need_snapshots = None
        self.need_next_snapshots = None
        self._latest_koopman_packet = None
        self._personalized_result = None
        self.koopman_dictionary_source_round = None
        self._dictionary_receive_diagnostics = (
            self._empty_dictionary_receive_diagnostics()
        )
        self._personalized_receive_diagnostics = (
            empty_personalized_diagnostics()
        )
        self._kstar_seconds = 0.0
        self._kstar_compute_seconds = 0.0
        self._kstar_visualization_seconds = 0.0
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
            'candidate_protocol': CANDIDATE_PROTOCOL,
            'local_koopman_round': int(self.local_koopman_round),
            'local_operator_col': self.local_operator_col,
            # Private observable snapshots are never sent to the server.
            'need_snapshots': self.need_snapshots,
            'need_next_snapshots': self.need_next_snapshots,
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
        self.need_snapshots = loaded.get('need_snapshots')
        self.need_next_snapshots = loaded.get('need_next_snapshots')
        self._latest_koopman_packet = None
        self._personalized_result = None
        self._dictionary_receive_diagnostics = (
            self._empty_dictionary_receive_diagnostics()
        )
        self._personalized_receive_diagnostics = (
            empty_personalized_diagnostics()
        )
        self._kstar_seconds = 0.0
        self._kstar_compute_seconds = 0.0
        self._kstar_visualization_seconds = 0.0
        self._dictionary_receive_seconds = 0.0

    def _receive_candidate_package(self, curr_rnd):
        self._personalized_receive_diagnostics = (
            empty_personalized_diagnostics()
        )
        self._kstar_seconds = 0.0
        self._kstar_compute_seconds = 0.0
        self._kstar_visualization_seconds = 0.0
        source_round = int(curr_rnd)
        if source_round == 0 or self.local_koopman_round != source_round:
            # Never use a package without matching private source-round data.
            return
        started = time.perf_counter()
        package = self.sd.get(f'candidate_package:{int(self.client_id)}')
        if (
            package is None
            or package.get('protocol') != CANDIDATE_PROTOCOL
            or int(package.get('source_round', -1)) != source_round
            or int(package.get('client_id', -1)) != int(self.client_id)
        ):
            raise RuntimeError(
                'Missing or stale personalized Koopman candidate package '
                f'for client {self.client_id}, source round {source_round}.'
            )
        if any(value is None for value in (
            self.local_operator_col, self.need_snapshots,
            self.need_next_snapshots,
        )):
            raise RuntimeError('Private Koopman snapshots are missing.')
        compute_started = time.perf_counter()
        result = fit_personalized_koopman(
            self.local_operator_col,
            self.need_snapshots,
            self.need_next_snapshots,
            dict(package),
            eta=float(self.args.feddgn_candidate_eta),
        )
        self._kstar_compute_seconds = time.perf_counter() - compute_started
        result['task_feedback'] = False
        self._personalized_result = result
        self._personalized_receive_diagnostics = dict(result['stats'])
        self._save_personalized_artifacts(result)
        self._kstar_seconds = time.perf_counter() - started

    def _receive_global_dictionary(self, curr_rnd):
        self._dictionary_receive_diagnostics = (
            self._empty_dictionary_receive_diagnostics()
        )
        self._dictionary_receive_seconds = 0.0
        expected_round = int(curr_rnd)
        if expected_round == 0:
            return
        started = time.perf_counter()
        message = self.sd.get('global_koopman_dictionary')
        if message is None:
            raise RuntimeError('Global Koopman dictionary is missing.')
        if (
            message.get('protocol') != KOOPMAN_DICTIONARY_PROTOCOL
            or int(message.get('source_round', -1)) != expected_round
        ):
            raise RuntimeError(
                'Received a stale or incompatible global dictionary: '
                f'expected source round {expected_round}.'
            )
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
        self._latest_koopman_packet = None
        self._receive_candidate_package(curr_rnd)
        # K-star must be reconstructed before this coordinate update.
        self._receive_global_dictionary(curr_rnd)

    def _save_personalized_artifacts(self, result):
        root = os.path.join(
            self.args.log_path, 'koopman', f'client_{self.client_id}',
            'personalized',
        )
        torch_save(root, 'latest.pt', result)
        dashboard = None
        history = None
        knowledge_flow = None
        source_round = int(result['source_round'])
        target_round = source_round + 1
        from models.feddgn0916.kstar_visualize import (
            append_personalized_history, render_client_history,
        )
        history_path = os.path.join(root, 'history.csv')
        append_personalized_history(result, history_path)
        visualize_every = int(self.args.feddgn_kstar_visualize_every)
        should_visualize = (
            target_round == int(self.args.n_rnds)
            or target_round % visualize_every == 0
        )
        if should_visualize:
            started = time.perf_counter()
            knowledge_flow = render_client_history(
                history_path,
                os.path.join(root, 'figures'),
                dpi=int(self.args.feddgn_kstar_visualization_dpi),
            )
            from models.feddgn0916.kstar_visualize import render_kstar_client
            report = render_kstar_client(
                result, os.path.join(root, 'figures'),
                dpi=int(self.args.feddgn_kstar_visualization_dpi),
            )
            self._kstar_visualization_seconds = (
                time.perf_counter() - started
            )
            dashboard = report['dashboard']
        else:
            latest_dashboard = os.path.join(root, 'figures', 'latest.html')
            latest_flow = os.path.join(root, 'figures', 'knowledge_flow.png')
            dashboard = (
                os.path.abspath(latest_dashboard)
                if os.path.exists(latest_dashboard) else None
            )
            knowledge_flow = (
                os.path.abspath(latest_flow)
                if os.path.exists(latest_flow) else None
            )
        history = str(os.path.abspath(history_path))
        self._write_json(os.path.join(root, 'latest.json'), {
            'protocol': CANDIDATE_PROTOCOL,
            'client_id': int(self.client_id),
            'source_round': int(result['source_round']),
            'reference_client_id': int(result['reference_client_id']),
            'A_condition': float(result['A_condition']),
            'task_feedback': False,
            'visualization_dashboard': dashboard,
            'visualization_knowledge_flow': knowledge_flow,
            'history_csv': history,
            'stats': result['stats'],
            'candidate_rows': result['candidate_rows'],
        })

    @staticmethod
    def _write_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)

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

    def _private_observable_snapshots(self, trajectory):
        current, following = self.koopman.snapshot_pairs(trajectory)
        limit = int(self.args.feddgn_candidate_need_snapshots)
        if current.size(0) > limit:
            indices = torch.linspace(
                0, current.size(0) - 1, steps=limit,
                device=current.device,
            ).round().long()
            current = current.index_select(0, indices)
            following = following.index_select(0, indices)
        with torch.no_grad():
            z = self.koopman.dictionary(current).detach().cpu().numpy().T.copy()
            z_next = (
                self.koopman.dictionary(following)
                .detach().cpu().numpy().T.copy()
            )
        return z, z_next

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
            'kstar_seconds': float(self._kstar_seconds),
            'kstar_compute_seconds': float(self._kstar_compute_seconds),
            'kstar_visualization_seconds': float(
                self._kstar_visualization_seconds
            ),
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
        self.need_snapshots, self.need_next_snapshots = (
            self._private_observable_snapshots(trajectory)
        )
        self.local_koopman_round = int(self.curr_rnd + 1)
        self._latest_koopman_packet = make_koopman_packet(
            result,
            client_id=int(self.client_id),
            round_number=int(self.local_koopman_round),
        )
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
        self._round_result.update(self._personalized_receive_diagnostics)
        self._round_result['koopman_seconds'] = float(fit_seconds)
        self._round_result['reskoop_seconds'] = float(
            time.perf_counter() - reskoop_started
        )

    def _upload_payload(self):
        # No task model, task optimizer, or K-star feedback leaves this client.
        dictionary = get_state_dict(self.koopman.dictionary)
        dictionary_bytes = int(sum(value.nbytes for value in dictionary.values()))
        if self._latest_koopman_packet is None:
            raise RuntimeError('Local Koopman packet was not prepared.')
        packet_bytes = int(self._latest_koopman_packet['bytes'])
        if self._client_round_started is None:
            raise RuntimeError('Client round timing was not started.')
        return {
            'client_id': int(self.client_id),
            'train_size': int(self._batch().train_mask.sum().item()),
            'upload_bytes': dictionary_bytes + packet_bytes,
            'koopman_dictionary_protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'koopman_dictionary': dictionary,
            'koopman_dictionary_bytes': dictionary_bytes,
            'koopman_packet': self._latest_koopman_packet,
            'koopman_upload_bytes': packet_bytes,
            'client_total_seconds': float(
                time.perf_counter() - self._client_round_started
            ),
            **self._round_result,
        }


__all__ = ['Client', 'KOOPMAN_DICTIONARY_PROTOCOL']
