"""FedAvg/O4 FedDGN client with a federated ResKoop dictionary."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from misc.utils import get_state_dict, set_state_dict, torch_load, torch_save
from models.feddgn.mode_graph import MODE_GRAPH_PROTOCOL, make_mode_upload
from models.feddgn.reskoop import (
    KOOPMAN_DICTIONARY_PROTOCOL,
    build_passive_reskoop,
)
from models.feddgn_avg.client import Client as FedAvgFedDGNClient


def _state_relative_delta(left, right):
    if set(left) != set(right):
        raise ValueError('Koopman dictionary state keys do not match.')
    numerator = 0.0
    denominator = 0.0
    for name in sorted(left):
        first = np.asarray(left[name], dtype=np.float64)
        second = np.asarray(right[name], dtype=np.float64)
        if first.shape != second.shape:
            raise ValueError(
                f'Koopman dictionary tensor {name} has incompatible shapes.'
            )
        numerator += float(np.square(first - second).sum())
        denominator += float(np.square(second).sum())
    return float(
        np.sqrt(numerator)
        / max(np.sqrt(denominator), np.finfo(np.float64).eps)
    )


class Client(FedAvgFedDGNClient):
    """FedAvg task dynamics with a private passive Koopman analysis branch.

    Every round installs the server-averaged A-DGN parameters through the
    ``feddgn_avg`` lifecycle.  The trainable ResKoop dictionary is averaged
    separately and its Adam state is reset after every global receive.  Each
    client still identifies its own analytic K from a detached local trajectory.
    Returned Functional Maps and mode banks remain analysis-only.
    """

    def __init__(self, args, w_id, g_id, sd):
        super().__init__(args, w_id, g_id, sd)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.seed) + 314159)
            self.koopman = build_passive_reskoop(args).cuda(g_id)
        self._initial_koopman = get_state_dict(self.koopman.dictionary)
        self._build_koopman_optimizer()
        self.personalized_mode_view = None
        self.global_mode_graph_view = None
        self._latest_mode_upload = None
        self.koopman_dictionary_source_round = None
        self._dictionary_receive_diagnostics = self._empty_receive_diagnostics()

    @staticmethod
    def _empty_receive_diagnostics():
        return {
            'koopman_dictionary_received': 0.0,
            'koopman_dictionary_optimizer_reset': 0.0,
            'koopman_dictionary_pre_receive_delta': 0.0,
            'koopman_dictionary_post_receive_delta': 0.0,
            'koopman_dictionary_source_round': 0.0,
        }

    def _build_koopman_optimizer(self):
        parameters = [
            parameter
            for parameter in self.koopman.dictionary.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            self.koopman_optimizer = None
            return
        self.koopman_optimizer = torch.optim.Adam(
            parameters,
            lr=float(self.args.feddgn_koopman_lr),
            weight_decay=float(self.args.feddgn_koopman_weight_decay),
        )

    def init_state(self):
        super().init_state()
        set_state_dict(
            self.koopman.dictionary, self._initial_koopman, self.gpu_id
        )
        self._build_koopman_optimizer()
        self.personalized_mode_view = None
        self.global_mode_graph_view = None
        self._latest_mode_upload = None
        self.koopman_dictionary_source_round = None
        self._dictionary_receive_diagnostics = self._empty_receive_diagnostics()

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload.update({
            'koopman_dictionary': get_state_dict(self.koopman.dictionary),
            'koopman_optimizer': (
                None
                if self.koopman_optimizer is None
                else self.koopman_optimizer.state_dict()
            ),
            'koopman_protocol': MODE_GRAPH_PROTOCOL,
            'koopman_dictionary_protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'koopman_dictionary_source_round': (
                self.koopman_dictionary_source_round
            ),
            'personalized_mode_view': self.personalized_mode_view,
        })
        return payload

    def load_state(self):
        loaded = torch_load(
            self.args.checkpt_path, self._checkpoint_filename()
        )
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.best = loaded['best']
        set_state_dict(
            self.koopman.dictionary,
            loaded['koopman_dictionary'],
            self.gpu_id,
        )
        self._build_koopman_optimizer()
        optimizer_state = loaded.get('koopman_optimizer')
        if self.koopman_optimizer is not None and optimizer_state is not None:
            self.koopman_optimizer.load_state_dict(optimizer_state)
        self.personalized_mode_view = loaded.get('personalized_mode_view')
        self.koopman_dictionary_source_round = loaded.get(
            'koopman_dictionary_source_round'
        )
        self._dictionary_receive_diagnostics = self._empty_receive_diagnostics()
        self.global_mode_graph_view = None
        self._latest_mode_upload = None

    def _receive_global_koopman_dictionary(self, curr_rnd):
        self._dictionary_receive_diagnostics = self._empty_receive_diagnostics()
        expected_round = int(curr_rnd)
        if expected_round == 0:
            return
        message = self.sd.get('global_koopman_dictionary')
        if message is None:
            raise RuntimeError(
                'Global Koopman dictionary is missing after round 1.'
            )
        if (
            str(message.get('protocol')) != KOOPMAN_DICTIONARY_PROTOCOL
            or int(message.get('source_round', -1)) != expected_round
        ):
            raise RuntimeError(
                'Received a stale or incompatible Koopman dictionary: '
                f'expected round {expected_round}.'
            )
        global_state = dict(message['state_dict'])
        local_state = get_state_dict(self.koopman.dictionary)
        pre_delta = _state_relative_delta(local_state, global_state)
        set_state_dict(
            self.koopman.dictionary, global_state, self.gpu_id
        )
        post_delta = _state_relative_delta(
            get_state_dict(self.koopman.dictionary), global_state
        )
        # The parameter point changed from a private local dictionary to the
        # global average.  Old Adam moments no longer belong to that point.
        self._build_koopman_optimizer()
        self.koopman_dictionary_source_round = expected_round
        self._dictionary_receive_diagnostics = {
            'koopman_dictionary_received': 1.0,
            'koopman_dictionary_optimizer_reset': 1.0,
            'koopman_dictionary_pre_receive_delta': float(pre_delta),
            'koopman_dictionary_post_receive_delta': float(post_delta),
            'koopman_dictionary_source_round': float(expected_round),
        }

    def on_receive_message(self, curr_rnd):
        super().on_receive_message(curr_rnd)
        self._latest_mode_upload = None
        self._receive_global_koopman_dictionary(curr_rnd)
        if not bool(self.args.feddgn_mode_graph_enabled):
            return

        # Graph round r is built after task round r and is therefore consumed
        # at the beginning of zero-based training round r.  Reject stale views
        # explicitly so partial client participation cannot be mistaken for a
        # fresh personalized response.
        expected_graph_round = int(curr_rnd)
        view = self.sd.get(f'mode_graph_view:{int(self.client_id)}')
        if (
            view is None
            or int(view.get('round', -1)) != expected_graph_round
            or str(view.get('protocol')) != MODE_GRAPH_PROTOCOL
        ):
            return
        self.personalized_mode_view = dict(view)
        global_view = self.sd.get('mode_graph_global')
        if (
            global_view is not None
            and int(global_view.get('round', -1)) == expected_graph_round
        ):
            self.global_mode_graph_view = dict(global_view)
        self._save_received_mode_view()

    def _should_save_koopman(self) -> bool:
        round_number = self.curr_rnd + 1
        return (
            round_number == int(self.args.n_rnds)
            or round_number % int(self.args.feddgn_koopman_save_every) == 0
        )

    @staticmethod
    def _write_json(path, payload):
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)

    def _save_koopman_artifacts(self, result):
        client_dir = os.path.join(
            self.args.log_path, 'koopman', f'client_{self.client_id}'
        )
        tensors = {
            'K.pt': result['operator'],
            'eigenvalues.pt': result['eigenvalues'],
            'eigenvectors.pt': result['eigenvectors'],
            'residual.pt': result['mode_residuals'],
            'modes.pt': result['modes'],
            'dictionary.pt': {
                name: value.detach().cpu()
                for name, value in self.koopman.dictionary.state_dict().items()
            },
        }
        for filename, value in tensors.items():
            if torch.is_tensor(value):
                value = value.detach().cpu()
            torch_save(client_dir, filename, value)

        diagnostics = {
            name: float(value.detach().cpu().item())
            for name, value in result['diagnostics'].items()
        }
        self._write_json(
            os.path.join(client_dir, 'metadata.json'),
            {
                'client_id': int(self.client_id),
                'round': int(self.curr_rnd + 1),
                'protocol': MODE_GRAPH_PROTOCOL,
                'state_dim': int(self.koopman.state_dim),
                'observable_dim': int(self.koopman.observable_dim),
                'trajectory_steps': int(self.args.feddgn_steps),
                'step_size': float(self.args.feddgn_step_size),
                'diagnostics': diagnostics,
            },
        )

    def _save_received_mode_view(self):
        client_dir = os.path.join(
            self.args.log_path, 'koopman', f'client_{self.client_id}'
        )
        torch_save(
            client_dir,
            'personalized_modes.pt',
            self.personalized_mode_view,
        )
        view = self.personalized_mode_view
        values = view['eigenvalues']
        map_report = view['functional_map_diagnostics']
        global_bank = view['global_mode_bank']
        summary = {
            'protocol': str(view['protocol']),
            'graph_round': int(view['round']),
            'client_id': int(view['client_id']),
            'comparison_space': str(view['comparison_space']),
            'reference_client_id': int(view['reference_client_id']),
            'functional_map': {
                'shape': list(view['functional_map'].shape),
                'intertwining_residual': float(
                    map_report['intertwining_residual']
                ),
                'orthogonality_error': float(
                    map_report['orthogonality_error']
                ),
                'initialization': str(map_report['initialization']),
                'accepted_refinement_steps': int(
                    map_report['accepted_refinement_steps']
                ),
            },
            'global_mode_bank': {
                'community_ids': [
                    int(value) for value in global_bank['community_ids']
                ],
                'member_counts': [
                    int(value) for value in global_bank['member_counts']
                ],
                'representative_mode_shape': list(
                    global_bank['representative_modes'].shape
                ),
            },
            'own_count': int(view['own_count']),
            'related_count': int(view['related_count']),
            'own_mode_ids': list(view['own_mode_ids']),
            'related_mode_ids': list(view['related_mode_ids']),
            'source_client_ids': [
                int(value) for value in view['source_client_ids']
            ],
            'community_ids': [
                int(value) for value in view['community_ids']
            ],
            'eigenvalues': [
                {'real': float(value.real), 'imag': float(value.imag)}
                for value in values
            ],
            'residuals': [float(value) for value in view['residuals']],
            'related_similarity': [
                float(value) for value in view['related_similarity']
            ],
            'task_feedback': False,
        }
        self._write_json(
            os.path.join(client_dir, 'personalized_modes.json'), summary
        )

    def _fit_koopman(self):
        # Recompute a post-task-update trajectory. no_grad plus the explicit
        # detach in snapshot_pairs makes the one-way dependency auditable.
        with torch.no_grad():
            trajectory = self.model.forward_with_aux(
                self._batch(), external_force=None, force_scale=0.0
            ).trajectory
        return self.koopman.fit_trajectory(
            trajectory,
            self.koopman_optimizer,
            outer_steps=int(self.args.feddgn_koopman_outer_steps),
            inner_steps=int(self.args.feddgn_koopman_inner_steps),
            batch_size=int(self.args.feddgn_koopman_batch_size),
            grad_clip=float(self.args.feddgn_koopman_grad_clip),
            sample_seed=(
                int(self.args.seed)
                + int(self.curr_rnd) * 1000
                + int(self.client_id)
            ),
        )

    def train(self):
        super().train()
        started = time.time()
        result = self._fit_koopman()
        self._round_result.update({
            name: float(value.detach().cpu().item())
            for name, value in result['diagnostics'].items()
        })
        self._round_result['koopman_seconds'] = float(time.time() - started)
        self._round_result.update(self._dictionary_receive_diagnostics)
        if bool(self.args.feddgn_mode_graph_enabled):
            self._latest_mode_upload = make_mode_upload(
                result,
                client_id=int(self.client_id),
                top_modes=int(self.args.feddgn_mode_graph_top_modes),
            )
        if self._should_save_koopman():
            self._save_koopman_artifacts(result)

    def _upload_payload(self):
        payload = super()._upload_payload()
        dictionary = get_state_dict(self.koopman.dictionary)
        payload['koopman_dictionary_protocol'] = KOOPMAN_DICTIONARY_PROTOCOL
        payload['koopman_dictionary'] = dictionary
        payload['koopman_dictionary_bytes'] = int(sum(
            value.nbytes for value in dictionary.values()
        ))
        payload['upload_bytes'] += payload['koopman_dictionary_bytes']
        if not bool(self.args.feddgn_mode_graph_enabled):
            return payload
        if self._latest_mode_upload is None:
            raise RuntimeError('Koopman modes were not prepared before upload.')
        payload['koopman_modes'] = self._latest_mode_upload
        payload['koopman_upload_bytes'] = int(
            self._latest_mode_upload['bytes']
        )
        payload['upload_bytes'] += payload['koopman_upload_bytes']
        return payload


__all__ = ['Client', 'KOOPMAN_DICTIONARY_PROTOCOL']
