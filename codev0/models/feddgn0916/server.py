"""Observable-dictionary FedAvg and timed, client-specific mode exchange."""

from __future__ import annotations

import json
import os
import time

import numpy as np

from misc.utils import torch_save
from models.feddgn0916.candidate_pipeline import (
    ConflictCandidateBuilder,
    save_candidate_artifacts,
)
from models.feddgn0916.reskoop import KOOPMAN_DICTIONARY_PROTOCOL
from modules.federated import ServerModule
from modules.structured_logger import StructuredLogger


def _aggregate_dictionary_states(states, weights):
    if not states or len(states) != len(weights):
        raise ValueError('Koopman dictionary aggregation inputs are invalid.')
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError('Client Koopman dictionary keys do not match.')
    numeric = np.asarray(weights, dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0.0)
        or not np.isclose(numeric.sum(), 1.0)
    ):
        raise ValueError('Koopman dictionary weights must sum to one.')
    aggregated = {}
    for name in sorted(keys):
        arrays = [np.asarray(state[name]) for state in states]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError(
                f'Client Koopman dictionary tensor {name} shapes do not match.'
            )
        value = np.sum([
            array.astype(np.float64, copy=False) * float(weight)
            for array, weight in zip(arrays, numeric)
        ], axis=0)
        if not np.isfinite(value).all():
            raise ValueError(
                f'Aggregated Koopman dictionary tensor {name} is non-finite.'
            )
        aggregated[name] = value.astype(arrays[0].dtype, copy=False)
    return aggregated


def _dictionary_relative_delta(state, reference):
    numerator = 0.0
    denominator = 0.0
    for name in sorted(reference):
        value = np.asarray(state[name], dtype=np.float64)
        target = np.asarray(reference[name], dtype=np.float64)
        numerator += float(np.square(value - target).sum())
        denominator += float(np.square(target).sum())
    return float(
        np.sqrt(numerator)
        / max(np.sqrt(denominator), np.finfo(np.float64).eps)
    )


class Server(ServerModule):
    """Never receive, aggregate, or broadcast task-backbone parameters."""

    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)
        self.structured = StructuredLogger(args)
        self.candidate_builder = ConflictCandidateBuilder(
            top_modes=int(args.feddgn_candidate_top_modes),
            max_candidates=int(args.feddgn_candidate_max_per_client),
            mode_residual_max=float(args.feddgn_candidate_mode_residual_max),
            conflict_threshold=float(args.feddgn_candidate_conflict_threshold),
            compatibility_temperature=float(
                args.feddgn_candidate_compat_temperature
            ),
            functional_map_steps=int(args.feddgn_functional_map_steps),
            functional_map_tolerance=float(
                args.feddgn_functional_map_tolerance
            ),
        )
        self.global_koopman_dictionary = None
        self.global_koopman_dictionary_round = 0
        self.latest_dictionary_summary = None
        self.latest_candidate_summary = None
        self.latest_reference_operator = None

    def on_round_begin(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        self.sd.pop('global', None)
        self.sd.pop('global_optimizer', None)
        self.structured.begin_round()
        if self.global_koopman_dictionary is None:
            self.sd.pop('global_koopman_dictionary', None)
            return
        self.sd['global_koopman_dictionary'] = {
            'protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'source_round': int(self.global_koopman_dictionary_round),
            'state_dict': {
                name: value.copy()
                for name, value in self.global_koopman_dictionary.items()
            },
        }

    def on_round_complete(self, updated):
        messages, timings = self.update(updated)
        started = time.perf_counter()
        self.save_state()
        timings['server_checkpoint_seconds'] = time.perf_counter() - started
        self.structured.record_feddgn_timing(
            self.curr_rnd, messages, timings,
            round_wall_seconds=(
                time.perf_counter() - self.structured.round_started
            ),
        )

    def _aggregation_weights(self, messages):
        if self.args.aggregation == 'equal':
            return [1.0 / len(messages)] * len(messages)
        sizes = np.asarray(
            [message['train_size'] for message in messages], dtype=np.float64
        )
        if not np.isfinite(sizes).all() or np.any(sizes <= 0.0):
            raise ValueError('Dictionary weighted FedAvg needs positive train sizes.')
        return (sizes / sizes.sum()).tolist()

    def update(self, updated):
        started = time.perf_counter()
        timings = {
            'client_parallel_wall_seconds': float(
                started - self.structured.round_started
            ),
        }
        messages = [dict(self.sd[c_id]) for c_id in sorted(updated)]
        timings['server_message_unpack_seconds'] = (
            time.perf_counter() - started
        )
        if not messages:
            raise RuntimeError('No client updates for candidate screening.')
        if any('model' in item or 'main_v_blocks' in item for item in messages):
            raise RuntimeError('Task model/optimizer upload violates local backbone.')
        for message in messages:
            if (
                message.get('koopman_dictionary_protocol')
                != KOOPMAN_DICTIONARY_PROTOCOL
                or 'koopman_dictionary' not in message
                or 'koopman_packet' not in message
            ):
                raise RuntimeError(
                    f"Client {message['client_id']} omitted a Koopman upload."
                )

        dictionary_started = time.perf_counter()
        weights = self._aggregation_weights(messages)
        states = [dict(message['koopman_dictionary']) for message in messages]
        global_dictionary = _aggregate_dictionary_states(states, weights)
        client_deltas = [
            _dictionary_relative_delta(state, global_dictionary)
            for state in states
        ]
        timings['server_dictionary_fedavg_seconds'] = (
            time.perf_counter() - dictionary_started
        )

        round_number = int(self.curr_rnd) + 1
        modes_started = time.perf_counter()
        analysis = self.candidate_builder.build(
            [message['koopman_packet'] for message in messages],
            round_number=round_number,
        )
        timings['server_modes_build_seconds'] = (
            time.perf_counter() - modes_started
        )

        self.global_koopman_dictionary = global_dictionary
        self.global_koopman_dictionary_round = round_number
        self.latest_dictionary_summary = {
            'protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'round': round_number,
            'aggregation': str(self.args.aggregation),
            'optimizer_on_receive': 'reset_adam',
            'client_count': len(messages),
            'tensor_count': len(global_dictionary),
            'parameter_count': int(sum(
                value.size for value in global_dictionary.values()
            )),
            'mean_client_relative_delta': float(np.mean(client_deltas)),
            'max_client_relative_delta': float(np.max(client_deltas)),
        }
        self.latest_reference_operator = analysis['reference_operator'].copy()
        self.latest_candidate_summary = {
            **analysis['summary'],
            'koopman_dictionary_aggregation': self.latest_dictionary_summary,
            'K_aggregation': False,
            'K_star_task_feedback': False,
            'task_parameter_aggregation': False,
            'task_optimizer_aggregation': False,
        }

        return_started = time.perf_counter()
        self.sd['global_koopman_dictionary'] = {
            'protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'source_round': round_number,
            'state_dict': {
                name: value.copy() for name, value in global_dictionary.items()
            },
        }
        for client_id, package in analysis['packages'].items():
            self.sd[f'candidate_package:{int(client_id)}'] = package
        timings['server_modes_return_seconds'] = (
            time.perf_counter() - return_started
        )

        artifact_started = time.perf_counter()
        save_candidate_artifacts(self.args.log_path, analysis)
        timings['server_numeric_artifacts_seconds'] = (
            time.perf_counter() - artifact_started
        )
        visualization_started = time.perf_counter()
        visualization_every = int(
            getattr(self.args, 'feddgn_server_visualize_every', 0)
        )
        if visualization_every > 0:
            round_number = int(analysis['round'])
            last_consumable_round = max(int(self.args.n_rnds) - 1, 1)
            should_render = (
                round_number == last_consumable_round
                or round_number % visualization_every == 0
            )
            if should_render:
                from models.feddgn0916.server_visualize import (
                    render_server_diagnostics,
                )
                render_server_diagnostics(
                    analysis,
                    os.path.join(
                        self.args.log_path, 'koopman', 'visualization', 'server'
                    ),
                    dpi=int(getattr(
                        self.args, 'feddgn_server_visualization_dpi', 90
                    )),
                    history_path=os.path.join(
                        self.args.log_path, 'koopman', 'candidates',
                        'server_rounds.csv',
                    ),
                )
        timings['server_visualization_seconds'] = (
            time.perf_counter() - visualization_started
        )
        logging_started = time.perf_counter()
        self.structured.record_round(
            self.curr_rnd, messages, None, self.args.aggregation
        )
        timings['server_metric_logging_seconds'] = (
            time.perf_counter() - logging_started
        )
        for client_id in updated:
            del self.sd[client_id]
        timings['server_update_seconds'] = time.perf_counter() - started
        return messages, timings

    def save_state(self):
        torch_save(
            self.args.checkpt_path, 'server_state.pt', {
                'round': int(self.curr_rnd) + 1,
                'aggregation': self.args.aggregation,
                'optimizer_state_mode': self.args.optimizer_state_mode,
                'task_model_present': False,
                'task_parameter_aggregation': False,
                'task_optimizer_aggregation': False,
                'global_koopman_dictionary': self.global_koopman_dictionary,
                'global_koopman_dictionary_round': int(
                    self.global_koopman_dictionary_round
                ),
                'koopman_dictionary_aggregation': self.latest_dictionary_summary,
                'candidate_summary': self.latest_candidate_summary,
                'candidate_reference_operator': self.latest_reference_operator,
                'candidate_K_star_task_feedback': False,
            },
        )

    @staticmethod
    def _write_result(path, result):
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)

    def finalize(self):
        result = self.structured.finalize()
        if self.latest_dictionary_summary is not None:
            result['koopman_dictionary_aggregation'] = (
                self.latest_dictionary_summary
            )
        if self.latest_candidate_summary is not None:
            result['candidate_analysis'] = self.latest_candidate_summary
        result['task_parameter_aggregation'] = False
        result['task_optimizer_aggregation'] = False
        result['K_star_task_feedback'] = False
        result['timing_csv'] = os.path.join(self.args.log_path, 'timing.csv')
        self._write_result(
            os.path.join(self.args.log_path, 'result.json'), result
        )
        return result


__all__ = ['Server', '_aggregate_dictionary_states']
