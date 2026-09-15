"""O4 task aggregation plus ResKoop dictionary FedAvg and mode analysis."""

from __future__ import annotations

import json
import os

import numpy as np

from models.feddgn_local.model import build_feddgn_model
from models.feddgn_avg.server import Server as FedAvgFedDGNServer
from models.feddgn.mode_graph import (
    MODE_GRAPH_PROTOCOL,
    ModeGraphBuilder,
    save_mode_graph_artifacts,
)
from models.feddgn.reskoop import KOOPMAN_DICTIONARY_PROTOCOL


def _aggregate_dictionary_states(states, weights):
    if not states or len(states) != len(weights):
        raise ValueError('Koopman dictionary aggregation inputs are invalid.')
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError('Client Koopman dictionary keys do not match.')
    if not np.isfinite(weights).all() or not np.isclose(sum(weights), 1.0):
        raise ValueError('Koopman dictionary weights must be finite and sum to 1.')
    aggregated = {}
    for name in sorted(keys):
        arrays = [np.asarray(state[name]) for state in states]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError(
                f'Client Koopman dictionary tensor {name} shapes do not match.'
            )
        value = np.sum([
            array.astype(np.float64, copy=False) * float(weight)
            for array, weight in zip(arrays, weights)
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


class Server(FedAvgFedDGNServer):
    """O4 task server plus dictionary FedAvg and Functional-Map exchange."""

    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)
        self.mode_graph_builder = ModeGraphBuilder(
            neighbors=int(args.feddgn_mode_graph_neighbors),
            personalized_modes=int(
                args.feddgn_mode_graph_personalized_modes
            ),
            resolution=float(args.feddgn_mode_graph_resolution),
            functional_map_steps=int(args.feddgn_functional_map_steps),
            functional_map_tolerance=float(
                args.feddgn_functional_map_tolerance
            ),
            seed=int(args.seed),
        )
        self.latest_mode_graph_summary = None
        self.latest_mode_graph_visualization = None
        self.global_koopman_dictionary = None
        self.global_koopman_dictionary_round = 0
        self.latest_dictionary_summary = None

    def build_model(self):
        return build_feddgn_model(self.args)

    def on_round_begin(self, curr_rnd):
        super().on_round_begin(curr_rnd)
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

    @staticmethod
    def _public_graph(graph):
        return {
            key: graph[key]
            for key in (
                'protocol', 'round', 'comparison_space',
                'functional_map_method', 'functional_map_refinement_steps',
                'functional_map_tolerance', 'reference_client_id',
                'reference_spectral_residual', 'reference_round',
                'functional_map_diagnostics', 'community_method',
                'edge_metric', 'edge_scope', 'neighbor_policy',
                'neighbors_per_mode', 'louvain_resolution',
                'community_count', 'client_ids',
                'community_client_matrix', 'cluster_quality',
                'global_mode_bank_summary', 'nodes', 'edges',
            )
        }

    def update(self, updated):
        # Capture the complete atomic client messages before the common update
        # consumes them from Manager.dict.
        messages = [dict(self.sd[c_id]) for c_id in sorted(updated)]
        missing_dictionary = [
            int(message['client_id']) for message in messages
            if (
                message.get('koopman_dictionary_protocol')
                != KOOPMAN_DICTIONARY_PROTOCOL
                or 'koopman_dictionary' not in message
            )
        ]
        if missing_dictionary:
            raise RuntimeError(
                'Clients missing Koopman dictionary uploads: '
                f'{missing_dictionary}.'
            )
        weights = self._aggregation_weights(messages)
        dictionary_states = [
            dict(message['koopman_dictionary']) for message in messages
        ]
        global_dictionary = _aggregate_dictionary_states(
            dictionary_states, weights
        )
        client_deltas = [
            _dictionary_relative_delta(state, global_dictionary)
            for state in dictionary_states
        ]
        super().update(updated)
        round_number = int(self.curr_rnd) + 1
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
        self.sd['global_koopman_dictionary'] = {
            'protocol': KOOPMAN_DICTIONARY_PROTOCOL,
            'source_round': round_number,
            'state_dict': {
                name: value.copy()
                for name, value in global_dictionary.items()
            },
        }
        if not bool(self.args.feddgn_mode_graph_enabled):
            return
        missing = [
            int(message['client_id']) for message in messages
            if 'koopman_modes' not in message
        ]
        if missing:
            raise RuntimeError(
                f'Clients missing Koopman mode uploads: {missing}.'
            )
        graph, views = self.mode_graph_builder.build(
            [message['koopman_modes'] for message in messages],
            round_number=round_number,
        )

        # Publish only after graph construction is complete.  A client accepts
        # a view only when its graph-round exactly matches the expected prior
        # training round, preventing stale feedback under partial sampling.
        self.sd['mode_graph_global'] = self._public_graph(graph)
        for client_id, view in views.items():
            self.sd[f'mode_graph_view:{int(client_id)}'] = view

        save_every = int(self.args.feddgn_mode_graph_save_every)
        archive = (
            round_number == int(self.args.n_rnds)
            or round_number % save_every == 0
        )
        artifact_root = save_mode_graph_artifacts(
            self.args.log_path, graph, views, archive=archive
        )
        self.latest_mode_graph_summary = {
            'protocol': MODE_GRAPH_PROTOCOL,
            'round': round_number,
            'node_count': len(graph['nodes']),
            'edge_count': len(graph['edges']),
            'community_count': int(graph['community_count']),
            'client_count': len(graph['client_ids']),
            'comparison_space': str(graph['comparison_space']),
            'functional_map_method': str(graph['functional_map_method']),
            'reference_client_id': int(graph['reference_client_id']),
            'reference_round': int(graph['reference_round']),
            'mean_functional_map_residual': float(sum(
                report['intertwining_residual']
                for report in graph['functional_map_diagnostics']
            ) / len(graph['functional_map_diagnostics'])),
            'max_functional_map_orthogonality_error': float(max(
                report['orthogonality_error']
                for report in graph['functional_map_diagnostics']
            )),
            'community_method': str(graph['community_method']),
            'edge_metric': str(graph['edge_metric']),
            'edge_scope': str(graph['edge_scope']),
            'louvain_resolution': float(graph['louvain_resolution']),
            'community_client_matrix': graph['community_client_matrix'],
            'cluster_quality': graph['cluster_quality'],
            'global_mode_bank_summary': graph['global_mode_bank_summary'],
            'koopman_dictionary_aggregation': self.latest_dictionary_summary,
            'artifact_root': artifact_root,
            'archived': bool(archive),
            'task_feedback': False,
        }
        if archive and bool(self.args.feddgn_mode_graph_visualize):
            round_dir = os.path.join(
                artifact_root, f'round_{round_number:04d}'
            )
            try:
                # Rendering is deliberately tied to archived graph rounds,
                # never to every training round.  With the default interval
                # this produces dashboards only at 10, 20, ..., 100.
                from models.feddgn.mode_graph_visualize import (
                    generate_mode_graph_visualizations,
                )

                self.latest_mode_graph_visualization = (
                    generate_mode_graph_visualizations(
                        self.args.log_path,
                        dpi=int(
                            self.args.feddgn_koopman_visualization_dpi
                        ),
                        graph_path=os.path.join(
                            round_dir, 'mode_graph.pt'
                        ),
                        output_directory=os.path.join(round_dir, 'figures'),
                    )
                )
                self.latest_mode_graph_summary['dashboard'] = (
                    self.latest_mode_graph_visualization['dashboard']
                )
                print(
                    '[feddgn] Mode graph round '
                    f'{round_number} dashboard written to '
                    f"{self.latest_mode_graph_visualization['dashboard']}"
                )
            except Exception as exc:
                error_path = os.path.join(
                    round_dir, 'visualization_error.txt'
                )
                with open(error_path, 'w', encoding='utf-8') as stream:
                    stream.write(f'{type(exc).__name__}: {exc}\n')
                print(
                    f'[feddgn] Mode graph round {round_number} '
                    f'visualization failed: {exc}'
                )

    def checkpoint_extra_state(self):
        return {
            'mode_graph_summary': self.latest_mode_graph_summary,
            'mode_graph_task_feedback': False,
            'functional_map_reference_client_id': (
                self.mode_graph_builder.reference_client_id
            ),
            'functional_map_reference_operator': (
                self.mode_graph_builder.reference_operator
            ),
            'functional_map_reference_round': (
                self.mode_graph_builder.reference_round
            ),
            'global_koopman_dictionary': self.global_koopman_dictionary,
            'global_koopman_dictionary_round': int(
                self.global_koopman_dictionary_round
            ),
            'koopman_dictionary_aggregation': self.latest_dictionary_summary,
        }

    @staticmethod
    def _write_result(path, result):
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)

    def finalize(self):
        result = super().finalize()
        if self.latest_dictionary_summary is not None:
            result['koopman_dictionary_aggregation'] = (
                self.latest_dictionary_summary
            )
        if self.latest_mode_graph_summary is not None:
            result['mode_graph_analysis'] = self.latest_mode_graph_summary
        if (
            self.latest_dictionary_summary is not None
            or self.latest_mode_graph_summary is not None
        ):
            self._write_result(
                os.path.join(self.args.log_path, 'result.json'), result
            )

        if (
            self.latest_mode_graph_summary is not None
            and bool(self.args.feddgn_mode_graph_visualize)
        ):
            try:
                mode_graph_report = self.latest_mode_graph_visualization
                if mode_graph_report is None:
                    from models.feddgn.mode_graph_visualize import (
                        generate_mode_graph_visualizations,
                    )

                    mode_graph_report = generate_mode_graph_visualizations(
                        self.args.log_path,
                        dpi=int(
                            self.args.feddgn_koopman_visualization_dpi
                        ),
                    )
                result['mode_graph_visualization'] = mode_graph_report
                self._write_result(
                    os.path.join(self.args.log_path, 'result.json'), result
                )
                print(
                    '[feddgn] Final mode graph dashboard is '
                    f"{mode_graph_report['dashboard']}"
                )
            except Exception as exc:
                error_path = os.path.join(
                    self.args.log_path,
                    'mode_graph_visualization_error.txt',
                )
                with open(error_path, 'w', encoding='utf-8') as stream:
                    stream.write(f'{type(exc).__name__}: {exc}\n')
                print(f'[feddgn] Mode graph visualization failed: {exc}')

        if not bool(self.args.feddgn_koopman_visualize):
            return result

        try:
            # Keep matplotlib out of worker imports; rendering happens once,
            # after all client processes have joined.
            from models.feddgn.visualize import generate_koopman_visualizations

            report = generate_koopman_visualizations(
                self.args.log_path,
                top_modes=int(self.args.feddgn_koopman_visualization_top_modes),
                dpi=int(self.args.feddgn_koopman_visualization_dpi),
            )
            result['koopman_analysis'] = report
            path = os.path.join(self.args.log_path, 'result.json')
            self._write_result(path, result)
            print(
                '[feddgn] Koopman figures written to '
                f"{report['figures_directory']}"
            )
        except Exception as exc:
            error_path = os.path.join(
                self.args.log_path, 'koopman_visualization_error.txt'
            )
            with open(error_path, 'w', encoding='utf-8') as stream:
                stream.write(f'{type(exc).__name__}: {exc}\n')
            print(f'[feddgn] Koopman visualization failed: {exc}')
        return result


__all__ = ['Server']
