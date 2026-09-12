"""Single, low-overhead logging pipeline for federated experiments."""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np


BINARY_DATASETS = {'Minesweeper', 'Tolokers', 'Questions'}


def _floating_vector(state_dict):
    arrays = []
    for value in state_dict.values():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            arrays.append(array.astype(np.float64, copy=False).reshape(-1))
    if not arrays:
        raise ValueError('Model state contains no floating parameters.')
    return np.concatenate(arrays)


class StructuredLogger:
    """Write compact default logs and opt-in parameter diagnostics.

    ``minimal`` is the normal training profile. It never flattens model
    parameters and produces only four files in the run directory:

    - config.json: resolved run configuration;
    - metrics.csv: one aggregate row per federated round;
    - client_best.csv: one row per client, selected by local validation;
    - result.json: the final best-validation-paired test summary.

    ``diagnostic`` additionally records per-client metrics and model-update
    diagnostics. These files are appended once per round instead of rewriting
    the full history.
    """

    V04_FIELDS = [
        'task_loss', 'train_accuracy',
        'probe_loss', 'probe_accuracy', 'probe_grad_norm',
        'field_anchor_loss', 'field_task_cosine', 'field_rms',
        'field_raw_rms', 'field_scale', 'task_direction_rms',
        'field_grad_norm', 'lie_weight', 'lie_loss', 'lie_cosine',
        'lie_target_rms', 'koopman_rhs_rms', 'lie_speed_ratio',
        'z0_norm', 'zt_norm', 'z_norm_ratio', 'rhs0_norm', 'nfe',
        'main_grad_norm', 'g0_symmetric_maxeig', 'g0_max_real_part',
        'g1_symmetric_maxeig', 'g1_max_real_part',
        'main_m_rms', 'main_v_rms', 'main_v_block_mean',
        'main_adaptive_step_rms', 'global_v_step',
        'global_v_block_mean',
    ]
    #: Method 1.1 Port-Hamiltonian diagnostics. Kept separate from V04_FIELDS so
    #: a non-v11 run logs zeros rather than reusing those columns with a
    #: different meaning. Names are prefixed to avoid colliding with the V0.4
    #: trajectory columns. Bands cover basis_degree up to 3; higher degrees
    #: simply log nothing.
    V11_FIELDS = [
        'manifold_loss', 'operator_speed_loss',
        'ph_z0_norm', 'ph_zT_norm',
        'trajectory_growth_ratio', 'trajectory_max_ratio',
        'hT_rms', 'auxT_rms',
        'band0_alpha', 'band0_beta', 'band0_gamma',
        'band0_J_fro', 'band0_R_fro', 'band0_sym_max_eig', 'band0_spectral_norm',
        'band1_alpha', 'band1_beta', 'band1_gamma',
        'band1_J_fro', 'band1_R_fro', 'band1_sym_max_eig', 'band1_spectral_norm',
        'band2_alpha', 'band2_beta', 'band2_gamma',
        'band2_J_fro', 'band2_R_fro', 'band2_sym_max_eig', 'band2_spectral_norm',
        'band3_alpha', 'band3_beta', 'band3_gamma',
        'band3_J_fro', 'band3_R_fro', 'band3_sym_max_eig', 'band3_spectral_norm',
    ]
    METRIC_FIELDS = [
        'round', 'optimizer_state_mode',
        'train_loss', 'val_loss', 'val_accuracy', 'test_loss',
        'test_accuracy', 'local_best_test_accuracy', 'round_seconds',
    ] + V04_FIELDS + V11_FIELDS
    CLIENT_FIELDS = [
        'round', 'client_id', 'train_loss', 'val_loss', 'val_accuracy',
        'test_loss', 'test_accuracy', 'test_f1', 'train_size',
    ] + V04_FIELDS + V11_FIELDS
    BEST_FIELDS = [
        'client_id', 'best_round', 'val_accuracy', 'test_accuracy', 'test_f1',
    ]
    UPDATE_FIELDS = ['round', 'client_id', 'update_norm']
    PAIR_FIELDS = ['round', 'client_i', 'client_j', 'update_cosine']
    SERVER_FIELDS = [
        'round', 'aggregation', 'mean_update_norm', 'std_update_norm',
        'update_variance_per_parameter', 'mean_pairwise_update_cosine',
        'min_pairwise_update_cosine', 'max_pairwise_update_cosine',
    ]

    def __init__(self, args):
        self.args = args
        self.log_path = args.log_path
        self.profile = str(args.log_profile)
        self.round_started = None
        self.client_best = {
            client_id: {
                'client_id': client_id,
                'best_round': 0,
                'val_accuracy': float('-inf'),
                'test_accuracy': 0.0,
                'test_f1': 0.0,
            }
            for client_id in range(args.n_clients)
        }
        os.makedirs(self.log_path, exist_ok=True)
        self._write_json(
            'config.json',
            {
                key: value
                for key, value in vars(args).items()
                if not key.startswith('_')
            },
        )

    def begin_round(self):
        self.round_started = time.perf_counter()

    def _write_json(self, filename, payload):
        path = os.path.join(self.log_path, filename)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write('\n')
        os.replace(temporary, path)

    def _append_csv(self, filename, rows, fieldnames):
        if not rows:
            return
        path = os.path.join(self.log_path, filename)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_csv(self, filename, rows, fieldnames):
        path = os.path.join(self.log_path, filename)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    def _update_client_best(self, rows):
        for row in rows:
            client_id = row['client_id']
            if row['val_accuracy'] <= self.client_best[client_id]['val_accuracy']:
                continue
            self.client_best[client_id] = {
                'client_id': client_id,
                'best_round': row['round'],
                'val_accuracy': row['val_accuracy'],
                'test_accuracy': row['test_accuracy'],
                'test_f1': row['test_f1'],
            }

    def _record_diagnostics(
        self, round_number, ordered, reference_state, aggregation
    ):
        reference = _floating_vector(reference_state)
        updates = np.stack([
            _floating_vector(item['model']) - reference for item in ordered
        ])
        norms = np.linalg.norm(updates, axis=1)
        gram = updates @ updates.T
        epsilon = np.finfo(np.float64).eps
        update_rows = []
        pair_rows = []
        pair_cosines = []
        for first, item in enumerate(ordered):
            update_rows.append({
                'round': round_number,
                'client_id': int(item['client_id']),
                'update_norm': float(norms[first]),
            })
            for second in range(first + 1, len(ordered)):
                cosine = gram[first, second] / (
                    norms[first] * norms[second] + epsilon
                )
                cosine = float(np.clip(cosine, -1.0, 1.0))
                pair_cosines.append(cosine)
                pair_rows.append({
                    'round': round_number,
                    'client_i': int(ordered[first]['client_id']),
                    'client_j': int(ordered[second]['client_id']),
                    'update_cosine': cosine,
                })
        centered = updates - updates.mean(axis=0, keepdims=True)
        server_row = {
            'round': round_number,
            'aggregation': str(aggregation),
            'mean_update_norm': float(norms.mean()),
            'std_update_norm': float(norms.std()),
            'update_variance_per_parameter': float(np.mean(centered ** 2)),
            'mean_pairwise_update_cosine': (
                float(np.mean(pair_cosines)) if pair_cosines else 1.0
            ),
            'min_pairwise_update_cosine': (
                float(np.min(pair_cosines)) if pair_cosines else 1.0
            ),
            'max_pairwise_update_cosine': (
                float(np.max(pair_cosines)) if pair_cosines else 1.0
            ),
        }
        self._append_csv(
            'client_update_diagnostics.csv', update_rows, self.UPDATE_FIELDS
        )
        self._append_csv(
            'pairwise_update_diagnostics.csv', pair_rows, self.PAIR_FIELDS
        )
        self._append_csv(
            'server_diagnostics.csv', [server_row], self.SERVER_FIELDS
        )

    def record_round(self, round_id, messages, reference_state, aggregation):
        round_number = int(round_id) + 1
        ordered = sorted(messages, key=lambda item: int(item['client_id']))
        client_rows = [
            {
                'round': round_number,
                'client_id': int(item['client_id']),
                'train_loss': float(item['train_loss']),
                'val_loss': float(item['val_loss']),
                'val_accuracy': float(item['val_accuracy']),
                'test_loss': float(item['test_loss']),
                'test_accuracy': float(item['test_accuracy']),
                'test_f1': float(item['test_f1']),
                'train_size': int(item['train_size']),
                **{
                    name: float(item.get(name, 0.0))
                    for name in self.V04_FIELDS + self.V11_FIELDS
                },
            }
            for item in ordered
        ]
        self._update_client_best(client_rows)
        metric_row = {
            'round': round_number,
            'optimizer_state_mode': str(self.args.optimizer_state_mode),
            'train_loss': float(np.mean([row['train_loss'] for row in client_rows])),
            'val_loss': float(np.mean([row['val_loss'] for row in client_rows])),
            'val_accuracy': float(np.mean([row['val_accuracy'] for row in client_rows])),
            'test_loss': float(np.mean([row['test_loss'] for row in client_rows])),
            'test_accuracy': float(np.mean([row['test_accuracy'] for row in client_rows])),
            'local_best_test_accuracy': float(np.mean([
                row['test_accuracy']
                for row in self.client_best.values()
                if row['best_round'] > 0
            ])),
            'round_seconds': float(time.perf_counter() - self.round_started),
            **{
                name: float(np.mean([row[name] for row in client_rows]))
                for name in self.V04_FIELDS + self.V11_FIELDS
            },
        }
        self._append_csv('metrics.csv', [metric_row], self.METRIC_FIELDS)
        self._write_csv(
            'client_best.csv',
            [self.client_best[index] for index in sorted(self.client_best)],
            self.BEST_FIELDS,
        )
        if self.profile == 'diagnostic':
            self._append_csv(
                'client_metrics.csv', client_rows, self.CLIENT_FIELDS
            )
            self._record_diagnostics(
                round_number, ordered, reference_state, aggregation
            )

    def finalize(self):
        completed = [
            self.client_best[index]
            for index in sorted(self.client_best)
            if self.client_best[index]['best_round'] > 0
        ]
        if not completed:
            raise RuntimeError('Cannot finalize an experiment with no client metrics.')
        test_values = np.asarray(
            [row['test_accuracy'] for row in completed], dtype=np.float64
        )
        f1_values = np.asarray(
            [row['test_f1'] for row in completed], dtype=np.float64
        )
        metric_name = 'AUC' if self.args.dataset in BINARY_DATASETS else 'ACC'
        result = {
            'dataset': self.args.dataset,
            'model': self.args.model,
            'metric': metric_name,
            'selection': 'per-client best validation round paired test metric',
            'n_clients': len(completed),
            'client_test_metrics': test_values.tolist(),
            'mean': float(test_values.mean()),
            'std': float(test_values.std()),
            'client_test_f1': f1_values.tolist(),
            'f1_mean': float(f1_values.mean()),
            'f1_std': float(f1_values.std()),
        }
        self._write_json('result.json', result)
        return result
