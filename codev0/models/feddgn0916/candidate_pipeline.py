"""Small-operator Functional Map alignment and client-conditioned conflicts.

Only Koopman operators and compact reliability metadata cross the server
boundary.  Private observable snapshots never enter this module.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from misc.utils import torch_save
from models.feddgn0916.functional_map import (
    identity_functional_map,
    solve_functional_map,
)


CANDIDATE_PROTOCOL = 'koopman_conflict_candidates_v1'


def _array(value, *, name):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.size == 0 or not np.all(
        np.isfinite(result.real) & np.isfinite(result.imag)
    ):
        raise ValueError(f'{name} must be finite and non-empty.')
    return result.copy()


def make_koopman_packet(result, *, client_id, round_number):
    """Convert row-EDMD output into a column-action server packet.

    ResKoop's spectral residuals describe the source eigenfunctions.  The
    server matches their eigenvalues to column-action eigenvectors; it does
    not pretend the two eigenvector arrays have identical indices.
    """

    row_operator = _array(result['operator'], name='Koopman operator')
    eigenvalues = _array(
        result['eigenvalues'], name='Koopman eigenvalues'
    ).reshape(-1)
    residuals = _array(
        result['mode_residuals'], name='Koopman mode residuals'
    ).reshape(-1)
    if (
        row_operator.ndim != 2
        or row_operator.shape[0] != row_operator.shape[1]
        or row_operator.shape[0] != eigenvalues.size
        or residuals.size != eigenvalues.size
        or np.any(residuals < 0.0)
    ):
        raise ValueError('Koopman operator and reliability metadata disagree.')
    spectral_residual = result['diagnostics']['koopman_spectral_residual']
    if hasattr(spectral_residual, 'detach'):
        spectral_residual = spectral_residual.detach().cpu().item()
    spectral_residual = float(spectral_residual)
    if not np.isfinite(spectral_residual) or spectral_residual < 0.0:
        raise ValueError('Client Koopman spectral residual is invalid.')
    packet = {
        'protocol': CANDIDATE_PROTOCOL,
        'client_id': int(client_id),
        'round': int(round_number),
        'operator': row_operator.T.copy(),
        'source_eigenvalues': eigenvalues.astype(np.complex64, copy=False),
        'source_mode_residuals': residuals.astype(np.float32, copy=False),
        'spectral_residual': spectral_residual,
        'operator_convention': 'column_action_transpose_of_row_edmd',
    }
    packet['bytes'] = int(sum(
        value.nbytes for value in packet.values()
        if isinstance(value, np.ndarray)
    ))
    return packet


def _spectrum_distance(left, right):
    cost = np.abs(left[:, None] - right[None, :])
    rows, columns = linear_sum_assignment(cost)
    scale = max(
        float((np.abs(left).mean() + np.abs(right).mean()) / 2.0),
        np.finfo(np.float64).eps,
    )
    return float(cost[rows, columns].mean() / scale)


def _operator_distance(left, right):
    numerator = np.linalg.norm(left - right, 'fro')
    denominator = np.linalg.norm(left, 'fro') + np.linalg.norm(right, 'fro')
    return float(numerator / max(denominator, np.finfo(np.float64).eps))


def _pairwise_distances(operators, client_ids):
    matrix = np.zeros((len(client_ids), len(client_ids)), dtype=np.float32)
    for row, left_id in enumerate(client_ids):
        for column in range(row + 1, len(client_ids)):
            right_id = client_ids[column]
            distance = _operator_distance(
                operators[left_id], operators[right_id]
            )
            matrix[row, column] = matrix[column, row] = distance
    return matrix


def _conflict(local_modes, candidate):
    """Return the largest overlap-weighted dynamical disagreement."""

    if not local_modes:
        return 0.0, 0.0, 0.0, None
    worst = (-1.0, 0.0, 0.0, None)
    for local in local_modes:
        overlap = local['U'].conj().T @ candidate['U']
        overlap_sq = float(np.linalg.norm(overlap, 'fro') ** 2)
        shared = overlap_sq / min(local['U'].shape[1], candidate['U'].shape[1])
        mismatch = local['T'] @ overlap - overlap @ candidate['T']
        scale = (
            np.linalg.norm(local['T'], 2) ** 2
            + np.linalg.norm(candidate['T'], 2) ** 2
        ) * overlap_sq + np.finfo(np.float64).eps
        disagreement = float(np.linalg.norm(mismatch, 'fro') ** 2 / scale)
        conflict = float(shared * disagreement)
        if conflict > worst[0]:
            worst = (
                conflict, float(shared), disagreement, local['mode_id']
            )
    return worst


class ConflictCandidateBuilder:
    """Generate individual non-conflicting mode packages, without a graph."""

    def __init__(
        self,
        *,
        top_modes=16,
        max_candidates=16,
        mode_residual_max=1.0,
        conflict_threshold=0.15,
        compatibility_temperature=0.1,
        functional_map_steps=8,
        functional_map_tolerance=1e-7,
    ):
        if min(int(top_modes), int(max_candidates)) <= 0:
            raise ValueError('Mode and candidate budgets must be positive.')
        if not np.isfinite(mode_residual_max) or not 0.0 < float(mode_residual_max):
            raise ValueError('Mode residual limit must be positive.')
        if any(
            not np.isfinite(value) or float(value) <= 0.0
            for value in (conflict_threshold, compatibility_temperature)
        ):
            raise ValueError('Conflict threshold and temperature must be positive.')
        self.top_modes = int(top_modes)
        self.max_candidates = int(max_candidates)
        self.mode_residual_max = float(mode_residual_max)
        self.conflict_threshold = float(conflict_threshold)
        self.compatibility_temperature = float(compatibility_temperature)
        self.functional_map_steps = int(functional_map_steps)
        self.functional_map_tolerance = float(functional_map_tolerance)

    def _extract_modes(self, packet, aligned_operator):
        values, vectors = np.linalg.eig(aligned_operator)
        source_values = packet['source_eigenvalues']
        cost = np.abs(values[:, None] - source_values[None, :])
        aligned_indices, source_indices = linear_sum_assignment(cost)
        reliability = np.empty(values.size, dtype=np.float64)
        source_index = np.empty(values.size, dtype=np.int64)
        reliability[aligned_indices] = packet['source_mode_residuals'][
            source_indices
        ]
        source_index[aligned_indices] = source_indices
        finite = (
            np.isfinite(values.real) & np.isfinite(values.imag)
            & np.isfinite(reliability)
            & (reliability >= 0.0)
            & (reliability <= self.mode_residual_max)
            & np.all(np.isfinite(vectors.real) & np.isfinite(vectors.imag), axis=0)
        )
        candidates = np.flatnonzero(finite)
        if candidates.size > 1:
            trivial = candidates[np.argmin(np.abs(values[candidates] - 1.0))]
            candidates = candidates[candidates != trivial]
        if candidates.size == 0:
            return [], float(cost[aligned_indices, source_indices].max())
        order = np.lexsort((
            -np.abs(values[candidates]),
            reliability[candidates],
        ))
        selected = candidates[order[:self.top_modes]]
        modes = []
        for index in selected:
            vector = vectors[:, index:index + 1]
            norm = np.linalg.norm(vector)
            if norm <= np.finfo(np.float64).eps:
                continue
            vector = vector / norm
            source_mode = int(source_index[index])
            modes.append({
                'mode_id': f"client_{packet['client_id']}:mode_{source_mode}",
                'source_client': int(packet['client_id']),
                'source_mode': source_mode,
                'U': vector.astype(np.complex64),
                'T': np.asarray([[values[index]]], dtype=np.complex64),
                'eigenvalue': complex(values[index]),
                'reliability': float(reliability[index]),
                'source_spectral_residual': float(
                    packet['spectral_residual']
                ),
            })
        return modes, float(cost[aligned_indices, source_indices].max())

    def build(self, packets, *, round_number):
        ordered = sorted(packets, key=lambda item: int(item['client_id']))
        if not ordered:
            raise ValueError('Candidate screening needs client Koopman packets.')
        client_ids = [int(item['client_id']) for item in ordered]
        if len(set(client_ids)) != len(client_ids):
            raise ValueError('Duplicate client Koopman packet.')
        dimensions = set()
        by_id = {}
        for item in ordered:
            if (
                item.get('protocol') != CANDIDATE_PROTOCOL
                or int(item.get('round', -1)) != int(round_number)
                or item.get('operator_convention')
                != 'column_action_transpose_of_row_edmd'
            ):
                raise ValueError('Stale or incompatible Koopman packet.')
            operator = _array(item['operator'], name='column Koopman operator')
            if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
                raise ValueError('Koopman operator must be square.')
            source_values = _array(
                item['source_eigenvalues'], name='source eigenvalues'
            ).reshape(-1)
            residuals = _array(
                item['source_mode_residuals'], name='mode residuals'
            ).reshape(-1)
            if (
                source_values.size != operator.shape[0]
                or residuals.size != source_values.size
                or np.any(residuals < 0.0)
            ):
                raise ValueError('Koopman packet reliability dimensions disagree.')
            dimensions.add(operator.shape[0])
            by_id[int(item['client_id'])] = {
                **dict(item),
                'operator': operator.astype(np.complex128),
                'source_eigenvalues': source_values.astype(np.complex128),
                'source_mode_residuals': residuals.astype(np.float64),
            }
        if len(dimensions) != 1:
            raise ValueError('Client Koopman operators have different dimensions.')

        raw_operators = {cid: by_id[cid]['operator'] for cid in client_ids}
        spectra = {
            cid: np.linalg.eigvals(raw_operators[cid])
            for cid in client_ids
        }
        medoid_costs = {
            cid: sum(
                _spectrum_distance(spectra[cid], spectra[other])
                for other in client_ids if other != cid
            )
            for cid in client_ids
        }
        reference_id = min(client_ids, key=lambda cid: (medoid_costs[cid], cid))
        reference = raw_operators[reference_id]
        maps = {}
        map_reports = []
        aligned = {}
        for cid in client_ids:
            if cid == reference_id:
                functional_map, report = identity_functional_map(reference)
            else:
                functional_map, report = solve_functional_map(
                    raw_operators[cid], reference,
                    refinement_steps=self.functional_map_steps,
                    tolerance=self.functional_map_tolerance,
                )
            functional_map = functional_map.astype(np.complex128)
            maps[cid] = functional_map
            aligned[cid] = (
                functional_map @ raw_operators[cid] @ functional_map.conj().T
            )
            map_reports.append({**report, 'client_id': cid})

        modes = {}
        matching_errors = {}
        for cid in client_ids:
            modes[cid], matching_errors[cid] = self._extract_modes(
                by_id[cid], aligned[cid]
            )

        packages = {}
        conflict_rows = []
        source_matrix = np.zeros(
            (len(client_ids), len(client_ids)), dtype=np.int64
        )
        for target_index, target_id in enumerate(client_ids):
            eligible = []
            for source_id in client_ids:
                if source_id == target_id:
                    continue
                for mode in modes[source_id]:
                    conflict, overlap, disagreement, local_mode_id = _conflict(
                        modes[target_id], mode
                    )
                    rejected = conflict > self.conflict_threshold
                    compatibility = float(np.exp(
                        -conflict / self.compatibility_temperature
                    ))
                    row = {
                        'target_client': target_id,
                        'source_client': source_id,
                        'source_mode': int(mode['source_mode']),
                        'mode_id': mode['mode_id'],
                        'overlap': overlap,
                        'disagreement': disagreement,
                        'conflict': conflict,
                        'compatibility': compatibility,
                        'reliability': mode['reliability'],
                        'worst_local_mode': local_mode_id,
                        'rejected': bool(rejected),
                    }
                    conflict_rows.append(row)
                    if not rejected:
                        eligible.append((row, mode))

            # Equalize the source-client opportunity before applying the
            # communication budget.  Ranking never uses spatial cosine.
            grouped = defaultdict(list)
            for row, mode in eligible:
                grouped[row['source_client']].append((row, mode))
            for source_id in grouped:
                grouped[source_id].sort(key=lambda item: (
                    item[0]['conflict'], item[0]['reliability'],
                    item[0]['source_mode'],
                ))
            chosen = []
            while len(chosen) < self.max_candidates and any(grouped.values()):
                for source_id in sorted(grouped):
                    if grouped[source_id] and len(chosen) < self.max_candidates:
                        chosen.append(grouped[source_id].pop(0))

            candidates = []
            for row, mode in chosen:
                source_matrix[target_index, client_ids.index(
                    row['source_client']
                )] += 1
                local_u = maps[target_id].conj().T @ mode['U']
                candidates.append({
                    **{key: row[key] for key in (
                        'mode_id', 'source_client', 'source_mode',
                        'overlap', 'disagreement', 'conflict',
                        'compatibility', 'reliability',
                    )},
                    'U_local': local_u.astype(np.complex64),
                    'T': mode['T'].copy(),
                })
            all_target = [
                row for row in conflict_rows
                if row['target_client'] == target_id
            ]
            packages[target_id] = {
                'protocol': CANDIDATE_PROTOCOL,
                'client_id': target_id,
                'source_round': int(round_number),
                'reference_client_id': reference_id,
                'functional_map': maps[target_id].astype(np.complex64),
                'functional_map_report': next(
                    dict(report) for report in map_reports
                    if report['client_id'] == target_id
                ),
                'candidates': candidates,
                'raw_external_count': len(all_target),
                'conflict_rejected_count': sum(
                    row['rejected'] for row in all_target
                ),
                'candidate_count': len(candidates),
            }

        total_external = len(conflict_rows)
        rejected = sum(row['rejected'] for row in conflict_rows)
        conflict_values = np.asarray(
            [row['conflict'] for row in conflict_rows], dtype=np.float64
        )
        summary = {
            'protocol': CANDIDATE_PROTOCOL,
            'round': int(round_number),
            'reference_client_id': reference_id,
            'reference_medoid_cost': float(medoid_costs[reference_id]),
            'client_ids': client_ids,
            'mode_count_per_client': {
                str(cid): len(modes[cid]) for cid in client_ids
            },
            'max_eigenvalue_matching_error': float(max(
                matching_errors.values()
            )),
            'raw_external_count': total_external,
            'conflict_rejected_count': rejected,
            'conflict_rejection_ratio': (
                rejected / total_external if total_external else 0.0
            ),
            'conflict_threshold': self.conflict_threshold,
            'compatibility_temperature': self.compatibility_temperature,
            'conflict_median': float(np.median(conflict_values)) if total_external else 0.0,
            'conflict_q95': float(np.quantile(conflict_values, 0.95)) if total_external else 0.0,
            'conflict_max': float(conflict_values.max()) if total_external else 0.0,
            'retained_count': int(source_matrix.sum()),
            'candidate_count_per_client': {
                str(cid): packages[cid]['candidate_count']
                for cid in client_ids
            },
            'mean_functional_map_residual': float(np.mean([
                report['intertwining_residual'] for report in map_reports
            ])),
            'max_functional_map_orthogonality_error': float(max(
                report['orthogonality_error'] for report in map_reports
            )),
            'task_feedback': False,
        }
        return {
            'protocol': CANDIDATE_PROTOCOL,
            'round': int(round_number),
            'client_ids': client_ids,
            'reference_client_id': reference_id,
            'reference_operator': reference.astype(np.complex64),
            'medoid_costs': medoid_costs,
            'functional_maps': {
                cid: maps[cid].astype(np.complex64) for cid in client_ids
            },
            'functional_map_reports': map_reports,
            'raw_operators': {
                cid: raw_operators[cid].astype(np.complex64)
                for cid in client_ids
            },
            'aligned_operators': {
                cid: aligned[cid].astype(np.complex64)
                for cid in client_ids
            },
            'raw_pairwise_distance': _pairwise_distances(
                raw_operators, client_ids
            ),
            'aligned_pairwise_distance': _pairwise_distances(
                aligned, client_ids
            ),
            'modes': modes,
            'conflict_rows': conflict_rows,
            'source_matrix': source_matrix,
            'packages': packages,
            'summary': summary,
        }


def save_candidate_artifacts(log_path, analysis):
    root = os.path.join(log_path, 'koopman', 'candidates')
    os.makedirs(root, exist_ok=True)
    torch_save(root, 'latest.pt', analysis)
    summary = dict(analysis['summary'])
    summary['source_matrix'] = analysis['source_matrix'].tolist()
    summary['functional_map_reports'] = analysis['functional_map_reports']
    temporary = os.path.join(root, 'latest.json.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, os.path.join(root, 'latest.json'))
    history = os.path.join(root, 'server_rounds.csv')
    new = not os.path.exists(history)
    with open(history, 'a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow([
                'round', 'reference_client_id', 'raw_external_count',
                'conflict_rejected_count', 'conflict_rejection_ratio',
                'conflict_threshold', 'conflict_median', 'conflict_q95',
                'conflict_max', 'retained_count', 'mean_functional_map_residual',
                'max_functional_map_orthogonality_error',
            ])
        writer.writerow([
            summary[key] for key in (
                'round', 'reference_client_id', 'raw_external_count',
                'conflict_rejected_count', 'conflict_rejection_ratio',
                'conflict_threshold', 'conflict_median', 'conflict_q95',
                'conflict_max', 'retained_count', 'mean_functional_map_residual',
                'max_functional_map_orthogonality_error',
            )
        ])
    return root


__all__ = [
    'CANDIDATE_PROTOCOL', 'ConflictCandidateBuilder',
    'make_koopman_packet', 'save_candidate_artifacts',
]
