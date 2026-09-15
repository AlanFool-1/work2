"""Functional-Map-aligned server Koopman mode graph for FedDGN.

This module is deliberately task-passive.  It turns detached client Koopman
analyses into a shared comparison space, discovers spatial communities, and
prepares a global mode bank plus client-specific views.  It never modifies a
task model, Koopman operator, or optimizer state.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from misc.utils import torch_save
from models.feddgn.functional_map import (
    FUNCTIONAL_MAP_METHOD,
    identity_functional_map,
    solve_functional_map,
)


MODE_GRAPH_PROTOCOL = 'mode_graph_v2_functional_map_louvain'


def _array(value, *, name: str) -> np.ndarray:
    """Convert a CPU/GPU tensor or array to an owned NumPy array."""

    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.size == 0:
        raise ValueError(f'Koopman {name} must not be empty.')
    return result.copy()


def _finite_complex(value: np.ndarray) -> np.ndarray:
    return np.isfinite(value.real) & np.isfinite(value.imag)


def make_mode_upload(
    result: Mapping[str, object], client_id: int, top_modes: int
) -> Dict[str, object]:
    """Select reliable non-constant modes and make a process-safe payload.

    Arrays are mode-major in the upload.  The fixed constant observable gives
    one near-unit eigenvalue; that trivial mode is removed before ranking.
    Remaining modes are ranked by normalized spectral residual, with spectral
    magnitude used only as a deterministic tie breaker.
    """

    if int(top_modes) <= 0:
        raise ValueError('mode_graph_top_modes must be positive.')
    eigenvalues = _array(result['eigenvalues'], name='eigenvalues').reshape(-1)
    eigenvectors = _array(result['eigenvectors'], name='eigenvectors')
    residuals = _array(result['mode_residuals'], name='residuals').reshape(-1)
    state_modes = _array(result['modes'], name='state modes')
    operator = _array(result['operator'], name='operator')
    count = eigenvalues.size
    if operator.ndim != 2 or operator.shape != (
        eigenvectors.shape[0], eigenvectors.shape[0]
    ):
        raise ValueError('Koopman operator and observable coordinates disagree.')
    if eigenvectors.ndim != 2 or eigenvectors.shape[1] != count:
        raise ValueError('Koopman eigenvectors must be [observable_dim, modes].')
    if state_modes.ndim != 2 or state_modes.shape[1] != count:
        raise ValueError('Koopman state modes must be [state_dim, modes].')
    if residuals.size != count:
        raise ValueError('Every Koopman mode must have one residual.')

    state_norms = np.linalg.norm(state_modes, axis=0)
    valid = (
        _finite_complex(eigenvalues)
        & np.isfinite(residuals)
        & (residuals >= 0.0)
        & np.isfinite(state_norms)
        & (state_norms > np.finfo(np.float32).eps)
        & np.all(_finite_complex(eigenvectors), axis=0)
        & np.all(_finite_complex(state_modes), axis=0)
    )
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        raise ValueError('No finite non-zero Koopman modes are available.')
    if candidates.size > 1:
        trivial = candidates[np.argmin(np.abs(eigenvalues[candidates] - 1.0))]
        candidates = candidates[candidates != trivial]
    if candidates.size == 0:
        raise ValueError('Only the constant Koopman mode is available.')

    order = np.lexsort((
        -np.abs(eigenvalues[candidates]),
        residuals[candidates],
    ))
    selected = candidates[order[:min(int(top_modes), candidates.size)]]
    diagnostics = result.get('diagnostics', {})
    spectral_residual = diagnostics.get('koopman_spectral_residual')
    if spectral_residual is None:
        spectral_residual = np.sqrt(np.mean(np.square(residuals)))
    if hasattr(spectral_residual, 'detach'):
        spectral_residual = spectral_residual.detach().cpu().item()
    spectral_residual = float(spectral_residual)
    if not np.isfinite(spectral_residual) or spectral_residual < 0.0:
        raise ValueError('Koopman spectral residual must be finite and non-negative.')
    payload = {
        'protocol': MODE_GRAPH_PROTOCOL,
        'client_id': int(client_id),
        'operator': operator.copy(),
        'spectral_residual': spectral_residual,
        'local_mode_indices': selected.astype(np.int64, copy=True),
        'eigenvalues': eigenvalues[selected].copy(),
        'eigenvectors': eigenvectors[:, selected].T.copy(),
        'residuals': residuals[selected].astype(np.float32, copy=True),
        'state_modes': state_modes[:, selected].T.copy(),
    }
    payload['bytes'] = int(sum(
        value.nbytes for value in payload.values()
        if isinstance(value, np.ndarray)
    ))
    return payload


def _stable_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber arbitrary cluster labels by their first graph-node index."""

    groups = sorted(
        np.unique(labels).tolist(),
        key=lambda label: int(np.flatnonzero(labels == label)[0]),
    )
    mapping = {label: index for index, label in enumerate(groups)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def _louvain_labels(
    adjacency: np.ndarray, resolution: float, seed: int
) -> np.ndarray:
    """Discover communities without preselecting their number."""

    import networkx as nx

    count = adjacency.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    sources, targets = np.nonzero(np.triu(adjacency, k=1) > 0.0)
    graph.add_weighted_edges_from(
        (
            int(source),
            int(target),
            float(adjacency[source, target]),
        )
        for source, target in zip(sources, targets)
    )
    if graph.number_of_edges() == 0:
        return np.arange(count, dtype=np.int64)
    communities = nx.community.louvain_communities(
        graph,
        weight='weight',
        resolution=float(resolution),
        threshold=1e-7,
        seed=int(seed),
    )
    labels = np.empty(count, dtype=np.int64)
    for label, members in enumerate(
        sorted(communities, key=lambda group: min(group))
    ):
        labels[list(members)] = label
    return _stable_labels(labels)


def _mode_id(client_id: int, local_mode_index: int) -> str:
    return f'client_{int(client_id)}:mode_{int(local_mode_index)}'


class ModeGraphBuilder:
    """Build the global graph and passive personalized mode views."""

    def __init__(
        self,
        *,
        neighbors: int = 5,
        personalized_modes: int = 16,
        resolution: float = 1.0,
        functional_map_steps: int = 8,
        functional_map_tolerance: float = 1e-7,
        seed: int = 0,
    ) -> None:
        if min(int(neighbors), int(personalized_modes)) <= 0:
            raise ValueError('Mode graph sizes must be positive.')
        if float(resolution) <= 0.0:
            raise ValueError('Louvain resolution must be positive.')
        if int(functional_map_steps) < 0:
            raise ValueError('Functional-map steps must be non-negative.')
        if float(functional_map_tolerance) <= 0.0:
            raise ValueError('Functional-map tolerance must be positive.')
        self.neighbors = int(neighbors)
        self.personalized_modes = int(personalized_modes)
        self.resolution = float(resolution)
        self.functional_map_steps = int(functional_map_steps)
        self.functional_map_tolerance = float(functional_map_tolerance)
        self.seed = int(seed)
        # Version 1 keeps the initial K_s fixed.  Re-selecting it every round
        # would move the comparison coordinates and make community evolution
        # uninterpretable.  Iterative K_s updates belong to a later version.
        self.reference_client_id = None
        self.reference_operator = None
        self.reference_spectral_residual = None
        self.reference_round = None

    @staticmethod
    def _combine_uploads(uploads: Iterable[Mapping[str, object]]):
        ordered = sorted(uploads, key=lambda item: int(item['client_id']))
        if not ordered:
            raise ValueError('Mode graph requires at least one client upload.')
        protocols = {str(item.get('protocol')) for item in ordered}
        if protocols != {MODE_GRAPH_PROTOCOL}:
            raise ValueError(f'Unexpected Koopman upload protocols: {protocols}.')
        client_ids = []
        local_indices = []
        eigenvalues = []
        eigenvectors = []
        residuals = []
        state_modes = []
        operators = {}
        spectral_residuals = {}
        state_dimensions = set()
        observable_dimensions = set()
        for item in ordered:
            client_id = int(item['client_id'])
            values = _array(item['eigenvalues'], name='uploaded eigenvalues').reshape(-1)
            indices = _array(
                item['local_mode_indices'], name='uploaded mode indices'
            ).reshape(-1)
            vectors = _array(
                item['eigenvectors'], name='uploaded eigenvectors'
            )
            modes = _array(item['state_modes'], name='uploaded state modes')
            errors = _array(item['residuals'], name='uploaded residuals').reshape(-1)
            operator = _array(item['operator'], name='uploaded operator')
            spectral_residual = float(item['spectral_residual'])
            count = values.size
            if (
                indices.size != count
                or errors.size != count
                or vectors.ndim != 2
                or vectors.shape[0] != count
                or modes.ndim != 2
                or modes.shape[0] != count
                or operator.ndim != 2
                or operator.shape[0] != operator.shape[1]
                or operator.shape[0] != vectors.shape[1]
            ):
                raise ValueError(f'Client {client_id} uploaded inconsistent modes.')
            if not (
                np.all(_finite_complex(values))
                and np.all(_finite_complex(vectors))
                and np.all(_finite_complex(modes))
                and np.all(np.isfinite(errors))
                and np.all(_finite_complex(operator))
                and np.isfinite(spectral_residual)
                and spectral_residual >= 0.0
            ):
                raise ValueError(f'Client {client_id} uploaded non-finite modes.')
            if client_id in operators:
                raise ValueError(f'Duplicate Koopman upload for client {client_id}.')
            state_dimensions.add(modes.shape[1])
            observable_dimensions.add(vectors.shape[1])
            client_ids.append(np.full(count, client_id, dtype=np.int64))
            local_indices.append(indices.astype(np.int64, copy=False))
            eigenvalues.append(values)
            eigenvectors.append(vectors)
            residuals.append(errors.astype(np.float64, copy=False))
            state_modes.append(modes)
            operators[client_id] = operator
            spectral_residuals[client_id] = spectral_residual
        if len(state_dimensions) != 1 or len(observable_dimensions) != 1:
            raise ValueError('Client mode coordinates have incompatible widths.')
        return {
            'client_ids': np.concatenate(client_ids),
            'local_mode_indices': np.concatenate(local_indices),
            'eigenvalues': np.concatenate(eigenvalues),
            'eigenvectors': np.concatenate(eigenvectors, axis=0),
            'residuals': np.concatenate(residuals),
            'state_modes': np.concatenate(state_modes, axis=0),
            'operators': operators,
            'spectral_residuals': spectral_residuals,
        }

    def _align_observable_modes(
        self, combined: Mapping[str, object], round_number: int
    ) -> Dict[str, object]:
        """Map every selected local eigenvector into one reference K-space."""

        residuals = combined['spectral_residuals']
        if self.reference_operator is None:
            reference_client_id = min(
                residuals,
                key=lambda client_id: (residuals[client_id], client_id),
            )
            self.reference_client_id = int(reference_client_id)
            self.reference_operator = combined['operators'][
                reference_client_id
            ].copy()
            self.reference_spectral_residual = float(
                residuals[reference_client_id]
            )
            self.reference_round = int(round_number)
        reference_client_id = int(self.reference_client_id)
        reference_operator = self.reference_operator
        client_ids = combined['client_ids']
        raw_modes = combined['eigenvectors'].astype(np.complex128, copy=False)
        transported = np.empty_like(raw_modes, dtype=np.complex64)
        functional_maps = {}
        diagnostics = []
        for client_id in sorted(combined['operators']):
            if np.array_equal(
                combined['operators'][client_id], reference_operator
            ):
                functional_map, report = identity_functional_map(
                    reference_operator
                )
            else:
                functional_map, report = solve_functional_map(
                    combined['operators'][client_id],
                    reference_operator,
                    refinement_steps=self.functional_map_steps,
                    tolerance=self.functional_map_tolerance,
                )
            indices = np.flatnonzero(client_ids == client_id)
            transported[indices] = (
                functional_map @ raw_modes[indices].T
            ).T
            functional_maps[int(client_id)] = functional_map
            diagnostics.append({
                **report,
                'client_id': int(client_id),
                'is_reference': bool(client_id == reference_client_id),
                'source_spectral_residual': float(residuals[client_id]),
            })
        return {
            'reference_client_id': int(reference_client_id),
            'reference_operator': reference_operator.copy(),
            'reference_spectral_residual': float(
                self.reference_spectral_residual
            ),
            'reference_round': int(self.reference_round),
            'functional_maps': functional_maps,
            'functional_map_diagnostics': diagnostics,
            'transported_modes': transported,
        }

    @staticmethod
    def _spatial_similarity(
        combined: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        modes = combined['transported_modes'].astype(
            np.complex128, copy=False
        )
        norms = np.linalg.norm(modes, axis=1, keepdims=True)
        normalized_modes = modes / np.maximum(
            norms, np.finfo(np.float64).eps
        )
        # The graph has exactly one signal after Functional Map alignment:
        # phase/sign-invariant direction similarity in shared observable space.
        # Eigenvalues and residuals never directly alter an edge weight.
        similarity = np.abs(normalized_modes.conj() @ normalized_modes.T)
        similarity = np.clip(similarity.real, 0.0, 1.0)
        np.fill_diagonal(similarity, 0.0)
        return similarity.astype(np.float32)

    def _balanced_cross_client_knn(
        self,
        combined: Mapping[str, np.ndarray],
        spatial_similarity: np.ndarray,
    ) -> np.ndarray:
        """Keep top spatial neighbors from distinct foreign clients only."""

        client_ids = combined['client_ids']
        unique_clients = np.unique(client_ids)
        adjacency = np.zeros_like(spatial_similarity, dtype=np.float32)
        for source in range(client_ids.size):
            candidates = []
            for other_client in unique_clients:
                if other_client == client_ids[source]:
                    continue
                client_nodes = np.flatnonzero(client_ids == other_client)
                weights = spatial_similarity[source, client_nodes]
                target = int(client_nodes[np.argmax(weights)])
                candidates.append((
                    float(spatial_similarity[source, target]), target
                ))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for weight, target in candidates[:self.neighbors]:
                if weight > 0.0:
                    adjacency[source, target] = weight
        # A union graph is less brittle than strict mutual kNN while every edge
        # still links two different clients and retains its raw spatial weight.
        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    @staticmethod
    def _cluster_quality(
        spatial_similarity: np.ndarray,
        adjacency: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Measure communities in the full geometry and sparse graph.

        Louvain modularity alone is circular because Louvain optimizes it on
        the sparsified graph.  The full-distance silhouette and within/between
        separation expose communities whose original geometry still overlaps.
        """

        count = labels.size
        off_diagonal = ~np.eye(count, dtype=bool)
        same = labels[:, None] == labels[None, :]
        within_mask = same & off_diagonal
        between_mask = (~same) & off_diagonal
        within_similarity = (
            float(spatial_similarity[within_mask].mean())
            if np.any(within_mask) else 0.0
        )
        between_similarity = (
            float(spatial_similarity[between_mask].mean())
            if np.any(between_mask) else 0.0
        )
        within_distance = 1.0 - within_similarity
        between_distance = 1.0 - between_similarity

        distance = 1.0 - spatial_similarity.astype(np.float64)
        np.fill_diagonal(distance, 0.0)
        silhouettes = np.zeros(count, dtype=np.float64)
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        for index in range(count):
            own = groups[int(labels[index])]
            own = own[own != index]
            if own.size == 0:
                continue
            intra = float(distance[index, own].mean())
            other_groups = [
                group for group in groups
                if labels[index] != labels[group[0]]
            ]
            if not other_groups:
                continue
            inter = min(
                float(distance[index, group].mean())
                for group in other_groups
            )
            denominator = max(intra, inter)
            if denominator > 0.0:
                silhouettes[index] = (inter - intra) / denominator

        degrees = adjacency.sum(axis=1, dtype=np.float64)
        total_edge_weight = float(degrees.sum() / 2.0)
        modularity = 0.0
        conductances = []
        if total_edge_weight > 0.0:
            total_volume = 2.0 * total_edge_weight
            for group in groups:
                internal = float(
                    adjacency[np.ix_(group, group)].sum() / 2.0
                )
                volume = float(degrees[group].sum())
                modularity += (
                    internal / total_edge_weight
                    - (volume / total_volume) ** 2
                )
                outside = np.setdiff1d(
                    np.arange(count), group, assume_unique=True
                )
                cut = float(adjacency[np.ix_(group, outside)].sum())
                denominator = min(volume, total_volume - volume)
                if denominator > 0.0:
                    conductances.append(cut / denominator)
        return {
            'within_similarity': within_similarity,
            'between_similarity': between_similarity,
            'similarity_gap': within_similarity - between_similarity,
            'within_distance': within_distance,
            'between_distance': between_distance,
            'between_within_distance_ratio': (
                between_distance
                / max(within_distance, np.finfo(np.float64).eps)
            ),
            'silhouette': float(silhouettes.mean()),
            'modularity': float(modularity),
            'mean_conductance': (
                float(np.mean(conductances)) if conductances else 1.0
            ),
        }

    @staticmethod
    def _global_mode_bank(
        combined: Mapping[str, np.ndarray], labels: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], list]:
        """Build phase-aligned community representatives in shared K-space."""

        transported = combined['transported_modes'].astype(
            np.complex128, copy=False
        )
        values = combined['eigenvalues']
        client_ids = combined['client_ids']
        residuals = combined['residuals']
        representatives = []
        representative_values = []
        member_counts = []
        summaries = []
        for community_id in np.unique(labels):
            members = np.flatnonzero(labels == community_id)
            modes = transported[members]
            norms = np.linalg.norm(modes, axis=1, keepdims=True)
            modes = modes / np.maximum(norms, np.finfo(np.float64).eps)
            similarities = np.abs(modes.conj() @ modes.T)
            anchor = modes[int(np.argmax(similarities.sum(axis=1)))]
            overlaps = modes @ anchor.conj()
            phases = np.exp(-1j * np.angle(overlaps))
            representative = (modes * phases[:, None]).mean(axis=0)
            representative /= max(
                np.linalg.norm(representative), np.finfo(np.float64).eps
            )
            representative_value = values[members].mean()
            representatives.append(representative)
            representative_values.append(representative_value)
            member_counts.append(members.size)
            summaries.append({
                'community_id': int(community_id),
                'member_count': int(members.size),
                'client_ids': np.unique(client_ids[members]).astype(int).tolist(),
                'representative_eigenvalue_real': float(
                    representative_value.real
                ),
                'representative_eigenvalue_imag': float(
                    representative_value.imag
                ),
                'mean_residual': float(residuals[members].mean()),
            })
        bank = {
            'community_ids': np.unique(labels).astype(np.int64),
            'representative_modes': np.asarray(
                representatives, dtype=np.complex64
            ),
            'representative_eigenvalues': np.asarray(
                representative_values, dtype=np.complex64
            ),
            'member_counts': np.asarray(member_counts, dtype=np.int64),
        }
        return bank, summaries

    def _public_graph(
        self,
        combined: Mapping[str, np.ndarray],
        adjacency: np.ndarray,
        labels: np.ndarray,
        cluster_quality: Mapping[str, float],
        mode_bank_summary: list,
        round_number: int,
    ) -> Dict[str, object]:
        client_ids = combined['client_ids']
        local_indices = combined['local_mode_indices']
        values = combined['eigenvalues']
        residuals = combined['residuals']
        nodes = []
        for index in range(values.size):
            candidates = np.flatnonzero(adjacency[index] > 0.0)
            selected = candidates[np.argsort(
                -adjacency[index, candidates], kind='stable'
            )]
            neighbors = [
                {
                    'mode_id': _mode_id(client_ids[other], local_indices[other]),
                    'client_id': int(client_ids[other]),
                    'similarity': float(adjacency[index, other]),
                }
                for other in selected
            ]
            nodes.append({
                'node_index': int(index),
                'mode_id': _mode_id(client_ids[index], local_indices[index]),
                'client_id': int(client_ids[index]),
                'local_mode_index': int(local_indices[index]),
                'eigenvalue_real': float(values[index].real),
                'eigenvalue_imag': float(values[index].imag),
                'eigenvalue_magnitude': float(abs(values[index])),
                'residual': float(residuals[index]),
                'community_id': int(labels[index]),
                'neighbors': neighbors,
            })
        sources, targets = np.nonzero(np.triu(adjacency, k=1) > 0.0)
        edges = [
            {
                'source': _mode_id(client_ids[source], local_indices[source]),
                'target': _mode_id(client_ids[target], local_indices[target]),
                'similarity': float(adjacency[source, target]),
            }
            for source, target in zip(sources, targets)
        ]
        unique_clients = np.unique(client_ids)
        community_count = int(labels.max()) + 1
        counts = np.zeros(
            (unique_clients.size, community_count), dtype=np.int64
        )
        for row, client_id in enumerate(unique_clients):
            counts[row] = np.bincount(
                labels[client_ids == client_id], minlength=community_count
            )
        return {
            'protocol': MODE_GRAPH_PROTOCOL,
            'round': int(round_number),
            'comparison_space': 'functional_map_shared_koopman_observable',
            'functional_map_method': FUNCTIONAL_MAP_METHOD,
            'functional_map_refinement_steps': int(
                self.functional_map_steps
            ),
            'functional_map_tolerance': float(
                self.functional_map_tolerance
            ),
            'reference_client_id': int(combined['reference_client_id']),
            'reference_spectral_residual': float(
                combined['reference_spectral_residual']
            ),
            'reference_round': int(combined['reference_round']),
            'functional_map_diagnostics': list(
                combined['functional_map_diagnostics']
            ),
            'community_method': 'louvain',
            'edge_metric': 'phase_invariant_aligned_koopman_cosine',
            'edge_scope': 'cross_client_only',
            'neighbor_policy': 'best_per_foreign_client_then_top_k',
            'neighbors_per_mode': int(self.neighbors),
            'louvain_resolution': float(self.resolution),
            'community_count': community_count,
            'client_ids': unique_clients.astype(int).tolist(),
            'community_client_matrix': counts.tolist(),
            'cluster_quality': dict(cluster_quality),
            'global_mode_bank_summary': mode_bank_summary,
            'nodes': nodes,
            'edges': edges,
        }

    def _client_views(
        self,
        combined: Mapping[str, np.ndarray],
        spatial_similarity: np.ndarray,
        labels: np.ndarray,
        global_mode_bank: Mapping[str, np.ndarray],
        round_number: int,
    ) -> Dict[int, Dict[str, object]]:
        client_ids = combined['client_ids']
        local_indices = combined['local_mode_indices']
        views = {}
        for client_id in np.unique(client_ids):
            own = np.flatnonzero(client_ids == client_id)
            foreign = np.flatnonzero(client_ids != client_id)
            communities = np.unique(labels[own])
            related = foreign[np.isin(labels[foreign], communities)]
            if related.size:
                scores = spatial_similarity[np.ix_(related, own)].max(axis=1)
                order = np.argsort(-scores, kind='stable')
                related = related[order[:min(
                    self.personalized_modes, related.size
                )]]
                scores = spatial_similarity[np.ix_(related, own)].max(axis=1)
            else:
                scores = np.empty(0, dtype=np.float32)
            bank = np.concatenate((own, related))
            views[int(client_id)] = {
                'protocol': MODE_GRAPH_PROTOCOL,
                'round': int(round_number),
                'client_id': int(client_id),
                'comparison_space': (
                    'functional_map_shared_koopman_observable'
                ),
                'reference_client_id': int(combined['reference_client_id']),
                'functional_map': combined['functional_maps'][
                    int(client_id)
                ].copy(),
                'functional_map_diagnostics': next(
                    dict(report)
                    for report in combined['functional_map_diagnostics']
                    if int(report['client_id']) == int(client_id)
                ),
                'global_mode_bank': {
                    name: value.copy()
                    for name, value in global_mode_bank.items()
                },
                'own_count': int(own.size),
                'related_count': int(related.size),
                'own_mode_ids': [
                    _mode_id(client_ids[index], local_indices[index])
                    for index in own
                ],
                'related_mode_ids': [
                    _mode_id(client_ids[index], local_indices[index])
                    for index in related
                ],
                'source_client_ids': client_ids[bank].copy(),
                'local_mode_indices': local_indices[bank].copy(),
                'community_ids': labels[bank].copy(),
                'eigenvalues': combined['eigenvalues'][bank].copy(),
                'eigenvectors': combined['eigenvectors'][bank].copy(),
                'transported_modes': combined['transported_modes'][bank].copy(),
                'residuals': combined['residuals'][bank].astype(
                    np.float32, copy=True
                ),
                'state_modes': combined['state_modes'][bank].copy(),
                'related_similarity': scores.astype(np.float32, copy=True),
            }
        return views

    def build(
        self,
        uploads: Iterable[Mapping[str, object]],
        *,
        round_number: int,
    ) -> Tuple[Dict[str, object], Dict[int, Dict[str, object]]]:
        combined = self._combine_uploads(uploads)
        combined.update(self._align_observable_modes(combined, round_number))
        spatial_similarity = self._spatial_similarity(combined)
        adjacency = self._balanced_cross_client_knn(
            combined, spatial_similarity
        )
        labels = _louvain_labels(
            adjacency,
            resolution=self.resolution,
            # Keep the optimizer seed fixed across rounds so apparent
            # community evolution comes from the graph, not random restarts.
            seed=self.seed,
        )
        cluster_quality = self._cluster_quality(
            spatial_similarity, adjacency, labels
        )
        global_mode_bank, mode_bank_summary = self._global_mode_bank(
            combined, labels
        )
        public = self._public_graph(
            combined,
            adjacency,
            labels,
            cluster_quality,
            mode_bank_summary,
            round_number,
        )
        graph = {
            **public,
            'similarity': adjacency,
            'spatial_similarity': spatial_similarity,
            'client_id_per_node': combined['client_ids'].copy(),
            'local_mode_index_per_node': combined[
                'local_mode_indices'
            ].copy(),
            'community_id_per_node': labels.copy(),
            'eigenvalues': combined['eigenvalues'].copy(),
            'eigenvectors': combined['eigenvectors'].copy(),
            'transported_modes': combined['transported_modes'].copy(),
            'residuals': combined['residuals'].astype(np.float32, copy=True),
            'state_modes': combined['state_modes'].copy(),
            'reference_operator': combined['reference_operator'].copy(),
            'functional_maps': {
                int(client_id): value.copy()
                for client_id, value in combined['functional_maps'].items()
            },
            'global_mode_bank': global_mode_bank,
        }
        views = self._client_views(
            combined,
            spatial_similarity,
            labels,
            global_mode_bank,
            round_number,
        )
        return graph, views


def _atomic_json(path: str, payload: Mapping[str, object]) -> None:
    temporary = f'{path}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def save_mode_graph_artifacts(
    log_path: str,
    graph: Mapping[str, object],
    client_views: Mapping[int, Mapping[str, object]],
    *,
    archive: bool,
    artifact_directory: str = 'mode_graph',
) -> str:
    """Atomically save the latest graph and optional round history."""

    root = os.path.join(log_path, 'koopman', str(artifact_directory))
    torch_save(root, 'latest.pt', dict(graph))
    public = {
        key: value
        for key, value in graph.items()
        if key not in {
            'similarity', 'spatial_similarity', 'client_id_per_node',
            'local_mode_index_per_node', 'community_id_per_node',
            'eigenvalues', 'eigenvectors', 'transported_modes',
            'residuals', 'state_modes', 'reference_operator',
            'functional_maps', 'global_mode_bank',
        }
    }
    _atomic_json(os.path.join(root, 'latest.json'), public)
    if not archive:
        return root

    round_dir = os.path.join(root, f"round_{int(graph['round']):04d}")
    torch_save(round_dir, 'mode_graph.pt', dict(graph))
    _atomic_json(os.path.join(round_dir, 'global_view.json'), public)
    view_dir = os.path.join(round_dir, 'client_views')
    for client_id, view in sorted(client_views.items()):
        torch_save(view_dir, f'client_{int(client_id)}.pt', dict(view))
    return root


__all__ = [
    'MODE_GRAPH_PROTOCOL',
    'ModeGraphBuilder',
    'make_mode_upload',
    'save_mode_graph_artifacts',
]
