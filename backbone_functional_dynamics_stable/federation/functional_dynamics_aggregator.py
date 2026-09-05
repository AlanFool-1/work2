"""Multi-prototype aggregation of aligned low-rank dynamics summaries.

Bootstrap sees individual low-dimensional summaries in this experimental
implementation. This is centralized diagnostic clustering, not cryptographic
secure aggregation.
"""

from __future__ import annotations

import numpy as np


def _polar(matrix):
    left, _, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def _procrustes_distance(first, second):
    affinity = np.linalg.svd(second @ first.T, compute_uv=False).sum()
    squared = np.sum(first * first) + np.sum(second * second) - 2.0 * affinity
    return float(np.sqrt(max(0.0, squared)))


class FunctionalDynamicsAggregator:
    def __init__(self, ema_a=0.9, ema_b=0.9, num_prototypes=1, min_cluster_size=1):
        self.ema_a = float(ema_a)
        self.ema_b = float(ema_b)
        self.num_prototypes = int(num_prototypes)
        self.min_cluster_size = int(min_cluster_size)
        if self.num_prototypes <= 0:
            raise ValueError('num_prototypes must be positive.')
        if self.min_cluster_size <= 0:
            raise ValueError('min_cluster_size must be positive.')
        self.A_stars = None
        self.B_stars = None
        self.bootstrap_assignments = {}
        self.cluster_counts = [0] * self.num_prototypes

    @staticmethod
    def _normalized_weights(weights, indices):
        selected = np.asarray([weights[index] for index in indices], dtype=np.float64)
        total = float(selected.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError('Cluster weights must have positive finite sum.')
        return selected / total

    @staticmethod
    def _weighted_sum(values, weights):
        return np.sum([
            np.asarray(value) * float(weight)
            for value, weight in zip(values, weights)
        ], axis=0)

    def _distance_matrix(self, descriptors):
        count = len(descriptors)
        distances = np.zeros((count, count), dtype=np.float64)
        for first in range(count):
            for second in range(first + 1, count):
                value = _procrustes_distance(descriptors[first], descriptors[second])
                distances[first, second] = distances[second, first] = value
        return distances

    def _k_medoids(self, distances, max_steps=50):
        count = distances.shape[0]
        if self.num_prototypes > count:
            raise ValueError('num_prototypes cannot exceed participating clients.')
        medoids = [int(np.argmin(distances.sum(axis=1)))]
        while len(medoids) < self.num_prototypes:
            nearest = distances[:, medoids].min(axis=1)
            nearest[medoids] = -1.0
            medoids.append(int(np.argmax(nearest)))
        assignments = np.full(count, -1, dtype=np.int64)
        for _ in range(max_steps):
            updated = np.argmin(distances[:, medoids], axis=1)
            new_medoids = []
            for cluster in range(self.num_prototypes):
                members = np.flatnonzero(updated == cluster)
                if len(members) == 0:
                    candidates = [i for i in range(count) if i not in new_medoids]
                    choice = max(candidates, key=lambda i: distances[i, medoids].min())
                else:
                    within = distances[np.ix_(members, members)].sum(axis=1)
                    choice = int(members[np.argmin(within)])
                new_medoids.append(choice)
            if new_medoids == medoids and np.array_equal(updated, assignments):
                assignments = updated
                break
            medoids, assignments = new_medoids, updated
        counts = np.bincount(assignments, minlength=self.num_prototypes)
        for cluster in range(self.num_prototypes):
            while counts[cluster] < self.min_cluster_size:
                candidates = [
                    index for index in range(count)
                    if counts[assignments[index]] > self.min_cluster_size
                ]
                if not candidates:
                    raise ValueError('Cannot satisfy the requested minimum cluster size.')
                choice = min(
                    candidates,
                    key=lambda index: (
                        distances[index, medoids[cluster]]
                        - distances[index, medoids[assignments[index]]],
                        index,
                    ),
                )
                counts[assignments[choice]] -= 1
                assignments[choice] = cluster
                counts[cluster] += 1
        for cluster in range(self.num_prototypes):
            members = np.flatnonzero(assignments == cluster)
            within = distances[np.ix_(members, members)].sum(axis=1)
            medoids[cluster] = int(members[np.argmin(within)])
        return assignments, medoids

    def _bootstrap(self, a_values, b_values, weights, client_ids):
        if self.num_prototypes == 1:
            assignments = np.zeros(len(b_values), dtype=np.int64)
            medoids = [0]
        else:
            assignments, medoids = self._k_medoids(self._distance_matrix(b_values))
        self.A_stars, self.B_stars = [], []
        for cluster in range(self.num_prototypes):
            indices = np.flatnonzero(assignments == cluster).tolist()
            reference = b_values[medoids[cluster]]
            aligned_a, aligned_b = [], []
            for index in indices:
                transform = _polar(reference @ b_values[index].T)
                aligned_a.append(transform @ a_values[index] @ transform.T)
                aligned_b.append(transform @ b_values[index])
            local_weights = self._normalized_weights(weights, indices)
            self.A_stars.append(self._weighted_sum(aligned_a, local_weights))
            self.B_stars.append(self._weighted_sum(aligned_b, local_weights))
        self.cluster_counts = np.bincount(
            assignments, minlength=self.num_prototypes
        ).astype(int).tolist()
        self.bootstrap_assignments = {
            int(client_ids[index]): int(assignments[index])
            for index in range(len(client_ids))
        }

    def update(self, a_values, b_values, weights=None, cluster_ids=None, client_ids=None):
        if not a_values or len(a_values) != len(b_values):
            raise ValueError('Functional A/B payload counts must match and be nonzero.')
        weights = [1.0 / len(a_values)] * len(a_values) if weights is None else list(weights)
        client_ids = list(range(len(a_values))) if client_ids is None else list(client_ids)
        a_values = [np.asarray(value) for value in a_values]
        b_values = [np.asarray(value) for value in b_values]
        if self.A_stars is None:
            self._bootstrap(a_values, b_values, weights, client_ids)
            return self.state_dict()
        if cluster_ids is None and self.num_prototypes == 1:
            cluster_ids = [0] * len(a_values)
        if cluster_ids is None or len(cluster_ids) != len(a_values):
            raise ValueError('Established prototypes require one cluster_id per client.')
        counts = [0] * self.num_prototypes
        for cluster in range(self.num_prototypes):
            indices = [i for i, value in enumerate(cluster_ids) if int(value) == cluster]
            counts[cluster] = len(indices)
            if not indices:
                continue
            local_weights = self._normalized_weights(weights, indices)
            a_bar = self._weighted_sum([a_values[i] for i in indices], local_weights)
            b_bar = self._weighted_sum([b_values[i] for i in indices], local_weights)
            self.A_stars[cluster] = self.ema_a * self.A_stars[cluster] + (1.0 - self.ema_a) * a_bar
            self.B_stars[cluster] = self.ema_b * self.B_stars[cluster] + (1.0 - self.ema_b) * b_bar
        self.cluster_counts = counts
        return self.state_dict()

    def state_dict(self):
        if self.num_prototypes == 1:
            a_star = None if self.A_stars is None else self.A_stars[0]
            b_star = None if self.B_stars is None else self.B_stars[0]
        else:
            a_star, b_star = self.A_stars, self.B_stars
        return {
            'A_star': a_star,
            'B_star': b_star,
            'num_prototypes': self.num_prototypes,
            'min_cluster_size': self.min_cluster_size,
            'cluster_counts': self.cluster_counts,
            'bootstrap_assignments': self.bootstrap_assignments,
            'clustering_mode': 'centralized_diagnostic_kmedoids',
        }
