import tempfile
import unittest

import numpy as np
import torch

from modules.initial_state_visualizer import (
    compute_distance_to_client0,
    compute_shared_pca,
    compute_shared_tsne,
    load_initial_state_snapshots,
    save_initial_state_snapshot,
)


class InitialStateVisualizerTest(unittest.TestCase):
    def test_snapshot_shape_and_shared_pca(self):
        with tempfile.TemporaryDirectory() as root:
            for client_id in range(2):
                save_initial_state_snapshot(root, client_id, torch.randn(12, 4), torch.arange(12) % 3, severity=client_id)
            snapshots = load_initial_state_snapshots(root, 2)
            self.assertEqual(snapshots[0]['h'].shape, (12, 4))
            self.assertEqual(snapshots[0]['y'].shape, (12,))
            pca, projected = compute_shared_pca(snapshots, random_state=3)
            self.assertEqual(pca.components_.shape, (2, 4))
            self.assertEqual(projected[1]['z'].shape, (12, 2))

    def test_reference_distance_zero_and_missing_class_safe(self):
        snapshots = [
            {'client_id': 0, 'severity': 0.0, 'h': np.asarray([[0., 0.], [1., 1.], [2., 2.]]), 'y': np.asarray([0, 1, 2])},
            {'client_id': 1, 'severity': 1.0, 'h': np.asarray([[1., 0.], [3., 2.]]), 'y': np.asarray([0, 2])},
        ]
        rows = compute_distance_to_client0(snapshots, 3)
        self.assertLess(rows[0]['h0_centroid_distance_to_client0'], 1e-8)
        self.assertTrue(np.isfinite(rows[1]['h0_centroid_distance_to_client0']))

    def test_shared_tsne_samples_and_splits_clients(self):
        rng = np.random.default_rng(7)
        snapshots = [
            {'client_id': client_id, 'severity': float(client_id), 'h': rng.normal(size=(18, 4)), 'y': np.arange(18) % 3}
            for client_id in range(2)
        ]
        projected = compute_shared_tsne(snapshots, max_points_per_client=12, random_state=3, perplexity=5)
        self.assertEqual(len(projected), 2)
        self.assertEqual(projected[0]['z'].shape, (12, 2))
        self.assertEqual(projected[1]['z'].shape, (12, 2))
        self.assertEqual(set(projected[0]['y']), {0, 1, 2})


if __name__ == '__main__':
    unittest.main()
