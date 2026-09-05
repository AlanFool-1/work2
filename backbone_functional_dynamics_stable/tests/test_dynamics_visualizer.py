import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from modules.dynamics_visualizer import (
    compute_aligned_dynamics_tsne,
    compute_joint_dynamics_tsne,
    load_dynamics_snapshots,
    plot_dynamics_tsne_html,
    save_dynamics_snapshot,
)


class DynamicsVisualizerTest(unittest.TestCase):
    def test_snapshot_joint_embedding_and_html(self):
        with tempfile.TemporaryDirectory() as root:
            for client_id in range(2):
                generator = torch.Generator().manual_seed(client_id)
                save_dynamics_snapshot(
                    root,
                    client_id,
                    torch.randn(3, 18, 4, generator=generator),
                    torch.arange(18) % 3,
                    severity=float(client_id),
                )
            snapshots = load_dynamics_snapshots(root, 2)
            self.assertEqual(snapshots[0]['states'].shape, (3, 18, 4))
            projected = compute_joint_dynamics_tsne(
                snapshots, max_points_per_client=12,
                random_state=3, perplexity=5,
            )
            self.assertEqual(projected[0]['z'].shape, (3, 12, 2))
            self.assertEqual(len(np.unique(projected[0]['sample_indices'])), 12)
            aligned = compute_aligned_dynamics_tsne(
                snapshots, max_points_per_client=12,
                random_state=3, perplexity=5,
            )
            self.assertEqual(aligned[1]['z'].shape, (3, 12, 2))
            output = plot_dynamics_tsne_html(
                root, 2, 3, max_points_per_client=12,
                random_state=3, perplexity=5,
            )
            self.assertTrue(Path(output).is_file())
            self.assertIn('plotly', Path(output).read_text(encoding='utf-8')[:1000].lower())


if __name__ == '__main__':
    unittest.main()
