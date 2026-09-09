import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.v02.diagnostics import RidgeGraphTransition, StableTransition, graph_gain, per_time_nmse, rollout
from models.v02.model import GraphKoopmanBackbone, graph_operator


class V02DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)
        self.data = Data(x=torch.randn(5, 3, dtype=torch.float64), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
        self.operator = graph_operator(self.data)

    def test_affine_fit_predicts_unseen_starts_and_matches_augmented_linear_map(self):
        dim = 2
        transition = RidgeGraphTransition(
            torch.tensor([[0.02, -0.05], [0.03, 0.04]], dtype=torch.float64),
            torch.tensor([[0.01, 0.02], [-0.03, 0.01]], dtype=torch.float64),
            torch.tensor([0.1, -0.2], dtype=torch.float64),
        )
        train = rollout(transition, torch.randn(12, 5, dim, dtype=torch.float64), self.operator, 4)
        fitted = RidgeGraphTransition.fit(train, self.operator, ridge=1e-10, affine=True)
        unseen = torch.randn(3, 5, dim, dtype=torch.float64)
        torch.testing.assert_close(rollout(fitted, unseen, self.operator, 8), rollout(transition, unseen, self.operator, 8), atol=1e-7, rtol=1e-7)
        before = {key: value.clone() for key, value in fitted.state_dict().items()}
        rollout(fitted, unseen, self.operator, 8)
        for key, value in fitted.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_graph_frequency_gain_equals_dense_rk4_operator_gain(self):
        model = GraphKoopmanBackbone(3, 2, state_dim=4, latent_dim=2, width=5, num_steps=3).double()
        transition = StableTransition(model.generator, model.step_size)
        basis = torch.eye(10, dtype=torch.float64).reshape(10, 5, 2)
        dense = rollout(transition, basis, self.operator, model.num_steps)[:, -1].reshape(10, 10)
        actual = graph_gain(model, self.operator)['final_operator_norm']
        self.assertAlmostEqual(actual, float(torch.linalg.svdvals(dense).max()), places=10)

    def test_per_time_normalization_exposes_small_initial_state_error(self):
        target = torch.tensor([[[[1.0]], [[10.0]]]])
        predicted = target - 1
        torch.testing.assert_close(per_time_nmse(predicted, target), torch.tensor([1.0, 0.01]))


if __name__ == '__main__':
    unittest.main()
