import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.v01.model import (
    LatentLinearGraphDynamics,
    normalize_graph,
    normalized_aggregate,
)


class V01ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.nodes = 6
        source = torch.arange(self.nodes)
        target = torch.roll(source, -1)
        self.data = Data(
            x=torch.randn(self.nodes, 5),
            edge_index=torch.stack([
                torch.cat([source, target]),
                torch.cat([target, source]),
            ]),
            y=torch.arange(self.nodes) % 3,
        )

    def make_model(self, steps=4, reconstruction=False):
        return LatentLinearGraphDynamics(
            input_dim=5,
            output_dim=3,
            latent_dim=7,
            encoder_width=11,
            num_steps=steps,
            step_size=0.1,
            gamma=0.2,
            reconstruction=reconstruction,
        )

    def normalized_graph(self):
        return normalize_graph(
            self.data.edge_index,
            None,
            self.nodes,
            self.data.x.dtype,
        )

    def test_shapes_and_optional_reconstruction(self):
        model = self.make_model(reconstruction=True)
        output = model.forward_with_aux(self.data)
        self.assertEqual(tuple(output.logits.shape), (self.nodes, 3))
        self.assertEqual(tuple(output.reconstruction.shape), (self.nodes, 5))
        self.assertEqual(
            tuple(output.latent_trajectory.shape), (5, self.nodes, 7)
        )

        without_decoder = self.make_model(reconstruction=False)
        self.assertIsNone(
            without_decoder.forward_with_aux(self.data).reconstruction
        )
        self.assertFalse(any(
            name.startswith('decoder.')
            for name in without_decoder.state_dict()
        ))

    def test_normalized_aggregate_matches_known_operator(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        normalized_index, normalized_weight = normalize_graph(
            edge_index, None, num_nodes=3, dtype=torch.float32
        )
        features = torch.tensor([[2.0], [4.0], [9.0]])
        actual = normalized_aggregate(
            features, normalized_index, normalized_weight
        )
        expected = torch.tensor([[3.0], [3.0], [9.0]])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_joint_node_permutation_equivariance(self):
        model = self.make_model().eval()
        original = model(self.data)

        permutation = torch.tensor([2, 5, 0, 3, 1, 4])
        old_to_new = torch.empty_like(permutation)
        old_to_new[permutation] = torch.arange(self.nodes)
        permuted = Data(
            x=self.data.x[permutation],
            edge_index=old_to_new[self.data.edge_index],
        )
        actual = model(permuted)
        self.assertTrue(torch.allclose(
            actual, original[permutation], atol=1e-6, rtol=1e-5
        ))

    def test_fixed_graph_propagation_is_linear(self):
        model = self.make_model(steps=3)
        normalized_index, normalized_weight = self.normalized_graph()
        first = torch.randn(self.nodes, model.latent_dim)
        second = torch.randn_like(first)
        scale = -1.7

        propagated_sum = model.propagate_latent(
            first + second, normalized_index, normalized_weight
        )
        sum_propagated = model.propagate_latent(
            first, normalized_index, normalized_weight
        ) + model.propagate_latent(
            second, normalized_index, normalized_weight
        )
        propagated_scaled = model.propagate_latent(
            scale * first, normalized_index, normalized_weight
        )
        scaled_propagated = scale * model.propagate_latent(
            first, normalized_index, normalized_weight
        )

        self.assertTrue(torch.allclose(
            propagated_sum, sum_propagated, atol=1e-6, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            propagated_scaled, scaled_propagated, atol=1e-6, rtol=1e-5
        ))

    def test_zero_steps_are_encoder_then_readout(self):
        model = self.make_model(steps=0).eval()
        normalized_index, normalized_weight = self.normalized_graph()
        initial = model.encode_nodes(
            self.data, normalized_index, normalized_weight
        )
        output = model.forward_with_aux(self.data)
        self.assertEqual(tuple(output.latent_trajectory.shape), (1, 6, 7))
        self.assertTrue(torch.allclose(output.latent_trajectory[0], initial))
        self.assertTrue(torch.allclose(output.logits, model.readout(initial)))

    def test_self_generator_has_required_symmetric_part(self):
        model = self.make_model()
        generator = model.self_generator()
        expected = -2.0 * model.gamma * torch.eye(model.latent_dim)
        self.assertTrue(torch.allclose(
            generator + generator.T, expected, atol=1e-7
        ))

    def test_supervised_path_has_finite_gradients(self):
        model = self.make_model()
        loss = F.cross_entropy(model(self.data), self.data.y)
        loss.backward()
        required = [
            model.encoder.net[0].weight,
            model.encoder.net[2].weight,
            model.self_generator_raw,
            model.neighbor_generator,
            model.readout.weight,
        ]
        self.assertTrue(all(parameter.grad is not None for parameter in required))
        self.assertTrue(all(
            torch.isfinite(parameter.grad).all() for parameter in required
        ))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
