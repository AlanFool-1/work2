import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.s0.model import build_s0_model
from proxy.graph_koopman_v0 import GraphKoopmanV0


class KoopmanProxyV0Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.nodes = 5
        self.state_dim = 4
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
             [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]],
            dtype=torch.long,
        )
        self.reference = torch.randn(self.nodes, self.state_dim)
        self.states = torch.randn(self.nodes, self.state_dim)

    def make_proxy(self):
        proxy = GraphKoopmanV0(
            state_dim=self.state_dim,
            output_dim=3,
            latent_dim=6,
            condition_dim=3,
            width=8,
            queries=2,
            step_size=0.1,
        )
        cache = proxy.make_edge_cache(self.edge_index, None, self.nodes)
        condition, neighborhood = proxy.conditions(self.reference, cache)
        return proxy, cache, condition, neighborhood

    def test_fixed_matrix_rollout(self):
        proxy, _, _, _ = self.make_proxy()
        proxy.step_size = 1.0
        with torch.no_grad():
            proxy.generator.zero_()
            proxy.generator[0, 0] = 1.0
        latent = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
        rollout = proxy.rollout(latent, 2)
        expected = torch.tensor(
            [[[1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
              [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
              [4.0, 2.0, 0.0, 0.0, 0.0, 0.0]]]
        )
        self.assertTrue(torch.allclose(rollout, expected))

    def test_joint_node_permutation_equivariance(self):
        proxy, cache, condition, neighborhood = self.make_proxy()
        proxy.eval()
        mean, logvar = proxy.encode(self.states, condition, neighborhood, cache)
        decoded = proxy.decode(mean, condition, neighborhood)
        permutation = torch.tensor([2, 0, 4, 1, 3])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(self.nodes)
        permuted_edges = inverse[self.edge_index]
        permuted_cache = proxy.make_edge_cache(permuted_edges, None, self.nodes)
        permuted_reference = self.reference[permutation]
        permuted_state = self.states[permutation]
        permuted_condition, permuted_neighborhood = proxy.conditions(
            permuted_reference, permuted_cache
        )
        permuted_mean, permuted_logvar = proxy.encode(
            permuted_state,
            permuted_condition,
            permuted_neighborhood,
            permuted_cache,
        )
        permuted_decoded = proxy.decode(
            permuted_mean, permuted_condition, permuted_neighborhood
        )
        self.assertTrue(torch.allclose(mean, permuted_mean, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(logvar, permuted_logvar, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(decoded[:, permutation], permuted_decoded, atol=1e-6, rtol=1e-5))

    def test_frozen_readout_keeps_proxy_gradients(self):
        proxy, cache, condition, neighborhood = self.make_proxy()
        readout = nn.Linear(self.state_dim, 3)
        for parameter in readout.parameters():
            parameter.requires_grad_(False)
        mean, _ = proxy.encode(self.states, condition, neighborhood, cache)
        output = readout(proxy.decode(mean, condition, neighborhood)).sum()
        output.backward()
        gradients = [parameter.grad for parameter in proxy.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None))
        self.assertTrue(all(parameter.grad is None for parameter in readout.parameters()))

    def test_reconstruction_warmup_loss_is_finite(self):
        proxy, cache, condition, neighborhood = self.make_proxy()
        readout = nn.Linear(self.state_dim, 3)
        trajectory = torch.randn(2, 3, self.nodes, self.state_dim)
        target_logits = torch.randn(2, 3, self.nodes, 3)
        scales = {'state_scale': torch.tensor(1.0), 'logit_scale': torch.tensor(1.0)}
        losses = proxy.reconstruction_loss(
            trajectory,
            target_logits,
            condition,
            neighborhood,
            cache,
            readout,
            scales,
        )
        self.assertTrue(torch.isfinite(losses['total']))
        losses['total'].backward()
        self.assertIsNotNone(proxy.condition_net[0].weight.grad)
        self.assertIsNone(proxy.generator.grad)

    def test_native_capture_matches_forward_output(self):
        args = SimpleNamespace(
            base_path=str(ROOT.parent),
            n_feat=3,
            n_clss=2,
            hidden_dim=4,
            ode_steps=3,
            adgn_step_size=0.1,
            ode_gamma=0.1,
            ode_activation='tanh',
        )
        model = build_s0_model(args).eval()
        data = Data(
            x=torch.randn(5, 3),
            edge_index=self.edge_index,
        )
        data.edge_weight = None
        output = model(data)
        states = model.encode_native_dynamics_states(data)
        self.assertEqual(tuple(states.shape), (4, 5, 4))
        self.assertTrue(torch.allclose(output, model.readout(states[-1]), atol=1e-6, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
