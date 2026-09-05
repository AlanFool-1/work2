import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from dynamics import build_trajectory_basis, fit_ridge_generator
from dynamics.trajectory_basis import shared_probe
from federation import FunctionalDynamicsAggregator
from functional_map import (
    normalized_descriptor, procrustes_distance, solve_orthogonal_fm,
    solve_regularized_fm,
)
from models.dissipative_injector import (
    low_rank_injection,
    minimal_dissipative_projection,
)
from models.s0.model import build_s0_model


class FunctionalDynamicsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_basis_is_orthonormal_and_deterministic(self):
        trajectory = torch.randn(17, 40, 12)
        probe = shared_probe(12, 3, 11, trajectory.device, trajectory.dtype)
        first, snapshot, indices = build_trajectory_basis(trajectory, probe, 8)
        second, _, _ = build_trajectory_basis(trajectory, probe, 8)
        self.assertEqual(indices, (0, 4, 8, 16))
        self.assertEqual(snapshot.shape, (40, 12))
        torch.testing.assert_close(first.T @ first, torch.eye(8), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(first, second)

    def test_generator_fit_is_finite(self):
        trajectory = torch.randn(17, 32, 10)
        basis, _, _ = build_trajectory_basis(
            trajectory, shared_probe(10, 3, 4, 'cpu', trajectory.dtype), 8
        )
        generator, metrics = fit_ridge_generator(trajectory, basis, 0.1, 1e-3)
        self.assertEqual(generator.shape, (8, 8))
        self.assertTrue(torch.isfinite(generator).all())
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_functional_map_remains_orthogonal(self):
        rank, descriptor_dim = 6, 12
        local_a = torch.randn(rank, rank) * 0.05
        local_b = normalized_descriptor(torch.eye(rank), torch.randn(rank, descriptor_dim))
        rotation, _ = torch.linalg.qr(torch.randn(rank, rank))
        canonical_a = rotation @ local_a @ rotation.T
        canonical_b = rotation @ local_b
        fitted, metrics = solve_orthogonal_fm(
            local_a, local_b, canonical_a, canonical_b, steps=20, learning_rate=0.05
        )
        torch.testing.assert_close(fitted.T @ fitted, torch.eye(rank), atol=1e-5, rtol=1e-5)
        self.assertLess(metrics['fm_desc_error'], 1e-3)
        self.assertLess(metrics['fm_dyn_error'], 1e-3)

    def test_procrustes_distance_is_gauge_invariant(self):
        descriptor = torch.randn(6, 12)
        rotation, _ = torch.linalg.qr(torch.randn(6, 6))
        distance, affinity = procrustes_distance(
            descriptor, rotation @ descriptor
        )
        self.assertLess(float(distance), 1e-3)
        self.assertGreater(float(affinity), 0.0)

    def test_regularized_functional_map_reduces_nonorthogonal_mismatch(self):
        rank = 6
        local_b = torch.randn(rank, 12)
        scale = torch.diag(torch.linspace(0.6, 1.5, rank))
        target_b = scale @ local_b
        local_a = torch.randn(rank, rank) * 0.05
        target_a = scale @ local_a @ torch.linalg.inv(scale)
        fitted, metrics = solve_regularized_fm(
            local_a, local_b, target_a, target_b,
            steps=30, learning_rate=0.2, ridge=1e-6,
            orthogonal_regularization=0.0, max_condition=20.0,
        )
        self.assertLess(metrics['fm_desc_error'], 1e-3)
        self.assertLess(metrics['fm_dyn_error'], 1e-3)
        self.assertLessEqual(metrics['fm_final_objective'], metrics['fm_initial_objective'])
        self.assertTrue(torch.isfinite(fitted).all())

    def test_dissipative_projection_and_gradient(self):
        raw = torch.randn(8, 8)
        safe, metrics = minimal_dissipative_projection(raw, 0.7)
        largest = torch.linalg.eigvalsh(0.5 * (safe + safe.T))[-1]
        self.assertLessEqual(float(largest), 1e-5)
        self.assertLessEqual(metrics['delta_safe_spectral_norm'], 0.70001)
        hidden = torch.randn(24, 10, requires_grad=True)
        basis, _ = torch.linalg.qr(torch.randn(24, 8), mode='reduced')
        low_rank_injection(hidden, basis.detach(), safe.detach(), 0.05).sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(float(torch.linalg.vector_norm(hidden.grad)), 0.0)

    def test_simulated_secure_aggregator_weighting_and_ema(self):
        aggregator = FunctionalDynamicsAggregator(ema_a=0.5, ema_b=0.25)
        state = aggregator.update(
            [np.ones((2, 2)), np.full((2, 2), 3.0)],
            [np.ones((2, 3)), np.full((2, 3), 5.0)],
            weights=[0.25, 0.75],
        )
        np.testing.assert_allclose(state['A_star'], 2.5)
        np.testing.assert_allclose(state['B_star'], 4.0)
        state = aggregator.update(
            [np.zeros((2, 2))], [np.zeros((2, 3))], weights=[1.0]
        )
        np.testing.assert_allclose(state['A_star'], 1.25)
        np.testing.assert_allclose(state['B_star'], 1.0)

    def test_two_prototype_bootstrap_recovers_gauge_invariant_groups(self):
        rng = np.random.default_rng(4)
        first = np.zeros((4, 8)); first[:, :4] = np.eye(4)
        second = np.zeros((4, 8)); second[:, 4:] = np.eye(4)
        descriptors, generators = [], []
        for index, base in enumerate([first, first, second, second]):
            rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
            descriptors.append(rotation @ base / np.linalg.norm(base))
            generators.append(rotation @ (np.eye(4) * (index + 1)) @ rotation.T)
        aggregator = FunctionalDynamicsAggregator(num_prototypes=2, ema_a=0.5, ema_b=0.5)
        state = aggregator.update(
            generators, descriptors, client_ids=[0, 1, 2, 3]
        )
        assignments = state['bootstrap_assignments']
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])
        self.assertEqual(sorted(state['cluster_counts']), [2, 2])
        self.assertEqual(len(state['A_star']), 2)

    def test_feature_flag_preserves_native_backbone_exactly(self):
        common = dict(
            base_path=str(Path(__file__).resolve().parents[2]),
            n_feat=5, n_clss=3, hidden_dim=8, ode_steps=4,
            adgn_step_size=0.1, ode_gamma=0.1, ode_activation='tanh',
        )
        torch.manual_seed(19)
        baseline = build_s0_model(SimpleNamespace(
            **common, enable_functional_dynamics=False
        ))
        torch.manual_seed(19)
        feature_model = build_s0_model(SimpleNamespace(
            **common, enable_functional_dynamics=True
        ))
        nodes = 14
        source = torch.arange(nodes)
        target = torch.roll(source, -1)
        graph = Data(
            x=torch.randn(nodes, 5),
            edge_index=torch.stack([
                torch.cat([source, target]), torch.cat([target, source])
            ]),
            edge_weight=None,
        )
        torch.testing.assert_close(baseline(graph), feature_model(graph), rtol=0, atol=0)
        baseline_states = baseline.encode_native_dynamics_states(graph)
        feature_states = feature_model.encode_native_dynamics_states(graph)
        torch.testing.assert_close(baseline_states, feature_states, rtol=0, atol=0)

        basis, _ = torch.linalg.qr(torch.randn(nodes, 4), mode='reduced')
        delta = -torch.eye(4)
        feature_model.set_functional_injection(basis, delta, beta=0.05)
        injected = feature_model(graph)
        self.assertFalse(torch.equal(injected, baseline(graph)))
        injected.sum().backward()
        self.assertIsNotNone(feature_model.emb.weight.grad)

        # Native calibration/visualization must ignore even a stale cached Q
        # from a different-size client graph.
        stale_basis, _ = torch.linalg.qr(torch.randn(nodes + 3, 4), mode='reduced')
        feature_model.set_functional_injection(stale_basis, delta, beta=0.05)
        native = feature_model.encode_native_dynamics_states(graph)
        torch.testing.assert_close(native, baseline_states, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
