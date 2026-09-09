import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.v02.model import GraphKoopmanBackbone, aggregate, graph_operator
from models.v02.training import LossWeights, checked_step, objective, task_loss, trajectory_mse


class V02ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        source = torch.arange(7)
        target = source.roll(-1)
        self.data = Data(
            x=torch.randn(7, 5),
            edge_index=torch.stack([torch.cat([source, target]), torch.cat([target, source])]),
            y=torch.arange(7) % 3,
            train_mask=torch.tensor([True, True, True, True, False, False, False]),
        )

    def model(self, **kwargs):
        return GraphKoopmanBackbone(5, 3, state_dim=8, latent_dim=4, width=9, num_steps=4, **kwargs)

    def test_deployment_never_reads_reference_and_requires_decoder(self):
        model = self.model().eval()
        expected = model.forward_with_aux(self.data).logits
        with patch.object(model, 'native_trajectory', side_effect=AssertionError('teacher called')):
            actual = model(self.data)
        torch.testing.assert_close(expected, actual)
        with torch.no_grad():
            model.native_self.weight.add_(50)
            model.native_readout.weight.add_(50)
        torch.testing.assert_close(model(self.data), actual)
        with patch.object(model.decoder, 'forward', side_effect=AssertionError('decoder called')):
            with self.assertRaisesRegex(AssertionError, 'decoder called'):
                model(self.data)

    def test_supervised_gradient_reaches_encoder_operator_decoder(self):
        model = self.model()
        task_loss(model(self.data), self.data.y, self.data.train_mask).backward()
        for module in (model.stem, model.encoder, model.generator, model.decoder, model.readout):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertIsNone(model.native_self.weight.grad)

    def test_auxiliary_targets_do_not_update_reference(self):
        model = self.model()
        output = model.forward_with_aux(self.data)
        _, values = objective(model, output, self.data.y, self.data.train_mask, LossWeights())
        sum(values[k] for k in ('reconstruction_loss', 'prediction_loss', 'linearity_loss')).backward()
        for module in (model.stem, model.native_self, model.native_neighbor, model.native_readout):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertIsNotNone(model.decoder[0].weight.grad)
        self.assertIsNotNone(model.generator.self_skew.grad)

    def test_targets_use_no_validation_or_test_labels(self):
        model = self.model()
        output = model.forward_with_aux(self.data)
        first, _ = objective(model, output, self.data.y, self.data.train_mask, LossWeights())
        changed = self.data.y.clone()
        changed[~self.data.train_mask] = 10000  # must never reach cross entropy
        second, _ = objective(model, output, changed, self.data.train_mask, LossWeights())
        torch.testing.assert_close(first, second)

    def test_linear_semigroup_and_all_window_free_rollouts(self):
        model = self.model()
        operator = graph_operator(self.data)
        first, second = torch.randn(7, 4), torch.randn(7, 4)
        torch.testing.assert_close(
            model.rollout(first - 2 * second, operator),
            model.rollout(first, operator) - 2 * model.rollout(second, operator),
        )
        output = model.forward_with_aux(self.data)
        for horizon, predicted in output.predictions.items():
            self.assertEqual(predicted.shape[0], 5 - horizon)
            for start in range(5 - horizon):
                rolled = model.rollout(output.encoded_targets[start], operator)
                torch.testing.assert_close(predicted[start], rolled[horizon])

    def test_generator_dissipativity_and_rk4_against_exact_exponential(self):
        model = self.model().double()
        data = self.data.clone()
        data.x = data.x.double()
        operator = graph_operator(data)
        # Build the full finite graph generator from its action on basis vectors.
        basis = torch.eye(28, dtype=torch.float64).reshape(28, 7, 4)
        matrices = model.generator.matrices()
        full = model.generator.field(basis, operator, matrices).reshape(28, 28)
        self.assertLessEqual(float(torch.linalg.eigvalsh((full + full.T) / 2).max()), 1e-10)
        self.assertLessEqual(float(torch.linalg.matrix_norm(full, ord=2)), model.generator.norm_bound + 1e-10)
        z = torch.randn(7, 4, dtype=torch.float64)
        step = model.generator.step(z, operator, model.step_size, matrices)
        exact = (z.reshape(-1) @ torch.matrix_exp(model.step_size * full)).reshape(7, 4)
        torch.testing.assert_close(step, exact, atol=1e-7, rtol=1e-7)
        self.assertLessEqual(float(step.norm()), float(z.norm()) + 1e-9)

    def test_equivariance_batched_aggregation_and_isolated_node(self):
        model = self.model().eval()
        permutation = torch.tensor([2, 5, 0, 3, 1, 6, 4])
        old_to_new = torch.empty_like(permutation)
        old_to_new[permutation] = torch.arange(7)
        permuted = Data(x=self.data.x[permutation], edge_index=old_to_new[self.data.edge_index])
        torch.testing.assert_close(model(permuted), model(self.data)[permutation])
        operator = graph_operator(self.data)
        states = torch.randn(3, 7, 4)
        torch.testing.assert_close(aggregate(operator, states), torch.stack([operator.to_dense() @ x for x in states]))
        empty_edges = Data(x=torch.randn(1, 5), edge_index=torch.empty(2, 0, dtype=torch.long))
        self.assertTrue(torch.isfinite(model(empty_edges)).all())

    def test_identity_correction_and_asymmetric_graph_guard(self):
        model = self.model(identity_dynamics=True)
        output = model.forward_with_aux(self.data)
        torch.testing.assert_close(output.latent_trajectory, output.latent_trajectory[:1].expand_as(output.latent_trajectory))
        with self.assertRaises(ValueError):
            self.model(identity_dynamics=True, correction_interval=2)
        corrected = self.model(correction_interval=2)
        self.assertTrue(torch.isfinite(corrected(self.data)).all())
        directed = self.data.clone()
        directed.edge_index = directed.edge_index[:, :7]
        with self.assertRaisesRegex(ValueError, 'undirected'):
            corrected(directed)

    def test_short_joint_training_is_finite_and_reduces_objective(self):
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(model, model.forward_with_aux(self.data), self.data.y, self.data.train_mask, LossWeights())
            losses.append(float(loss))
            checked_step(loss, model, optimizer)
        self.assertLess(losses[-1], losses[0])

    def test_bounded_control_has_same_initial_function_and_can_model_growth(self):
        torch.manual_seed(71)
        dissipative = self.model()
        torch.manual_seed(71)
        bounded = self.model(generator_mode='bounded')
        torch.testing.assert_close(dissipative(self.data), bounded(self.data))
        with torch.no_grad():
            bounded.generator.self_matrix.copy_(0.2 * torch.eye(4))
            bounded.generator.neighbor_matrix.zero_()
        operator = graph_operator(self.data)
        z = torch.randn(7, 4)
        evolved = bounded.rollout(z, operator)
        self.assertGreater(float(evolved[-1].norm()), float(z.norm()))
        torch.testing.assert_close(bounded.rollout(2 * z, operator), 2 * evolved)
        task_loss(bounded(self.data), self.data.y, self.data.train_mask).backward()
        self.assertTrue(torch.isfinite(bounded.generator.self_matrix.grad).all())
        with torch.no_grad():
            bounded.generator.self_matrix.mul_(1000)
        matrices = bounded.generator.matrices()
        self.assertLessEqual(float(sum(m.norm() for m in matrices)), bounded.generator.norm_bound + 1e-5)

    def test_time_balanced_loss_retains_initial_error_and_detaches_target(self):
        target = torch.tensor([[[1.0]], [[10.0]]], requires_grad=True)
        predicted = torch.tensor([[[0.0]], [[9.0]]], requires_grad=True)
        loss = trajectory_mse(predicted, target, 'per_time')
        self.assertAlmostEqual(float(loss), 0.505, places=6)
        self.assertGreater(float(loss), float(trajectory_mse(predicted, target)))
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all())


if __name__ == '__main__':
    unittest.main()
