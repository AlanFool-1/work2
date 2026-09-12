"""Executable Gate-0 checks without recreating a repository tests directory.

Run from ``codev0`` with ``python -m models.v04.diagnostics``.
"""

from __future__ import annotations

import json
import math

import torch
from torch_geometric.data import Data
from torchdiffeq import odeint

from models.v04.generator import StableBernsteinGraphGenerator
from modules.fed_optimizer import (
    FederatedAdamW,
    aggregate_block_second_moments,
    initialize_second_moment_blocks,
)
from models.v04.lie_closure import GraphLieClosureLoss
from models.v04.model import NeuralFieldClosedGraphKoopman
from models.v04.observable import StatePreservingObservable
from models.v04.spectral_basis import bernstein_k1, graph_operator
from models.v04.training import (
    update_main_model,
    update_probe,
    update_reference_field,
)


def _toy_data(dtype=torch.float32):
    # Undirected four-node cycle, stored as both directed edge orientations.
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ])
    return Data(
        x=torch.tensor([
            [1.0, 0.0, 0.2, -0.1],
            [0.9, 0.1, 0.0, 0.2],
            [-0.1, 0.8, 0.2, 0.1],
            [0.0, 1.0, -0.1, 0.3],
        ], dtype=dtype),
        edge_index=edge_index,
        y=torch.tensor([0, 0, 1, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        val_mask=torch.tensor([True, False, True, False]),
        test_mask=torch.tensor([False, True, False, True]),
    )


def _changed(before, parameters, tolerance=0.0):
    return any(
        not torch.allclose(old, new.detach(), atol=tolerance, rtol=0.0)
        for old, new in zip(before, parameters)
    )


def _snapshot(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def _model(dtype=torch.float32):
    return NeuralFieldClosedGraphKoopman(
        input_dim=4,
        output_dim=2,
        state_dim=4,
        encoder_width=8,
        observable_aux_dim=2,
        observable_width=6,
        field_hidden_dim=6,
        generator_gamma=1e-3,
        generator_low_damping_init=1e-3,
        generator_high_damping_init=0.5,
        integration_time=0.2,
        step_size=0.02,
        classifier_width=6,
        dropout=0.0,
    ).to(dtype=dtype)


def check_observable_and_jvp():
    torch.manual_seed(7)
    observable = StatePreservingObservable(3, 2, 5).double()
    state = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(state)
    latent, product = torch.autograd.functional.jvp(
        observable, state, direction, create_graph=True
    )
    if not torch.equal(latent[:, :3], state):
        raise AssertionError('Observable identity block is not exact.')

    flat_state = state.detach().reshape(-1).requires_grad_(True)

    def flattened(value):
        return observable(value.reshape_as(state)).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(flattened, flat_state)
    explicit = (jacobian @ direction.reshape(-1)).reshape_as(product)
    error = float((product - explicit).abs().max())
    if error > 1e-10:
        raise AssertionError(f'JVP mismatch: {error}')
    return error


def check_basis_generator_and_solver():
    torch.manual_seed(11)
    data = _toy_data(dtype=torch.float64)
    operator = graph_operator(data)
    latent = torch.randn(4, 3, dtype=torch.float64)
    generator = StableBernsteinGraphGenerator(
        3, 1e-3, 1e-3, 0.5
    ).double()
    low, high = bernstein_k1(latent, operator)
    partition_error = float((low + high - latent).abs().max())
    if partition_error > 1e-12:
        raise AssertionError(f'K=1 partition error: {partition_error}')
    factorized = generator(latent, operator)
    one_tap = generator.one_tap(latent, operator)
    one_tap_error = float((factorized - one_tap).abs().max())
    if one_tap_error > 1e-12:
        raise AssertionError(f'K=1 one-tap mismatch: {one_tap_error}')

    matrices = generator.matrices()
    for matrix in matrices:
        symmetric = 0.5 * (matrix + matrix.T)
        if float(torch.linalg.eigvalsh(symmetric).max()) > -1e-3 + 1e-10:
            raise AssertionError('Per-band dissipativity certificate failed.')

    dense = operator.to_dense()
    identity = torch.eye(dense.shape[0], dtype=dense.dtype)
    basis_low = 0.5 * (identity + dense)
    basis_high = 0.5 * (identity - dense)
    effective = (
        torch.kron(basis_low, matrices[0])
        + torch.kron(basis_high, matrices[1])
    )
    max_real = float(torch.linalg.eigvals(effective).real.max())
    if max_real > -1e-3 + 1e-9:
        raise AssertionError(f'Full generator is not stable: {max_real}')

    class RHS(torch.nn.Module):
        def forward(self, _time, value):
            return generator(value, operator, matrices)

    horizon = 0.2
    numerical = odeint(
        RHS(),
        latent,
        torch.tensor([0.0, horizon], dtype=torch.float64),
        method='rk4',
        options={'step_size': 0.005},
    )[-1]
    exact = (
        torch.matrix_exp(horizon * effective) @ latent.reshape(-1)
    ).reshape_as(latent)
    solver_error = float(
        (numerical - exact).norm() / exact.norm().clamp_min(1e-12)
    )
    if solver_error > 1e-8:
        raise AssertionError(f'RK4/exponential mismatch: {solver_error}')
    return {
        'partition_error': partition_error,
        'one_tap_error': one_tap_error,
        'effective_max_real_part': max_real,
        'rk4_relative_error': solver_error,
    }


def check_lie_stop_gradient():
    torch.manual_seed(13)
    model = _model()
    data = _toy_data()
    operator = graph_operator(data)
    state = model.encoder(data.x)
    closure = GraphLieClosureLoss()(
        state,
        data.train_mask,
        operator,
        model.observable,
        model.reference_field,
        model.generator,
    )
    closure.loss.backward()
    field_has_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.reference_field.parameters()
    )
    main_has_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.observable.parameters()
    )
    if field_has_grad or not main_has_grad:
        raise AssertionError('Lie stop-gradient boundary is incorrect.')
    return float(closure.cosine.detach())


def check_optimizer_isolation_and_anchor():
    torch.manual_seed(17)
    model = _model()
    data = _toy_data()
    main_parameters = model.main_parameters()
    field_parameters = model.field_parameters()
    probe_parameters = model.probe_parameters()
    optimizers = {
        'main': torch.optim.Adam(main_parameters, lr=1e-2),
        'field': torch.optim.Adam(field_parameters, lr=1e-2),
        'probe': torch.optim.Adam(probe_parameters, lr=1e-2),
    }

    before_main = _snapshot(main_parameters)
    before_field = _snapshot(field_parameters)
    before_probe = _snapshot(probe_parameters)
    update_probe(model, data, optimizers['probe'], 'Cora', 5.0)
    if (
        _changed(before_main, main_parameters)
        or _changed(before_field, field_parameters)
        or not _changed(before_probe, probe_parameters)
    ):
        raise AssertionError('Probe optimizer isolation failed.')

    before_main = _snapshot(main_parameters)
    before_field = _snapshot(field_parameters)
    before_probe = _snapshot(probe_parameters)
    first = update_reference_field(
        model, data, optimizers['field'], 'Cora', 5.0
    )
    if (
        _changed(before_main, main_parameters)
        or not _changed(before_field, field_parameters)
        or _changed(before_probe, probe_parameters)
    ):
        raise AssertionError('Field optimizer isolation failed.')

    for _ in range(24):
        latest = update_reference_field(
            model, data, optimizers['field'], 'Cora', 5.0
        )
    if float(latest['field_task_cosine']) <= float(first['field_task_cosine']):
        raise AssertionError('Direction anchor did not improve on the toy graph.')
    if float(latest['field_rms']) < 1e-5:
        raise AssertionError('Reference field collapsed on the toy graph.')

    before_main = _snapshot(main_parameters)
    before_field = _snapshot(field_parameters)
    before_probe = _snapshot(probe_parameters)
    update_main_model(
        model, data, optimizers['main'], 'Cora', 0.05, 5.0
    )
    if (
        not _changed(before_main, main_parameters)
        or _changed(before_field, field_parameters)
        or _changed(before_probe, probe_parameters)
    ):
        raise AssertionError('Main optimizer isolation failed.')
    return {
        'anchor_cosine_before': float(first['field_task_cosine']),
        'anchor_cosine_after': float(latest['field_task_cosine']),
        'field_rms': float(latest['field_rms']),
    }


def check_fedadamw_protocols():
    """Check O4 block-v injection, split clocks, and aggregation."""

    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = FederatedAdamW(
        [('weight', parameter)],
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        global_v_step=100,
    )
    initialize_second_moment_blocks(
        optimizer, [('weight', parameter)], {'weight': 0.0123}
    )
    state = optimizer.state[parameter]
    if int(torch.count_nonzero(state['exp_avg'])) != 0:
        raise AssertionError('FedAdamW first moment was not reset.')
    if state['local_m_step'] != 0:
        raise AssertionError('FedAdamW local first-moment clock was not reset.')
    if not torch.allclose(
        state['exp_avg_sq'], torch.full_like(parameter, 0.0123)
    ):
        raise AssertionError('FedAdamW block-v injection failed.')

    gradient = torch.tensor([2.0], dtype=torch.float64)
    parameter.grad = gradient.clone()
    initial = parameter.detach().clone()
    optimizer.step()
    m = 0.1 * gradient
    v = 0.999 * torch.tensor([0.0123], dtype=torch.float64)
    v = v + 0.001 * gradient.square()
    correction1 = 1.0 - 0.9 ** 1
    correction2 = 1.0 - 0.999 ** 101
    expected = initial - 0.1 * (
        (m / correction1) / ((v / correction2).sqrt() + 1e-8)
    )
    if not torch.allclose(parameter.detach(), expected, atol=1e-12, rtol=0.0):
        raise AssertionError('FedAdamW split-clock update is incorrect.')
    if not math.isclose(
        optimizer.last_bias_correction1, correction1, rel_tol=0.0, abs_tol=1e-15
    ) or not math.isclose(
        optimizer.last_bias_correction2, correction2, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AssertionError('FedAdamW bias-correction clocks are coupled.')

    equal = aggregate_block_second_moments(
        [{'weight': 1.0}, {'weight': 3.0}], [0.5, 0.5]
    )
    weighted = aggregate_block_second_moments(
        [{'weight': 1.0}, {'weight': 3.0}], [0.25, 0.75]
    )
    if equal['weight'] != 2.0 or weighted['weight'] != 2.5:
        raise AssertionError('Server block-v aggregation is incorrect.')
    state_keys = set(_model().state_dict())
    named_keys = {name for name, _ in _model().named_main_parameters()}
    if not named_keys or not named_keys.issubset(state_keys):
        raise AssertionError('Named main parameters do not match state_dict.')
    return {
        'fedadamw_bias_correction_m': optimizer.last_bias_correction1,
        'fedadamw_bias_correction_v': optimizer.last_bias_correction2,
        'fedadamw_equal_block_v': equal['weight'],
        'fedadamw_weighted_block_v': weighted['weight'],
    }


def run_gate0():
    results = {
        'jvp_max_error': check_observable_and_jvp(),
        **check_basis_generator_and_solver(),
        'lie_cosine_smoke': check_lie_stop_gradient(),
        **check_optimizer_isolation_and_anchor(),
        **check_fedadamw_protocols(),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    print('[Gate 0] PASS')
    return results


if __name__ == '__main__':
    run_gate0()
