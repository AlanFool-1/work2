"""Frozen-coordinate controls for diagnosing V0.2, not new backbone defaults."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.v02.model import aggregate


def per_time_nmse(prediction, target):
    """B,T,N,D -> T errors, each normalized by that target time's energy."""
    error = (prediction - target).square().mean(dim=(0, 2, 3))
    scale = target.square().mean(dim=(0, 2, 3)).clamp_min(1e-8)
    return error / scale


def encode_states(model, states, operator):
    return model.encode(states, operator)


class RidgeGraphTransition(nn.Module):
    """Discrete Z' = Z + Z B0 + PZ B1 + b. b is optional and node-shared."""
    def __init__(self, self_delta, neighbor, bias):
        super().__init__()
        self.register_buffer('self_delta', self_delta)
        self.register_buffer('neighbor', neighbor)
        self.register_buffer('bias', bias)

    def forward(self, z, operator):
        return z + z @ self.self_delta + aggregate(operator, z) @ self.neighbor + self.bias

    @classmethod
    @torch.no_grad()
    def fit(cls, states, operator, ridge, affine):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError('Ridge regularization must be finite and positive.')
        x = states[:, :-1]
        y = states[:, 1:] - x
        pieces = [x, aggregate(operator, x)]
        if affine:
            pieces.append(torch.ones_like(x[..., :1]))
        design = torch.cat(pieces, dim=-1).reshape(-1, sum(v.shape[-1] for v in pieces)).double()
        target = y.reshape(-1, y.shape[-1]).double()
        # RMS scale features using training data only. Fit a residual around I;
        # regularize the residual matrices but not the optional intercept.
        scale = design.square().mean(0).sqrt().clamp_min(1e-8)
        design = design / scale
        gram = design.T @ design / design.shape[0]
        penalty = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * ridge
        if affine:
            penalty[-1, -1] = 0
        coefficients = torch.linalg.solve(gram + penalty, design.T @ target / design.shape[0])
        coefficients = (coefficients / scale[:, None]).to(states.dtype)
        dim = states.shape[-1]
        bias = coefficients[-1] if affine else states.new_zeros(dim)
        return cls(coefficients[:dim], coefficients[dim:2 * dim], bias)


class StableTransition(nn.Module):
    def __init__(self, generator, step_size):
        super().__init__()
        self.generator = generator
        self.step_size = step_size

    def forward(self, z, operator):
        return self.generator.step(z, operator, self.step_size, self.generator.matrices())


def rollout(transition, initial, operator, steps):
    states, z = [initial], initial
    for _ in range(steps):
        z = transition(z, operator)
        states.append(z)
    return torch.stack(states, dim=1)


@torch.no_grad()
def graph_gain(model, operator):
    """Exact graph-frequency RK4 singular gains, for small frozen graphs only."""
    eigenvalues = torch.linalg.eigvalsh(operator.to_dense().double())
    a0, a1 = [matrix.double() for matrix in model.generator.matrices()]
    field = a0.unsqueeze(0) + eigenvalues[:, None, None] * a1.unsqueeze(0)
    count = max(1, math.ceil(model.step_size * model.generator.norm_bound / 0.5))
    q = field * (model.step_size / count)
    identity = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype).expand_as(q)
    q2 = q @ q
    rk4 = identity + q + q2 / 2 + q2 @ q / 6 + q2 @ q2 / 24
    one_step = torch.linalg.matrix_power(rk4, count)
    final_step = torch.linalg.matrix_power(one_step, model.num_steps)
    return {
        'one_step_operator_norm': float(torch.linalg.svdvals(one_step).max()),
        'final_operator_norm': float(torch.linalg.svdvals(final_step).max()),
    }


@torch.no_grad()
def contraction_audit(model, states, operator):
    encoded = encode_states(model, states, operator)
    initial = encoded[:, 0].flatten(1).norm(dim=1)
    target = encoded[:, -1].flatten(1).norm(dim=1)
    gain = graph_gain(model, operator)
    lower = (target - gain['final_operator_norm'] * initial).clamp_min(0).square().sum()
    lower = lower / target.square().sum().clamp_min(1e-12)
    return {
        **gain,
        'target_final_to_initial_norm': float(target.norm() / initial.norm().clamp_min(1e-12)),
        'frozen_operator_nmse_lower_bound': float(lower),
        'note': 'Bound fixes the encoder and this operator norm; it is not a bound after representation relearning.',
    }
