"""V0.2: nonlinear reference trajectories and a deployed E--K_G--D path.

The time coordinate is graph propagation depth, not physical time. Reference
states supervise the proxy but never enter its free-running prediction path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from models.v01.model import log_degree_feature, normalize_graph


def graph_operator(data) -> torch.Tensor:
    """Build a symmetric normalized operator; reject unsupported graph types."""
    weights = getattr(data, 'edge_weight', None)
    if weights is not None and (not torch.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError('V0.2 requires finite non-negative edge weights.')
    index, weight = normalize_graph(
        data.edge_index, weights, data.x.shape[0], data.x.dtype,
    )
    operator = torch.sparse_coo_tensor(
        index.flip(0), weight, (data.x.shape[0], data.x.shape[0]),
        device=data.x.device,
    ).coalesce()
    difference = (operator - operator.transpose(0, 1)).coalesce().values()
    if difference.numel() and difference.abs().max() > 1e-6:
        raise ValueError('V0.2 dissipativity requires an undirected symmetric graph.')
    return operator


def aggregate(operator: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Apply P to node rows of (..., N, d), including batched time windows."""
    node_first = states.movedim(-2, 0)
    result = torch.sparse.mm(operator, node_first.reshape(node_first.shape[0], -1))
    return result.reshape(node_first.shape).movedim(0, -2)


def mlp(input_dim: int, width: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, output_dim))


class StableGraphGenerator(nn.Module):
    """Dissipative continuous generator with bounded matrix norm.

    A0 = S0 - D0 - D1, A1 = S1 + D1, Sj = -Sj.T, Dj >= 0.
    For symmetric P with eigenvalues in [-1, 1], each graph-frequency block
    has symmetric part -2 D0 - 2 (1-lambda) D1. RK4 is an approximation;
    continuous dissipativity is not an unconditional discrete stability claim.
    """

    def __init__(self, dim: int, damping: float = 0.1, norm_bound: float = 4.0):
        super().__init__()
        if dim <= 0 or not math.isfinite(damping) or damping < 0:
            raise ValueError('Generator dimension must be positive and damping non-negative.')
        if not math.isfinite(norm_bound) or norm_bound <= 0:
            raise ValueError('Generator norm bound must be finite and positive.')
        self.damping = float(damping)
        self.norm_bound = float(norm_bound)
        self.self_skew = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.neighbor_skew = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.self_damping = nn.Parameter(torch.eye(dim) * math.sqrt(0.01))
        self.neighbor_damping = nn.Parameter(torch.eye(dim) * math.sqrt(0.1))
        self.register_buffer('_identity', torch.eye(dim), persistent=False)

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        d0 = self.self_damping @ self.self_damping.T + self.damping * self._identity
        d1 = self.neighbor_damping @ self.neighbor_damping.T
        a0 = self.self_skew - self.self_skew.T - d0 - d1
        a1 = self.neighbor_skew - self.neighbor_skew.T + d1
        # Frobenius norms upper-bound spectral norms. Positive joint scaling
        # preserves the dissipativity proof and bounds ||I x A0 + P x A1||.
        bound = a0.norm() + a1.norm()
        scale = (self.norm_bound / bound.clamp_min(1e-12)).clamp(max=1.0)
        return scale * a0, scale * a1

    @staticmethod
    def field(z, operator, matrices):
        a0, a1 = matrices
        return z @ a0 + aggregate(operator, z) @ a1

    def step(self, z, operator, step_size, matrices):
        # Substeps are fixed by configuration, never by hidden target states.
        substeps = max(1, math.ceil(step_size * self.norm_bound / 0.5))
        dt = step_size / substeps
        for _ in range(substeps):
            k1 = self.field(z, operator, matrices)
            k2 = self.field(z + 0.5 * dt * k1, operator, matrices)
            k3 = self.field(z + 0.5 * dt * k2, operator, matrices)
            k4 = self.field(z + dt * k3, operator, matrices)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z


@dataclass
class V02Output:
    logits: torch.Tensor
    native_logits: torch.Tensor
    native_states: torch.Tensor
    encoded_targets: torch.Tensor
    reconstructions: torch.Tensor
    latent_trajectory: torch.Tensor
    decoded_trajectory: torch.Tensor
    predictions: dict[int, torch.Tensor]


class GraphKoopmanBackbone(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, state_dim: int = 64,
        latent_dim: int = 32, width: int = 64, num_steps: int = 16,
        step_size: float = 0.1, damping: float = 0.1,
        generator_norm_bound: float = 4.0, correction_interval: int = 0,
        identity_dynamics: bool = False,
    ):
        super().__init__()
        if min(input_dim, output_dim, state_dim, latent_dim, width, num_steps) <= 0:
            raise ValueError('Dimensions and propagation steps must be positive.')
        if not math.isfinite(step_size) or step_size <= 0 or correction_interval < 0:
            raise ValueError('Step size must be positive and correction interval non-negative.')
        if identity_dynamics and correction_interval:
            raise ValueError('Identity ablation must not introduce nonlinear reencoding.')
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.correction_interval = int(correction_interval)
        self.identity_dynamics = bool(identity_dynamics)
        self.stem = mlp(2 * input_dim + 1, width, state_dim)
        self.native_self = nn.Linear(state_dim, state_dim, bias=False)
        self.native_neighbor = nn.Linear(state_dim, state_dim)
        self.native_readout = nn.Linear(state_dim, output_dim)
        self.encoder = mlp(2 * state_dim, width, latent_dim)
        # Nodewise decoder has no X, node-ID, reference state, or depth shortcut.
        self.decoder = mlp(latent_dim, width, state_dim)
        self.readout = nn.Linear(state_dim, output_dim)
        self.generator = StableGraphGenerator(latent_dim, damping, generator_norm_bound)
        self.horizons = tuple(sorted({min(s, num_steps) for s in (1, 4, 8, num_steps)}))

    def initial_state(self, data, operator):
        degree = log_degree_feature(
            data.edge_index, getattr(data, 'edge_weight', None),
            data.x.shape[0], data.x.dtype,
        )
        return self.stem(torch.cat([data.x, aggregate(operator, data.x), degree], dim=-1))

    def encode(self, h, operator):
        return self.encoder(torch.cat([h, aggregate(operator, h)], dim=-1))

    def native_trajectory(self, initial, operator):
        h = initial
        states = [h]
        for _ in range(self.num_steps):
            h = h + self.step_size * torch.tanh(
                self.native_self(h) + self.native_neighbor(aggregate(operator, h))
            )
            states.append(h)
        return torch.stack(states)

    def rollout(self, initial, operator, matrices=None, *, correction_interval=None):
        matrices = self.generator.matrices() if matrices is None else matrices
        interval = self.correction_interval if correction_interval is None else correction_interval
        z, states = initial, [initial]
        for step in range(1, self.num_steps + 1):
            if not self.identity_dynamics:
                z = self.generator.step(z, operator, self.step_size, matrices)
            if interval and step % interval == 0:
                z = self.encode(self.decoder(z), operator)
            states.append(z)
        return torch.stack(states)

    def forward(self, data):
        """Deployment: no reference transitions or future observations required."""
        operator = graph_operator(data)
        initial = self.encode(self.initial_state(data, operator), operator)
        z = self.rollout(initial, operator)[-1]
        return self.readout(self.decoder(z))

    def forward_native(self, data):
        operator = graph_operator(data)
        native = self.native_trajectory(self.initial_state(data, operator), operator)
        return self.native_readout(native[-1])

    def forward_with_aux(self, data, *, auxiliary=True):
        operator = graph_operator(data)
        initial = self.initial_state(data, operator)
        native = self.native_trajectory(initial, operator)
        matrices = self.generator.matrices()
        latent = self.rollout(self.encode(initial, operator), operator, matrices)
        decoded = self.decoder(latent)
        # Auxiliary fitting cannot change the reference state coordinates or
        # vector field directly. The shared stem can still evolve through CE.
        encoded = self.encode(native.detach(), operator)
        predictions = {}
        if auxiliary:
            predicted = encoded
            # All valid starts, free-running without teacher forcing. No
            # course correction here: these losses test the linear operator.
            for horizon in range(1, self.num_steps + 1):
                predicted = predicted[:-1]
                if not self.identity_dynamics:
                    predicted = self.generator.step(
                        predicted, operator, self.step_size, matrices,
                    )
                if horizon in self.horizons:
                    predictions[horizon] = predicted
        return V02Output(
            self.readout(decoded[-1]), self.native_readout(native[-1]), native,
            encoded, self.decoder(encoded), latent, decoded, predictions,
        )

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        training_only = sum(p.numel() for module in (
            self.native_self, self.native_neighbor, self.native_readout,
        ) for p in module.parameters())
        return {'training_parameters': total, 'deployment_parameters': total - training_only}


def build_v02_model(args):
    return GraphKoopmanBackbone(
        input_dim=args.n_feat, output_dim=args.n_clss, state_dim=args.hidden_dim,
        latent_dim=args.latent_dim, width=args.encoder_width,
        num_steps=args.linear_steps, step_size=args.linear_step_size,
        damping=args.linear_gamma, generator_norm_bound=args.generator_norm_bound,
        correction_interval=args.correction_interval,
        identity_dynamics=args.identity_dynamics,
    )
