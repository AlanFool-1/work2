"""Method V0.1: deterministic node-level linear graph dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm


@dataclass
class V01ForwardOutput:
    """Full V0.1 output used by training and dynamics diagnostics."""

    logits: torch.Tensor
    reconstruction: torch.Tensor | None
    latent_trajectory: torch.Tensor


def normalize_graph(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the self-looped symmetric GCN normalization of a COO graph."""
    normalized_index, normalized_weight = gcn_norm(
        edge_index,
        edge_weight,
        num_nodes=num_nodes,
        improved=False,
        add_self_loops=True,
        flow='source_to_target',
        dtype=dtype,
    )
    if normalized_weight is None:
        raise RuntimeError('GCN normalization did not return edge weights.')
    return normalized_index, normalized_weight


def normalized_aggregate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a normalized sparse graph operator to node-row features."""
    source, target = edge_index
    messages = x[source] * edge_weight.to(dtype=x.dtype).unsqueeze(-1)
    aggregated = torch.zeros_like(x)
    aggregated.index_add_(0, target, messages)
    return aggregated


def log_degree_feature(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute log(1 + weighted in-degree) before adding normalization loops."""
    target = edge_index[1]
    if edge_weight is None:
        weights = torch.ones(target.numel(), device=target.device, dtype=dtype)
    else:
        weights = edge_weight.to(device=target.device, dtype=dtype)
    degree = torch.zeros(num_nodes, device=target.device, dtype=dtype)
    degree.index_add_(0, target, weights)
    return torch.log1p(degree).unsqueeze(-1)


class NodeEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim + 1, width),
            nn.SiLU(),
            nn.Linear(width, latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        neighborhood_x: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([x, neighborhood_x, degree], dim=-1))


class LatentLinearGraphDynamics(nn.Module):
    """A node encoder followed by shared linear graph-dynamics steps."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 32,
        encoder_width: int = 64,
        num_steps: int = 16,
        step_size: float = 0.1,
        gamma: float = 0.1,
        reconstruction: bool = False,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, latent_dim, encoder_width) <= 0:
            raise ValueError('Input, output, latent, and encoder dimensions must be positive.')
        if num_steps < 0:
            raise ValueError('num_steps must be non-negative.')
        if step_size <= 0.0:
            raise ValueError('step_size must be positive.')
        if gamma < 0.0:
            raise ValueError('gamma must be non-negative.')

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.gamma = float(gamma)

        self.encoder = NodeEncoder(input_dim, encoder_width, latent_dim)
        self.self_generator_raw = nn.Parameter(
            torch.empty(latent_dim, latent_dim)
        )
        self.neighbor_generator = nn.Parameter(
            torch.empty(latent_dim, latent_dim)
        )
        self.readout = nn.Linear(latent_dim, output_dim)
        self.decoder = (
            nn.Sequential(
                nn.Linear(latent_dim, encoder_width),
                nn.SiLU(),
                nn.Linear(encoder_width, input_dim),
            )
            if reconstruction else None
        )
        self.register_buffer(
            '_identity', torch.eye(latent_dim), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        nn.init.normal_(self.self_generator_raw, mean=0.0, std=1e-2)
        nn.init.normal_(self.neighbor_generator, mean=0.0, std=1e-2)
        self.readout.reset_parameters()
        if self.decoder is not None:
            for module in self.decoder.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    def self_generator(self) -> torch.Tensor:
        identity = self._identity.to(
            device=self.self_generator_raw.device,
            dtype=self.self_generator_raw.dtype,
        )
        return (
            self.self_generator_raw
            - self.self_generator_raw.transpose(0, 1)
            - self.gamma * identity
        )

    def encode_nodes(
        self,
        data,
        normalized_index: torch.Tensor,
        normalized_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = data.x
        neighborhood_x = normalized_aggregate(
            x, normalized_index, normalized_weight
        )
        degree = log_degree_feature(
            data.edge_index,
            getattr(data, 'edge_weight', None),
            x.shape[0],
            x.dtype,
        )
        return self.encoder(x, neighborhood_x, degree)

    def propagate_latent(
        self,
        initial: torch.Tensor,
        normalized_index: torch.Tensor,
        normalized_weight: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        z = initial
        trajectory = [z] if return_trajectory else None
        a0 = self.self_generator()
        for _ in range(self.num_steps):
            neighborhood_z = normalized_aggregate(
                z, normalized_index, normalized_weight
            )
            z = z + self.step_size * (
                z @ a0 + neighborhood_z @ self.neighbor_generator
            )
            if trajectory is not None:
                trajectory.append(z)
        if trajectory is not None:
            return torch.stack(trajectory, dim=0)
        return z

    def forward_with_aux(self, data) -> V01ForwardOutput:
        normalized_index, normalized_weight = normalize_graph(
            data.edge_index,
            getattr(data, 'edge_weight', None),
            data.x.shape[0],
            data.x.dtype,
        )
        initial = self.encode_nodes(
            data, normalized_index, normalized_weight
        )
        trajectory = self.propagate_latent(
            initial,
            normalized_index,
            normalized_weight,
            return_trajectory=True,
        )
        logits = self.readout(trajectory[-1])
        reconstruction = (
            self.decoder(initial) if self.decoder is not None else None
        )
        return V01ForwardOutput(logits, reconstruction, trajectory)

    def forward(self, data) -> torch.Tensor:
        normalized_index, normalized_weight = normalize_graph(
            data.edge_index,
            getattr(data, 'edge_weight', None),
            data.x.shape[0],
            data.x.dtype,
        )
        initial = self.encode_nodes(
            data, normalized_index, normalized_weight
        )
        final = self.propagate_latent(
            initial, normalized_index, normalized_weight
        )
        return self.readout(final)


def build_v01_model(args) -> LatentLinearGraphDynamics:
    return LatentLinearGraphDynamics(
        input_dim=args.n_feat,
        output_dim=args.n_clss,
        latent_dim=args.latent_dim,
        encoder_width=args.encoder_width,
        num_steps=args.linear_steps,
        step_size=args.linear_step_size,
        gamma=args.linear_gamma,
        reconstruction=float(args.reconstruction_weight) > 0.0,
    )
