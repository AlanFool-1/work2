"""Graph-aware variational coordinates with a fixed latent linear operator."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _edge_cache(edge_index: Tensor, edge_weight: Optional[Tensor], nodes: int):
    """Build a permutation-equivariant normalized sparse aggregation cache."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must have shape [2, edges].')
    device = edge_index.device
    row, col = edge_index.long()
    self_nodes = torch.arange(nodes, device=device)
    row = torch.cat((row, self_nodes))
    col = torch.cat((col, self_nodes))
    if edge_weight is None:
        weight = torch.ones(row.numel(), device=device)
    else:
        weight = torch.cat((edge_weight.to(device), torch.ones(nodes, device=device)))
    degree = torch.zeros(nodes, device=device, dtype=weight.dtype)
    degree.index_add_(0, col, weight)
    normalizer = degree[row].clamp_min(1e-12).rsqrt() * degree[col].clamp_min(1e-12).rsqrt()
    return row, col, weight * normalizer


def _aggregate(states: Tensor, cache: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    row, col, weight = cache
    if states.ndim == 2:
        output = torch.zeros_like(states)
        output.index_add_(0, col, states[row] * weight.unsqueeze(-1))
        return output
    if states.ndim != 3:
        raise ValueError('states must have shape [nodes, channels] or [batch, nodes, channels].')
    output = torch.zeros_like(states)
    messages = states[:, row, :] * weight.view(1, -1, 1)
    output.scatter_add_(1, col.view(1, -1, 1).expand(states.shape[0], -1, states.shape[2]), messages)
    return output


def compute_normalization(
    states: Tensor,
    logits: Tensor,
    response: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Compute training-only scales used by all proxy losses."""
    state_mean = states.mean(dim=(0, 1), keepdim=True)
    state_scale = (states - state_mean).square().mean().clamp_min(1e-8)
    logit_mean = logits.mean(dim=(0, 1), keepdim=True)
    logit_scale = (logits - logit_mean).square().mean().clamp_min(1e-8)
    result = {'state_scale': state_scale, 'logit_scale': logit_scale}
    if response is not None:
        result['response_scale'] = response.square().mean().clamp_min(1e-8)
    return result


class GraphKoopmanV0(nn.Module):
    """V0 proxy: graph encoder -> Gaussian latent -> fixed linear rollout -> decoder."""

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        latent_dim: int = 32,
        condition_dim: int = 16,
        width: int = 64,
        queries: int = 8,
        step_size: float = 0.1,
        logvar_bounds: Tuple[float, float] = (-8.0, 4.0),
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or condition_dim <= 0 or width <= 0 or queries <= 0:
            raise ValueError('Proxy dimensions must be positive.')
        self.state_dim = int(state_dim)
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.width = int(width)
        self.queries = int(queries)
        self.step_size = float(step_size)
        self.logvar_bounds = tuple(float(value) for value in logvar_bounds)

        condition_input = 2 * self.state_dim + 1
        self.condition_net = nn.Sequential(
            nn.Linear(condition_input, width),
            nn.SiLU(),
            nn.Linear(width, condition_dim),
        )
        self.encoder_input = nn.Linear(state_dim + condition_dim, width)
        self.encoder_mix = nn.Linear(2 * width, width)
        self.queries_bank = nn.Parameter(torch.randn(queries, width) / math.sqrt(width))
        self.mean_head = nn.Linear(queries * width, latent_dim)
        self.logvar_head = nn.Linear(queries * width, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 2 * condition_dim, width),
            nn.SiLU(),
            nn.Linear(width, state_dim),
        )
        self.generator = nn.Parameter(torch.zeros(latent_dim, latent_dim))

    def make_edge_cache(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        nodes: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return _edge_cache(edge_index, edge_weight, nodes)

    def conditions(
        self,
        reference_state: Tensor,
        edge_cache: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if reference_state.ndim != 2:
            raise ValueError('reference_state must have shape [nodes, state_dim].')
        degree_features = edge_cache[2].new_zeros(reference_state.shape[0])
        degree_features.index_add_(0, edge_cache[1], edge_cache[2].abs())
        degree_features = torch.log1p(degree_features).unsqueeze(-1)
        neighborhood = _aggregate(reference_state, edge_cache)
        condition = self.condition_net(
            torch.cat((reference_state, neighborhood, degree_features), dim=-1)
        )
        return condition, _aggregate(condition, edge_cache)

    def encode(
        self,
        states: Tensor,
        condition: Tensor,
        neighborhood_condition: Tensor,
        edge_cache: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        batch, nodes, _ = states.shape
        static = condition.unsqueeze(0).expand(batch, -1, -1)
        local = torch.cat((states, static), dim=-1)
        first = torch.nn.functional.silu(self.encoder_input(local))
        mixed = torch.cat((first, _aggregate(first, edge_cache)), dim=-1)
        features = torch.nn.functional.silu(self.encoder_mix(mixed))
        scores = torch.einsum('pw,bnw->bpn', self.queries_bank, features) / math.sqrt(self.width)
        attention = torch.softmax(scores, dim=-1)
        pooled = torch.einsum('bpn,bnw->bpw', attention, features).flatten(1)
        mean = self.mean_head(pooled)
        logvar = self.logvar_head(pooled).clamp(*self.logvar_bounds)
        return mean, logvar

    def decode(
        self,
        latent: Tensor,
        condition: Tensor,
        neighborhood_condition: Tensor,
    ) -> Tensor:
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        batch = latent.shape[0]
        static = torch.cat((condition, neighborhood_condition), dim=-1)
        decoder_input = torch.cat(
            (
                latent[:, None, :].expand(-1, static.shape[0], -1),
                static.unsqueeze(0).expand(batch, -1, -1),
            ),
            dim=-1,
        )
        return self.decoder(decoder_input)

    def sample(self, mean: Tensor, logvar: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(mean)
        return mean + torch.exp(0.5 * logvar) * noise

    def operator(self) -> Tensor:
        return torch.eye(self.latent_dim, device=self.generator.device, dtype=self.generator.dtype) + self.step_size * self.generator

    def rollout(self, latent: Tensor, steps: int) -> Tensor:
        if steps < 0:
            raise ValueError('steps must be non-negative.')
        values = [latent]
        current = latent
        operator = self.operator()
        for _ in range(int(steps)):
            current = current @ operator.T
            values.append(current)
        return torch.stack(values, dim=1)

    def encode_trajectory(
        self,
        trajectory: Tensor,
        condition: Tensor,
        neighborhood_condition: Tensor,
        edge_cache: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        batch, time, nodes, channels = trajectory.shape
        mean, logvar = self.encode(
            trajectory.reshape(batch * time, nodes, channels),
            condition,
            neighborhood_condition,
            edge_cache,
        )
        return mean.reshape(batch, time, -1), logvar.reshape(batch, time, -1)

    def decode_rollout(
        self,
        latents: Tensor,
        condition: Tensor,
        neighborhood_condition: Tensor,
    ) -> Tensor:
        batch, time, latent_dim = latents.shape
        decoded = self.decode(
            latents.reshape(batch * time, latent_dim),
            condition,
            neighborhood_condition,
        )
        return decoded.reshape(batch, time, condition.shape[0], self.state_dim)

    def reconstruction_loss(
        self,
        trajectory: Tensor,
        target_logits: Tensor,
        condition: Tensor,
        neighborhood_condition: Tensor,
        edge_cache: Tuple[Tensor, Tensor, Tensor],
        readout: Callable[[Tensor], Tensor],
        scales: Dict[str, Tensor],
        kl_weight: float = 1e-4,
    ) -> Dict[str, Tensor]:
        mean, logvar = self.encode_trajectory(trajectory, condition, neighborhood_condition, edge_cache)
        latent = self.sample(mean[:, 0], logvar[:, 0])
        decoded = self.decode(latent, condition, neighborhood_condition)
        predicted_logits = readout(decoded)
        rec = (decoded - trajectory[:, 0]).square().mean() / scales['state_scale']
        output = (predicted_logits - target_logits[:, 0]).square().mean() / scales['logit_scale']
        kl = 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).mean()
        return {
            'total': rec + output + kl_weight * kl,
            'rec': rec.detach(),
            'pred': trajectory.new_zeros(()),
            'linear': trajectory.new_zeros(()),
            'output': output.detach(),
            'response': trajectory.new_zeros(()),
            'kl': kl.detach(),
        }

    def loss(
        self,
        trajectory: Tensor,
        target_logits: Tensor,
        base_trajectory: Tensor,
        base_logits: Tensor,
        origin: int,
        horizon: int,
        condition: Tensor,
        neighborhood_condition: Tensor,
        edge_cache: Tuple[Tensor, Tensor, Tensor],
        readout: Callable[[Tensor], Tensor],
        scales: Dict[str, Tensor],
        kl_weight: float = 1e-4,
        linear_weight: float = 0.1,
    ) -> Dict[str, Tensor]:
        mean, logvar = self.encode_trajectory(trajectory, condition, neighborhood_condition, edge_cache)
        base_mean, base_logvar = self.encode_trajectory(base_trajectory, condition, neighborhood_condition, edge_cache)
        noise = torch.randn_like(mean[:, origin])
        latent_origin = self.sample(mean[:, origin], logvar[:, origin], noise)
        base_latent_origin = self.sample(base_mean[:, origin], base_logvar[:, origin], noise[:base_mean.shape[0]])
        latent_rollout = self.rollout(latent_origin, horizon)
        base_latent_rollout = self.rollout(base_latent_origin, horizon)
        decoded = self.decode_rollout(latent_rollout, condition, neighborhood_condition)
        base_decoded = self.decode_rollout(base_latent_rollout, condition, neighborhood_condition)
        target_states = trajectory[:, origin:origin + horizon + 1]
        target_y = target_logits[:, origin:origin + horizon + 1]
        base_target_states = base_trajectory[:, origin:origin + horizon + 1]
        base_target_y = base_logits[:, origin:origin + horizon + 1]
        rec = (decoded[:, 0] - target_states[:, 0]).square().mean() / scales['state_scale']
        pred = (decoded[:, 1:] - target_states[:, 1:]).square().mean() / scales['state_scale']
        linear = (latent_rollout[:, 1:] - mean[:, origin + 1:origin + horizon + 1]).square().mean()
        predicted_logits = readout(decoded.reshape(-1, decoded.shape[-2], decoded.shape[-1])).reshape_as(target_y)
        base_predicted_logits = readout(base_decoded.reshape(-1, base_decoded.shape[-2], base_decoded.shape[-1])).reshape_as(base_target_y)
        output = (predicted_logits - target_y).square().mean() / scales['logit_scale']
        response_target = target_y - base_target_y
        response_prediction = predicted_logits - base_predicted_logits
        response_scale = scales.get('response_scale', response_target.detach().square().mean().clamp_min(1e-8))
        response = (response_prediction - response_target).square().mean() / response_scale
        kl = 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).mean()
        total = rec + pred + linear_weight * linear + output + response + kl_weight * kl
        return {
            'total': total,
            'rec': rec.detach(),
            'pred': pred.detach(),
            'linear': linear.detach(),
            'output': output.detach(),
            'response': response.detach(),
            'kl': kl.detach(),
        }
