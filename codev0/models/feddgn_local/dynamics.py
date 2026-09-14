"""Clean anti-symmetric graph dynamics with a reserved external-force port.

The intrinsic update follows the A-DGN discretization

    H^{l+1} = H^l + eps * sigma((W-W^T-gamma I)H^l + Phi(H^l, G) + b)

with weights shared across recurrent steps.  The only extension is an explicit
external-force port kept *outside* the intrinsic vector field:

    dH/dt = F_int(H, G) + beta * F_ext(...).

For the current local-only stage F_ext is always ``None`` and therefore zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import init
from torch_geometric.nn import GCNConv


ForceLike = Union[torch.Tensor, Callable[..., torch.Tensor]]


@dataclass
class DGNDiagnostics:
    """Scalar diagnostics from the last recurrent step."""

    intrinsic_rms: torch.Tensor
    force_rms: torch.Tensor
    force_intrinsic_ratio: torch.Tensor
    state_rms: torch.Tensor


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x.square().mean())


def _edge_weight(data) -> Optional[torch.Tensor]:
    """Return scalar edge weights when the dataset provides them.

    The reference codebase feeds ``data.edge_attr`` into GCNConv.  Some PyG
    datasets, however, store vector edge attributes.  A-DGN's graph coupling
    here only consumes scalar weights, so vector attributes are intentionally
    ignored rather than being silently reduced.
    """

    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is not None:
        return edge_weight

    edge_attr = getattr(data, 'edge_attr', None)
    if edge_attr is None:
        return None
    if edge_attr.ndim == 1:
        return edge_attr
    if edge_attr.ndim == 2 and edge_attr.size(-1) == 1:
        return edge_attr.reshape(-1)
    return None


class AntiSymmetricGraphField(nn.Module):
    """Shared-weight A-DGN vector field.

    ``self.raw_weight`` parameterizes the anti-symmetric self interaction,
    while ``self.graph_conv`` realizes the generic neighborhood aggregation
    Phi from the A-DGN paper.  Self loops are disabled in Phi because the
    explicit anti-symmetric term already supplies the node-wise self dynamics.
    """

    def __init__(
        self,
        hidden_dim: int,
        gamma: float = 0.1,
        activation: str = 'tanh',
        graph_bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive.')
        if gamma < 0:
            raise ValueError('gamma must be non-negative.')

        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.raw_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.graph_conv = GCNConv(
            hidden_dim,
            hidden_dim,
            improved=False,
            cached=False,
            normalize=True,
            bias=bool(graph_bias),
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        activation = str(activation).lower()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'gelu':
            self.activation = torch.nn.functional.gelu
        else:
            raise ValueError('activation must be tanh, relu, or gelu.')

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match the released A-DGN implementation exactly.
        init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        self.graph_conv.reset_parameters()
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def anti_symmetric_weight(self) -> torch.Tensor:
        """Return W - W^T - gamma I from the current raw parameter."""

        eye = torch.eye(
            self.hidden_dim,
            device=self.raw_weight.device,
            dtype=self.raw_weight.dtype,
        )
        return self.raw_weight - self.raw_weight.transpose(0, 1) - self.gamma * eye

    def preactivation(self, state: torch.Tensor, data) -> torch.Tensor:
        weight = self.anti_symmetric_weight()
        self_term = torch.nn.functional.linear(state, weight)
        neighbor_term = self.graph_conv(
            state,
            data.edge_index,
            _edge_weight(data),
        )
        return self_term + neighbor_term + self.bias

    def forward(self, state: torch.Tensor, data) -> torch.Tensor:
        return self.activation(self.preactivation(state, data))


class ForcedAntiSymmetricFlow(nn.Module):
    """Short recurrent rollout of intrinsic A-DGN dynamics plus force port.

    External-force protocol
    -----------------------
    ``external_force`` can be either:
      * ``None``: exactly zero force (the current local-only setting);
      * a tensor broadcastable to ``state``;
      * a callable invoked as
        ``external_force(state=..., intrinsic=..., step_id=..., data=...)``.

    This mirrors the tensor/callable port used in the PH-DGN adaptation while
    retaining the state-dependent forcing semantics used by SONAR/IGNS.  The
    interface is deliberately model-agnostic so a future Koopman/TANGO module
    can be plugged in without modifying the intrinsic backbone.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_steps: int = 16,
        step_size: float = 0.1,
        gamma: float = 0.1,
        activation: str = 'tanh',
    ) -> None:
        super().__init__()
        if int(num_steps) <= 0:
            raise ValueError('num_steps must be positive.')
        if float(step_size) <= 0:
            raise ValueError('step_size must be positive.')
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.field = AntiSymmetricGraphField(
            hidden_dim=hidden_dim,
            gamma=gamma,
            activation=activation,
        )

    def intrinsic_vector_field(self, state: torch.Tensor, data) -> torch.Tensor:
        """Evaluate only the local/client-intrinsic vector field."""

        return self.field(state, data)

    def _resolve_force(
        self,
        external_force: Optional[ForceLike],
        state: torch.Tensor,
        intrinsic: torch.Tensor,
        step_id: int,
        data,
    ) -> torch.Tensor:
        if external_force is None:
            return torch.zeros_like(state)

        if callable(external_force):
            force = external_force(
                state=state,
                intrinsic=intrinsic,
                step_id=int(step_id),
                data=data,
            )
        else:
            force = external_force

        if not torch.is_tensor(force):
            force = state.new_tensor(force)
        force = force.to(device=state.device, dtype=state.dtype)
        if force.ndim == 1:
            force = force.unsqueeze(0)
        try:
            force = torch.broadcast_to(force, state.shape)
        except RuntimeError as exc:
            raise ValueError(
                'external force must be broadcastable to the hidden-state shape '
                f'{tuple(state.shape)}, got {tuple(force.shape)}.'
            ) from exc
        return force

    def vector_field(
        self,
        state: torch.Tensor,
        data,
        external_force: Optional[ForceLike] = None,
        force_scale: float = 1.0,
        step_id: int = 0,
    ):
        """Return total, intrinsic, and external vector fields at one state."""

        intrinsic = self.intrinsic_vector_field(state, data)
        force = self._resolve_force(
            external_force, state, intrinsic, step_id, data
        )
        applied_force = float(force_scale) * force
        total = intrinsic + applied_force
        return total, intrinsic, applied_force

    def step(
        self,
        state: torch.Tensor,
        data,
        step_id: int,
        external_force: Optional[ForceLike] = None,
        force_scale: float = 1.0,
    ):
        total, intrinsic, force = self.vector_field(
            state,
            data,
            external_force=external_force,
            force_scale=force_scale,
            step_id=step_id,
        )
        next_state = state + self.step_size * total
        ratio = _rms(force) / _rms(intrinsic).clamp_min(1e-8)
        diagnostics = DGNDiagnostics(
            intrinsic_rms=_rms(intrinsic),
            force_rms=_rms(force),
            force_intrinsic_ratio=ratio,
            state_rms=_rms(next_state),
        )
        return next_state, diagnostics

    def forward(
        self,
        state: torch.Tensor,
        data,
        external_force: Optional[ForceLike] = None,
        force_scale: float = 1.0,
        return_trajectory: bool = False,
    ):
        trajectory = [state] if return_trajectory else None
        diagnostics = None
        for step_id in range(self.num_steps):
            state, diagnostics = self.step(
                state,
                data,
                step_id=step_id,
                external_force=external_force,
                force_scale=force_scale,
            )
            if trajectory is not None:
                trajectory.append(state)
        return state, diagnostics, trajectory


__all__ = [
    'AntiSymmetricGraphField',
    'DGNDiagnostics',
    'ForcedAntiSymmetricFlow',
]
