"""Task-anchored neural graph vector field for Method V0.4."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.v04.spectral_basis import aggregate


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(float(value)))


class NeuralGraphVectorField(nn.Module):
    """Two-layer graph field with a positive, non-zero RMS scale."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        scale_init: float,
        scale_min: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        if min(state_dim, hidden_dim) <= 0:
            raise ValueError('V0.4 field dimensions must be positive.')
        if not 0 < scale_min < scale_init:
            raise ValueError('field_scale_init must exceed field_scale_min > 0.')
        self.self_first = nn.Linear(state_dim, hidden_dim)
        self.neighbor_first = nn.Linear(state_dim, hidden_dim, bias=False)
        self.self_second = nn.Linear(hidden_dim, state_dim, bias=False)
        self.neighbor_second = nn.Linear(hidden_dim, state_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(
            _softplus_inverse(scale_init - scale_min), dtype=torch.float32
        ))
        self.scale_min = float(scale_min)
        self.eps = float(eps)

    def scale(self):
        return self.scale_min + F.softplus(self.log_scale)

    def raw_field(self, state, operator):
        hidden = F.silu(
            self.self_first(state)
            + aggregate(operator, self.neighbor_first(state))
        )
        return (
            self.self_second(hidden)
            + aggregate(operator, self.neighbor_second(hidden))
        )

    def forward_with_diagnostics(self, state, operator):
        raw = self.raw_field(state, operator)
        raw_rms = torch.sqrt(raw.square().mean() + self.eps)
        field = self.scale() * raw / raw_rms
        diagnostics = {
            'field_raw_rms': raw_rms,
            'field_rms': torch.sqrt(field.square().mean() + self.eps),
            'field_scale': self.scale(),
        }
        return field, diagnostics

    def forward(self, state, operator):
        return self.forward_with_diagnostics(state, operator)[0]
