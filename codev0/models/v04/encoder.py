"""Node-wise state encoder for Method V0.4."""

from __future__ import annotations

from torch import nn
import torch.nn.functional as F


class NodeStateEncoder(nn.Module):
    """Residual MLP that deliberately does not access graph topology."""

    def __init__(self, input_dim: int, width: int, state_dim: int):
        super().__init__()
        if min(input_dim, width, state_dim) <= 0:
            raise ValueError('V0.4 encoder dimensions must be positive.')
        self.hidden = nn.Linear(input_dim, width)
        self.output = nn.Linear(width, state_dim)
        self.residual = nn.Linear(input_dim, state_dim, bias=False)
        self.normalization = nn.LayerNorm(state_dim)

    def forward(self, features):
        hidden = F.silu(self.hidden(features))
        return self.normalization(
            self.output(hidden) + self.residual(features)
        )
