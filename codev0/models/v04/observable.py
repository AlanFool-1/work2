"""State-preserving observable dictionary for Method V0.4."""

from __future__ import annotations

import torch
from torch import nn


class StatePreservingObservable(nn.Module):
    """Lift ``H`` to ``[H, rho(H)]`` without mixing the identity block."""

    def __init__(self, state_dim: int, aux_dim: int, width: int):
        super().__init__()
        if min(state_dim, width) <= 0 or aux_dim < 0:
            raise ValueError('V0.4 observable dimensions are invalid.')
        self.state_dim = int(state_dim)
        self.aux_dim = int(aux_dim)
        if self.aux_dim:
            self.auxiliary = nn.Sequential(
                nn.Linear(state_dim, width),
                nn.SiLU(),
                nn.Linear(width, aux_dim),
            )
        else:
            self.auxiliary = None

    @property
    def output_dim(self):
        return self.state_dim + self.aux_dim

    def forward(self, state):
        if self.auxiliary is None:
            return state
        return torch.cat((state, self.auxiliary(state)), dim=-1)
