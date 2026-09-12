from typing import List

import torch
from torch import nn


class StateEncoder(nn.Module):
    """Node-wise initial-state encoder with an explicit scale gauge fix.

    No graph propagation happens here.  Optional non-affine LayerNorm on the
    final physical state removes a trivial per-client scale degree of freedom,
    which stabilizes Koopman coordinates and makes later operator comparison
    less sensitive to encoder norm growth.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        num_layers: int = 2,
        input_dropout: float = 0.20,
        dropout: float = 0.35,
        gauge_fix: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_dropout = nn.Dropout(input_dropout)

        dims: List[int] = [in_dim]
        if num_layers == 1:
            dims.append(hidden_dim)
        else:
            dims.extend([mlp_hidden_dim] * (num_layers - 1))
            dims.append(hidden_dim)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.final_norm = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False) if gauge_fix else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        return self.final_norm(self.net(x))


class KoopmanObservable(nn.Module):
    """State-preserving nonlinear lifting Psi(H)=[H, rho(H)].

    There is intentionally no stochastic dropout in rho: a Koopman coordinate
    system should be deterministic.  Generalization is handled by the encoder,
    physical-only readout, low-rank/shared-core operator, and training protocol.
    """

    def __init__(
        self,
        hidden_dim: int,
        aux_dim: int,
        rho_hidden_dim: int = 64,
        activation: str = "tanh",
        aux_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aux_dim = aux_dim
        self.aux_scale = float(aux_scale)

        if activation == "tanh":
            act1: nn.Module = nn.Tanh()
            act2: nn.Module = nn.Tanh()
        elif activation == "gelu":
            act1 = nn.GELU()
            act2 = nn.Tanh()  # bounded observable output fixes latent scale
        else:
            raise ValueError(f"Unknown activation={activation}")

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, rho_hidden_dim),
            act1,
            nn.Linear(rho_hidden_dim, aux_dim),
            act2,
        )

    @property
    def observable_dim(self) -> int:
        return self.hidden_dim + self.aux_dim

    def aux(self, h: torch.Tensor) -> torch.Tensor:
        return self.aux_scale * self.rho(h)

    def lift(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, self.aux(h)], dim=-1)

    def split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = z[..., : self.hidden_dim]
        aux = z[..., self.hidden_dim :]
        return h, aux

    def closure_target(self, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.split(z)
        return self.aux(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lift(h)
