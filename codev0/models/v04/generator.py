"""Structurally stable K=1 Bernstein Graph-Koopman generator."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.v04.spectral_basis import aggregate, bernstein_k1


class DissipativeGenerator(nn.Module):
    """Dense latent generator ``A-A.T-BB.T-gamma*I``."""

    def __init__(
        self,
        dim: int,
        gamma: float,
        damping_init: float,
        skew_init: float = 1e-3,
    ):
        super().__init__()
        if dim <= 0 or gamma <= 0 or damping_init < 0 or skew_init < 0:
            raise ValueError('Invalid dissipative generator configuration.')
        self.dim = int(dim)
        self.gamma = float(gamma)
        self.skew_raw = nn.Parameter(torch.empty(dim, dim))
        self.dissipation_raw = nn.Parameter(torch.empty(dim, dim))
        self.reset_parameters(damping_init, skew_init)

    def reset_parameters(self, damping_init: float, skew_init: float):
        nn.init.normal_(self.skew_raw, mean=0.0, std=float(skew_init))
        with torch.no_grad():
            self.dissipation_raw.zero_()
            diagonal = math.sqrt(float(damping_init))
            if diagonal:
                self.dissipation_raw.diagonal().fill_(diagonal)

    def matrix(self):
        skew = self.skew_raw - self.skew_raw.transpose(-1, -2)
        damping = self.dissipation_raw @ self.dissipation_raw.transpose(-1, -2)
        identity = torch.eye(
            self.dim,
            device=self.skew_raw.device,
            dtype=self.skew_raw.dtype,
        )
        return skew - damping - self.gamma * identity


    @torch.no_grad()
    def stability_diagnostics(self):
        matrix = self.matrix()
        symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
        symmetric_max = torch.linalg.eigvalsh(symmetric).max().real
        max_real = torch.linalg.eigvals(matrix).real.max()
        return {
            'symmetric_max_eigenvalue': symmetric_max,
            'max_real_part': max_real,
        }


class StableBernsteinGraphGenerator(nn.Module):
    """Matrix-free K=1 topology-aware continuous generator."""

    def __init__(
        self,
        dim: int,
        gamma: float,
        low_damping_init: float,
        high_damping_init: float,
    ):
        super().__init__()
        self.dim = int(dim)
        self.generators = nn.ModuleList([
            DissipativeGenerator(dim, gamma, low_damping_init),
            DissipativeGenerator(dim, gamma, high_damping_init),
        ])


    def matrices(self):
        return tuple(generator.matrix() for generator in self.generators)

    def forward(self, latent, operator, matrices=None):
        low, high = bernstein_k1(latent, operator)
        if matrices is None:
            matrices = self.matrices()
        return low @ matrices[0].transpose(-1, -2) + high @ matrices[1].transpose(-1, -2)

    def one_tap(self, latent, operator, matrices=None):
        """Equivalent self/neighbor representation used by correctness tests."""

        if matrices is None:
            matrices = self.matrices()
        self_matrix = 0.5 * (matrices[0] + matrices[1])
        neighbor_matrix = 0.5 * (matrices[0] - matrices[1])
        return (
            latent @ self_matrix.transpose(-1, -2)
            + aggregate(operator, latent)
            @ neighbor_matrix.transpose(-1, -2)
        )

    @torch.no_grad()
    def stability_diagnostics(self):
        output = {}
        for index, generator in enumerate(self.generators):
            values = generator.stability_diagnostics()
            output[f'g{index}_symmetric_maxeig'] = values[
                'symmetric_max_eigenvalue'
            ]
            output[f'g{index}_max_real_part'] = values['max_real_part']
        return output
