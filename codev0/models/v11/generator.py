from typing import Dict, List

import torch
from torch import nn

from .graph_ops import BernsteinGraphBasis, GraphCache


class LowRankPHCore(nn.Module):
    """Low-rank port-Hamiltonian channel core.

    J = U V^T - V U^T       (skew, conservative)
    R = C C^T               (PSD dissipation basis)

    Dissipation *strength* is controlled by a separate band gain beta. Keeping C
    at a healthy scale while beta starts tiny avoids the v1.0 failure mode where
    C itself collapses to zero and becomes difficult to re-grow.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        conservative_init_scale: float = 0.05,
        dissipation_basis_init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if rank > dim:
            raise ValueError("rank cannot exceed dim")
        self.dim = int(dim)
        self.rank = int(rank)
        self.U = nn.Parameter(conservative_init_scale * torch.randn(dim, rank))
        self.V = nn.Parameter(conservative_init_scale * torch.randn(dim, rank))
        self.C = nn.Parameter(dissipation_basis_init_scale * torch.randn(dim, rank))

    def apply_J_transpose(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) @ self.U.T - (x @ self.U) @ self.V.T

    def apply_R(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.C) @ self.C.T

    def dense_J(self) -> torch.Tensor:
        return self.U @ self.V.T - self.V @ self.U.T

    def dense_R(self) -> torch.Tensor:
        return self.C @ self.C.T


class FullRankPHCore(nn.Module):
    """Full-rank port-Hamiltonian capacity upper bound."""

    def __init__(
        self,
        dim: int,
        conservative_init_scale: float = 0.05,
        dissipation_basis_init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.W = nn.Parameter(conservative_init_scale * torch.randn(dim, dim))
        self.C = nn.Parameter(dissipation_basis_init_scale * torch.randn(dim, dim))

    def apply_J_transpose(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T - x @ self.W

    def apply_R(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.C) @ self.C.T

    def dense_J(self) -> torch.Tensor:
        return self.W - self.W.T

    def dense_R(self) -> torch.Tensor:
        return self.C @ self.C.T


class PHBandCoefficients(nn.Module):
    """Small band-specific gains.

    G_q = alpha_q J_q - beta_q R_q - gamma_q I.

    beta and gamma use squared parameters rather than softplus near zero.  This
    preserves non-negativity while avoiding the vanishing-gradient behavior of a
    very negative softplus raw parameter.
    """

    def __init__(
        self,
        alpha_init: float,
        beta_init: float,
        gamma_init: float,
        min_gamma: float,
        learnable_gamma: bool,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init) ** 0.5))
        self.min_gamma = float(min_gamma)
        gamma_raw = torch.tensor(max(float(gamma_init - min_gamma), 0.0) ** 0.5)
        if learnable_gamma:
            self.gamma_raw = nn.Parameter(gamma_raw)
        else:
            self.register_buffer("gamma_raw", gamma_raw)

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_raw.square()

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_raw.square() + self.min_gamma


class GraphKoopmanVectorField(nn.Module):
    """Structured Graph-Koopman ODE.

    dZ/dt = sum_q B_q(L/2) Z G_q^T

    Default v1.1 uses one shared low-rank PH channel core with band-specific
    gains. This regularizes tiny local-client datasets while preserving spectral
    graph response. `per_band` remains an upper-capacity ablation.
    """

    def __init__(
        self,
        observable_dim: int,
        basis_degree: int,
        operator_rank: int,
        generator_type: str = "low_rank_ph",
        band_parameterization: str = "shared_core",
        gamma_init: float = 1e-4,
        min_gamma: float = 0.0,
        learnable_gamma: bool = True,
        dissipation_basis_init_scale: float = 0.05,
        dissipation_gain_init: float = 1e-3,
        conservative_init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.observable_dim = int(observable_dim)
        self.basis_degree = int(basis_degree)
        self.operator_rank = int(operator_rank)
        self.generator_type = generator_type
        self.band_parameterization = band_parameterization
        self.basis = BernsteinGraphBasis(basis_degree)

        def make_core() -> nn.Module:
            if generator_type == "low_rank_ph":
                return LowRankPHCore(
                    observable_dim,
                    operator_rank,
                    conservative_init_scale=conservative_init_scale,
                    dissipation_basis_init_scale=dissipation_basis_init_scale,
                )
            if generator_type == "full_rank_ph":
                return FullRankPHCore(
                    observable_dim,
                    conservative_init_scale=conservative_init_scale,
                    dissipation_basis_init_scale=dissipation_basis_init_scale,
                )
            raise ValueError(f"Unknown generator_type={generator_type}")

        qn = basis_degree + 1
        if qn == 1:
            alpha_init = [1.0]
        else:
            alpha_init = torch.linspace(0.5, 1.5, qn).tolist()

        self.band_coeffs = nn.ModuleList(
            [
                PHBandCoefficients(
                    alpha_init=alpha_init[q],
                    beta_init=dissipation_gain_init,
                    gamma_init=gamma_init,
                    min_gamma=min_gamma,
                    learnable_gamma=learnable_gamma,
                )
                for q in range(qn)
            ]
        )

        if band_parameterization == "shared_core":
            self.shared_core = make_core()
            self.cores = None
        elif band_parameterization == "per_band":
            self.shared_core = None
            self.cores = nn.ModuleList([make_core() for _ in range(qn)])
        else:
            raise ValueError(f"Unknown band_parameterization={band_parameterization}")

    def core_for_band(self, q: int) -> nn.Module:
        if self.shared_core is not None:
            return self.shared_core
        assert self.cores is not None
        return self.cores[q]

    def apply_band_generator_transpose(self, q: int, x: torch.Tensor) -> torch.Tensor:
        core = self.core_for_band(q)
        coeff = self.band_coeffs[q]
        return (
            coeff.alpha * core.apply_J_transpose(x)
            - coeff.beta * core.apply_R(x)
            - coeff.gamma * x
        )

    def dense_generator(self, q: int) -> torch.Tensor:
        core = self.core_for_band(q)
        coeff = self.band_coeffs[q]
        eye = torch.eye(self.observable_dim, device=coeff.alpha.device, dtype=coeff.alpha.dtype)
        return coeff.alpha * core.dense_J() - coeff.beta * core.dense_R() - coeff.gamma * eye

    def forward(self, t: torch.Tensor, z: torch.Tensor, graph: GraphCache) -> torch.Tensor:
        del t
        dz = torch.zeros_like(z)
        for q in range(self.basis_degree + 1):
            z_band = self.basis.apply(q, z, graph)
            dz = dz + self.apply_band_generator_transpose(q, z_band)
        return dz

    def dense_generators(self, detach: bool = False) -> List[torch.Tensor]:
        out = [self.dense_generator(q) for q in range(self.basis_degree + 1)]
        if detach:
            out = [x.detach().clone() for x in out]
        return out

    def operator_speed_penalty(self) -> torch.Tensor:
        mats = self.dense_generators(detach=False)
        denom = float(self.observable_dim * self.observable_dim)
        return torch.stack([g.square().sum() / denom for g in mats]).mean()

    def operator_state_dict(self, detach: bool = True, cpu: bool = True) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for name, value in self.state_dict().items():
            tensor = value.detach().clone() if detach else value.clone()
            if cpu:
                tensor = tensor.cpu()
            result[name] = tensor
        return result

    def load_operator_state_dict(self, payload: Dict[str, torch.Tensor], strict: bool = True) -> None:
        self.load_state_dict(payload, strict=strict)
