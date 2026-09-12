from dataclasses import dataclass
from typing import Literal


@dataclass
class Method1LocalConfig:
    """Method 1.1 local PH-Koopman backbone configuration.

    Design goals:
    1) one propagated trajectory only;
    2) nonlinear state-preserving observable;
    3) linear low-rank port-Hamiltonian Koopman dynamics;
    4) physical-state readout by default to prevent the auxiliary observable from
       becoming a label-memorization shortcut;
    5) no forced dissipation floor.
    """

    in_dim: int
    num_classes: int

    # Initial physical state
    hidden_dim: int = 64
    encoder_hidden_dim: int = 128
    encoder_layers: int = 2
    input_dropout: float = 0.20
    encoder_dropout: float = 0.35
    gauge_fix_initial_state: bool = True

    # Nonlinear state-preserving observable Psi(H)=[H,rho(H)]
    aux_dim: int = 32
    observable_hidden_dim: int = 64
    observable_activation: Literal["tanh", "gelu"] = "tanh"
    aux_scale: float = 1.0

    # Graph-Koopman dynamics
    basis_degree: int = 2
    operator_rank: int = 8
    generator_type: Literal["low_rank_ph", "full_rank_ph"] = "low_rank_ph"
    band_parameterization: Literal["shared_core", "per_band"] = "shared_core"

    # Port-Hamiltonian dissipation. Keep defaults near-conservative; do not force
    # the network to dissipate when the task does not need it.
    gamma_init: float = 1e-4
    min_gamma: float = 0.0
    learnable_gamma: bool = True
    dissipation_basis_init_scale: float = 0.05
    dissipation_gain_init: float = 1e-3
    conservative_init_scale: float = 0.05

    # ODE integration
    ode_horizon: float = 1.0
    ode_steps: int = 8
    solver: Literal["euler", "heun", "rk4"] = "rk4"

    # Graph preprocessing
    symmetrize_edges: bool = False
    remove_self_loops: bool = True

    # Readout: default h-only is deliberate; auxiliary coordinates support
    # linearization but cannot directly memorize labels.
    classifier_on: Literal["h", "z"] = "h"
    classifier_dropout: float = 0.50

    # Regularization. Manifold loss is relative/normalized in v1.1.
    manifold_weight: float = 1e-3
    label_smoothing: float = 0.05
    operator_speed_weight: float = 1e-4

    @property
    def observable_dim(self) -> int:
        return self.hidden_dim + self.aux_dim

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.aux_dim <= 0:
            raise ValueError("hidden_dim and aux_dim must be positive")
        if self.encoder_layers < 1:
            raise ValueError("encoder_layers must be >= 1")
        if self.basis_degree < 0:
            raise ValueError("basis_degree must be >= 0")
        if self.operator_rank <= 0:
            raise ValueError("operator_rank must be positive")
        if self.operator_rank > self.observable_dim:
            raise ValueError("operator_rank cannot exceed observable_dim")
        if self.ode_steps <= 0:
            raise ValueError("ode_steps must be positive")
        if self.ode_horizon <= 0:
            raise ValueError("ode_horizon must be positive")
        if self.gamma_init < 0 or self.min_gamma < 0:
            raise ValueError("gamma values must be non-negative")
        if self.gamma_init < self.min_gamma:
            raise ValueError("gamma_init must be >= min_gamma")
        if self.dissipation_basis_init_scale < 0 or self.dissipation_gain_init < 0:
            raise ValueError("dissipation initialization values must be non-negative")
        if not 0 <= self.input_dropout < 1:
            raise ValueError("input_dropout must be in [0,1)")
        if not 0 <= self.encoder_dropout < 1:
            raise ValueError("encoder_dropout must be in [0,1)")
        if not 0 <= self.classifier_dropout < 1:
            raise ValueError("classifier_dropout must be in [0,1)")
        if self.aux_scale <= 0:
            raise ValueError("aux_scale must be positive")
        if self.manifold_weight < 0 or self.operator_speed_weight < 0:
            raise ValueError("regularization weights must be non-negative")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0,1)")
