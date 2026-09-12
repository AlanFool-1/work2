from typing import Any, Dict, Optional

import torch
from torch import nn

from .config import Method1LocalConfig
from .generator import GraphKoopmanVectorField
from .graph_ops import build_normalized_adjacency
from .observable import KoopmanObservable, StateEncoder
from .ode import integrate_fixed


class Method1LocalKoopmanGNN(nn.Module):
    """Method 1.1 local Port-Hamiltonian Graph-Koopman backbone.

    X -> node-wise encoder -> H0
      -> deterministic nonlinear lift Psi(H0)=[H0,rho(H0)] -> Z0
      -> low-rank PH Graph-Koopman ODE -> Z(T)
      -> physical-state classifier by default

    There is only one propagated trajectory.  Nonlinearity comes from the
    learned coordinate map; the continuous latent flow is linear and structured.
    """

    def __init__(self, config: Method1LocalConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.encoder = StateEncoder(
            in_dim=config.in_dim,
            hidden_dim=config.hidden_dim,
            mlp_hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_layers,
            input_dropout=config.input_dropout,
            dropout=config.encoder_dropout,
            gauge_fix=config.gauge_fix_initial_state,
        )
        self.observable = KoopmanObservable(
            hidden_dim=config.hidden_dim,
            aux_dim=config.aux_dim,
            rho_hidden_dim=config.observable_hidden_dim,
            activation=config.observable_activation,
            aux_scale=config.aux_scale,
        )
        self.koopman_field = GraphKoopmanVectorField(
            observable_dim=config.observable_dim,
            basis_degree=config.basis_degree,
            operator_rank=config.operator_rank,
            generator_type=config.generator_type,
            band_parameterization=config.band_parameterization,
            gamma_init=config.gamma_init,
            min_gamma=config.min_gamma,
            learnable_gamma=config.learnable_gamma,
            dissipation_basis_init_scale=config.dissipation_basis_init_scale,
            dissipation_gain_init=config.dissipation_gain_init,
            conservative_init_scale=config.conservative_init_scale,
        )

        readout_dim = config.observable_dim if config.classifier_on == "z" else config.hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(readout_dim, config.num_classes),
        )

    def encode_initial_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.encoder(x)
        z0 = self.observable(h0)
        return h0, z0

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Any:
        graph = build_normalized_adjacency(
            edge_index=edge_index,
            num_nodes=x.shape[0],
            edge_weight=edge_weight,
            dtype=x.dtype,
            device=x.device,
            symmetrize=self.config.symmetrize_edges,
            remove_self_loops=self.config.remove_self_loops,
        )

        h0, z0 = self.encode_initial_state(x)

        def vf(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return self.koopman_field(t, z, graph)

        trajectory = integrate_fixed(
            vf,
            z0,
            horizon=self.config.ode_horizon,
            steps=self.config.ode_steps,
            solver=self.config.solver,
            return_trajectory=True,
        )
        zT = trajectory[-1]
        hT, auxT = self.observable.split(zT)
        readout = zT if self.config.classifier_on == "z" else hT
        logits = self.classifier(readout)

        if not return_details:
            return logits

        return {
            "logits": logits,
            "h0": h0,
            "z0": z0,
            "trajectory": trajectory,
            "zT": zT,
            "hT": hT,
            "auxT": auxT,
        }

    def export_operator_state(self, cpu: bool = True) -> Dict[str, torch.Tensor]:
        return self.koopman_field.operator_state_dict(detach=True, cpu=cpu)

    def load_operator_state(self, payload: Dict[str, torch.Tensor], strict: bool = True) -> None:
        self.koopman_field.load_operator_state_dict(payload, strict=strict)

    @torch.no_grad()
    def dense_koopman_generators(self, cpu: bool = True) -> list[torch.Tensor]:
        mats = self.koopman_field.dense_generators(detach=True)
        return [m.cpu() for m in mats] if cpu else mats
