"""Build Method 1.1 from codev0's argparse namespace.

Kept separate from the reference package so `config.py`, `model.py` and the
other backbone files stay byte-identical to the upstream Method 1.1 sources.
"""

from __future__ import annotations

from models.v11.config import Method1LocalConfig
from models.v11.model import Method1LocalKoopmanGNN


def build_v11_config(args) -> Method1LocalConfig:
    return Method1LocalConfig(
        in_dim=int(args.n_feat),
        num_classes=int(args.n_clss),
        hidden_dim=int(args.v11_hidden_dim),
        encoder_hidden_dim=int(args.v11_encoder_hidden),
        encoder_layers=int(args.v11_encoder_layers),
        input_dropout=float(args.v11_input_dropout),
        encoder_dropout=float(args.v11_encoder_dropout),
        gauge_fix_initial_state=bool(args.v11_gauge_fix),
        aux_dim=int(args.v11_aux_dim),
        observable_hidden_dim=int(args.v11_observable_hidden),
        observable_activation=str(args.v11_observable_activation),
        aux_scale=float(args.v11_aux_scale),
        basis_degree=int(args.v11_basis_degree),
        operator_rank=int(args.v11_operator_rank),
        generator_type=str(args.v11_generator_type),
        band_parameterization=str(args.v11_band_parameterization),
        gamma_init=float(args.v11_gamma_init),
        min_gamma=float(args.v11_min_gamma),
        learnable_gamma=bool(args.v11_learnable_gamma),
        dissipation_basis_init_scale=float(args.v11_dissipation_basis_init),
        dissipation_gain_init=float(args.v11_dissipation_gain_init),
        conservative_init_scale=float(args.v11_conservative_init),
        ode_horizon=float(args.v11_ode_horizon),
        ode_steps=int(args.v11_ode_steps),
        solver=str(args.v11_solver),
        symmetrize_edges=bool(args.v11_symmetrize_edges),
        remove_self_loops=bool(args.v11_remove_self_loops),
        classifier_on=str(args.v11_classifier_on),
        classifier_dropout=float(args.v11_classifier_dropout),
        manifold_weight=float(args.v11_manifold_weight),
        label_smoothing=float(args.v11_label_smoothing),
        operator_speed_weight=float(args.v11_operator_speed_weight),
    )


def build_v11_model(args) -> Method1LocalKoopmanGNN:
    return Method1LocalKoopmanGNN(build_v11_config(args))
