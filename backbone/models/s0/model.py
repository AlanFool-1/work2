"""Official Anti-Symmetric DGN backbone used by the FedAvg baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from torch import nn


def load_official_adgn_class(base_path: str):
    """Import ``GraphAntiSymmetricNN`` directly from the extracted project."""
    source = (
        Path(base_path).resolve()
        / 'Anti-SymmetricDGN-main'
        / 'graph_heteropily'
        / 'models'
        / 'antisymmetric_dgn.py'
    )
    if not source.is_file():
        raise FileNotFoundError(f'Official A-DGN source is missing: {source}')

    module_name = '_official_antisymmetric_dgn'
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load official A-DGN source: {source}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.GraphAntiSymmetricNN


def build_s0_model(args) -> nn.Module:
    """Build the sole baseline backbone: official A-DGN with fixed Euler."""
    official_class = load_official_adgn_class(args.base_path)
    return official_class(
        input_dim=args.n_feat,
        output_dim=args.n_clss,
        hidden_dim=args.hidden_dim,
        num_layers=args.ode_steps,
        epsilon=args.adgn_step_size,
        gamma=args.ode_gamma,
        activ_fun=args.ode_activation,
        gcn_norm=True,
        bias=True,
    )
