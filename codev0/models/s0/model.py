"""Official Anti-Symmetric DGN backbone used by the FedAvg baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MethodType

import torch
from torch import nn


def resolve_official_adgn_source(base_path: str) -> Path:
    """Resolve the extracted official A-DGN source without moving user data.

    Older runs placed the repository directly below ``base_path``.  The
    algorithm workspace now keeps third-party repositories below ``GitHub``.
    Prefer the legacy location when both exist so old commands retain their
    original behavior.
    """
    relative = Path(
        'Anti-SymmetricDGN-main/graph_heteropily/models/antisymmetric_dgn.py'
    )
    candidates = [
        Path(base_path).resolve() / relative,
        Path(base_path).resolve() / 'GitHub' / relative,
    ]
    for source in candidates:
        if source.is_file():
            return source
    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f'Official A-DGN source is missing; searched: {searched}'
    )


def load_official_adgn_class(base_path: str):
    """Import ``GraphAntiSymmetricNN`` directly from the extracted project."""
    source = resolve_official_adgn_source(base_path)

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
    model = official_class(
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

    @torch.no_grad()
    def encode_initial_state(self, data):
        """Return the hidden state immediately before native Euler updates."""
        if self.emb is None:
            return data.x.detach().clone()
        captured = {}

        def capture(_module, _inputs, output):
            captured['h0'] = output.detach().clone()

        handle = self.emb.register_forward_hook(capture)
        was_training = self.training
        self.eval()
        try:
            self(data)
        finally:
            handle.remove()
            self.train(was_training)
        if 'h0' not in captured:
            raise RuntimeError('Failed to capture the native A-DGN initial state.')
        return captured['h0']

    @torch.no_grad()
    def encode_native_dynamics_states(self, data):
        """Capture the exact native states H_0,...,H_T without changing updates."""
        if self.conv.gcn_conv is None:
            raise RuntimeError('Native trajectory capture requires gcn_norm=True.')
        states = []
        final_state = {}

        def capture_step_input(_module, inputs, _output):
            states.append(inputs[0].detach().clone())

        def capture_final(_module, _inputs, output):
            final_state['hT'] = output.detach().clone()

        step_handle = self.conv.gcn_conv.register_forward_hook(capture_step_input)
        final_handle = self.conv.register_forward_hook(capture_final)
        was_training = self.training
        self.eval()
        try:
            self(data)
        finally:
            step_handle.remove()
            final_handle.remove()
            self.train(was_training)
        if len(states) != int(self.conv.num_iters) or 'hT' not in final_state:
            raise RuntimeError('Failed to capture every native A-DGN Euler state.')
        states.append(final_state['hT'])
        return torch.stack(states, dim=0)

    model.encode_initial_state = MethodType(encode_initial_state, model)
    model.encode_native_dynamics_states = MethodType(
        encode_native_dynamics_states, model
    )
    return model
