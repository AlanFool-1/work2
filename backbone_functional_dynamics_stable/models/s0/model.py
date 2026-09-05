"""Official Anti-Symmetric DGN backbone used by the FedAvg baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MethodType

import torch
from torch import nn
from models.dissipative_injector import low_rank_injection


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

    if bool(getattr(args, 'enable_functional_dynamics', False)):
        conv = model.conv
        conv._fd_enabled = False
        conv._fd_basis = None
        conv._fd_delta = None
        conv._fd_beta = 0.0
        conv._fd_last_ratio = 0.0

        def functional_forward(
            self, x, edge_index, edge_weight=None
        ):
            antisymmetric_w = self.W - self.W.T - self.gamma * self.I
            native_norms = []
            injected_norms = []
            for _ in range(self.num_iters):
                if self.gcn_conv is None:
                    neigh_x = self.lin(x)
                    neigh_x = self.propagate(
                        edge_index, x=neigh_x, edge_weight=edge_weight
                    )
                else:
                    neigh_x = self.gcn_conv(
                        x, edge_index=edge_index, edge_weight=edge_weight
                    )
                field = x @ antisymmetric_w.T + neigh_x
                if self.bias is not None:
                    field = field + self.bias
                native = self.activation(field)
                if (
                    self._fd_enabled
                    and self._fd_basis is not None
                    and self._fd_delta is not None
                ):
                    injected = low_rank_injection(
                        x, self._fd_basis, self._fd_delta, self._fd_beta
                    )
                    native_norms.append(torch.linalg.matrix_norm(native).detach())
                    injected_norms.append(
                        torch.linalg.matrix_norm(injected).detach()
                    )
                    native = native + injected
                x = x + self.epsilon * native
            if native_norms:
                numerator = torch.stack(injected_norms).mean()
                denominator = torch.stack(native_norms).mean() + 1e-12
                self._fd_last_ratio = float((numerator / denominator).item())
            else:
                self._fd_last_ratio = 0.0
            return x

        conv.forward = MethodType(functional_forward, conv)

    @torch.no_grad()
    def encode_initial_state(self, data):
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
            raise RuntimeError('Failed to capture the official A-DGN initial state H0.')
        return captured['h0']

    @torch.no_grad()
    def encode_dynamics_states(self, data):
        """Capture H0,...,HT without modifying the official Euler updates."""
        states = []
        final_state = {}

        def capture_step_input(_module, inputs, _output):
            states.append(inputs[0].detach().clone())

        def capture_final(_module, _inputs, output):
            final_state['hT'] = output.detach().clone()

        step_handle = self.conv.gcn_conv.register_forward_hook(
            capture_step_input
        )
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
            raise RuntimeError(
                'Failed to capture every official A-DGN Euler state.'
            )
        states.append(final_state['hT'])
        return torch.stack(states, dim=0)

    def set_functional_injection(self, basis=None, delta=None, beta=0.0):
        if not bool(getattr(args, 'enable_functional_dynamics', False)):
            return
        self.conv._fd_basis = None if basis is None else basis.detach()
        self.conv._fd_delta = None if delta is None else delta.detach()
        self.conv._fd_beta = float(beta)
        self.conv._fd_enabled = basis is not None and delta is not None

    @torch.no_grad()
    def encode_native_dynamics_states(self, data):
        if not bool(getattr(args, 'enable_functional_dynamics', False)):
            return self.encode_dynamics_states(data)
        was_enabled = self.conv._fd_enabled
        self.conv._fd_enabled = False
        try:
            return self.encode_dynamics_states(data)
        finally:
            self.conv._fd_enabled = was_enabled

    def functional_injection_ratio(self):
        return float(getattr(self.conv, '_fd_last_ratio', 0.0))

    model.encode_initial_state = MethodType(encode_initial_state, model)
    model.encode_dynamics_states = MethodType(encode_dynamics_states, model)
    model.set_functional_injection = MethodType(set_functional_injection, model)
    model.encode_native_dynamics_states = MethodType(
        encode_native_dynamics_states, model
    )
    model.functional_injection_ratio = MethodType(
        functional_injection_ratio, model
    )
    return model
