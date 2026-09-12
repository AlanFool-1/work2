"""Task-anchored neural-field-closed Graph-Koopman backbone."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torchdiffeq import odeint

from models.v04.encoder import NodeStateEncoder
from models.v04.generator import StableBernsteinGraphGenerator
from models.v04.lie_closure import GraphLieClosureLoss
from models.v04.observable import StatePreservingObservable
from models.v04.reference_field import NeuralGraphVectorField
from models.v04.spectral_basis import graph_operator


class _ODEFunction(nn.Module):
    def __init__(self, generator, operator, matrices):
        super().__init__()
        self.generator = generator
        self.operator = operator
        self.matrices = matrices
        self.nfe = 0

    def forward(self, _time, latent):
        self.nfe += 1
        return self.generator(latent, self.operator, self.matrices)


@dataclass
class V04Output:
    logits: torch.Tensor
    h0: torch.Tensor
    z0: torch.Tensor
    zt: torch.Tensor
    lie_loss: torch.Tensor
    diagnostics: dict


class NeuralFieldClosedGraphKoopman(nn.Module):
    """V0.4 model with one deployment-time Graph-Koopman trajectory."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int = 64,
        encoder_width: int = 128,
        observable_aux_dim: int = 32,
        observable_width: int = 64,
        field_hidden_dim: int = 64,
        field_scale_init: float = 0.1,
        field_scale_min: float = 0.001,
        generator_gamma: float = 1e-4,
        generator_low_damping_init: float = 1e-4,
        generator_high_damping_init: float = 2.0,
        integration_time: float = 1.0,
        step_size: float = 0.1,
        solver: str = 'rk4',
        classifier_width: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        if min(input_dim, output_dim, state_dim, classifier_width) <= 0:
            raise ValueError('V0.4 model dimensions must be positive.')
        if not math.isfinite(integration_time) or integration_time <= 0:
            raise ValueError('integration_time must be finite and positive.')
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError('solver_step_size must be finite and positive.')
        if solver not in {'rk4', 'dopri5'}:
            raise ValueError('V0.4 solver must be rk4 or dopri5.')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must lie in [0, 1).')
        steps = int(round(float(integration_time) / float(step_size)))
        if steps < 1 or not math.isclose(
            steps * float(step_size),
            float(integration_time),
            rel_tol=1e-7,
            abs_tol=1e-10,
        ):
            raise ValueError(
                'integration_time must be an integer multiple of step_size.'
            )

        self.state_dim = int(state_dim)
        self.integration_time = float(integration_time)
        self.step_size = float(step_size)
        self.num_steps = steps
        self.solver = solver
        self.encoder = NodeStateEncoder(
            input_dim, encoder_width, state_dim
        )
        self.observable = StatePreservingObservable(
            state_dim, observable_aux_dim, observable_width
        )
        koopman_dim = self.observable.output_dim
        self.generator = StableBernsteinGraphGenerator(
            koopman_dim,
            generator_gamma,
            generator_low_damping_init,
            generator_high_damping_init,
        )
        self.reference_field = NeuralGraphVectorField(
            state_dim,
            field_hidden_dim,
            field_scale_init,
            field_scale_min,
        )
        self.probe = nn.Linear(state_dim, output_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(koopman_dim),
            nn.Linear(koopman_dim, classifier_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_width, output_dim),
        )
        self.lie_closure = GraphLieClosureLoss()


    def named_main_parameters(self):
        modules = (
            ('encoder', self.encoder),
            ('observable', self.observable),
            ('generator', self.generator),
            ('classifier', self.classifier),
        )
        for prefix, module in modules:
            for name, parameter in module.named_parameters():
                yield f'{prefix}.{name}', parameter


    def named_field_parameters(self):
        for name, parameter in self.reference_field.named_parameters():
            yield f'reference_field.{name}', parameter

    def named_probe_parameters(self):
        for name, parameter in self.probe.named_parameters():
            yield f'probe.{name}', parameter

    def main_parameters(self):
        return [parameter for _, parameter in self.named_main_parameters()]


    def field_parameters(self):
        return [parameter for _, parameter in self.named_field_parameters()]

    def probe_parameters(self):
        return [parameter for _, parameter in self.named_probe_parameters()]

    def _evolve(self, initial, operator):
        matrices = self.generator.matrices()
        function = _ODEFunction(self.generator, operator, matrices)
        times = torch.tensor(
            [0.0, self.integration_time],
            device=initial.device,
            dtype=initial.dtype,
        )
        options = {'step_size': self.step_size} if self.solver == 'rk4' else None
        trajectory = odeint(
            function,
            initial,
            times,
            method=self.solver,
            options=options,
        )
        return trajectory[-1], function.nfe

    def encode_and_lift(self, data):
        state = self.encoder(data.x)
        latent = self.observable(state)
        if not torch.equal(latent[..., :self.state_dim], state):
            raise AssertionError('V0.4 observable lost its identity block.')
        return state, latent

    def forward(self, data):
        """Deployment path: no probe, reference field, labels, or JVP."""

        operator = graph_operator(data)
        _, initial = self.encode_and_lift(data)
        terminal, _ = self._evolve(initial, operator)
        return self.classifier(terminal)

    def forward_with_aux(self, data, compute_lie: bool):
        operator = graph_operator(data)
        state, initial = self.encode_and_lift(data)
        if compute_lie:
            lie = self.lie_closure(
                state,
                data.train_mask,
                operator,
                self.observable,
                self.reference_field,
                self.generator,
            )
            lie_loss = lie.loss
            lie_diagnostics = {
                'lie_cosine': lie.cosine,
                'lie_target_rms': lie.target_rms,
                'koopman_rhs_rms': lie.rhs_rms,
                'lie_speed_ratio': lie.speed_ratio,
            }
        else:
            zero = initial.new_zeros(())
            lie_loss = zero
            lie_diagnostics = {
                'lie_cosine': zero,
                'lie_target_rms': zero,
                'koopman_rhs_rms': zero,
                'lie_speed_ratio': zero,
            }
        rhs_initial = self.generator(initial, operator)
        terminal, nfe = self._evolve(initial, operator)
        z0_norm = initial.norm()
        zt_norm = terminal.norm()
        diagnostics = {
            **lie_diagnostics,
            'z0_norm': z0_norm,
            'zt_norm': zt_norm,
            'z_norm_ratio': zt_norm / z0_norm.clamp_min(1e-8),
            'rhs0_norm': rhs_initial.norm(),
            'nfe': initial.new_tensor(float(nfe)),
        }
        return V04Output(
            logits=self.classifier(terminal),
            h0=state,
            z0=initial,
            zt=terminal,
            lie_loss=lie_loss,
            diagnostics=diagnostics,
        )


def build_v04_model(args):
    if int(args.spectral_order) != 1:
        raise ValueError('V0.4 Gate 0-4 require spectral_order == 1.')
    return NeuralFieldClosedGraphKoopman(
        input_dim=args.n_feat,
        output_dim=args.n_clss,
        state_dim=args.state_dim,
        encoder_width=args.encoder_width,
        observable_aux_dim=args.observable_aux_dim,
        observable_width=args.observable_width,
        field_hidden_dim=args.field_hidden_dim,
        field_scale_init=args.field_scale_init,
        field_scale_min=args.field_scale_min,
        generator_gamma=args.generator_gamma,
        generator_low_damping_init=args.generator_low_damping_init,
        generator_high_damping_init=args.generator_high_damping_init,
        integration_time=args.integration_time,
        step_size=args.solver_step_size,
        solver=args.solver,
        classifier_width=args.classifier_width,
        dropout=args.dropout,
    )
