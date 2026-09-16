"""FedDGN local backbone: short A-DGN dynamics with an external-force port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from models.feddgn_local.dynamics import ForcedAntiSymmetricFlow, ForceLike


@dataclass
class FedDGNOutput:
    logits: torch.Tensor
    h0: torch.Tensor
    ht: torch.Tensor
    trajectory: tuple
    diagnostics: dict


class FedDGNBackbone(nn.Module):
    """A compact local dynamical GNN.

    Design intent:
      1) keep only the A-DGN-style intrinsic state dynamics;
      2) use a short trajectory for local representation learning rather than
         targeting very deep/long-range propagation;
      3) reserve an explicit external-force port for the later federated stage;
      4) expose the trajectory and instantaneous vector field for future
         Koopman/ResKoopNet compression without changing the classifier path.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 16,
        step_size: float = 0.1,
        gamma: float = 0.1,
        activation: str = 'tanh',
        input_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
        head: str = 'linear',
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(float(input_dropout))
        self.flow = ForcedAntiSymmetricFlow(
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            step_size=step_size,
            gamma=gamma,
            activation=activation,
        )

        if str(head).lower() == 'mlp':
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(classifier_dropout)),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(float(classifier_dropout)),
                nn.Linear(hidden_dim, output_dim),
            )

    def encode(self, data) -> torch.Tensor:
        x = data.x.float()
        return self.encoder(self.input_dropout(x))

    def compute_vector_field(
        self,
        state: torch.Tensor,
        data,
        external_force: Optional[ForceLike] = None,
        force_scale: float = 1.0,
        step_id: int = 0,
    ):
        """Evaluate dynamics at a fixed state without advancing the rollout."""

        return self.flow.vector_field(
            state,
            data,
            external_force=external_force,
            force_scale=force_scale,
            step_id=step_id,
        )

    def intrinsic_vector_field(self, state: torch.Tensor, data) -> torch.Tensor:
        return self.flow.intrinsic_vector_field(state, data)

    def rollout(
        self,
        data,
        external_force: Optional[ForceLike] = None,
        force_scale: float = 1.0,
        initial_state: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ):
        h0 = self.encode(data) if initial_state is None else initial_state
        ht, diagnostics, trajectory = self.flow(
            h0,
            data,
            external_force=external_force,
            force_scale=force_scale,
            return_trajectory=return_trajectory,
        )
        return h0, ht, diagnostics, trajectory

    def forward(
        self,
        data,
        external_force: Optional[ForceLike] = None,
        force_scale: float = 1.0,
        is_proxy: bool = False,
    ) -> torch.Tensor:
        h0, ht, _, _ = self.rollout(
            data,
            external_force=external_force,
            force_scale=force_scale,
            return_trajectory=False,
        )
        if is_proxy:
            return ht
        return self.classifier(ht)

    def forward_with_aux(
        self,
        data,
        external_force: Optional[ForceLike] = None,
        force_scale: float = 1.0,
        initial_state: Optional[torch.Tensor] = None,
    ) -> FedDGNOutput:
        h0, ht, diagnostics, trajectory = self.rollout(
            data,
            external_force=external_force,
            force_scale=force_scale,
            initial_state=initial_state,
            return_trajectory=True,
        )
        if diagnostics is None:
            zero = ht.new_zeros(())
            diag = {
                'intrinsic_rms': zero,
                'force_rms': zero,
                'force_intrinsic_ratio': zero,
                'state_rms': torch.sqrt(ht.square().mean() + 1e-12),
            }
        else:
            diag = {
                'intrinsic_rms': diagnostics.intrinsic_rms,
                'force_rms': diagnostics.force_rms,
                'force_intrinsic_ratio': diagnostics.force_intrinsic_ratio,
                'state_rms': diagnostics.state_rms,
            }
        return FedDGNOutput(
            logits=self.classifier(ht),
            h0=h0,
            ht=ht,
            trajectory=tuple(trajectory),
            diagnostics=diag,
        )

    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _arg(args, name, default, fallback=None):
    if hasattr(args, name):
        return getattr(args, name)
    if fallback is not None and hasattr(args, fallback):
        return getattr(args, fallback)
    return default


def build_feddgn_model(args) -> FedDGNBackbone:
    """Build with safe defaults so no parser changes are required initially."""

    return FedDGNBackbone(
        input_dim=int(args.n_feat),
        output_dim=int(args.n_clss),
        hidden_dim=int(_arg(args, 'feddgn_hidden_dim', 64, 'n_dims')),
        num_steps=int(_arg(args, 'feddgn_steps', 16)),
        step_size=float(_arg(args, 'feddgn_step_size', 0.1)),
        gamma=float(_arg(args, 'feddgn_gamma', 0.1)),
        activation=str(_arg(args, 'feddgn_activation', 'tanh')),
        input_dropout=float(_arg(args, 'feddgn_input_dropout', 0.0)),
        classifier_dropout=float(
            _arg(args, 'feddgn_classifier_dropout', 0.0)
        ),
        head=str(_arg(args, 'feddgn_head', 'linear')),
    )


__all__ = ['FedDGNBackbone', 'FedDGNOutput', 'build_feddgn_model']
