"""Stop-gradient Lie/JVP direction closure for Method V0.4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def direction_cosine(first, second, mask, eps=1e-8):
    if int(mask.sum().item()) == 0:
        raise ValueError('Direction cosine requires a non-empty mask.')
    first = first[mask]
    second = second[mask]
    first = first / (first.norm(dim=-1, keepdim=True) + eps)
    second = second / (second.norm(dim=-1, keepdim=True) + eps)
    return (first * second).sum(dim=-1).mean()


@dataclass
class LieClosureOutput:
    loss: torch.Tensor
    cosine: torch.Tensor
    target_rms: torch.Tensor
    rhs_rms: torch.Tensor
    speed_ratio: torch.Tensor


class GraphLieClosureLoss(nn.Module):
    """Align an anchored field's observable derivative with Koopman RHS."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        state,
        mask,
        operator,
        observable,
        reference_field,
        generator,
    ):
        with torch.no_grad():
            field_reference = reference_field(state.detach(), operator)
        latent, target = torch.autograd.functional.jvp(
            observable,
            state,
            field_reference.detach(),
            create_graph=True,
            strict=False,
        )
        rhs = generator(latent, operator)
        cosine = direction_cosine(target, rhs, mask, self.eps)
        target_rms = torch.sqrt(target[mask].square().mean() + self.eps)
        rhs_rms = torch.sqrt(rhs[mask].square().mean() + self.eps)
        speed_ratio = rhs_rms / (target_rms + self.eps)
        return LieClosureOutput(
            loss=1.0 - cosine,
            cosine=cosine,
            target_rms=target_rms,
            rhs_rms=rhs_rms,
            speed_ratio=speed_ratio,
        )
