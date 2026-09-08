"""Shared local/federated V0.2 objective and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    native: float = 1.0
    reconstruction: float = 1.0
    prediction: float = 1.0
    linear: float = 0.1

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError('V0.2 loss weights must be finite and non-negative.')

    @property
    def auxiliary(self):
        """Whether additional free-running training windows are needed."""
        return self.prediction > 0 or self.linear > 0


def task_loss(logits, labels, mask):
    if not mask.any():
        raise ValueError('Supervised training requires a non-empty training mask.')
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits[mask].reshape(-1), labels[mask].float().reshape(-1),
        )
    return F.cross_entropy(logits[mask], labels[mask].long())


def relative_mse(prediction, target):
    """Normalize by detached reference energy, never prediction energy."""
    target = target.detach()
    return (prediction - target).square().mean() / target.square().mean().clamp_min(1e-6)


def objective(model, output, labels, train_mask, weights: LossWeights):
    values = {
        'task_loss': task_loss(output.logits, labels, train_mask),
        'native_task_loss': task_loss(output.native_logits, labels, train_mask),
        'reconstruction_loss': relative_mse(output.reconstructions, output.native_states),
    }
    prediction_terms, linear_terms = [], []
    for horizon, predicted in output.predictions.items():
        prediction_terms.append(relative_mse(
            model.decoder(predicted), output.native_states[horizon:],
        ))
        linear_terms.append(relative_mse(predicted, output.encoded_targets[horizon:]))
    zero = values['task_loss'].new_zeros(())
    values['prediction_loss'] = torch.stack(prediction_terms).mean() if prediction_terms else zero
    values['linearity_loss'] = torch.stack(linear_terms).mean() if linear_terms else zero
    loss = (
        values['task_loss'] + weights.native * values['native_task_loss']
        + weights.reconstruction * values['reconstruction_loss']
        + weights.prediction * values['prediction_loss']
        + weights.linear * values['linearity_loss']
    )
    return loss, values


def checked_step(loss, model, optimizer, max_grad_norm=5.0):
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite V0.2 loss.')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True,
    )
    optimizer.step()
    return float(norm.detach())


@torch.no_grad()
def trajectory_diagnostics(model, output):
    z = output.latent_trajectory
    norms = z.flatten(1).norm(dim=1)
    centered = z.flatten(0, 1)
    singular = torch.linalg.svdvals(centered - centered.mean(0, keepdim=True))
    energy = singular.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    if energy.sum() < 1e-12:
        effective_rank = energy.new_zeros(())
    target = output.native_states.detach()
    values = {
        'latent_norm_ratio': float(norms[-1] / norms[0].clamp_min(1e-12)),
        'max_latent_growth_ratio': float((norms[1:] / norms[:-1].clamp_min(1e-12)).max()),
        'latent_effective_rank': float(effective_rank),
        'reconstruction_nmse': float(relative_mse(output.reconstructions, target)),
        'latent_norms': norms.cpu().tolist(),
        'native_relative_change': float(
            (target[-1] - target[0]).norm() / target[0].norm().clamp_min(1e-12)
        ),
    }
    for horizon in model.horizons:
        error = (output.decoded_trajectory[horizon] - target[horizon]).square().mean()
        increment = (target[horizon] - target[0]).square().mean().clamp_min(1e-6)
        values[f'prediction_nmse_h{horizon}'] = float(relative_mse(
            output.decoded_trajectory[horizon], target[horizon],
        ))
        values[f'increment_nrmse_h{horizon}'] = float((error / increment).sqrt())
        values[f'linearity_nmse_h{horizon}'] = float(relative_mse(
            z[horizon], output.encoded_targets[horizon],
        ))
    return values
