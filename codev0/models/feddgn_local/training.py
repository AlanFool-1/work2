"""Task-only training for the local FedDGN backbone."""

from __future__ import annotations

import torch
import torch.nn.functional as F


BINARY_DATASETS = {'Minesweeper', 'Tolokers', 'Questions'}


def task_loss(logits, labels, mask, dataset, label_smoothing=0.0):
    if int(mask.sum().item()) == 0:
        raise ValueError('FedDGN requires a non-empty supervised mask.')
    if dataset in BINARY_DATASETS:
        return F.binary_cross_entropy_with_logits(
            logits[mask].reshape(-1),
            labels[mask].float().reshape(-1),
        )
    return F.cross_entropy(
        logits[mask],
        labels[mask].long(),
        label_smoothing=float(label_smoothing),
    )


def task_accuracy(logits, labels, mask, dataset):
    if dataset in BINARY_DATASETS:
        prediction = (logits[mask].reshape(-1) > 0).long()
        target = labels[mask].reshape(-1).long()
    else:
        prediction = logits[mask].argmax(dim=-1)
        target = labels[mask].long()
    return (prediction == target).float().mean()


def train_feddgn_step(model, batch, optimizer, args):
    """One task-only update; the reserved external-force port is kept at zero."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_with_aux(
        batch,
        external_force=None,
        force_scale=0.0,
    )
    loss = task_loss(
        output.logits,
        batch.y,
        batch.train_mask,
        args.dataset,
        label_smoothing=float(getattr(args, 'feddgn_label_smoothing', 0.0)),
    )
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite FedDGN task loss.')
    loss.backward()
    max_grad_norm = float(getattr(args, 'feddgn_max_grad_norm', 0.0))
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm if max_grad_norm > 0.0 else float('inf'),
        error_if_nonfinite=True,
    )
    optimizer.step()

    accuracy = task_accuracy(
        output.logits.detach(), batch.y, batch.train_mask, args.dataset
    )
    diagnostics = {
        'train_loss': loss.detach(),
        'task_loss': loss.detach(),
        'train_accuracy': accuracy.detach(),
        'main_grad_norm': grad_norm.detach(),
        **{name: value.detach() for name, value in output.diagnostics.items()},
    }
    for name, value in diagnostics.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite FedDGN diagnostic {name}.')
    return diagnostics


__all__ = [
    'BINARY_DATASETS',
    'task_accuracy',
    'task_loss',
    'train_feddgn_step',
]
