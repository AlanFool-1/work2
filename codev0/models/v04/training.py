"""Three optimizer-isolated local updates for Method V0.4."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.v04.lie_closure import direction_cosine
from models.v04.spectral_basis import graph_operator


BINARY_DATASETS = {'Minesweeper', 'Tolokers', 'Questions'}


def task_loss(logits, labels, mask, dataset):
    if int(mask.sum().item()) == 0:
        raise ValueError('V0.4 requires a non-empty supervised mask.')
    if dataset in BINARY_DATASETS:
        return F.binary_cross_entropy_with_logits(
            logits[mask].reshape(-1),
            labels[mask].float().reshape(-1),
        )
    return F.cross_entropy(logits[mask], labels[mask].long())


def task_accuracy(logits, labels, mask, dataset):
    if dataset in BINARY_DATASETS:
        prediction = (logits[mask].reshape(-1) > 0).long()
        target = labels[mask].reshape(-1).long()
    else:
        prediction = logits[mask].argmax(dim=-1)
        target = labels[mask].long()
    return (prediction == target).float().mean()


def lie_weight_for_round(args, round_id):
    round_number = int(round_id) + 1
    warmup = int(args.lie_warmup_rounds)
    ramp_end = int(args.lie_ramp_end_round)
    maximum = float(args.lie_weight_max)
    if round_number <= warmup or maximum == 0.0:
        return 0.0
    if round_number >= ramp_end:
        return maximum
    return maximum * (round_number - warmup) / (ramp_end - warmup)


def update_probe(model, batch, optimizer, dataset, max_grad_norm):
    """Phase A: update only the training-time linear task probe."""

    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        state = model.encoder(batch.x)
    logits = model.probe(state.detach())
    loss = task_loss(logits, batch.y, batch.train_mask, dataset)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite V0.4 probe loss.')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.probe_parameters(),
        float(max_grad_norm),
        error_if_nonfinite=True,
    )
    optimizer.step()
    accuracy = task_accuracy(
        logits.detach(), batch.y, batch.train_mask, dataset
    )
    optimizer.zero_grad(set_to_none=True)
    return {
        'probe_loss': loss.detach(),
        'probe_accuracy': accuracy.detach(),
        'probe_grad_norm': grad_norm.detach(),
    }


def _task_direction(model, batch, dataset):
    """Return detached negative probe-loss gradient with respect to ``H``."""

    requires_grad = [parameter.requires_grad for parameter in model.probe.parameters()]
    for parameter in model.probe.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        state = model.encoder(batch.x)
    state = state.detach().requires_grad_(True)
    logits = model.probe(state)
    loss = task_loss(logits, batch.y, batch.train_mask, dataset)
    direction = -torch.autograd.grad(loss, state, create_graph=False)[0]
    for parameter, original in zip(model.probe.parameters(), requires_grad):
        parameter.requires_grad_(original)
    return state.detach(), direction.detach()


def update_reference_field(
    model,
    batch,
    optimizer,
    dataset,
    max_grad_norm,
):
    """Phase B: update only the graph field against detached task descent."""

    optimizer.zero_grad(set_to_none=True)
    operator = graph_operator(batch)
    state, direction = _task_direction(model, batch, dataset)
    field, field_diagnostics = model.reference_field.forward_with_diagnostics(
        state, operator
    )
    cosine = direction_cosine(field, direction, batch.train_mask)
    loss = 1.0 - cosine
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite V0.4 field anchor loss.')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.field_parameters(),
        float(max_grad_norm),
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    direction_rms = torch.sqrt(
        direction[batch.train_mask].square().mean() + 1e-8
    )
    return {
        'field_anchor_loss': loss.detach(),
        'field_task_cosine': cosine.detach(),
        'field_rms': field_diagnostics['field_rms'].detach(),
        'field_raw_rms': field_diagnostics['field_raw_rms'].detach(),
        'field_scale': field_diagnostics['field_scale'].detach(),
        'task_direction_rms': direction_rms.detach(),
        'field_grad_norm': grad_norm.detach(),
    }


def update_main_model(
    model,
    batch,
    optimizer,
    dataset,
    lie_weight,
    max_grad_norm,
):
    """Phase C: task plus stop-gradient Lie direction closure."""

    optimizer.zero_grad(set_to_none=True)
    output = model.forward_with_aux(batch, compute_lie=lie_weight > 0.0)
    classification = task_loss(
        output.logits, batch.y, batch.train_mask, dataset
    )
    loss = classification + float(lie_weight) * output.lie_loss
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite V0.4 main loss.')
    loss.backward()
    for parameter in model.reference_field.parameters():
        if parameter.grad is not None and torch.count_nonzero(parameter.grad):
            raise AssertionError('Lie closure wrote gradients into the field.')
    for parameter in model.probe.parameters():
        if parameter.grad is not None and torch.count_nonzero(parameter.grad):
            raise AssertionError('Main phase wrote gradients into the probe.')
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.main_parameters(),
        float(max_grad_norm),
        error_if_nonfinite=True,
    )
    optimizer.step()
    accuracy = task_accuracy(
        output.logits.detach(), batch.y, batch.train_mask, dataset
    )
    optimizer.zero_grad(set_to_none=True)
    return {
        'train_loss': loss.detach(),
        'task_loss': classification.detach(),
        'train_accuracy': accuracy.detach(),
        'lie_weight': output.logits.new_tensor(float(lie_weight)),
        'lie_loss': output.lie_loss.detach(),
        'main_grad_norm': grad_norm.detach(),
        **{
            name: value.detach()
            for name, value in output.diagnostics.items()
        },
    }


def _zero_reference_diagnostics(like):
    zero = like.new_zeros(())
    return {
        'probe_loss': zero,
        'probe_accuracy': zero,
        'probe_grad_norm': zero,
        'field_anchor_loss': zero,
        'field_task_cosine': zero,
        'field_rms': zero,
        'field_raw_rms': zero,
        'field_scale': zero,
        'task_direction_rms': zero,
        'field_grad_norm': zero,
    }


def train_v04_step(
    model,
    batch,
    optimizers,
    args,
    round_id,
):
    """Run A/B/C in order and return detached scalar diagnostics."""

    model.train()
    if args.reference_mode == 'anchor':
        probe = update_probe(
            model,
            batch,
            optimizers['probe'],
            args.dataset,
            args.max_grad_norm,
        )
        field = update_reference_field(
            model,
            batch,
            optimizers['field'],
            args.dataset,
            args.max_grad_norm,
        )
    elif args.reference_mode == 'off':
        probe = _zero_reference_diagnostics(batch.x)
        field = {}
    else:
        raise ValueError(f'Unknown reference_mode: {args.reference_mode}')

    lie_weight = lie_weight_for_round(args, round_id)
    if args.reference_mode == 'off' and lie_weight:
        raise ValueError('Lie closure requires reference_mode=anchor.')
    main = update_main_model(
        model,
        batch,
        optimizers['main'],
        args.dataset,
        lie_weight,
        args.max_grad_norm,
    )
    stability = model.generator.stability_diagnostics()
    diagnostics = {**probe, **field, **main, **stability}
    for name, value in diagnostics.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f'Non-finite V0.4 diagnostic {name}.'
            )
    return diagnostics
