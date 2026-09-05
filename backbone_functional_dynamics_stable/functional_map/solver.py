"""Conditioned orthogonal and general Functional Map solvers."""

from __future__ import annotations

import torch


def _polar(matrix):
    left, _, right = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right


def descriptor_procrustes(local_descriptor, canonical_descriptor):
    return _polar(canonical_descriptor @ local_descriptor.T)


def procrustes_distance(local_descriptor, canonical_descriptor):
    """Return the exact orthogonal-Procrustes distance and affinity."""
    cross = canonical_descriptor @ local_descriptor.T
    affinity = torch.linalg.svdvals(cross).sum()
    transform = _polar(cross)
    distance = torch.linalg.matrix_norm(
        transform @ local_descriptor - canonical_descriptor
    )
    return distance, affinity


def solve_orthogonal_fm(
    local_generator,
    local_descriptor,
    canonical_generator,
    canonical_descriptor,
    steps=10,
    learning_rate=0.1,
    lambda_desc=1.0,
    lambda_dyn=1.0,
    warm_start=None,
    normalize_terms=True,
    line_search_steps=8,
):
    procrustes = descriptor_procrustes(local_descriptor, canonical_descriptor)
    desc_scale = (
        torch.linalg.matrix_norm(local_descriptor) ** 2
        + torch.linalg.matrix_norm(canonical_descriptor) ** 2
    ) if normalize_terms else local_descriptor.new_tensor(1.0)
    dyn_scale = (
        torch.linalg.matrix_norm(local_generator) ** 2
        + torch.linalg.matrix_norm(canonical_generator) ** 2
    ) if normalize_terms else local_generator.new_tensor(1.0)
    desc_scale = torch.clamp(desc_scale, min=1e-12)
    dyn_scale = torch.clamp(dyn_scale, min=1e-12)

    def objective(candidate):
        desc = candidate @ local_descriptor - canonical_descriptor
        dyn = canonical_generator @ candidate - candidate @ local_generator
        return (
            float(lambda_desc) * torch.sum(desc * desc) / desc_scale
            + float(lambda_dyn) * torch.sum(dyn * dyn) / dyn_scale
        )

    if warm_start is None:
        c_map = procrustes
    else:
        warm = _polar(warm_start)
        c_map = warm if objective(warm) <= objective(procrustes) else procrustes
    initial_desc = torch.linalg.matrix_norm(
        c_map @ local_descriptor - canonical_descriptor
    )
    initial_dyn = torch.linalg.matrix_norm(
        canonical_generator @ c_map - c_map @ local_generator
    )
    initial_objective = objective(c_map)
    accepted_steps = 0
    for _ in range(int(steps)):
        residual_desc = c_map @ local_descriptor - canonical_descriptor
        residual_dyn = canonical_generator @ c_map - c_map @ local_generator
        gradient = (
            2.0 * float(lambda_desc) * residual_desc @ local_descriptor.T / desc_scale
            + 2.0 * float(lambda_dyn) * (
                canonical_generator.T @ residual_dyn
                - residual_dyn @ local_generator.T
            ) / dyn_scale
        )
        symmetric = 0.5 * (
            c_map.T @ gradient + gradient.T @ c_map
        )
        tangent = gradient - c_map @ symmetric
        current_objective = objective(c_map)
        step_size = float(learning_rate)
        accepted = False
        for _ in range(int(line_search_steps)):
            candidate = _polar(c_map - step_size * tangent)
            if objective(candidate) <= current_objective + 1e-12:
                c_map = candidate
                accepted = True
                accepted_steps += 1
                break
            step_size *= 0.5
        if not accepted:
            break
    residual_desc = c_map @ local_descriptor - canonical_descriptor
    residual_dyn = canonical_generator @ c_map - c_map @ local_generator
    identity = torch.eye(c_map.shape[0], device=c_map.device, dtype=c_map.dtype)
    metrics = {
        'fm_desc_error': float(torch.linalg.matrix_norm(residual_desc).item()),
        'fm_dyn_error': float(torch.linalg.matrix_norm(residual_dyn).item()),
        'fm_orth_error': float(
            torch.linalg.matrix_norm(c_map.T @ c_map - identity).item()
        ),
        'fm_initial_desc_error': float(initial_desc.item()),
        'fm_initial_dyn_error': float(initial_dyn.item()),
        'fm_initial_objective': float(initial_objective.item()),
        'fm_final_objective': float(objective(c_map).item()),
        'fm_accepted_steps': float(accepted_steps),
    }
    return c_map, metrics


def solve_regularized_fm(
    local_generator,
    local_descriptor,
    canonical_generator,
    canonical_descriptor,
    steps=20,
    learning_rate=0.1,
    lambda_desc=1.0,
    lambda_dyn=1.0,
    ridge=1e-3,
    orthogonal_regularization=1e-2,
    max_condition=20.0,
    warm_start=None,
):
    """Solve a conditioned, non-orthogonal functional map with line search."""
    rank = local_descriptor.shape[0]
    identity = torch.eye(rank, device=local_descriptor.device, dtype=local_descriptor.dtype)
    gram = local_descriptor @ local_descriptor.T
    scale = torch.trace(gram) / rank
    descriptor_init = torch.linalg.solve(
        gram + float(ridge) * scale * identity,
        (canonical_descriptor @ local_descriptor.T).T,
    ).T

    desc_scale = torch.clamp(
        torch.linalg.matrix_norm(local_descriptor) ** 2
        + torch.linalg.matrix_norm(canonical_descriptor) ** 2, min=1e-12,
    )
    dyn_scale = torch.clamp(
        torch.linalg.matrix_norm(local_generator) ** 2
        + torch.linalg.matrix_norm(canonical_generator) ** 2, min=1e-12,
    )

    def project(candidate):
        left, singular, right = torch.linalg.svd(candidate, full_matrices=False)
        upper = float(max_condition) ** 0.5
        lower = 1.0 / upper
        return (left * torch.clamp(singular, min=lower, max=upper).unsqueeze(0)) @ right

    def objective(candidate):
        desc = candidate @ local_descriptor - canonical_descriptor
        dyn = canonical_generator @ candidate - candidate @ local_generator
        orth = candidate.T @ candidate - identity
        return (
            float(lambda_desc) * torch.sum(desc * desc) / desc_scale
            + float(lambda_dyn) * torch.sum(dyn * dyn) / dyn_scale
            + float(orthogonal_regularization) * torch.sum(orth * orth) / rank
        )

    descriptor_init = project(descriptor_init)
    if warm_start is None:
        c_map = descriptor_init
    else:
        warm = project(warm_start)
        c_map = warm if objective(warm) <= objective(descriptor_init) else descriptor_init
    initial_desc = torch.linalg.matrix_norm(c_map @ local_descriptor - canonical_descriptor)
    initial_dyn = torch.linalg.matrix_norm(canonical_generator @ c_map - c_map @ local_generator)
    initial_objective = objective(c_map)
    accepted_steps = 0
    for _ in range(int(steps)):
        residual_desc = c_map @ local_descriptor - canonical_descriptor
        residual_dyn = canonical_generator @ c_map - c_map @ local_generator
        orth = c_map.T @ c_map - identity
        gradient = (
            2.0 * float(lambda_desc) * residual_desc @ local_descriptor.T / desc_scale
            + 2.0 * float(lambda_dyn) * (
                canonical_generator.T @ residual_dyn - residual_dyn @ local_generator.T
            ) / dyn_scale
            + 4.0 * float(orthogonal_regularization) * c_map @ orth / rank
        )
        current = objective(c_map)
        step_size = float(learning_rate)
        for attempt in range(10):
            candidate = project(c_map - step_size * gradient)
            if objective(candidate) <= current + 1e-12:
                c_map = candidate
                accepted_steps += 1
                break
            step_size *= 0.5
        else:
            break
    residual_desc = c_map @ local_descriptor - canonical_descriptor
    residual_dyn = canonical_generator @ c_map - c_map @ local_generator
    singular = torch.linalg.svdvals(c_map)
    metrics = {
        'fm_desc_error': float(torch.linalg.matrix_norm(residual_desc).item()),
        'fm_dyn_error': float(torch.linalg.matrix_norm(residual_dyn).item()),
        'fm_orth_error': float(torch.linalg.matrix_norm(c_map.T @ c_map - identity).item()),
        'fm_initial_desc_error': float(initial_desc.item()),
        'fm_initial_dyn_error': float(initial_dyn.item()),
        'fm_initial_objective': float(initial_objective.item()),
        'fm_final_objective': float(objective(c_map).item()),
        'fm_accepted_steps': float(accepted_steps),
        'fm_condition': float((singular.max() / singular.min()).item()),
    }
    return c_map, metrics
