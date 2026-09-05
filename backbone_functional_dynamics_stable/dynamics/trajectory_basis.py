"""Trajectory-induced functional basis with deterministic QR gauge."""

from __future__ import annotations

import torch


def shared_probe(hidden_dim, proj_dim, seed, device, dtype):
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    matrix = torch.randn(int(hidden_dim), int(proj_dim), generator=generator)
    probe, _ = torch.linalg.qr(matrix, mode='reduced')
    return probe.to(device=device, dtype=dtype)


def snapshot_indices(step_count):
    last = int(step_count) - 1
    return sorted(set([0, last // 4, last // 2, last]))


def build_snapshot_matrix(trajectory, probe, indices=None, eps=1e-12):
    if trajectory.ndim != 3:
        raise ValueError('trajectory must be [steps, nodes, hidden_dim].')
    indices = snapshot_indices(trajectory.shape[0]) if indices is None else indices
    blocks = []
    for index in indices:
        block = trajectory[int(index)] @ probe
        norms = torch.linalg.vector_norm(block, dim=0, keepdim=True)
        blocks.append(block / torch.clamp(norms, min=float(eps)))
    return torch.cat(blocks, dim=1), tuple(int(i) for i in indices)


def build_trajectory_basis(trajectory, probe, rank, indices=None, eps=1e-12):
    snapshot, used_indices = build_snapshot_matrix(
        trajectory, probe, indices=indices, eps=eps
    )
    rank = int(rank)
    if rank > snapshot.shape[1] or rank > snapshot.shape[0]:
        raise ValueError('rank must not exceed nodes or snapshots*probe_dim.')
    q_full, r_qr = torch.linalg.qr(snapshot, mode='reduced')
    signs = torch.sign(torch.diagonal(r_qr))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q_full = q_full * signs.unsqueeze(0)
    return q_full[:, :rank], snapshot, used_indices
