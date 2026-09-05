"""Minimal dissipative projection and low-rank functional injection."""

from __future__ import annotations

import torch


def minimal_dissipative_projection(delta, spectral_limit, eps=1e-12):
    skew = 0.5 * (delta - delta.T)
    symmetric = 0.5 * (delta + delta.T)
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    clipped = torch.minimum(eigenvalues, torch.zeros_like(eigenvalues))
    dissipative = skew + (vectors * clipped.unsqueeze(0)) @ vectors.T
    spectral_norm = torch.linalg.matrix_norm(dissipative, ord=2)
    scale = torch.clamp(
        float(spectral_limit) / (spectral_norm + float(eps)), max=1.0
    )
    safe = dissipative * scale
    max_symmetric_eigenvalue = torch.linalg.eigvalsh(
        0.5 * (safe + safe.T)
    )[-1]
    return safe, {
        'delta_raw_norm': float(torch.linalg.matrix_norm(delta).item()),
        'delta_safe_norm': float(torch.linalg.matrix_norm(safe).item()),
        'delta_safe_spectral_norm': float(
            torch.linalg.matrix_norm(safe, ord=2).item()
        ),
        'delta_safe_max_sym_eig': float(max_symmetric_eigenvalue.item()),
        'delta_dissipative_spectral_norm': float(spectral_norm.item()),
        'delta_clip_scale': float(scale.item()),
    }


def low_rank_injection(hidden, basis, delta_safe, beta):
    reduced = basis.T @ hidden
    return float(beta) * basis @ (delta_safe @ reduced)
