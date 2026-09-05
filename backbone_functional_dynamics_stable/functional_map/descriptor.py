"""Label-independent normalized trajectory descriptor."""

from __future__ import annotations

import torch


def normalized_descriptor(basis, snapshot_matrix, eps=1e-12):
    descriptor = basis.T @ snapshot_matrix
    return descriptor / (
        torch.linalg.matrix_norm(descriptor) + float(eps)
    )
