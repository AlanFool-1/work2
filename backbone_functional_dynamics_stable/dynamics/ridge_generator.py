"""Scale-adaptive ridge fit of a reduced trajectory generator."""

from __future__ import annotations

import torch


def fit_ridge_generator(trajectory, basis, step_size, lambda0, eps=1e-12):
    z = torch.einsum('nr,tnd->trd', basis, trajectory)
    x = z[:-1].permute(1, 0, 2).reshape(basis.shape[1], -1)
    y = ((z[1:] - z[:-1]) / float(step_size)).permute(1, 0, 2).reshape(
        basis.shape[1], -1
    )
    xxt = x @ x.T
    ridge_scale = torch.trace(xxt) / int(basis.shape[1])
    regularizer = float(lambda0) * ridge_scale
    system = xxt + regularizer * torch.eye(
        basis.shape[1], device=x.device, dtype=x.dtype
    )
    rhs = y @ x.T
    generator = torch.linalg.solve(system, rhs.T).T
    residual = y - generator @ x
    fit_error = torch.linalg.matrix_norm(residual) / (
        torch.linalg.matrix_norm(y) + float(eps)
    )
    eigenvalues = torch.linalg.eigvalsh(system)
    cond_proxy = eigenvalues[-1] / torch.clamp(eigenvalues[0], min=float(eps))
    metrics = {
        'dyn_fit_error': float(fit_error.item()),
        'ridge_scale': float(ridge_scale.item()),
        'cond_proxy': float(cond_proxy.item()),
        'A_norm': float(torch.linalg.matrix_norm(generator).item()),
    }
    return generator, metrics
