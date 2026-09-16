"""Small-operator Functional Maps for server-side Koopman alignment.

The matrices handled here live in the ResKoop observable space (81 by 81 for
the default FedDGN configuration).  No graph Laplacian, adjacency matrix, or
node-feature matrix is factorized.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


FUNCTIONAL_MAP_METHOD = 'spectral_init_unitary_projected_gradient'


def _complex_matrix(value, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f'{name} must be a square matrix.')
    if not np.all(np.isfinite(result.real) & np.isfinite(result.imag)):
        raise ValueError(f'{name} contains non-finite values.')
    return result.copy()


def _project_unitary(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest unitary matrix through a small polar factor."""

    left, _, right_h = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_h


def _normalized_residual(
    functional_map: np.ndarray,
    local_operator: np.ndarray,
    reference_operator: np.ndarray,
) -> float:
    residual = (
        functional_map @ local_operator
        - reference_operator @ functional_map
    )
    denominator = np.sqrt(
        np.linalg.norm(local_operator, 'fro') ** 2
        + np.linalg.norm(reference_operator, 'fro') ** 2
    )
    return float(
        np.linalg.norm(residual, 'fro')
        / max(denominator, np.finfo(np.float64).eps)
    )


def _orthogonality_error(functional_map: np.ndarray) -> float:
    dimension = functional_map.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    return float(
        np.linalg.norm(
            functional_map.conj().T @ functional_map - identity,
            'fro',
        )
        / np.sqrt(dimension)
    )


def _spectral_initialization(
    local_operator: np.ndarray,
    reference_operator: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Match operator spectra, then project the induced map to unitary.

    If ``K_m V_m = V_m Lambda_m`` and ``K_s V_s = V_s Lambda_s``, the
    unconstrained correspondence is ``V_s P pinv(V_m)``.  Its polar factor is
    a stable initialization satisfying the document's orthogonality
    constraint exactly (up to numerical precision).
    """

    local_values, local_vectors = np.linalg.eig(local_operator)
    reference_values, reference_vectors = np.linalg.eig(reference_operator)
    cost = np.abs(
        reference_values[:, None] - local_values[None, :]
    )
    reference_index, local_index = linear_sum_assignment(cost)
    permutation = np.zeros(cost.shape, dtype=np.complex128)
    permutation[reference_index, local_index] = 1.0
    unconstrained = (
        reference_vectors
        @ permutation
        @ np.linalg.pinv(local_vectors)
    )
    return _project_unitary(unconstrained), float(cost[
        reference_index, local_index
    ].mean())


def solve_functional_map(
    local_operator,
    reference_operator,
    *,
    refinement_steps: int = 8,
    tolerance: float = 1e-7,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Solve ``min_C ||C K_m - K_s C||`` under ``C^H C = I``.

    A spectrum-matched polar initialization avoids the zero-map degeneracy.
    A few deterministic projected-gradient steps then refine the dynamical
    consistency while every iterate remains unitary.  The returned diagnostics
    expose both the identity baseline and final residual, so a failed alignment
    cannot silently look successful.
    """

    if int(refinement_steps) < 0:
        raise ValueError('Functional-map refinement steps must be non-negative.')
    if float(tolerance) <= 0.0:
        raise ValueError('Functional-map tolerance must be positive.')
    local = _complex_matrix(local_operator, name='local Koopman operator')
    reference = _complex_matrix(
        reference_operator, name='reference Koopman operator'
    )
    if local.shape != reference.shape:
        raise ValueError('Local and reference Koopman operators must agree in size.')

    dimension = local.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    identity_residual = _normalized_residual(identity, local, reference)
    spectral_map, spectral_assignment_cost = _spectral_initialization(
        local, reference
    )
    spectral_residual = _normalized_residual(
        spectral_map, local, reference
    )
    if identity_residual < spectral_residual:
        current = identity
        initialization = 'identity'
        current_residual = identity_residual
    else:
        current = spectral_map
        initialization = 'spectral_polar'
        current_residual = spectral_residual

    operator_scale = (
        np.linalg.norm(local, 2) + np.linalg.norm(reference, 2)
    ) ** 2
    base_step = 0.5 / max(operator_scale, np.finfo(np.float64).eps)
    step_size = base_step
    accepted_steps = 0
    converged = False
    for _ in range(int(refinement_steps)):
        residual = current @ local - reference @ current
        gradient = 2.0 * (
            residual @ local.conj().T
            - reference.conj().T @ residual
        )
        # Remove the normal component to obtain a tangent direction on U(n).
        cotangent = current.conj().T @ gradient
        symmetric = 0.5 * (cotangent + cotangent.conj().T)
        tangent = gradient - current @ symmetric
        tangent_norm = float(np.linalg.norm(tangent, 'fro'))
        if tangent_norm <= float(tolerance) * np.sqrt(dimension):
            converged = True
            break

        previous = current_residual
        accepted = False
        trial_step = step_size
        for _ in range(8):
            trial = _project_unitary(current - trial_step * tangent)
            trial_residual = _normalized_residual(trial, local, reference)
            if trial_residual <= current_residual + 1e-14:
                current = trial
                current_residual = trial_residual
                step_size = min(base_step, trial_step * 1.2)
                accepted_steps += 1
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            break
        if previous - current_residual <= float(tolerance) * max(previous, 1.0):
            converged = True
            break

    diagnostics = {
        'method': FUNCTIONAL_MAP_METHOD,
        'dimension': int(dimension),
        'initialization': initialization,
        'identity_residual': float(identity_residual),
        'spectral_initial_residual': float(spectral_residual),
        'spectral_assignment_cost': float(spectral_assignment_cost),
        'intertwining_residual': float(current_residual),
        'orthogonality_error': _orthogonality_error(current),
        'accepted_refinement_steps': int(accepted_steps),
        'converged': bool(converged),
    }
    return current.astype(np.complex64), diagnostics


def identity_functional_map(operator) -> Tuple[np.ndarray, Dict[str, object]]:
    """Construct the exact reference-client map and matching diagnostics."""

    value = _complex_matrix(operator, name='reference Koopman operator')
    identity = np.eye(value.shape[0], dtype=np.complex64)
    return identity, {
        'method': FUNCTIONAL_MAP_METHOD,
        'dimension': int(value.shape[0]),
        'initialization': 'reference_identity',
        'identity_residual': 0.0,
        'spectral_initial_residual': 0.0,
        'spectral_assignment_cost': 0.0,
        'intertwining_residual': 0.0,
        'orthogonality_error': 0.0,
        'accepted_refinement_steps': 0,
        'converged': True,
    }


__all__ = [
    'FUNCTIONAL_MAP_METHOD',
    'identity_functional_map',
    'solve_functional_map',
]
