"""Private need estimation and closed-form personalized Koopman fitting.

All observable snapshots stay on the client.  This module is task-passive:
it returns K-star and diagnostics, not an A-DGN force or gradient.
"""

from __future__ import annotations

import numpy as np

from models.feddgn0916.candidate_pipeline import CANDIDATE_PROTOCOL


def _finite(value, *, name, ndim=None):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.complex128)
    if (
        result.size == 0
        or (ndim is not None and result.ndim != ndim)
        or not np.all(np.isfinite(result.real) & np.isfinite(result.imag))
    ):
        raise ValueError(f'{name} must be finite with the expected dimensions.')
    return result.copy()


def _normalize_block(u, t):
    """QR-normalize U and change T into the corresponding block basis."""

    if u.ndim != 2 or t.shape != (u.shape[1], u.shape[1]):
        raise ValueError('Candidate U and T block dimensions disagree.')
    q, r = np.linalg.qr(u, mode='reduced')
    singular_values = np.linalg.svd(r, compute_uv=False)
    if singular_values[-1] <= 1e-10 * max(singular_values[0], 1.0):
        raise ValueError('Candidate invariant block is rank deficient.')
    # Q = U R^{-1}, therefore T_Q = R T_U R^{-1}; solve the right equation.
    transformed = np.linalg.solve(r.T, (r @ t).T).T
    return q, transformed


def empty_personalized_diagnostics():
    return {
        'candidate_received': 0.0,
        'candidate_kstar_computed': 0.0,
        'candidate_source_round': 0.0,
        'candidate_total': 0.0,
        'candidate_server_raw_external': 0.0,
        'candidate_server_conflict_rejected': 0.0,
        'candidate_server_nonconflicting': 0.0,
        'candidate_positive_need': 0.0,
        'candidate_active': 0.0,
        'candidate_mean_compatibility': 0.0,
        'candidate_mean_need': 0.0,
        'candidate_mean_alpha': 0.0,
        'candidate_delta_k_relative': 0.0,
        'candidate_prediction_residual_before': 0.0,
        'candidate_prediction_residual_after': 0.0,
        'candidate_prediction_residual_gain': 0.0,
        'candidate_spectral_radius_k': 0.0,
        'candidate_spectral_radius_kstar': 0.0,
        'candidate_mean_mode_residual_before': 0.0,
        'candidate_mean_mode_residual_after': 0.0,
    }


def fit_personalized_koopman(
    local_operator, snapshots, next_snapshots, package, *, eta
):
    """Estimate private need gates and solve K-star A = B jointly.

    Input operators act on column observable snapshots: Z+ ~= K Z.
    The derivative D is with respect to beta = eta * alpha, so the need score
    and the final reconstruction cannot silently multiply eta twice.
    """

    k = _finite(local_operator, name='local K', ndim=2)
    z = _finite(snapshots, name='local observable snapshots', ndim=2)
    z_next = _finite(
        next_snapshots, name='next observable snapshots', ndim=2
    )
    if (
        k.shape[0] != k.shape[1]
        or z.shape != z_next.shape
        or z.shape[0] != k.shape[0]
    ):
        raise ValueError('Local K and private snapshot dimensions disagree.')
    if package.get('protocol') != CANDIDATE_PROTOCOL:
        raise ValueError('Incompatible Koopman candidate package.')
    if not np.isfinite(eta) or float(eta) < 0.0:
        raise ValueError('Koopman absorption eta must be finite and non-negative.')

    residual = z_next - k @ z
    residual_norm = float(np.linalg.norm(z_next, 'fro'))
    initial_prediction = float(
        np.linalg.norm(residual, 'fro')
        / max(residual_norm, np.finfo(np.float64).eps)
    )
    candidate_terms = []
    need_values = []
    for candidate in package['candidates']:
        u = _finite(candidate['U_local'], name='candidate U', ndim=2)
        t = _finite(candidate['T'], name='candidate T', ndim=2)
        if u.shape[0] != k.shape[0]:
            raise ValueError('Candidate U has the wrong observable width.')
        u, t = _normalize_block(u, t)
        compatibility = float(candidate['compatibility'])
        if not np.isfinite(compatibility) or not 0.0 <= compatibility <= 1.0:
            raise ValueError('Candidate compatibility must lie in [0, 1].')
        mismatch = k @ u - u @ t
        direction = -(mismatch @ u.conj().T)
        predicted_change = direction @ z
        need = float(np.real(np.vdot(residual, predicted_change)))
        cosine = need / max(
            float(np.linalg.norm(residual, 'fro'))
            * float(np.linalg.norm(predicted_change, 'fro')),
            np.finfo(np.float64).eps,
        )
        candidate_terms.append((candidate, u, t, compatibility, need, cosine))
        need_values.append(max(need, 0.0))

    max_positive = max(need_values, default=0.0)
    denominator = max_positive + np.finfo(np.float64).eps
    weights = []
    rows = []
    for candidate, u, t, compatibility, need, cosine in candidate_terms:
        need_gate = max(need, 0.0) / denominator
        alpha = compatibility * need_gate
        weights.append(alpha)
        rows.append({
            'mode_id': str(candidate['mode_id']),
            'source_client': int(candidate['source_client']),
            'source_mode': int(candidate['source_mode']),
            'overlap': float(candidate['overlap']),
            'disagreement': float(candidate['disagreement']),
            'conflict': float(candidate['conflict']),
            'compatibility': compatibility,
            'reliability': float(candidate['reliability']),
            'need': need,
            'need_gate': float(need_gate),
            'residual_alignment_cosine': float(cosine),
            'alpha': float(alpha),
        })

    a = np.eye(k.shape[0], dtype=np.complex128)
    b = k.copy()
    for (candidate, u, t, _, _, _), alpha in zip(candidate_terms, weights):
        beta = float(eta) * float(alpha)
        if beta <= 0.0:
            continue
        a += beta * (u @ u.conj().T)
        b += beta * (u @ t @ u.conj().T)
    # K-star A = B.  Taking adjoints turns this right solve into a left solve.
    kstar = np.linalg.solve(a.conj().T, b.conj().T).conj().T
    if not np.all(np.isfinite(kstar.real) & np.isfinite(kstar.imag)):
        raise FloatingPointError('Personalized Koopman K-star is non-finite.')
    delta = kstar - k
    final_prediction = float(
        np.linalg.norm(z_next - kstar @ z, 'fro')
        / max(residual_norm, np.finfo(np.float64).eps)
    )
    active_before = []
    active_after = []
    for row, (_, u, t, _, _, _) in zip(rows, candidate_terms):
        if row['alpha'] <= 0.0:
            row['mode_residual_before'] = 0.0
            row['mode_residual_after'] = 0.0
            continue
        scale = max(
            float(np.linalg.norm(u @ t, 'fro')),
            np.finfo(np.float64).eps,
        )
        before = float(np.linalg.norm(k @ u - u @ t, 'fro') / scale)
        after = float(np.linalg.norm(kstar @ u - u @ t, 'fro') / scale)
        row['mode_residual_before'] = before
        row['mode_residual_after'] = after
        active_before.append(before)
        active_after.append(after)

    stats = {
        **empty_personalized_diagnostics(),
        'candidate_received': 1.0,
        'candidate_kstar_computed': 1.0,
        'candidate_source_round': float(package['source_round']),
        'candidate_total': float(len(rows)),
        'candidate_server_raw_external': float(
            package.get('raw_external_count', len(rows))
        ),
        'candidate_server_conflict_rejected': float(
            package.get('conflict_rejected_count', 0)
        ),
        'candidate_server_nonconflicting': float(
            max(
                package.get('raw_external_count', len(rows))
                - package.get('conflict_rejected_count', 0),
                0,
            )
        ),
        'candidate_positive_need': float(sum(
            row['need'] > 0.0 for row in rows
        )),
        'candidate_active': float(len(active_before)),
        'candidate_mean_compatibility': float(np.mean([
            row['compatibility'] for row in rows
        ])) if rows else 0.0,
        'candidate_mean_need': float(np.mean([
            row['need'] for row in rows
        ])) if rows else 0.0,
        'candidate_mean_alpha': float(np.mean(weights)) if weights else 0.0,
        'candidate_delta_k_relative': float(
            np.linalg.norm(delta, 'fro')
            / max(np.linalg.norm(k, 'fro'), np.finfo(np.float64).eps)
        ),
        'candidate_prediction_residual_before': initial_prediction,
        'candidate_prediction_residual_after': final_prediction,
        'candidate_prediction_residual_gain': float(
            (initial_prediction - final_prediction)
            / max(initial_prediction, np.finfo(np.float64).eps)
        ),
        'candidate_spectral_radius_k': float(max(abs(np.linalg.eigvals(k)))),
        'candidate_spectral_radius_kstar': float(
            max(abs(np.linalg.eigvals(kstar)))
        ),
        'candidate_mean_mode_residual_before': float(np.mean(
            active_before
        )) if active_before else 0.0,
        'candidate_mean_mode_residual_after': float(np.mean(
            active_after
        )) if active_after else 0.0,
    }
    if not np.all(np.isfinite(list(stats.values()))):
        raise FloatingPointError('Personalized Koopman diagnostics are non-finite.')
    return {
        'protocol': CANDIDATE_PROTOCOL,
        'client_id': int(package['client_id']),
        'source_round': int(package['source_round']),
        'reference_client_id': int(package['reference_client_id']),
        'K': k.astype(np.complex64),
        'K_star': kstar.astype(np.complex64),
        'delta_K': delta.astype(np.complex64),
        'A_condition': float(np.linalg.cond(a)),
        'candidate_rows': rows,
        'stats': stats,
        'task_feedback': False,
    }


__all__ = [
    'empty_personalized_diagnostics', 'fit_personalized_koopman',
]
