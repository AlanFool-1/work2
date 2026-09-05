"""Post-run diagnostic plots for the functional-dynamics mechanism."""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path):
    with open(path, newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _series(rows, field):
    result = {}
    for row in rows:
        value = float(row[field])
        if not np.isfinite(value):
            continue
        result.setdefault(int(row['client_id']), []).append(
            (int(row['round']), value)
        )
    return result


def plot_client_validation_curves(log_dir):
    """Plot every client's validation accuracy over communication rounds."""
    csv_path = os.path.join(log_dir, 'client_metrics.csv')
    if not os.path.isfile(csv_path):
        return None
    rows = _read_rows(csv_path)
    curves = _series(rows, 'val_accuracy')
    if not curves:
        return None
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    all_rounds = sorted({round_id for points in curves.values() for round_id, _ in points})
    aligned = []
    for client_id, points in sorted(curves.items()):
        mapping = dict(points)
        values = [mapping.get(round_id, np.nan) for round_id in all_rounds]
        aligned.append(values)
        axis.plot(
            all_rounds, values, linewidth=1.15, alpha=0.72,
            label=f'Client {client_id}',
        )
    mean_curve = np.nanmean(np.asarray(aligned, dtype=np.float64), axis=0)
    axis.plot(
        all_rounds, mean_curve, color='black', linewidth=2.8,
        label='Client mean', zorder=20,
    )
    axis.set_title('Per-client validation accuracy over federated rounds')
    axis.set_xlabel('Round')
    axis.set_ylabel('Validation accuracy')
    axis.set_xlim(min(all_rounds), max(all_rounds))
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    output = os.path.join(log_dir, 'client_validation_accuracy.png')
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_cluster_assignments(log_dir):
    csv_path = os.path.join(log_dir, 'functional_dynamics.csv')
    if not os.path.isfile(csv_path):
        return None
    rows = _read_rows(csv_path)
    valid = [row for row in rows if float(row.get('cluster_id', 'nan')) >= 0]
    if not valid:
        return None
    rounds = sorted({int(row['round']) for row in valid})
    clients = sorted({int(row['client_id']) for row in valid})
    matrix = np.full((len(clients), len(rounds)), np.nan)
    round_index = {value: index for index, value in enumerate(rounds)}
    client_index = {value: index for index, value in enumerate(clients)}
    for row in valid:
        matrix[client_index[int(row['client_id'])], round_index[int(row['round'])]] = float(row['cluster_id'])
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    image_plot = axis.imshow(matrix, aspect='auto', interpolation='nearest', cmap='tab10')
    axis.set_title('Client dynamics-regime assignments')
    axis.set_xlabel('Round')
    axis.set_ylabel('Client')
    axis.set_yticks(range(len(clients)), labels=clients)
    tick_positions = np.linspace(0, len(rounds) - 1, min(10, len(rounds)), dtype=int)
    axis.set_xticks(tick_positions, labels=[rounds[index] for index in tick_positions])
    figure.colorbar(image_plot, ax=axis, label='Prototype ID', shrink=0.8)
    output = os.path.join(log_dir, 'functional_cluster_assignments.png')
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_functional_dynamics_diagnostics(
    log_dir, a_star=None, b_star=None, canonical_suffix='', plot_curves=True
):
    """Render mechanism curves and final canonical summaries, if available."""
    csv_path = os.path.join(log_dir, 'functional_dynamics.csv')
    if not os.path.isfile(csv_path):
        return []
    rows = _read_rows(csv_path)
    if not rows:
        return []

    panels = [
        ('dyn_fit_error', 'Native dynamics fit error'),
        ('fm_desc_error', 'Descriptor alignment residual'),
        ('fm_dyn_error', 'Dynamics intertwining residual'),
        ('basis_staleness', 'Basis overlap (higher is stable)'),
        ('injection_native_ratio', 'Injection / native field norm'),
        ('delta_safe_max_sym_eig', 'Max eigenvalue of sym(Delta safe)'),
    ]
    outputs = []
    if plot_curves:
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for axis, (field, title) in zip(axes.reshape(-1), panels):
            for client_id, points in sorted(_series(rows, field).items()):
                x, y = zip(*points)
                axis.plot(x, y, linewidth=1.0, alpha=0.75, label=f'C{client_id}')
            axis.set_title(title)
            axis.set_xlabel('Round')
            axis.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc='outside lower center', ncol=min(10, len(labels)))
        curves_path = os.path.join(log_dir, 'functional_dynamics_diagnostics.png')
        figure.savefig(curves_path, dpi=180)
        plt.close(figure)
        outputs.append(curves_path)

    if a_star is None or b_star is None:
        return outputs
    a_star = np.asarray(a_star)
    b_star = np.asarray(b_star)
    eigenvalues = np.linalg.eigvals(a_star)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    image_a = axes[0].imshow(a_star, cmap='coolwarm', aspect='auto')
    axes[0].set_title('Final canonical generator A*')
    figure.colorbar(image_a, ax=axes[0], shrink=0.8)
    axes[1].scatter(eigenvalues.real, eigenvalues.imag, s=35)
    axes[1].axvline(0.0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Eigenvalues of A*')
    axes[1].set_xlabel('Real')
    axes[1].set_ylabel('Imaginary')
    axes[1].grid(alpha=0.25)
    image_b = axes[2].imshow(b_star, cmap='viridis', aspect='auto')
    axes[2].set_title('Final canonical descriptor B*')
    figure.colorbar(image_b, ax=axes[2], shrink=0.8)
    canonical_path = os.path.join(
        log_dir, f'canonical_functional_dynamics{canonical_suffix}.png'
    )
    figure.savefig(canonical_path, dpi=180)
    plt.close(figure)
    outputs.append(canonical_path)
    return outputs
