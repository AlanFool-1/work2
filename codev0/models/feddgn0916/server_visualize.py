"""Server-side diagnostic plots for the passive Koopman candidate pipeline.

The renderer consumes the already persisted candidate analysis.  It never
reads private observable snapshots and never creates a mode graph.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np


def _finite(value, *, ndim=None):
    array = np.asarray(value, dtype=np.complex128)
    if ndim is not None and array.ndim != ndim:
        raise ValueError('Visualization array has an unexpected rank.')
    if not np.all(np.isfinite(array.real) & np.isfinite(array.imag)):
        raise ValueError('Visualization array contains non-finite values.')
    return array


def _render_functional_map(analysis, output, *, dpi):
    import matplotlib.pyplot as plt

    reports = sorted(
        analysis.get('functional_map_reports', []),
        key=lambda item: int(item['client_id']),
    )
    clients = [int(item['client_id']) for item in reports]
    positions = np.arange(len(clients))
    figure, axis = plt.subplots(figsize=(max(7, len(clients) * 0.7), 4.5),
                                layout='constrained')
    width = 0.24
    for offset, (field, label, color) in enumerate((
        ('identity_residual', 'identity', 'tab:gray'),
        ('spectral_initial_residual', 'spectral init', 'tab:blue'),
        ('intertwining_residual', 'final map', 'tab:orange'),
    )):
        axis.bar(
            positions + (offset - 1) * width,
            [float(item.get(field, 0.0)) for item in reports],
            width=width, label=label, color=color,
        )
    axis.set_xticks(positions, [str(client) for client in clients])
    axis.set_xlabel('client')
    axis.set_ylabel('normalized residual')
    axis.set_title('Functional Map alignment quality')
    axis.grid(axis='y', alpha=0.25)
    unitary_axis = axis.twinx()
    unitary_axis.plot(
        positions,
        [float(item.get('orthogonality_error', 0.0)) for item in reports],
        color='black', marker='x', linestyle=':', label='unitary error',
    )
    unitary_axis.set_ylabel('unitary error')
    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = unitary_axis.get_legend_handles_labels()
    axis.legend(handles + extra_handles, labels + extra_labels, fontsize=8)
    residual_path = output / 'functional_map_residuals.png'
    figure.savefig(residual_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return residual_path


def _render_correspondence(analysis, output, client_id, *, dpi):
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    raw = {
        int(key): _finite(value, ndim=2)
        for key, value in analysis['raw_operators'].items()
    }
    maps = {
        int(key): _finite(value, ndim=2)
        for key, value in analysis['functional_maps'].items()
    }
    reference_id = int(analysis['reference_client_id'])
    _, reference_vectors = np.linalg.eig(raw[reference_id])
    _, local_vectors = np.linalg.eig(raw[int(client_id)])
    correspondence = np.abs(
        reference_vectors.conj().T @ maps[int(client_id)] @ local_vectors
    ) ** 2
    # Reorder local modes onto their best-matching reference mode so a clean
    # one-to-one correspondence appears as a diagonal rather than as a
    # scattered permutation. An assignment always yields a permutation, even
    # from pure noise, so the mean best-match mass is reported alongside it:
    # near 1/N means the diagonal carries no information.
    shares = correspondence / np.maximum(
        correspondence.sum(axis=1, keepdims=True), np.finfo(float).tiny
    )
    _, columns = linear_sum_assignment(-correspondence)
    concentration = float(shares.max(axis=1).mean())
    chance = 1.0 / correspondence.shape[0]
    client_dir = output / f'client_{int(client_id):02d}'
    client_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 5.5), layout='constrained')
    image = axis.imshow(correspondence[:, columns], cmap='viridis',
                        aspect='auto')
    axis.set_xlabel('local mode (aligned to reference)')
    axis.set_ylabel('reference mode')
    axis.set_title(
        f'Mode correspondence: client {int(client_id)}\n'
        f'mean best-match mass {concentration:.3f} (chance {chance:.3f})'
    )
    figure.colorbar(image, ax=axis, label='|v_ref^H C v_local|^2')
    path = client_dir / 'functional_map_mode_matching.png'
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return path


def _render_conflict(analysis, output, client_id, *, dpi):
    import matplotlib.pyplot as plt

    rows = [
        row for row in analysis.get('conflict_rows', [])
        if int(row.get('target_client', -1)) == int(client_id)
    ]
    client_dir = output / f'client_{int(client_id):02d}'
    client_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 5.0), layout='constrained')
    accepted = [row for row in rows if not bool(row.get('rejected', False))]
    rejected = [row for row in rows if bool(row.get('rejected', False))]
    for subset, label, color in (
        (accepted, 'accepted', 'tab:blue'),
        (rejected, 'rejected', 'tab:red'),
    ):
        if subset:
            axis.scatter(
                [float(row.get('overlap', 0.0)) for row in subset],
                [float(row.get('disagreement', 0.0)) for row in subset],
                s=30, alpha=0.75, label=label, color=color,
            )
    threshold = float(analysis['summary'].get('conflict_threshold', 0.0))
    # Scale to the observed cloud rather than to the nominal [0, 1] x [0, tau]
    # box. With a permissive threshold the boundary sits far above every point
    # and a fixed box leaves the plot almost entirely empty.
    x_values = [float(row.get('overlap', 0.0)) for row in rows]
    y_values = [float(row.get('disagreement', 0.0)) for row in rows]
    x_limit = max(max(x_values, default=0.0) * 1.10, 1e-3)
    y_limit = max(max(y_values, default=0.0) * 1.20, 1e-4)
    boundary_x = np.linspace(x_limit * 0.01, x_limit, 400)
    axis.plot(boundary_x, threshold / boundary_x, '--', color='black',
              linewidth=1.0, label='conflict boundary d=tau/s')
    rejected_count = len(rejected)
    ratio = rejected_count / max(len(rows), 1)
    axis.set(
        xlim=(0.0, x_limit), ylim=(0.0, y_limit),
        xlabel='subspace overlap', ylabel='dynamical disagreement',
        title=(f'Mode conflict map: target client {int(client_id)}, '
               f'server round {int(analysis["round"])}\n'
               f'reference client {int(analysis["reference_client_id"])} | '
               f'threshold={threshold:g} | evaluated={len(rows)} | '
               f'rejected={rejected_count} ({ratio:.1%})'),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    path = client_dir / 'conflict_map.png'
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return path


def _render_server_funnel(analysis, output, *, dpi):
    import matplotlib.pyplot as plt

    client_ids = [int(value) for value in analysis['client_ids']]
    packages = analysis['packages']
    figure, axes = plt.subplots(
        len(client_ids), 1, figsize=(8, max(3.0, len(client_ids) * 1.2)),
        squeeze=False, layout='constrained',
    )
    for axis, client_id in zip(axes.flat, client_ids):
        package = packages[client_id]
        raw = int(package.get('raw_external_count', 0))
        rejected = int(package.get('conflict_rejected_count', 0))
        values = [raw, max(raw - rejected, 0), int(package.get('candidate_count', 0))]
        axis.barh(np.arange(3), values, color=['0.55', 'tab:blue', 'tab:orange'])
        axis.set_yticks(np.arange(3), ['evaluated', 'non-conflicting', 'sent'])
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.set_title(f'target client {client_id}', loc='left', fontsize=9)
        axis.grid(axis='x', alpha=0.2)
        for index, value in enumerate(values):
            axis.text(value, index, f' {value}', va='center', fontsize=8)
    axes[-1, 0].set_xlabel('candidate count')
    path = output / 'candidate_funnel_server.png'
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return path


def render_server_history(history_path, figure_root, *, dpi=90):
    """Render round-wise alignment, conflict, and candidate flow."""
    path = Path(history_path)
    if not path.exists():
        return None
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def values(field):
        return np.asarray(
            [float(row.get(field, 0.0) or 0.0) for row in rows],
            dtype=np.float64,
        )

    rounds = values('round')
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), layout='constrained')
    axes[0].plot(rounds, values('mean_functional_map_residual'),
                 marker='o', markersize=3, label='mean map residual')
    axes[0].plot(rounds, values('max_functional_map_orthogonality_error'),
                 marker='o', markersize=3, label='max unitary error')
    axes[0].set_title('Functional Map quality')
    axes[1].plot(rounds, values('conflict_rejection_ratio'),
                 marker='o', markersize=3, label='rejection ratio')
    axes[1].plot(rounds, values('conflict_median'), label='conflict median')
    axes[1].plot(rounds, values('conflict_q95'), label='conflict q95')
    axes[1].set_title('Conflict dynamics')
    axes[2].plot(rounds, values('raw_external_count'), label='evaluated')
    axes[2].plot(rounds, values('retained_count'), label='sent')
    axes[2].set_title('Server candidate flow')
    for axis in axes:
        axis.set_xlabel('round')
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    root = Path(figure_root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / 'server_knowledge_flow.png'
    figure.savefig(output, dpi=int(dpi), bbox_inches='tight')
    plt.close(figure)
    return str(output.resolve())


def render_server_diagnostics(
    analysis, figure_root, *, dpi=90, history_path=None
):
    """Render one server round's alignment, conflict, and funnel figures."""
    if int(dpi) <= 0:
        raise ValueError('Visualization DPI must be positive.')
    import matplotlib
    matplotlib.use('Agg')

    round_number = int(analysis['round'])
    root = Path(figure_root)
    round_dir = root / f'round_{round_number:04d}'
    round_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        'functional_map_residuals': _render_functional_map(
            analysis, round_dir, dpi=int(dpi)
        ),
        'candidate_funnel': _render_server_funnel(
            analysis, round_dir, dpi=int(dpi)
        ),
    }
    history_figure = (
        render_server_history(history_path, root, dpi=int(dpi))
        if history_path is not None else None
    )
    if history_figure is not None:
        figures['knowledge_flow'] = Path(history_figure)
    per_client = {}
    for client_id in analysis['client_ids']:
        per_client[str(int(client_id))] = {
            'correspondence': _render_correspondence(
                analysis, round_dir, int(client_id), dpi=int(dpi)
            ),
            'conflict': _render_conflict(
                analysis, round_dir, int(client_id), dpi=int(dpi)
            ),
        }

    html_path = round_dir / 'index.html'
    links = [
        f'<h2>Server round {round_number}</h2>',
        '<p>Only Koopman operators and compact mode metadata are plotted; '
        'private snapshots remain on clients.</p>',
        f'<h3>Functional Map quality</h3><img src="{html.escape(figures["functional_map_residuals"].name)}">',
        f'<h3>Candidate funnel</h3><img src="{html.escape(figures["candidate_funnel"].name)}">',
    ]
    if 'knowledge_flow' in figures:
        links.append(
            f'<h3>Round-wise knowledge flow</h3><img src="../{html.escape(figures["knowledge_flow"].name)}">'
        )
    for client_id, paths in per_client.items():
        links.extend((
            f'<h3>Target client {client_id}: conflict map</h3>',
            f'<img src="client_{int(client_id):02d}/{html.escape(paths["conflict"].name)}">',
            f'<h3>Target client {client_id}: mode correspondence</h3>',
            f'<img src="client_{int(client_id):02d}/{html.escape(paths["correspondence"].name)}">',
        ))
    html_path.write_text(
        '<!doctype html><meta charset="utf-8"><title>FedDGN server diagnostics</title>'
        '<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;'
        'padding:0 20px;line-height:1.5}img{max-width:100%;height:auto}'
        'h3{border-top:1px solid #ddd;padding-top:18px}</style>'
        + ''.join(links),
        encoding='utf-8',
    )
    return {
        'round_dir': str(round_dir.resolve()),
        'dashboard': str(html_path.resolve()),
        'figures': {
            key: str(value.resolve()) for key, value in figures.items()
        },
        'clients': {
            client: {key: str(value.resolve()) for key, value in paths.items()}
            for client, paths in per_client.items()
        },
    }


__all__ = ['render_server_diagnostics', 'render_server_history']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dpi', type=int, default=90)
    args = parser.parse_args()
    import torch

    run_dir = Path(args.run_dir).resolve()
    analysis_path = run_dir / 'koopman' / 'candidates' / 'latest.pt'
    if not analysis_path.exists():
        raise FileNotFoundError(f'Candidate analysis not found: {analysis_path}')
    analysis = torch.load(analysis_path, map_location='cpu', weights_only=False)
    report = render_server_diagnostics(
        analysis,
        run_dir / 'koopman' / 'visualization' / 'server',
        dpi=args.dpi,
        history_path=run_dir / 'koopman' / 'candidates' / 'server_rounds.csv',
    )
    print(report['dashboard'])


if __name__ == '__main__':
    main()
