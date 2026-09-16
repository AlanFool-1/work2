"""Client-private K-star, candidate, and round-wise diagnostic plots.

The active client calls this after private K-star reconstruction. No graph,
client-to-client distance, force, or task-backbone state is plotted.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch


HISTORY_FIELDS = (
    'source_round', 'target_client', 'reference_client_id',
    'candidate_total', 'candidate_positive_need', 'candidate_active',
    'candidate_mean_compatibility', 'candidate_mean_need',
    'candidate_mean_alpha', 'candidate_delta_k_relative',
    'candidate_prediction_residual_before',
    'candidate_prediction_residual_after',
    'candidate_prediction_residual_gain',
    'candidate_mean_mode_residual_before',
    'candidate_mean_mode_residual_after',
    'candidate_spectral_radius_k',
    'candidate_spectral_radius_kstar', 'A_condition',
)


def _stats(result):
    value = result.get('stats', {})
    return value if isinstance(value, dict) else {}


def _rows(result):
    value = result.get('candidate_rows', [])
    return value if isinstance(value, list) else []


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _operator(value, name):
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'{name} must be a square Koopman operator.')
    if not np.all(np.isfinite(matrix.real) & np.isfinite(matrix.imag)):
        raise ValueError(f'{name} must be finite.')
    return matrix


def operator_comparison(result):
    """Compare this client's K and K-star in their shared private frame."""
    local = _operator(result['K'], 'K')
    target = _operator(result['K_star'], 'K-star')
    if local.shape != target.shape:
        raise ValueError('K and K-star have different observable widths.')
    delta = target - local
    local_values = np.linalg.eigvals(local)
    target_values = np.linalg.eigvals(target)
    delta_norm = float(np.linalg.norm(delta, 'fro'))
    summary = {
        'client_id': int(result['client_id']),
        'source_round': int(result['source_round']),
        'observable_dim': int(local.shape[0]),
        'K_fro': float(np.linalg.norm(local, 'fro')),
        'delta_K_fro': delta_norm,
        'relative_delta_K': float(
            delta_norm / max(np.linalg.norm(local, 'fro'), np.finfo(float).eps)
        ),
        'K_spectral_radius': float(np.abs(local_values).max()),
        'K_star_spectral_radius': float(np.abs(target_values).max()),
        'delta_K_imaginary_fraction': float(
            np.linalg.norm(delta.imag, 'fro')
            / max(delta_norm, np.finfo(float).eps)
        ),
        'coordinate': 'same_previous_round_private_observable_dictionary',
        'task_feedback': False,
    }
    return {
        'K': local, 'K_star': target, 'delta_K': delta,
        'summary': summary,
    }


def append_personalized_history(result, history_path):
    """Append one completed private K-star reconstruction to CSV history."""
    stats = _stats(result)
    row = {
        'source_round': int(result['source_round']),
        'target_client': int(result['client_id']),
        'reference_client_id': int(result.get('reference_client_id', -1)),
        'A_condition': _number(result.get('A_condition')),
    }
    for field in HISTORY_FIELDS:
        if field in row:
            continue
        row[field] = _number(stats.get(field))
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({field: row[field] for field in HISTORY_FIELDS})
    return str(path.resolve())


def _load_history(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def render_client_history(history_path, figure_root, *, dpi=90):
    """Render round-wise client knowledge-flow curves from history.csv."""
    rows = _load_history(history_path)
    if not rows:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rounds = np.asarray([_number(row.get('source_round')) for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    panels = (
        (('candidate_active', 'candidate_total'), 'candidate survival'),
        (('candidate_mean_alpha', 'candidate_delta_k_relative'),
         'absorption strength'),
        (('candidate_prediction_residual_before',
          'candidate_prediction_residual_after'),
         'private prediction residual'),
        (('candidate_prediction_residual_gain',), 'private residual gain'),
    )
    labels = {
        'candidate_active': 'active',
        'candidate_total': 'sent',
        'candidate_mean_alpha': 'mean alpha',
        'candidate_delta_k_relative': 'relative delta K',
        'candidate_prediction_residual_before': 'before',
        'candidate_prediction_residual_after': 'after',
        'candidate_prediction_residual_gain': 'gain',
    }
    for axis, (fields, title) in zip(axes.flat, panels):
        for field in fields:
            axis.plot(
                rounds,
                [_number(row.get(field)) for row in rows],
                marker='o', markersize=3, label=labels[field],
            )
        axis.set_title(title)
        axis.set_xlabel('source round')
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle('Client round-wise Koopman knowledge flow')
    root = Path(figure_root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / 'knowledge_flow.png'
    figure.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return str(output.resolve())


def _render_client_diagnostics(result, round_dir, *, dpi):
    """Render candidate-level diagnostics from one private result."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stats = _stats(result)
    rows = _rows(result)
    outputs = {}

    # Compatibility × necessity: service-side compatibility and client-side
    # need remain separate axes, with alpha as the point weight. Every
    # candidate that reaches a client already passed the server's hard
    # conflict veto, so compatibility sits in a narrow band near 1 and the
    # nominal [0, 1] axis would collapse every point onto a single column.
    # The axis is therefore zoomed to the observed band and the range is
    # stated in the title.
    figure, axis = plt.subplots(figsize=(6.5, 5.0), layout='constrained')
    if rows:
        compatibility = np.asarray(
            [_number(row.get('compatibility')) for row in rows]
        )
        sources = np.asarray([int(row.get('source_client', -1)) for row in rows])
        scatter = axis.scatter(
            compatibility,
            [_number(row.get('need_gate')) for row in rows],
            s=30 + 170 * np.asarray([max(_number(row.get('alpha')), 0.0)
                                     for row in rows]),
            c=sources, cmap='tab10', alpha=0.85, edgecolors='white',
            linewidths=0.4,
        )
        figure.colorbar(scatter, ax=axis, label='source client')
        low = float(compatibility.min())
        high = float(compatibility.max())
        pad = max((high - low) * 0.15, 1e-4)
        axis.set_xlim(max(low - pad, 0.0), min(high + pad, 1.0))
        title = ('Compatibility × necessity\n'
                 f'compatibility ∈ [{low:.5f}, {high:.5f}]')
    else:
        axis.text(0.5, 0.5, 'No candidates', ha='center', va='center')
        title = 'Compatibility × necessity'
    axis.set(ylim=(-0.03, 1.03),
             xlabel='compatibility', ylabel='need gate', title=title)
    axis.axhline(0.5, color='0.7', linewidth=0.8)
    axis.grid(alpha=0.2)
    output = Path(round_dir) / 'compatibility_need.png'
    figure.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    outputs['compatibility_need'] = str(output.resolve())

    # Private trajectory residual before/after. Lower is better, so the arrow
    # colour and the title state the direction instead of leaving the reader to
    # infer it from the sign of the gain.
    before = _number(stats.get('candidate_prediction_residual_before'))
    after = _number(stats.get('candidate_prediction_residual_after'))
    gain = _number(stats.get('candidate_prediction_residual_gain'))
    improved = gain > 0.0
    figure, axis = plt.subplots(figsize=(6.5, 3.8), layout='constrained')
    axis.annotate(
        '', xy=(after, 0), xytext=(before, 0),
        arrowprops={
            'arrowstyle': '-|>', 'linewidth': 2,
            'color': 'tab:green' if improved else 'tab:red',
        },
    )
    axis.scatter([before], [0], s=70, zorder=3, color='tab:blue',
                 label='before (K)')
    axis.scatter([after], [0], s=70, zorder=3, color='tab:orange',
                 label='after (K*)')
    axis.text(before, 0.5, 'before', ha='center', fontsize=8, color='tab:blue')
    axis.text(after, 0.5, 'after', ha='center', fontsize=8, color='tab:orange')
    axis.set(
        yticks=[], ylim=(-0.6, 1.2),
        xlabel='normalized private prediction residual (lower is better)',
        title=(f'Private prediction residual: gain={gain:+.4f} → '
               f'{"improved" if improved else "worsened"}\n'
               f'before={before:.5f}, after={after:.5f}'),
    )
    axis.grid(axis='x', alpha=0.25)
    axis.legend(loc='upper right')
    output = Path(round_dir) / 'private_residual_before_after.png'
    figure.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    outputs['private_residual'] = str(output.resolve())

    # Per-active-mode constraint residuals.
    active = [row for row in rows if _number(row.get('alpha')) > 0.0]
    figure, axis = plt.subplots(figsize=(7.0, 4.5), layout='constrained')
    if active:
        positions = np.arange(len(active))
        for position, row in zip(positions, active):
            left = _number(row.get('mode_residual_before'))
            right = _number(row.get('mode_residual_after'))
            axis.plot([left, right], [position, position], color='0.65')
            axis.scatter([left], [position], color='tab:blue', s=35)
            axis.scatter([right], [position], color='tab:orange', s=35)
        axis.set_yticks(positions)
        axis.set_yticklabels([str(row.get('mode_id', 'mode')) for row in active],
                             fontsize=8)
        axis.scatter([], [], color='tab:blue', s=35, label='before')
        axis.scatter([], [], color='tab:orange', s=35, label='after')
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, 'No active candidates', ha='center', va='center')
        axis.set_yticks([])
    axis.set_xlabel('normalized mode constraint residual')
    axis.set_title('Active candidate constraints')
    axis.grid(axis='x', alpha=0.2)
    output = Path(round_dir) / 'active_mode_residuals.png'
    figure.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    outputs['active_mode_residuals'] = str(output.resolve())

    # Candidate funnel. Server counts are copied into client stats when the
    # package is consumed, so this plot preserves the server/client boundary.
    sent = _number(stats.get('candidate_total'))
    raw = _number(stats.get('candidate_server_raw_external', sent), sent)
    rejected = _number(stats.get('candidate_server_conflict_rejected'))
    nonconflicting = _number(
        stats.get('candidate_server_nonconflicting', max(raw - rejected, 0.0)),
        max(raw - rejected, 0.0),
    )
    stages = [
        ('conflict evaluated', raw),
        ('non-conflicting', nonconflicting),
        ('sent', sent),
        ('positive need', _number(stats.get('candidate_positive_need'))),
        ('active', _number(stats.get('candidate_active'))),
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.2), layout='constrained')
    values = [value for _, value in stages]
    axis.barh(np.arange(len(stages)), values, color='#168b8b')
    axis.set_yticks(np.arange(len(stages)), [label for label, _ in stages])
    axis.invert_yaxis()
    axis.set_xlabel('candidate count')
    axis.set_title('Candidate funnel')
    for position, value in enumerate(values):
        axis.text(value, position, f' {value:.0f}', va='center')
    axis.grid(axis='x', alpha=0.2)
    output = Path(round_dir) / 'candidate_funnel.png'
    figure.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    outputs['candidate_funnel'] = str(output.resolve())
    return outputs


def render_kstar_client(result, figure_root, *, dpi=90):
    """Write one matrix comparison for one client/source round only."""
    if int(dpi) <= 0:
        raise ValueError('Visualization DPI must be positive.')
    comparison = operator_comparison(result)
    summary = dict(comparison['summary'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(figure_root)
    round_dir = root / f"source_round_{summary['source_round']:04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    # Clean the obsolete spectrum image if a previous renderer populated this
    # exact client/round directory. It is no longer part of the protocol.
    (round_dir / 'K_vs_Kstar_spectrum.png').unlink(missing_ok=True)

    local = comparison['K']
    target = comparison['K_star']
    delta = comparison['delta_K']
    # K and K-star use the same scale. Delta has its own scale so a small
    # but structured update remains visible instead of looking identical.
    local_log = np.log10(np.abs(local) + 1e-8)
    target_log = np.log10(np.abs(target) + 1e-8)
    delta_log = np.log10(np.abs(delta) + 1e-8)
    shared_min = float(min(local_log.min(), target_log.min()))
    shared_max = float(max(local_log.max(), target_log.max()))
    figure = plt.figure(figsize=(14, 4.5), layout='constrained')
    grid = figure.add_gridspec(
        1, 5, width_ratios=(1, 1, 0.045, 1, 0.045)
    )
    axes = [figure.add_subplot(grid[0, index]) for index in (0, 1, 3)]
    shared_colorbar_axis = figure.add_subplot(grid[0, 2])
    update_colorbar_axis = figure.add_subplot(grid[0, 4])
    images = [
        axes[0].imshow(local_log, cmap='viridis',
                       vmin=shared_min, vmax=shared_max),
        axes[1].imshow(target_log, cmap='viridis',
                       vmin=shared_min, vmax=shared_max),
        axes[2].imshow(delta_log, cmap='magma'),
    ]
    for axis, title in zip(axes, ('log10 |K|', 'log10 |K*|',
                                  'log10 |K* - K|')):
        axis.set_title(title)
        axis.set_xlabel('observable index')
        axis.set_ylabel('observable index')
    figure.colorbar(images[0], cax=shared_colorbar_axis,
                    label='same magnitude scale')
    figure.colorbar(images[2], cax=update_colorbar_axis,
                    label='update magnitude scale')
    figure.suptitle(
        f"Client {summary['client_id']}, source round {summary['source_round']}: "
        f"||ΔK||/||K|| = {100*summary['relative_delta_K']:.4f}%"
    )
    matrix_file = round_dir / 'K_vs_Kstar_matrix.png'
    figure.savefig(matrix_file, dpi=dpi, bbox_inches='tight')
    plt.close(figure)

    diagnostic_figures = _render_client_diagnostics(
        result, round_dir, dpi=int(dpi)
    )
    summary['figures'] = {
        'matrix': str(matrix_file.resolve()),
        **diagnostic_figures,
    }
    (round_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    dashboard = round_dir / 'index.html'
    history_figure = root / 'knowledge_flow.png'
    history_panel = ''
    if history_figure.exists():
        summary['figures']['knowledge_flow'] = str(history_figure.resolve())
        history_panel = (
            '<div class="panel"><h2>Round-wise knowledge flow</h2>'
            '<img src="../knowledge_flow.png"></div>'
        )
        (round_dir / 'summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    dashboard.write_text(
        '<!doctype html><meta charset="utf-8"><title>Client K vs K*</title>'
        '<style>body{font-family:sans-serif;max-width:1000px;margin:30px auto;'
        'line-height:1.6;padding:0 20px}img{max-width:100%;height:auto}'
        '.panel{margin:28px 0;border-top:1px solid #ddd;padding-top:16px}'
        '</style>'
        f'<h1>客户端 {summary["client_id"]}：K 与 K*</h1>'
        f'<p>同一私有观测坐标，来源轮次 {summary["source_round"]}；'
        f'相对更新 {100*summary["relative_delta_K"]:.4f}%。'
        'K* 仅被动重构，不指导任务训练。</p>'
        f'<div class="panel"><h2>矩阵幅度及差值</h2>'
        f'<img src="{html.escape(matrix_file.name)}"></div>'
        f'<div class="panel"><h2>Compatibility × necessity</h2>'
        f'<img src="{html.escape(Path(diagnostic_figures["compatibility_need"]).name)}"></div>'
        f'<div class="panel"><h2>Private residual before/after</h2>'
        f'<img src="{html.escape(Path(diagnostic_figures["private_residual"]).name)}"></div>'
        f'<div class="panel"><h2>Active mode constraints</h2>'
        f'<img src="{html.escape(Path(diagnostic_figures["active_mode_residuals"]).name)}"></div>'
        f'<div class="panel"><h2>Candidate funnel</h2>'
        f'<img src="{html.escape(Path(diagnostic_figures["candidate_funnel"]).name)}"></div>'
        f'{history_panel}'
        '<p><a href="summary.json">数值摘要</a></p>',
        encoding='utf-8',
    )
    (root / 'latest.html').write_text(
        '<!doctype html><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={html.escape(round_dir.name)}/index.html">'
        f'<a href="{html.escape(round_dir.name)}/index.html">最新 K/K* 图</a>',
        encoding='utf-8',
    )
    return {'dashboard': str(dashboard.resolve()), 'summary': summary}


def main():
    """Immediate preview for already-running campaigns using the same renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dpi', type=int, default=90)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve() / 'koopman'
    paths = sorted(root.glob('client_*/personalized/latest.pt'))
    if not paths:
        raise FileNotFoundError('No private client K/K-star artifacts found.')
    for path in paths:
        result = torch.load(path, map_location='cpu', weights_only=False)
        if int(path.parent.parent.name.split('_', 1)[1]) != int(result['client_id']):
            raise ValueError(f'Artifact client ID mismatch: {path}.')
        report = render_kstar_client(
            result, path.parent / 'figures', dpi=args.dpi
        )
        print(f"[client {result['client_id']}] {report['dashboard']}")


if __name__ == '__main__':
    main()
