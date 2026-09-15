"""Artifact-driven visual dashboard for the server-side Koopman mode graph."""

from __future__ import annotations

import argparse
import html
import json
import os
from glob import glob

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import torch

from models.feddgn.mode_graph import MODE_GRAPH_PROTOCOL


SUPPORTED_PROTOCOLS = {
    MODE_GRAPH_PROTOCOL,
    'mode_graph_v1_spatial_louvain',
    'mode_graph_v0_passive_view',
}


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_latest_mode_graph(log_path: str, graph_path: str = None):
    path = graph_path or os.path.join(
        log_path, 'koopman', 'mode_graph', 'latest.pt'
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Mode graph artifact not found: {path}')
    graph = _load(path)
    if str(graph.get('protocol')) not in SUPPORTED_PROTOCOLS:
        raise ValueError(f'Unexpected mode graph protocol in {path}.')
    count = len(graph.get('nodes', []))
    similarity = np.asarray(graph['similarity'])
    if similarity.shape != (count, count):
        raise ValueError('Mode similarity matrix and node table disagree.')
    return graph


def _save(figure, path, dpi):
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def _colors(indices, count, palette):
    denominator = max(1, int(count) - 1)
    return plt.get_cmap(palette)(np.asarray(indices) / denominator)


def _plot_global_graph(graph, path, dpi):
    nodes = graph['nodes']
    network = nx.Graph()
    lookup = {}
    for node in nodes:
        node_id = str(node['mode_id'])
        lookup[node_id] = node
        network.add_node(node_id)
    for edge in graph['edges']:
        network.add_edge(
            str(edge['source']),
            str(edge['target']),
            weight=float(edge['similarity']),
        )
    positions = nx.spring_layout(
        network,
        seed=42,
        weight='weight',
        iterations=120,
    )
    community_count = max(1, int(graph['community_count']))
    client_ids = list(graph['client_ids'])
    client_lookup = {
        int(client_id): index for index, client_id in enumerate(client_ids)
    }
    ordered_ids = list(network.nodes())
    node_communities = [lookup[node_id]['community_id'] for node_id in ordered_ids]
    node_clients = [client_lookup[lookup[node_id]['client_id']] for node_id in ordered_ids]
    sizes = np.full(len(ordered_ids), 70.0)

    figure, axis = plt.subplots(figsize=(11.5, 9.0))
    if network.number_of_edges():
        weights = np.asarray([
            float(data['weight'])
            for _, _, data in network.edges(data=True)
        ])
        nx.draw_networkx_edges(
            network,
            positions,
            ax=axis,
            edge_color='0.48',
            alpha=np.clip(0.12 + 0.55 * weights, 0.12, 0.7),
            width=0.4 + 1.8 * weights,
        )
    nx.draw_networkx_nodes(
        network,
        positions,
        nodelist=ordered_ids,
        ax=axis,
        node_color=_colors(node_communities, community_count, 'tab10'),
        edgecolors=_colors(node_clients, len(client_ids), 'tab20'),
        node_size=sizes,
        linewidths=1.4,
        alpha=0.92,
    )
    if len(nodes) <= 30:
        labels = {
            node_id: (
                f"c{lookup[node_id]['client_id']}:"
                f"m{lookup[node_id]['local_mode_index']}"
            )
            for node_id in ordered_ids
        }
        nx.draw_networkx_labels(
            network, positions, labels=labels, ax=axis, font_size=7
        )
    community_handles = [
        Patch(
            facecolor=_colors([community], community_count, 'tab10')[0],
            label=f'community {community}',
        )
        for community in range(community_count)
    ]
    client_handles = [
        Line2D(
            [0], [0], marker='o', linestyle='none', markerfacecolor='white',
            markeredgecolor=_colors([index], len(client_ids), 'tab20')[0],
            markeredgewidth=1.5, label=f'client {client_id}',
        )
        for index, client_id in enumerate(client_ids)
    ]
    first_legend = axis.legend(
        handles=community_handles,
        title='Node fill',
        loc='upper left',
        fontsize=8,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=client_handles,
        title='Node border',
        loc='upper right',
        ncol=2 if len(client_handles) > 5 else 1,
        fontsize=8,
    )
    axis.set_title(
        f"Global Koopman mode graph — round {graph['round']}\n"
        'fill = community, border = source client, aligned K-space edges'
    )
    axis.axis('off')
    _save(figure, path, dpi)


def _plot_similarity(graph, path, dpi):
    similarity = np.asarray(
        graph.get('spatial_similarity', graph['similarity'])
    )
    communities = np.asarray(graph['community_id_per_node'])
    clients = np.asarray(graph['client_id_per_node'])
    order = np.lexsort((clients, communities))
    ordered = similarity[np.ix_(order, order)]
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    image = axis.imshow(ordered, vmin=0.0, vmax=1.0, cmap='magma')
    ordered_communities = communities[order]
    boundaries = np.flatnonzero(np.diff(ordered_communities)) + 0.5
    for boundary in boundaries:
        axis.axhline(boundary, color='cyan', linewidth=0.8, alpha=0.8)
        axis.axvline(boundary, color='cyan', linewidth=0.8, alpha=0.8)
    if ordered.size <= 30:
        local_indices = np.asarray(graph['local_mode_index_per_node'])[order]
        labels = [
            f'c{client}:m{mode}'
            for client, mode in zip(clients[order], local_indices)
        ]
        axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=6)
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_xlabel('Modes ordered by community, then client')
    axis.set_ylabel('Modes ordered by community, then client')
    axis.set_title(
        'Full aligned K-space similarity (cyan = community boundary)'
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save(figure, path, dpi)


def _plot_client_community_matrix(graph, path, dpi):
    matrix = np.asarray(graph['community_client_matrix'], dtype=np.int64)
    clients = list(graph['client_ids'])
    figure, axis = plt.subplots(
        figsize=(max(6.4, matrix.shape[1] * 0.9), max(4.8, len(clients) * 0.48))
    )
    image = axis.imshow(matrix, cmap='Blues', aspect='auto')
    axis.set_xticks(
        np.arange(matrix.shape[1]),
        [f'C{index}' for index in range(matrix.shape[1])],
    )
    axis.set_yticks(
        np.arange(len(clients)), [f'client {client}' for client in clients]
    )
    threshold = matrix.max(initial=0) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha='center',
                va='center',
                color='white' if matrix[row, column] > threshold else 'black',
                fontsize=8,
            )
    axis.set_xlabel('Mode community')
    axis.set_ylabel('Client')
    axis.set_title('Client–community mode counts')
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save(figure, path, dpi)


def _plot_client_distribution(graph, path, dpi):
    matrix = np.asarray(graph['community_client_matrix'], dtype=np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    fractions = matrix / np.maximum(totals, 1.0)
    clients = list(graph['client_ids'])
    positions = np.arange(len(clients))
    bottom = np.zeros(len(clients), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(max(7.0, len(clients) * 0.72), 5.2))
    colors = _colors(np.arange(matrix.shape[1]), matrix.shape[1], 'tab10')
    for community in range(matrix.shape[1]):
        axis.bar(
            positions,
            fractions[:, community],
            bottom=bottom,
            color=colors[community],
            label=f'community {community}',
        )
        bottom += fractions[:, community]
    axis.set_xticks(positions, [str(client) for client in clients])
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel('Client')
    axis.set_ylabel('Fraction of uploaded modes')
    axis.set_title('Per-client mode community composition')
    axis.legend(ncol=min(4, matrix.shape[1]), fontsize=8)
    _save(figure, path, dpi)


def _plot_functional_map_quality(graph, path, dpi):
    reports = list(graph.get('functional_map_diagnostics', []))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    if not reports:
        for axis in axes:
            axis.text(
                0.5, 0.5, 'Not available for this legacy graph',
                ha='center', va='center', transform=axis.transAxes,
            )
            axis.axis('off')
        _save(figure, path, dpi)
        return
    clients = [int(report['client_id']) for report in reports]
    positions = np.arange(len(clients))
    identity = np.asarray([
        max(float(report['identity_residual']), 1e-12)
        for report in reports
    ])
    aligned = np.asarray([
        max(float(report['intertwining_residual']), 1e-12)
        for report in reports
    ])
    orthogonality = np.asarray([
        max(float(report['orthogonality_error']), 1e-12)
        for report in reports
    ])
    width = 0.38
    axes[0].bar(
        positions - width / 2, identity, width,
        label='Identity (unaligned)', color='#9aa5b1',
    )
    axes[0].bar(
        positions + width / 2, aligned, width,
        label='Functional Map', color='#2463a8',
    )
    axes[0].set_yscale('log')
    axes[0].set_xticks(positions, [str(client) for client in clients])
    axes[0].set_xlabel('Client')
    axes[0].set_ylabel('Normalized intertwining residual (log)')
    axes[0].set_title('Dynamical consistency: C K_m vs K_s C')
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, orthogonality, color='#d17a22')
    axes[1].set_yscale('log')
    axes[1].set_xticks(positions, [str(client) for client in clients])
    axes[1].set_xlabel('Client')
    axes[1].set_ylabel('||C^H C - I|| / sqrt(d) (log)')
    axes[1].set_title('Functional Map orthogonality')
    _save(figure, path, dpi)


def _archived_graphs(log_path, latest, artifact_root=None):
    root = artifact_root or os.path.join(
        log_path, 'koopman', 'mode_graph'
    )
    by_round = {}
    for path in sorted(glob(os.path.join(root, 'round_*', 'mode_graph.pt'))):
        graph = _load(path)
        by_round[int(graph['round'])] = graph
    by_round[int(latest['round'])] = latest
    return [by_round[round_number] for round_number in sorted(by_round)]


def _mean_cross_client_similarity(graph):
    similarity = np.asarray(graph['similarity'], dtype=np.float64)
    clients = np.asarray(graph['client_id_per_node'])
    mask = (clients[:, None] != clients[None, :]) & (similarity > 0.0)
    return float(similarity[mask].mean()) if np.any(mask) else 0.0


def _plot_evolution(log_path, latest, path, dpi, artifact_root=None):
    graphs = _archived_graphs(
        log_path, latest, artifact_root=artifact_root
    )
    rounds = np.asarray([int(graph['round']) for graph in graphs])
    communities = np.asarray([
        int(graph['community_count']) for graph in graphs
    ])
    similarities = np.asarray([
        _mean_cross_client_similarity(graph) for graph in graphs
    ])
    figure, left = plt.subplots(figsize=(8.2, 4.8))
    left.plot(rounds, communities, marker='o', color='#1f77b4')
    left.set_xlabel('Training round (archived snapshots)')
    left.set_ylabel('Community count', color='#1f77b4')
    left.tick_params(axis='y', labelcolor='#1f77b4')
    left.set_ylim(bottom=0.0)
    right = left.twinx()
    right.plot(rounds, similarities, marker='s', color='#d62728')
    right.set_ylabel('Mean cross-client W', color='#d62728')
    right.tick_params(axis='y', labelcolor='#d62728')
    right.set_ylim(0.0, 1.0)
    left.set_title('Mode-community evolution')
    _save(figure, path, dpi)


def _write_dashboard(path, report):
    quality = report.get('cluster_quality', {})
    cards = [
        ('Round', report['round']),
        ('Modes', report['node_count']),
        ('Edges', report['edge_count']),
        ('Communities', report['community_count']),
        ('Clients', report['client_count']),
        (
            'Mean spatial edge W',
            f"{report['mean_cross_client_similarity']:.4f}",
        ),
        ('Reference client', report.get('reference_client_id', 'legacy')),
        (
            'Mean map residual',
            f"{report.get('mean_functional_map_residual', 0.0):.4f}",
        ),
        ('Silhouette', f"{quality.get('silhouette', 0.0):.4f}"),
        (
            'Between/within D',
            f"{quality.get('between_within_distance_ratio', 0.0):.4f}",
        ),
    ]
    figures = ''.join(
        '<section><h2>{}</h2><img src="{}" alt="{}"></section>'.format(
            html.escape(name.replace('_', ' ').title()),
            html.escape(os.path.basename(figure_path)),
            html.escape(name),
        )
        for name, figure_path in report['figures'].items()
    )
    card_html = ''.join(
        f'<div class="card"><span>{html.escape(str(name))}</span>'
        f'<strong>{html.escape(str(value))}</strong></div>'
        for name, value in cards
    )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FedDGN mode graph dashboard</title>
<style>
body {{ margin: 0 auto; max-width: 1280px; padding: 28px; font-family: system-ui,
sans-serif; color: #18212f; background: #f4f6f9; }}
h1 {{ margin-bottom: 4px; }} p {{ color: #526070; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
gap: 12px; margin: 22px 0; }}
.card, section {{ background: white; border-radius: 10px; padding: 16px;
box-shadow: 0 2px 10px #17203318; }}
.card span {{ display: block; color: #667085; font-size: 13px; }}
.card strong {{ display: block; margin-top: 6px; font-size: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(500px,1fr));
gap: 18px; }} section h2 {{ font-size: 17px; margin: 0 0 10px; }}
img {{ width: 100%; height: auto; display: block; }}
</style></head><body>
<h1>FedDGN mode graph</h1>
<p>Functional-Map-aligned comparison only: returned modes are not injected into task dynamics.</p>
<div class="cards">{card_html}</div><div class="grid">{figures}</div>
</body></html>'''
    temporary = f'{path}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        stream.write(document)
    os.replace(temporary, path)


def generate_mode_graph_visualizations(
    log_path: str,
    dpi: int = 160,
    graph_path: str = None,
    output_directory: str = None,
):
    """Render one server mode graph and archived-round evolution."""

    if int(dpi) <= 0:
        raise ValueError('Mode graph visualization dpi must be positive.')
    graph = load_latest_mode_graph(log_path, graph_path=graph_path)
    artifact_root = None
    if graph_path is not None:
        parent = os.path.dirname(os.path.abspath(graph_path))
        if os.path.basename(parent).startswith('round_'):
            artifact_root = os.path.dirname(parent)
        else:
            artifact_root = parent
    figures = output_directory or os.path.join(
        log_path, 'koopman', 'mode_graph', 'figures'
    )
    os.makedirs(figures, exist_ok=True)
    # This plot was intentionally retired: eigenvalues do not define graph
    # edges, and coloring the spectrum by Louvain community encouraged a
    # temporal interpretation that the spatial-only graph does not support.
    legacy_spectrum = os.path.join(figures, 'community_spectrum.png')
    if os.path.isfile(legacy_spectrum):
        os.remove(legacy_spectrum)
    paths = {
        'global_mode_graph': os.path.join(figures, 'global_mode_graph.png'),
        'mode_similarity_matrix': os.path.join(
            figures, 'mode_similarity_matrix.png'
        ),
        'client_community_matrix': os.path.join(
            figures, 'client_community_matrix.png'
        ),
        'client_community_distribution': os.path.join(
            figures, 'client_community_distribution.png'
        ),
        'functional_map_quality': os.path.join(
            figures, 'functional_map_quality.png'
        ),
        'community_evolution': os.path.join(
            figures, 'community_evolution.png'
        ),
    }
    _plot_global_graph(graph, paths['global_mode_graph'], dpi)
    _plot_similarity(graph, paths['mode_similarity_matrix'], dpi)
    _plot_client_community_matrix(
        graph, paths['client_community_matrix'], dpi
    )
    _plot_client_distribution(
        graph, paths['client_community_distribution'], dpi
    )
    _plot_functional_map_quality(
        graph, paths['functional_map_quality'], dpi
    )
    _plot_evolution(
        log_path,
        graph,
        paths['community_evolution'],
        dpi,
        artifact_root=artifact_root,
    )

    report = {
        'protocol': str(graph['protocol']),
        'round': int(graph['round']),
        'node_count': len(graph['nodes']),
        'edge_count': len(graph['edges']),
        'community_count': int(graph['community_count']),
        'client_count': len(graph['client_ids']),
        'mean_cross_client_similarity': _mean_cross_client_similarity(graph),
        'reference_client_id': graph.get('reference_client_id'),
        'mean_functional_map_residual': float(np.mean([
            float(item['intertwining_residual'])
            for item in graph.get('functional_map_diagnostics', [])
        ])) if graph.get('functional_map_diagnostics') else 0.0,
        'cluster_quality': dict(graph.get('cluster_quality', {})),
        'figures_directory': os.path.abspath(figures),
        'figures': {
            name: os.path.abspath(path) for name, path in paths.items()
        },
    }
    dashboard = os.path.join(figures, 'index.html')
    report['dashboard'] = os.path.abspath(dashboard)
    _write_dashboard(dashboard, report)
    summary = os.path.join(figures, 'summary.json')
    temporary = f'{summary}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, summary)
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Render a saved FedDGN server mode graph.'
    )
    parser.add_argument('log_path')
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--graph-path', type=str, default=None)
    parser.add_argument('--output-directory', type=str, default=None)
    args = parser.parse_args()
    report = generate_mode_graph_visualizations(
        args.log_path,
        dpi=args.dpi,
        graph_path=args.graph_path,
        output_directory=args.output_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()


__all__ = ['generate_mode_graph_visualizations', 'load_latest_mode_graph']
