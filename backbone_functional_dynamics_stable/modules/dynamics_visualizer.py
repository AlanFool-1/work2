"""Interactive joint-t-SNE visualization of every official A-DGN state."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from modules.initial_state_visualizer import _stratified_sample_indices


def save_dynamics_snapshot(log_dir, client_id, states, labels, severity=None):
    raw_dir = Path(log_dir) / 'dynamics' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    hidden = states.detach().cpu().numpy()
    targets = labels.detach().cpu().numpy().reshape(-1)
    if hidden.ndim != 3 or hidden.shape[1] != targets.shape[0]:
        raise ValueError('Dynamics states must be [steps, nodes, hidden_dim].')
    with tempfile.NamedTemporaryFile(
        dir=raw_dir, suffix='.npz', delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            states=hidden,
            y=targets,
            severity=np.nan if severity is None else float(severity),
            client_id=int(client_id),
        )
        os.replace(temporary, raw_dir / f'client_{int(client_id)}.npz')
    finally:
        temporary.unlink(missing_ok=True)


def load_dynamics_snapshots(log_dir, n_clients):
    raw_dir = Path(log_dir) / 'dynamics' / 'raw'
    snapshots = []
    expected_steps = None
    for client_id in range(int(n_clients)):
        path = raw_dir / f'client_{client_id}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing dynamics snapshot: {path}')
        with np.load(path) as item:
            states = item['states'].copy()
            if expected_steps is None:
                expected_steps = states.shape[0]
            if states.shape[0] != expected_steps:
                raise ValueError('Clients have inconsistent dynamics step counts.')
            snapshots.append({
                'client_id': client_id,
                'states': states,
                'y': item['y'].copy(),
                'severity': float(item['severity']),
            })
    return snapshots


def compute_joint_dynamics_tsne(
    snapshots, max_points_per_client=60, random_state=0, perplexity=30.0
):
    """Fit one t-SNE over all clients and steps using fixed node samples."""
    rng = np.random.default_rng(int(random_state))
    sampled = []
    flat_parts = []
    for item in snapshots:
        indices = _stratified_sample_indices(
            item['y'], max_points_per_client, rng
        )
        states = item['states'][:, indices, :]
        sampled.append({
            **item, 'states': states, 'y': item['y'][indices],
            'sample_indices': indices,
        })
        flat_parts.append(states.reshape(-1, states.shape[-1]))
    matrix = np.concatenate(flat_parts, axis=0)
    if matrix.shape[1] > 30:
        matrix = PCA(n_components=30, random_state=int(random_state)).fit_transform(
            matrix
        )
    effective_perplexity = min(
        float(perplexity), max(1.0, float(len(matrix) - 1) / 3.0)
    )
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate='auto',
        init='pca',
        random_state=int(random_state),
    ).fit_transform(matrix)
    offset = 0
    for item in sampled:
        step_count, node_count = item['states'].shape[:2]
        count = step_count * node_count
        item['z'] = embedding[offset:offset + count].reshape(
            step_count, node_count, 2
        )
        offset += count
    return sampled


def _align_embedding(reference, moving):
    """Similarity-align a t-SNE map using identical nodes as landmarks."""
    reference_center = reference.mean(axis=0, keepdims=True)
    moving_center = moving.mean(axis=0, keepdims=True)
    reference_zero = reference - reference_center
    moving_zero = moving - moving_center
    left, _, right = np.linalg.svd(moving_zero.T @ reference_zero)
    rotation = left @ right
    rotated = moving_zero @ rotation
    denominator = float(np.sum(rotated ** 2))
    scale = (
        float(np.sum(rotated * reference_zero)) / denominator
        if denominator > 0.0 else 1.0
    )
    return rotated * scale + reference_center


def compute_aligned_dynamics_tsne(
    snapshots, max_points_per_client=60, random_state=0, perplexity=30.0
):
    """Fit each state separately, then align maps through fixed-node landmarks."""
    rng = np.random.default_rng(int(random_state))
    sampled = []
    for item in snapshots:
        indices = _stratified_sample_indices(
            item['y'], max_points_per_client, rng
        )
        sampled.append({
            **item,
            'states': item['states'][:, indices, :],
            'y': item['y'][indices],
            'sample_indices': indices,
        })
    step_count = sampled[0]['states'].shape[0]
    node_counts = [item['states'].shape[1] for item in sampled]
    total_nodes = sum(node_counts)
    effective_perplexity = min(
        float(perplexity), max(1.0, float(total_nodes - 1) / 3.0)
    )
    embeddings = []
    for step in range(step_count):
        matrix = np.concatenate(
            [item['states'][step] for item in sampled], axis=0
        )
        if matrix.shape[1] > 30:
            matrix = PCA(
                n_components=30, random_state=int(random_state)
            ).fit_transform(matrix)
        embedding = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate='auto',
            init='pca',
            random_state=int(random_state),
        ).fit_transform(matrix)
        if embeddings:
            embedding = _align_embedding(embeddings[0], embedding)
        embeddings.append(embedding)
    for item in sampled:
        item['z'] = np.empty(
            (step_count, item['states'].shape[1], 2), dtype=np.float32
        )
    for step, embedding in enumerate(embeddings):
        offset = 0
        for item, count in zip(sampled, node_counts):
            item['z'][step] = embedding[offset:offset + count]
            offset += count
    return sampled


def _trace(item, step, n_classes):
    points = item['z'][step]
    labels = item['y']
    return go.Scattergl(
        x=points[:, 0], y=points[:, 1], mode='markers',
        customdata=np.stack([labels, item['sample_indices']], axis=1),
        hovertemplate='class=%{customdata[0]}<br>node=%{customdata[1]}<extra></extra>',
        marker={
            'size': 6, 'opacity': 0.72, 'color': labels,
            'colorscale': 'Turbo', 'cmin': 0, 'cmax': int(n_classes) - 1,
            'showscale': item['client_id'] == 0,
            'colorbar': {'title': 'Class', 'len': 0.45},
        },
        showlegend=False,
    )


def plot_dynamics_tsne_html(
    log_dir, n_clients, n_classes, max_points_per_client=60,
    random_state=0, perplexity=30.0,
):
    snapshots = load_dynamics_snapshots(log_dir, n_clients)
    projected = compute_aligned_dynamics_tsne(
        snapshots, max_points_per_client, random_state, perplexity
    )
    step_count = projected[0]['states'].shape[0]
    columns = min(5, len(projected))
    rows = int(np.ceil(len(projected) / columns))
    titles = [
        (
            f"Client {item['client_id']} · λ={item['severity']:.2f}"
            if np.isfinite(item['severity'])
            else f"Client {item['client_id']}"
        )
        for item in projected
    ]
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for index, item in enumerate(projected):
        figure.add_trace(
            _trace(item, 0, n_classes),
            row=index // columns + 1, col=index % columns + 1,
        )
    frames = []
    for step in range(step_count):
        frames.append(go.Frame(
            name=str(step),
            data=[_trace(item, step, n_classes) for item in projected],
            traces=list(range(len(projected))),
        ))
    figure.frames = frames
    slider_steps = [{
        'label': f'H{step}',
        'method': 'animate',
        'args': [[str(step)], {
            'mode': 'immediate',
            'frame': {'duration': 0, 'redraw': True},
            'transition': {'duration': 0},
        }],
    } for step in range(step_count)]
    figure.update_layout(
        title=(
            'A-DGN hidden-state evolution · per-state t-SNE aligned by fixed nodes'
        ),
        template='plotly_white', height=760, width=1500,
        sliders=[{'active': 0, 'currentvalue': {'prefix': 'State: '},
                  'steps': slider_steps, 'pad': {'t': 45}}],
        updatemenus=[{
            'type': 'buttons', 'direction': 'left', 'x': 0.0, 'y': -0.08,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 700, 'redraw': True},
                                  'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'mode': 'immediate',
                                    'frame': {'duration': 0, 'redraw': False}}]},
            ],
        }],
    )
    all_z = np.concatenate([item['z'].reshape(-1, 2) for item in projected])
    lower, upper = all_z.min(axis=0), all_z.max(axis=0)
    margin = np.maximum((upper - lower) * 0.03, 1e-6)
    figure.update_xaxes(range=[lower[0] - margin[0], upper[0] + margin[0]],
                        showticklabels=False)
    figure.update_yaxes(range=[lower[1] - margin[1], upper[1] + margin[1]],
                        showticklabels=False)
    out_dir = Path(log_dir) / 'dynamics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'adgn_hidden_dynamics_tsne.html'
    figure.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f'[dynamics] interactive_html={out_path}')
    return out_path
