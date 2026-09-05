"""Local-only snapshots and shared-embedding diagnostics for Graph ODE initial states."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_initial_state_snapshot(log_dir, client_id, h0, labels, stage='before', severity=None):
    raw_dir = Path(log_dir) / 'initial_state' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    hidden = h0.detach().cpu().numpy()
    targets = labels.detach().cpu().numpy().reshape(-1)
    if hidden.ndim != 2 or hidden.shape[0] != targets.shape[0]:
        raise ValueError('H0 must be [num_nodes, hidden_dim] and align with labels.')
    with tempfile.NamedTemporaryFile(dir=raw_dir, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            h=hidden,
            y=targets,
            severity=np.nan if severity is None else float(severity),
            client_id=int(client_id),
        )
        os.replace(temporary, raw_dir / f'client_{int(client_id)}_{stage}.npz')
    finally:
        temporary.unlink(missing_ok=True)


def load_initial_state_snapshots(log_dir, n_clients, stage='before'):
    raw_dir = Path(log_dir) / 'initial_state' / 'raw'
    snapshots = []
    for client_id in range(int(n_clients)):
        path = raw_dir / f'client_{client_id}_{stage}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing initial-state snapshot: {path}')
        with np.load(path) as item:
            snapshots.append({
                'client_id': client_id,
                'h': item['h'].copy(),
                'y': item['y'].copy(),
                'severity': float(item['severity']),
            })
    return snapshots


def compute_shared_pca(snapshots, n_components=2, max_points_per_client=None, random_state=0):
    rng = np.random.default_rng(int(random_state))
    fit_parts = []
    for item in snapshots:
        hidden = item['h']
        if max_points_per_client is not None and len(hidden) > int(max_points_per_client):
            indices = rng.choice(len(hidden), size=int(max_points_per_client), replace=False)
            hidden = hidden[indices]
        fit_parts.append(hidden)
    pca = PCA(n_components=int(n_components), random_state=int(random_state))
    pca.fit(np.concatenate(fit_parts, axis=0))
    projected = [{**item, 'z': pca.transform(item['h'])} for item in snapshots]
    return pca, projected


def _stratified_sample_indices(labels, max_points, rng):
    if max_points is None or len(labels) <= int(max_points):
        return np.arange(len(labels))
    classes = np.unique(labels)
    selected = []
    remaining = int(max_points)
    for position, class_id in enumerate(classes):
        candidates = np.flatnonzero(labels == class_id)
        classes_left = len(classes) - position
        count = min(len(candidates), max(1, remaining // classes_left))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
        remaining -= count
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(available, size=min(remaining, len(available)), replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def compute_shared_tsne(snapshots, max_points_per_client=300, random_state=0, perplexity=30.0):
    rng = np.random.default_rng(int(random_state))
    sampled = []
    hidden_parts = []
    for item in snapshots:
        indices = _stratified_sample_indices(item['y'], max_points_per_client, rng)
        sampled_item = {
            **item,
            'h': item['h'][indices],
            'y': item['y'][indices],
            'sample_indices': indices,
        }
        sampled.append(sampled_item)
        hidden_parts.append(sampled_item['h'])
    hidden = np.concatenate(hidden_parts, axis=0)
    effective_perplexity = min(float(perplexity), max(1.0, float(len(hidden) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate='auto',
        init='pca',
        random_state=int(random_state),
    ).fit_transform(hidden)
    projected = []
    offset = 0
    for item in sampled:
        count = len(item['h'])
        projected.append({**item, 'z': embedding[offset:offset + count]})
        offset += count
    return projected


def compute_distance_to_client0(snapshots, n_classes):
    reference = snapshots[0]
    reference_centroids = {}
    for class_id in range(int(n_classes)):
        mask = reference['y'] == class_id
        if mask.any():
            reference_centroids[class_id] = reference['h'][mask].mean(axis=0)
    rows = []
    for item in snapshots:
        distances = []
        for class_id, centroid in reference_centroids.items():
            mask = item['y'] == class_id
            if mask.any():
                distances.append(np.linalg.norm(item['h'][mask].mean(axis=0) - centroid))
        rows.append({
            'client_id': item['client_id'],
            'severity': item['severity'],
            'h0_centroid_distance_to_client0': float(np.mean(distances)) if distances else np.nan,
            'num_nodes': int(len(item['h'])),
        })
    return rows


def _plot_pca_grid(projected, out_path, n_classes):
    columns = min(5, len(projected))
    rows = int(np.ceil(len(projected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.2 * rows), squeeze=False, sharex=True, sharey=True)
    all_points = np.concatenate([item['z'] for item in projected], axis=0)
    lower = all_points.min(axis=0); upper = all_points.max(axis=0)
    margin = np.maximum((upper - lower) * 0.05, 1e-6)
    for axis, item in zip(axes.flat, projected):
        axis.scatter(item['z'][:, 0], item['z'][:, 1], c=item['y'], cmap='tab10', vmin=0, vmax=n_classes - 1, s=6, alpha=0.5)
        axis.set_title(f"Client {item['client_id']}  λ={item['severity']:.2f}")
        axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
        axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    for axis in axes.flat[len(projected):]:
        axis.axis('off')
    figure.tight_layout(); figure.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close(figure)


def _plot_tsne_grid(projected, out_path, n_classes):
    columns = min(5, len(projected))
    rows = int(np.ceil(len(projected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.0 * rows), squeeze=False, sharex=True, sharey=True)
    all_points = np.concatenate([item['z'] for item in projected], axis=0)
    lower = all_points.min(axis=0); upper = all_points.max(axis=0)
    margin = np.maximum((upper - lower) * 0.05, 1e-6)
    for axis, item in zip(axes.flat, projected):
        axis.scatter(item['z'][:, 0], item['z'][:, 1], c=item['y'], cmap='tab10', vmin=0, vmax=n_classes - 1, s=8, alpha=0.65, linewidths=0)
        axis.set_title(f"C{item['client_id']}  λ={item['severity']:.2f}", fontsize=10)
        axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
        axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
        axis.set_xticks([]); axis.set_yticks([])
    for axis in axes.flat[len(projected):]:
        axis.axis('off')
    figure.tight_layout(); figure.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close(figure)


def _plot_all_clients(projected, out_path):
    figure, axis = plt.subplots(figsize=(8, 6))
    for item in projected:
        axis.scatter(item['z'][:, 0], item['z'][:, 1], s=5, alpha=0.25, label=f"C{item['client_id']}")
    axis.set_xlabel('PC1'); axis.set_ylabel('PC2'); axis.legend(ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(); figure.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close(figure)


def _plot_centroid_trajectories(projected, out_path, n_classes):
    figure, axis = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap('tab10')
    for class_id in range(int(n_classes)):
        points = []
        for item in projected:
            mask = item['y'] == class_id
            if mask.any():
                points.append(item['z'][mask].mean(axis=0))
        if not points:
            continue
        points = np.asarray(points)
        axis.plot(points[:, 0], points[:, 1], marker='o', markersize=4, linewidth=1.2, color=colors(class_id), label=f'Class {class_id}')
    axis.set_xlabel('PC1'); axis.set_ylabel('PC2'); axis.legend(ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(); figure.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close(figure)


def _plot_distance(rows, out_path):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot([row['severity'] for row in rows], [row['h0_centroid_distance_to_client0'] for row in rows], marker='o')
    axis.set_xlabel('Severity'); axis.set_ylabel('H0 distance to client 0')
    figure.tight_layout(); figure.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close(figure)


def plot_initial_state_diagnostics(
    log_dir,
    n_clients,
    n_classes,
    stage='before',
    random_state=0,
    max_points_per_client=1000,
    tsne_max_points_per_client=300,
    tsne_perplexity=30.0,
):
    snapshots = load_initial_state_snapshots(log_dir, n_clients, stage)
    pca, projected = compute_shared_pca(snapshots, 2, max_points_per_client, random_state)
    out_dir = Path(log_dir) / 'initial_state'; out_dir.mkdir(parents=True, exist_ok=True)
    _plot_pca_grid(projected, out_dir / f'initial_state_{stage}_pca_grid.png', n_classes)
    tsne_projected = compute_shared_tsne(snapshots, tsne_max_points_per_client, random_state, tsne_perplexity)
    _plot_tsne_grid(tsne_projected, out_dir / f'initial_state_{stage}_tsne_grid.png', n_classes)
    _plot_all_clients(projected, out_dir / f'initial_state_{stage}_pca_all_clients.png')
    _plot_centroid_trajectories(projected, out_dir / f'initial_state_{stage}_class_centroid_trajectories.png', n_classes)
    rows = compute_distance_to_client0(snapshots, n_classes)
    _plot_distance(rows, out_dir / f'initial_state_{stage}_distance_to_client0.png')
    for row in rows:
        row['pca_explained_variance_pc1'] = float(pca.explained_variance_ratio_[0])
        row['pca_explained_variance_pc2'] = float(pca.explained_variance_ratio_[1])
    with (out_dir / f'initial_state_{stage}_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'[initial-state] diagnostics={out_dir}')
