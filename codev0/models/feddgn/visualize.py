"""Artifact-driven visualizations for client-local Koopman dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class ClientKoopman:
    client_id: int
    operator: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residuals: np.ndarray
    modes: np.ndarray
    metadata: dict


def _torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _atomic_json(path, payload):
    temporary = f'{path}.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def _client_id(name):
    try:
        return int(name.rsplit('_', 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f'Invalid Koopman client directory: {name}') from exc


def load_client_artifacts(log_path: str) -> List[ClientKoopman]:
    root = os.path.join(log_path, 'koopman')
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Koopman artifact directory not found: {root}')
    clients = []
    for name in sorted(
        (entry for entry in os.listdir(root) if entry.startswith('client_')),
        key=_client_id,
    ):
        directory = os.path.join(root, name)
        required = {
            'operator': 'K.pt',
            'eigenvalues': 'eigenvalues.pt',
            'eigenvectors': 'eigenvectors.pt',
            'residuals': 'residual.pt',
            'modes': 'modes.pt',
        }
        paths = {
            key: os.path.join(directory, filename)
            for key, filename in required.items()
        }
        missing = [path for path in paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f'Incomplete Koopman artifacts for {name}: {missing}'
            )
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, encoding='utf-8') as stream:
            metadata = json.load(stream)
        arrays = {
            key: _torch_load(path).detach().cpu().numpy()
            for key, path in paths.items()
        }
        clients.append(ClientKoopman(
            client_id=_client_id(name), metadata=metadata, **arrays
        ))
    if not clients:
        raise RuntimeError('No complete client Koopman artifacts were found.')
    observable_dims = {client.operator.shape for client in clients}
    state_dims = {client.modes.shape[0] for client in clients}
    if len(observable_dims) != 1 or len(state_dims) != 1:
        raise ValueError('Client Koopman artifacts have incompatible dimensions.')
    return clients


def _nontrivial_indices(client: ClientKoopman, top_modes: int) -> np.ndarray:
    count = client.eigenvalues.size
    candidates = np.arange(count)
    if count > 1:
        # Psi contains a fixed constant observable. Remove the eigenvalue
        # closest to one from comparisons so it cannot dominate every client.
        trivial = int(np.argmin(np.abs(client.eigenvalues - 1.0)))
        candidates = candidates[candidates != trivial]
    order = np.lexsort((
        -np.abs(client.eigenvalues[candidates]),
        client.residuals[candidates],
    ))
    return candidates[order[:min(int(top_modes), candidates.size)]]


def _save_figure(figure, path, dpi):
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def _plot_spectrum(clients, path, dpi):
    figure, axis = plt.subplots(figsize=(7.2, 6.4))
    angles = np.linspace(0.0, 2.0 * np.pi, 512)
    axis.plot(np.cos(angles), np.sin(angles), '--', color='0.6', linewidth=1)
    colors = plt.get_cmap('tab10')
    for index, client in enumerate(clients):
        values = client.eigenvalues
        axis.scatter(
            values.real,
            values.imag,
            s=24,
            alpha=0.75,
            color=colors(index % 10),
            label=f'client {client.client_id}',
        )
    axis.axhline(0.0, color='0.8', linewidth=0.8)
    axis.axvline(0.0, color='0.8', linewidth=0.8)
    axis.set_xlabel('Re(lambda)')
    axis.set_ylabel('Im(lambda)')
    axis.set_title('Client Koopman spectra')
    axis.set_aspect('equal', adjustable='datalim')
    axis.legend(ncol=2, fontsize=8)
    _save_figure(figure, path, dpi)


def _plot_magnitudes(clients, path, dpi):
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for client in clients:
        magnitudes = np.sort(np.abs(client.eigenvalues))[::-1]
        axis.plot(
            np.arange(1, magnitudes.size + 1),
            magnitudes,
            marker='.',
            linewidth=1,
            label=f'client {client.client_id}',
        )
    axis.axhline(1.0, color='0.4', linestyle='--', linewidth=1)
    axis.set_xlabel('Mode rank')
    axis.set_ylabel('|lambda|')
    axis.set_title('Koopman eigenvalue magnitude / decay')
    axis.legend(ncol=2, fontsize=8)
    _save_figure(figure, path, dpi)


def _plot_residuals(clients, path, dpi):
    ids = [client.client_id for client in clients]
    fit = [
        float(client.metadata['diagnostics']['koopman_fit_residual'])
        for client in clients
    ]
    spectral = [
        float(client.metadata['diagnostics']['koopman_spectral_residual'])
        for client in clients
    ]
    positions = np.arange(len(clients))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(positions - width / 2, fit, width, label='EDMD fit')
    axis.bar(positions + width / 2, spectral, width, label='spectral')
    axis.set_xticks(positions, [str(value) for value in ids])
    axis.set_xlabel('Client')
    axis.set_ylabel('Relative residual')
    axis.set_title('Koopman reconstruction and spectral residuals')
    axis.legend()
    _save_figure(figure, path, dpi)


def _descriptor(client, top_modes):
    indices = _nontrivial_indices(client, top_modes)
    values = client.eigenvalues[indices]
    residuals = client.residuals[indices]
    features = np.stack(
        [values.real, values.imag, np.abs(values), residuals], axis=1
    ).reshape(-1)
    target = int(top_modes) * 4
    if features.size < target:
        features = np.pad(features, (0, target - features.size))
    return features


def _pca_embedding(clients, top_modes):
    matrix = np.stack([
        _descriptor(client, top_modes) for client in clients
    ])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    normalized = centered / np.where(scale > 1e-12, scale, 1.0)
    if len(clients) == 1 or not np.any(normalized):
        return np.zeros((len(clients), 2)), [0.0, 0.0]
    # This SVD is only over the tiny client descriptor matrix (normally
    # 10 x <=64), never over a graph, Laplacian, or node-feature matrix.
    left, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    dimensions = min(2, singular.size)
    embedding = np.zeros((len(clients), 2))
    embedding[:, :dimensions] = left[:, :dimensions] * singular[:dimensions]
    variance = np.square(singular)
    ratios = variance / max(float(variance.sum()), np.finfo(float).eps)
    explained = np.zeros(2)
    explained[:dimensions] = ratios[:dimensions]
    return embedding, explained.tolist()


def _plot_embedding(clients, embedding, explained, path, dpi):
    figure, axis = plt.subplots(figsize=(7.0, 5.6))
    axis.scatter(embedding[:, 0], embedding[:, 1], s=70)
    for point, client in zip(embedding, clients):
        axis.annotate(
            f'client {client.client_id}',
            (point[0], point[1]),
            xytext=(5, 4),
            textcoords='offset points',
            fontsize=8,
        )
    axis.set_xlabel(f'PC1 ({100.0 * explained[0]:.1f}%)')
    axis.set_ylabel(f'PC2 ({100.0 * explained[1]:.1f}%)')
    axis.set_title('Client Koopman embedding')
    _save_figure(figure, path, dpi)


def _selected_modes(client, top_modes):
    indices = _nontrivial_indices(client, top_modes)
    modes = client.modes[:, indices].T
    norms = np.linalg.norm(modes, axis=1, keepdims=True)
    return modes / np.maximum(norms, np.finfo(float).eps)


def _mode_similarity(clients, top_modes):
    selected = [_selected_modes(client, top_modes) for client in clients]
    matrix = np.eye(len(clients), dtype=np.float64)
    for first in range(len(clients)):
        for second in range(first + 1, len(clients)):
            pair = np.abs(selected[first].conj() @ selected[second].T)
            forward = float(pair.max(axis=1).mean())
            backward = float(pair.max(axis=0).mean())
            matrix[first, second] = matrix[second, first] = (
                0.5 * (forward + backward)
            )
    return matrix


def _plot_similarity(clients, similarity, path, dpi):
    figure, axis = plt.subplots(figsize=(7.0, 6.2))
    image = axis.imshow(similarity, vmin=0.0, vmax=1.0, cmap='viridis')
    labels = [str(client.client_id) for client in clients]
    axis.set_xticks(np.arange(len(clients)), labels)
    axis.set_yticks(np.arange(len(clients)), labels)
    axis.set_xlabel('Client')
    axis.set_ylabel('Client')
    axis.set_title('State-space Koopman mode similarity')
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save_figure(figure, path, dpi)


def _plot_operator_heatmaps(clients, path, dpi):
    columns = min(4, len(clients))
    rows = int(math.ceil(len(clients) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 3.0 * rows), squeeze=False
    )
    maximum = max(float(np.abs(client.operator).max()) for client in clients)
    image = None
    for axis, client in zip(axes.flat, clients):
        image = axis.imshow(
            np.abs(client.operator), vmin=0.0, vmax=maximum, cmap='magma'
        )
        axis.set_title(f'client {client.client_id}')
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(clients):]:
        axis.axis('off')
    figure.suptitle('|K| heatmaps')
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def _plot_mode_energy(clients, top_modes, path, dpi):
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for client in clients:
        energy = np.square(np.linalg.norm(client.modes, axis=0))
        reliability = 1.0 / (1.0 + client.residuals)
        weighted = np.sort(energy * reliability)[::-1][:top_modes]
        total = max(float(weighted.sum()), np.finfo(float).eps)
        axis.plot(
            np.arange(1, weighted.size + 1),
            weighted / total,
            marker='.',
            linewidth=1,
            label=f'client {client.client_id}',
        )
    axis.set_xlabel('Mode energy rank')
    axis.set_ylabel('Normalized reliability-weighted energy')
    axis.set_title('Koopman mode energy distribution')
    axis.legend(ncol=2, fontsize=8)
    _save_figure(figure, path, dpi)


def _write_embedding(path, clients, embedding):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['client_id', 'pc1', 'pc2'])
        for client, point in zip(clients, embedding):
            writer.writerow([client.client_id, float(point[0]), float(point[1])])


def _write_similarity(path, clients, similarity):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['client_id'] + [client.client_id for client in clients])
        for client, row in zip(clients, similarity):
            writer.writerow([client.client_id] + [float(value) for value in row])


def generate_koopman_visualizations(
    log_path: str, top_modes: int = 16, dpi: int = 160
):
    """Generate all phase-one plots from saved, client-local artifacts."""

    if top_modes <= 0 or dpi <= 0:
        raise ValueError('top_modes and dpi must be positive.')
    clients = load_client_artifacts(log_path)
    available = min(client.eigenvalues.size - 1 for client in clients)
    top_modes = min(int(top_modes), max(1, int(available)))
    figures = os.path.join(log_path, 'koopman', 'figures')
    os.makedirs(figures, exist_ok=True)

    paths = {
        'spectrum': os.path.join(figures, 'koopman_spectrum.png'),
        'magnitude': os.path.join(figures, 'koopman_magnitude.png'),
        'residual': os.path.join(figures, 'koopman_residual.png'),
        'embedding': os.path.join(figures, 'koopman_embedding.png'),
        'mode_similarity': os.path.join(
            figures, 'koopman_mode_similarity.png'
        ),
        'operator_heatmap': os.path.join(
            figures, 'koopman_operator_heatmaps.png'
        ),
        'mode_energy': os.path.join(figures, 'koopman_mode_energy.png'),
    }
    _plot_spectrum(clients, paths['spectrum'], dpi)
    _plot_magnitudes(clients, paths['magnitude'], dpi)
    _plot_residuals(clients, paths['residual'], dpi)
    embedding, explained = _pca_embedding(clients, top_modes)
    _plot_embedding(clients, embedding, explained, paths['embedding'], dpi)
    similarity = _mode_similarity(clients, top_modes)
    _plot_similarity(clients, similarity, paths['mode_similarity'], dpi)
    _plot_operator_heatmaps(clients, paths['operator_heatmap'], dpi)
    _plot_mode_energy(clients, top_modes, paths['mode_energy'], dpi)

    _write_embedding(
        os.path.join(figures, 'client_embedding.csv'), clients, embedding
    )
    _write_similarity(
        os.path.join(figures, 'mode_similarity.csv'), clients, similarity
    )
    fit_residuals = [
        float(client.metadata['diagnostics']['koopman_fit_residual'])
        for client in clients
    ]
    spectral_residuals = [
        float(client.metadata['diagnostics']['koopman_spectral_residual'])
        for client in clients
    ]
    report = {
        'protocol': 'passive_dictionary_fedavg_local_operator_no_feedback',
        'n_clients': len(clients),
        'client_ids': [client.client_id for client in clients],
        'top_modes': top_modes,
        'pca_explained_variance_ratio': explained,
        'mean_fit_residual': float(np.mean(fit_residuals)),
        'mean_spectral_residual': float(np.mean(spectral_residuals)),
        'mean_off_diagonal_mode_similarity': (
            float(similarity[~np.eye(len(clients), dtype=bool)].mean())
            if len(clients) > 1 else 1.0
        ),
        'figures_directory': os.path.abspath(figures),
        'figures': {name: os.path.abspath(path) for name, path in paths.items()},
    }
    _atomic_json(os.path.join(figures, 'summary.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Render saved FedDGN client Koopman artifacts.'
    )
    parser.add_argument('log_path')
    parser.add_argument('--top-modes', type=int, default=16)
    parser.add_argument('--dpi', type=int, default=160)
    args = parser.parse_args()
    report = generate_koopman_visualizations(
        args.log_path, top_modes=args.top_modes, dpi=args.dpi
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()


__all__ = ['generate_koopman_visualizations', 'load_client_artifacts']
