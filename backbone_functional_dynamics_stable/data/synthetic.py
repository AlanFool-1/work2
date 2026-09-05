"""Controlled contextual-SBM benchmark for federated graph heterogeneity.

The benchmark intentionally separates three generative factors:

    Y_m ~ pi_m
    X_m | Y_m=c ~ N(mu_{m,c}, sigma_m^2 I)
    A_m | (Y_i,Y_j) ~ SBM(B_m)

Client 0 is always the reference distribution.  Client m uses severity
m/(M-1), so synthetic heterogeneity increases monotonically with client id.
The graph itself is generated with NetworkX's mature stochastic block model.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


SYNTHETIC_SCENARIOS = (
    'iid',
    'label_shift',
    'feature_shift',
    'feature_mixed',
    'structure_homophily',
    'structure_degree',
    'structure_mixed',
    'mixed',
)


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('Probability vector must be one-dimensional and non-empty.')
    if np.any(values < 0.0):
        raise ValueError('Probability vector cannot contain negative entries.')
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError('Probability vector must have a positive finite sum.')
    return values / total


def _default_skew_target(n_classes: int) -> np.ndarray:
    """A smooth long-tail target with no exactly missing class."""
    ranks = np.arange(int(n_classes), dtype=np.float64)
    raw = np.exp(-0.43 * ranks)
    # Keep every class observable in the default 1000-node setting.
    raw = raw + 0.045 * raw.max()
    return _normalize_probability(raw)


def _severity(client_id: int, n_clients: int) -> float:
    if int(n_clients) <= 1:
        return 0.0
    return float(client_id) / float(n_clients - 1)


def _scenario_flags(scenario: str) -> Dict[str, bool]:
    if scenario not in SYNTHETIC_SCENARIOS:
        raise ValueError(f'Unknown synthetic scenario: {scenario}')
    return {
        'label': scenario in {'label_shift', 'feature_mixed', 'mixed'},
        'feature': scenario in {'feature_shift', 'feature_mixed', 'mixed'},
        'homophily': scenario in {'structure_homophily', 'structure_mixed', 'mixed'},
        'degree': scenario in {'structure_degree', 'structure_mixed', 'mixed'},
    }


def _proportions_for_client(
    client_id: int,
    n_clients: int,
    n_classes: int,
    scenario: str,
) -> np.ndarray:
    flags = _scenario_flags(scenario)
    base = np.full(int(n_classes), 1.0 / float(n_classes), dtype=np.float64)
    if not flags['label']:
        return base
    lam = _severity(client_id, n_clients)
    target = _default_skew_target(n_classes)
    return _normalize_probability((1.0 - lam) * base + lam * target)


def _integer_class_counts(probabilities: np.ndarray, n_nodes: int) -> np.ndarray:
    probabilities = _normalize_probability(probabilities)
    raw = probabilities * int(n_nodes)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(n_nodes) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    if np.any(counts <= 0):
        raise ValueError(
            'Synthetic configuration produced an empty class. Increase nodes or '
            'reduce label skew.'
        )
    if int(counts.sum()) != int(n_nodes):
        raise AssertionError('Class count rounding failed to preserve node count.')
    return counts


def _solve_sbm_probabilities(
    counts: np.ndarray,
    target_avg_degree: float,
    target_homophily: float,
) -> Tuple[float, float]:
    """Solve p_in/p_out from expected degree and expected edge homophily."""
    counts = np.asarray(counts, dtype=np.float64)
    n_nodes = int(counts.sum())
    same_pairs = float(np.sum(counts * (counts - 1.0) / 2.0))
    all_pairs = float(n_nodes * (n_nodes - 1) / 2.0)
    cross_pairs = all_pairs - same_pairs
    if same_pairs <= 0.0 or cross_pairs <= 0.0:
        raise ValueError('SBM requires at least two non-empty blocks.')

    expected_edges = float(n_nodes) * float(target_avg_degree) / 2.0
    p_in = float(target_homophily) * expected_edges / same_pairs
    p_out = (1.0 - float(target_homophily)) * expected_edges / cross_pairs
    if not (0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0):
        raise ValueError(
            'Requested degree/homophily is infeasible for the current class sizes: '
            f'p_in={p_in:.4f}, p_out={p_out:.4f}. '
            'Reduce avg degree or use a less extreme homophily.'
        )
    return p_in, p_out


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    p = _normalize_probability(first)
    q = _normalize_probability(second)
    mix = 0.5 * (p + q)
    eps = np.finfo(np.float64).tiny
    kl_pm = float(np.sum(p * (np.log(np.maximum(p, eps)) - np.log(mix))))
    kl_qm = float(np.sum(q * (np.log(np.maximum(q, eps)) - np.log(mix))))
    return 0.5 * (kl_pm + kl_qm)


def _gaussian_w2_isotropic(
    base_means: np.ndarray,
    shifted_means: np.ndarray,
    base_std: float,
    shifted_std: float,
) -> float:
    mean_sq = np.sum((shifted_means - base_means) ** 2, axis=1)
    dim = int(base_means.shape[1])
    variance_term = dim * (float(shifted_std) - float(base_std)) ** 2
    per_class_sq = mean_sq + variance_term
    return float(np.sqrt(np.mean(per_class_sq)))


def _base_class_means(seed: int, n_classes: int, n_features: int, separation: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(int(n_classes), int(n_features)))
    directions -= directions.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    return float(separation) * directions


def _feature_parameters_for_client(
    client_id: int,
    n_clients: int,
    scenario: str,
    base_means: np.ndarray,
    base_std: float,
    mean_shift: float,
    std_shift: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    flags = _scenario_flags(scenario)
    if not flags['feature'] or int(client_id) == 0:
        return base_means.copy(), float(base_std)

    lam = _severity(client_id, n_clients)
    rng = np.random.default_rng(int(seed) + 7919 * int(client_id))
    directions = rng.normal(size=base_means.shape)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    shifted = base_means + lam * float(mean_shift) * directions
    shifted_std = float(base_std) * (1.0 + lam * float(std_shift))
    return shifted, shifted_std


def _structure_targets_for_client(
    client_id: int,
    n_clients: int,
    scenario: str,
    base_degree: float,
    max_degree_delta: float,
    base_homophily: float,
    max_homophily_delta: float,
) -> Tuple[float, float]:
    flags = _scenario_flags(scenario)
    lam = _severity(client_id, n_clients)
    degree = float(base_degree)
    homophily = float(base_homophily)
    if flags['degree']:
        degree = degree + lam * float(max_degree_delta)
    if flags['homophily']:
        homophily = homophily - lam * float(max_homophily_delta)
    return degree, homophily


def _stratified_masks(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    train = np.zeros(labels.size, dtype=bool)
    val = np.zeros(labels.size, dtype=bool)
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        count = int(indices.size)
        n_train = max(1, int(round(count * float(train_ratio))))
        n_val = max(1, int(round(count * float(val_ratio))))
        if n_train + n_val >= count:
            n_val = max(1, count - n_train - 1)
        if n_train + n_val >= count:
            n_train = max(1, count - n_val - 1)
        train[indices[:n_train]] = True
        val[indices[n_train:n_train + n_val]] = True
    test = ~(train | val)
    if not train.any() or not val.any() or not test.any():
        raise RuntimeError('Synthetic split produced an empty train/val/test mask.')
    return (
        torch.from_numpy(train),
        torch.from_numpy(val),
        torch.from_numpy(test),
    )


def _networkx_sbm_edge_index(
    counts: np.ndarray,
    p_in: float,
    p_out: float,
    seed: int,
) -> torch.Tensor:
    import networkx as nx

    n_classes = int(len(counts))
    probabilities = [
        [float(p_in if row == col else p_out) for col in range(n_classes)]
        for row in range(n_classes)
    ]
    graph = nx.stochastic_block_model(
        sizes=[int(value) for value in counts],
        p=probabilities,
        seed=int(seed),
        directed=False,
        selfloops=False,
        sparse=True,
    )
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    reverse = edges[:, ::-1]
    directed_edges = np.concatenate([edges, reverse], axis=0)
    return torch.from_numpy(directed_edges.T.copy()).long().contiguous()


def _actual_graph_metrics(edge_index: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    n_nodes = int(labels.numel())
    if edge_index.numel() == 0:
        return 0.0, 0.0, 0.0
    row, col = edge_index
    avg_degree = float(edge_index.size(1)) / float(n_nodes)
    same = labels[row] == labels[col]
    homophily = float(same.float().mean().item())
    degrees = torch.bincount(row, minlength=n_nodes).float()
    degree_cv = float(degrees.std(unbiased=False).item() / max(degrees.mean().item(), 1e-12))
    return avg_degree, homophily, degree_cv


def _empirical_centroid_shift(
    features: np.ndarray,
    labels: np.ndarray,
    base_means: np.ndarray,
) -> float:
    distances = []
    for class_id in range(base_means.shape[0]):
        mask = labels == class_id
        if not np.any(mask):
            continue
        centroid = features[mask].mean(axis=0)
        distances.append(float(np.linalg.norm(centroid - base_means[class_id])))
    return float(np.mean(distances)) if distances else 0.0


def _configuration(args) -> dict:
    return {
        'scenario': str(args.synthetic_scenario),
        'seed': int(args.synthetic_seed),
        'n_clients': int(args.n_clients),
        'nodes_per_client': int(args.synthetic_nodes_per_client),
        'n_features': int(args.n_feat),
        'n_classes': int(args.n_clss),
        'train_ratio': float(args.synthetic_train_ratio),
        'val_ratio': float(args.synthetic_val_ratio),
        'class_separation': float(args.synthetic_class_separation),
        'feature_std': float(args.synthetic_feature_std),
        'feature_mean_shift': float(args.synthetic_feature_mean_shift),
        'feature_std_shift': float(args.synthetic_feature_std_shift),
        'base_degree': float(args.synthetic_base_degree),
        'max_degree_delta': float(args.synthetic_degree_delta),
        'base_homophily': float(args.synthetic_base_homophily),
        'max_homophily_delta': float(args.synthetic_homophily_delta),
    }


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def _dataset_is_current(root: Path, configuration: dict, n_clients: int, require_viz: bool) -> bool:
    metadata_path = root / 'metadata.json'
    if not metadata_path.is_file():
        return False
    try:
        with open(metadata_path, encoding='utf-8') as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get('configuration') != configuration:
        return False
    if any(not (root / f'partition_{client_id}.pt').is_file() for client_id in range(int(n_clients))):
        return False
    if require_viz and not (root / 'diagnostics' / '01_label_proportions.png').is_file():
        return False
    return True


def _safe_import_data_class():
    from torch_geometric.data import Data
    return Data


def _generate_visualizations(
    diagnostics_dir: Path,
    client_arrays: List[dict],
    summary_rows: List[dict],
    n_classes: int,
    seed: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - runtime dependency guard
        print(f'[synthetic] visualization skipped: matplotlib unavailable ({error})')
        return

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    client_ids = np.asarray([row['client_id'] for row in summary_rows])
    severities = np.asarray([row['severity'] for row in summary_rows], dtype=float)

    # 1) Label proportions: the most direct view of prior shift.
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(summary_rows), dtype=float)
    for class_id in range(int(n_classes)):
        values = np.asarray([row[f'class_{class_id}_ratio'] for row in summary_rows], dtype=float)
        ax.bar(client_ids, values, bottom=bottom, label=f'class {class_id}')
        bottom += values
    ax.set_xlabel('Client id (heterogeneity increases left to right)')
    ax.set_ylabel('Label proportion')
    ax.set_title('Synthetic client label distributions')
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(diagnostics_dir / '01_label_proportions.png', dpi=180)
    plt.close(fig)

    # Shared PCA basis from a bounded sample of all clients.
    rng = np.random.default_rng(int(seed) + 404)
    samples = []
    sample_clients = []
    sample_classes = []
    for record in client_arrays:
        x = record['x']
        y = record['y']
        take = min(300, x.shape[0])
        idx = rng.choice(x.shape[0], size=take, replace=False)
        samples.append(x[idx])
        sample_clients.extend([record['client_id']] * take)
        sample_classes.extend(y[idx].tolist())
    matrix = np.concatenate(samples, axis=0)
    matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(matrix_centered, full_matrices=False)
    coords = matrix_centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=np.asarray(sample_clients), s=8, alpha=0.45)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Feature PCA colored by client')
    fig.colorbar(scatter, ax=ax, label='Client id')
    fig.tight_layout()
    fig.savefig(diagnostics_dir / '02_feature_pca_by_client.png', dpi=180)
    plt.close(fig)

    # Client 0 versus maximally heterogeneous client, colored by class.
    first = client_arrays[0]
    last = client_arrays[-1]
    pair_x = np.concatenate([first['x'], last['x']], axis=0)
    pair_y = np.concatenate([first['y'], last['y']], axis=0)
    pair_client = np.concatenate([
        np.zeros(first['x'].shape[0], dtype=int),
        np.ones(last['x'].shape[0], dtype=int),
    ])
    pair_centered = pair_x - matrix.mean(axis=0, keepdims=True)
    pair_coords = pair_centered @ vt[:2].T
    fig, ax = plt.subplots(figsize=(8, 6))
    for marker_client, marker in [(0, 'o'), (1, 'x')]:
        mask = pair_client == marker_client
        ax.scatter(
            pair_coords[mask, 0],
            pair_coords[mask, 1],
            c=pair_y[mask],
            marker=marker,
            s=12,
            alpha=0.45,
            label=('client 0' if marker_client == 0 else f'client {len(client_arrays)-1}'),
        )
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Reference vs strongest-shift client, colored by class')
    ax.legend()
    fig.tight_layout()
    fig.savefig(diagnostics_dir / '03_feature_pca_reference_vs_last.png', dpi=180)
    plt.close(fig)

    # Class-centroid displacement is much less cluttered than raw PCA points.
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for class_id in range(int(n_classes)):
        first_mask = first['y'] == class_id
        last_mask = last['y'] == class_id
        first_centroid = (first['x'][first_mask].mean(axis=0) - matrix.mean(axis=0)) @ vt[:2].T
        last_centroid = (last['x'][last_mask].mean(axis=0) - matrix.mean(axis=0)) @ vt[:2].T
        line = ax.plot(
            [first_centroid[0], last_centroid[0]],
            [first_centroid[1], last_centroid[1]],
            marker='o',
            label=f'class {class_id}',
        )[0]
        ax.annotate(
            '',
            xy=(last_centroid[0], last_centroid[1]),
            xytext=(first_centroid[0], first_centroid[1]),
            arrowprops={'arrowstyle': '->', 'color': line.get_color(), 'alpha': 0.7},
        )
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Class-centroid displacement: client 0 to strongest-shift client')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagnostics_dir / '03b_feature_class_centroid_shift.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(severities, [row['actual_avg_degree'] for row in summary_rows], marker='o')
    ax.set_xlabel('Ground-truth severity')
    ax.set_ylabel('Average degree')
    ax.set_title('Structural heterogeneity: average degree')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagnostics_dir / '04_average_degree.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(severities, [row['actual_homophily'] for row in summary_rows], marker='o')
    ax.set_xlabel('Ground-truth severity')
    ax.set_ylabel('Edge homophily')
    ax.set_title('Structural heterogeneity: homophily')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagnostics_dir / '05_homophily.png', dpi=180)
    plt.close(fig)

    # Normalize heterogeneous components only for a single readable severity overview.
    component_names = [
        ('label_js_to_client0', 'Label JS'),
        ('gaussian_w2_to_client0', 'Class-cond. Gaussian W2'),
        ('degree_abs_delta_to_client0', 'Degree shift'),
        ('homophily_abs_delta_to_client0', 'Homophily shift'),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for key, label in component_names:
        values = np.asarray([row[key] for row in summary_rows], dtype=float)
        maximum = float(values.max())
        normalized = values / maximum if maximum > 0.0 else values
        ax.plot(severities, normalized, marker='o', label=label)
    ax.set_xlabel('Ground-truth severity')
    ax.set_ylabel('Normalized heterogeneity component')
    ax.set_title('Controlled heterogeneity relative to client 0')
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagnostics_dir / '06_heterogeneity_components.png', dpi=180)
    plt.close(fig)

    # Adjacency matrices sorted by class make SBM changes immediately visible.
    for record, suffix in [(first, 'client0'), (last, f'client{len(client_arrays)-1}')]:
        n_nodes = int(record['x'].shape[0])
        edge = record['edge_index']
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
        if edge.size:
            adjacency[edge[0], edge[1]] = 1
        order = np.argsort(record['y'], kind='stable')
        adjacency = adjacency[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.imshow(adjacency, interpolation='nearest', aspect='equal')
        ax.set_xlabel('Nodes sorted by class')
        ax.set_ylabel('Nodes sorted by class')
        ax.set_title(f'Adjacency pattern: {suffix}')
        fig.tight_layout()
        fig.savefig(diagnostics_dir / f'07_adjacency_{suffix}.png', dpi=180)
        plt.close(fig)


def ensure_synthetic_dataset(args) -> None:
    """Generate or reuse the controlled synthetic federated benchmark."""
    if str(args.dataset) != 'Synthetic':
        return
    if str(args.synthetic_scenario) not in SYNTHETIC_SCENARIOS:
        raise ValueError(f'Unknown synthetic scenario: {args.synthetic_scenario}')
    if int(args.n_clss) != 8 or int(args.n_feat) != 128:
        raise ValueError('Current Synthetic benchmark is fixed to 8 classes and 128 features.')

    root = (
        Path(args.data_path)
        / f'{args.data_name}_{args.mode}'
        / str(args.n_clients)
    )
    diagnostics_dir = root / 'diagnostics'
    configuration = _configuration(args)
    args.synthetic_dataset_root = str(root)
    args.synthetic_summary_path = str(root / 'synthetic_summary.csv')
    args.synthetic_diagnostics_dir = str(diagnostics_dir)

    require_viz = not bool(args.synthetic_no_viz)
    if (
        not bool(args.synthetic_regenerate)
        and _dataset_is_current(root, configuration, args.n_clients, require_viz)
    ):
        print(
            f'[synthetic] reuse scenario={args.synthetic_scenario} '
            f'root={root}'
        )
        return

    print(
        f'[synthetic] generate scenario={args.synthetic_scenario}, '
        f'clients={args.n_clients}, nodes/client={args.synthetic_nodes_per_client}, '
        f'features={args.n_feat}, classes={args.n_clss}'
    )
    root.mkdir(parents=True, exist_ok=True)
    Data = _safe_import_data_class()

    n_clients = int(args.n_clients)
    n_nodes = int(args.synthetic_nodes_per_client)
    n_classes = int(args.n_clss)
    n_features = int(args.n_feat)
    seed = int(args.synthetic_seed)
    base_means = _base_class_means(
        seed=seed + 17,
        n_classes=n_classes,
        n_features=n_features,
        separation=float(args.synthetic_class_separation),
    )
    base_probabilities = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    base_std = float(args.synthetic_feature_std)

    summary_rows: List[dict] = []
    client_arrays: List[dict] = []
    reference_actual_degree = None
    reference_actual_homophily = None

    for client_id in range(n_clients):
        lam = _severity(client_id, n_clients)
        probabilities = _proportions_for_client(
            client_id, n_clients, n_classes, args.synthetic_scenario
        )
        counts = _integer_class_counts(probabilities, n_nodes)
        labels_np = np.repeat(np.arange(n_classes, dtype=np.int64), counts)

        means, feature_std = _feature_parameters_for_client(
            client_id=client_id,
            n_clients=n_clients,
            scenario=args.synthetic_scenario,
            base_means=base_means,
            base_std=base_std,
            mean_shift=float(args.synthetic_feature_mean_shift),
            std_shift=float(args.synthetic_feature_std_shift),
            seed=seed + 101,
        )
        rng = np.random.default_rng(seed + 1009 * (client_id + 1))
        features_np = np.empty((n_nodes, n_features), dtype=np.float32)
        offset = 0
        for class_id, count in enumerate(counts.tolist()):
            block = rng.normal(
                loc=means[class_id],
                scale=feature_std,
                size=(int(count), n_features),
            )
            features_np[offset:offset + count] = block.astype(np.float32)
            offset += count

        target_degree, target_homophily = _structure_targets_for_client(
            client_id=client_id,
            n_clients=n_clients,
            scenario=args.synthetic_scenario,
            base_degree=float(args.synthetic_base_degree),
            max_degree_delta=float(args.synthetic_degree_delta),
            base_homophily=float(args.synthetic_base_homophily),
            max_homophily_delta=float(args.synthetic_homophily_delta),
        )
        p_in, p_out = _solve_sbm_probabilities(
            counts=counts,
            target_avg_degree=target_degree,
            target_homophily=target_homophily,
        )
        edge_index = _networkx_sbm_edge_index(
            counts=counts,
            p_in=p_in,
            p_out=p_out,
            seed=seed + 2003 * (client_id + 1),
        )
        labels = torch.from_numpy(labels_np).long()
        train_mask, val_mask, test_mask = _stratified_masks(
            labels=labels_np,
            train_ratio=float(args.synthetic_train_ratio),
            val_ratio=float(args.synthetic_val_ratio),
            seed=seed + 3001 * (client_id + 1),
        )
        actual_degree, actual_homophily, degree_cv = _actual_graph_metrics(edge_index, labels)
        if client_id == 0:
            reference_actual_degree = actual_degree
            reference_actual_homophily = actual_homophily

        gaussian_w2 = _gaussian_w2_isotropic(
            base_means=base_means,
            shifted_means=means,
            base_std=base_std,
            shifted_std=feature_std,
        )
        label_js = _jensen_shannon(probabilities, base_probabilities)
        empirical_centroid_shift = _empirical_centroid_shift(
            features_np, labels_np, base_means
        )

        data = Data(
            x=torch.from_numpy(features_np).float(),
            y=labels,
            edge_index=edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        data.client_id = torch.tensor([client_id], dtype=torch.long)
        data.heterogeneity_severity = torch.tensor([lam], dtype=torch.float32)
        data.target_avg_degree = torch.tensor([target_degree], dtype=torch.float32)
        data.target_homophily = torch.tensor([target_homophily], dtype=torch.float32)
        data.label_js_to_client0 = torch.tensor([label_js], dtype=torch.float32)
        data.gaussian_w2_to_client0 = torch.tensor([gaussian_w2], dtype=torch.float32)

        _atomic_torch_save(
            root / f'partition_{client_id}.pt',
            {
                'client_data': data,
                'client_id': client_id,
                'synthetic_scenario': str(args.synthetic_scenario),
                'severity': lam,
            },
        )

        row = {
            'client_id': client_id,
            'severity': lam,
            'label_js_to_client0': label_js,
            'gaussian_w2_to_client0': gaussian_w2,
            'empirical_centroid_shift_to_client0_model': empirical_centroid_shift,
            'target_avg_degree': target_degree,
            'actual_avg_degree': actual_degree,
            'target_homophily': target_homophily,
            'actual_homophily': actual_homophily,
            'degree_cv': degree_cv,
            'p_in': p_in,
            'p_out': p_out,
            'feature_std': feature_std,
        }
        for class_id in range(n_classes):
            row[f'class_{class_id}_count'] = int(counts[class_id])
            row[f'class_{class_id}_ratio'] = float(counts[class_id] / n_nodes)
        summary_rows.append(row)
        client_arrays.append({
            'client_id': client_id,
            'x': features_np,
            'y': labels_np,
            'edge_index': edge_index.cpu().numpy(),
        })
        print(
            f'[synthetic] client={client_id:02d} severity={lam:.3f} '
            f'label_JS={label_js:.4f} gaussian_W2={gaussian_w2:.4f} '
            f'degree={actual_degree:.2f} homophily={actual_homophily:.3f}'
        )

    for row in summary_rows:
        row['degree_abs_delta_to_client0'] = abs(
            float(row['actual_avg_degree']) - float(reference_actual_degree)
        )
        row['homophily_abs_delta_to_client0'] = abs(
            float(row['actual_homophily']) - float(reference_actual_homophily)
        )

    _write_csv(root / 'synthetic_summary.csv', summary_rows)
    metadata = {
        'generator': 'contextual stochastic block model (Gaussian features + NetworkX SBM)',
        'reference_client': 0,
        'configuration': configuration,
        'summary_file': str(root / 'synthetic_summary.csv'),
        'diagnostics_dir': str(diagnostics_dir),
    }
    _write_json(root / 'metadata.json', metadata)

    if require_viz:
        _generate_visualizations(
            diagnostics_dir=diagnostics_dir,
            client_arrays=client_arrays,
            summary_rows=summary_rows,
            n_classes=n_classes,
            seed=seed,
        )
        print(f'[synthetic] diagnostics={diagnostics_dir}')


def plot_synthetic_run_diagnostics(args, client_best: Iterable[dict], log_path: str) -> None:
    """Plot final accuracy against the known synthetic heterogeneity severity."""
    if str(getattr(args, 'dataset', '')) != 'Synthetic':
        return
    summary_path = Path(getattr(args, 'synthetic_summary_path', ''))
    if not summary_path.is_file():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f'[synthetic] run visualization skipped: matplotlib unavailable ({error})')
        return

    with open(summary_path, newline='', encoding='utf-8') as stream:
        summary_rows = list(csv.DictReader(stream))
    best_by_client = {int(row['client_id']): row for row in client_best}
    merged = []
    for row in summary_rows:
        client_id = int(row['client_id'])
        if client_id not in best_by_client:
            continue
        best = best_by_client[client_id]
        merged.append({
            'client_id': client_id,
            'severity': float(row['severity']),
            'label_js': float(row['label_js_to_client0']),
            'feature_w2': float(row['gaussian_w2_to_client0']),
            'degree_shift': float(row['degree_abs_delta_to_client0']),
            'homophily_shift': float(row['homophily_abs_delta_to_client0']),
            'test_accuracy': float(best['test_accuracy']),
            'val_accuracy': float(best['val_accuracy']),
            'best_round': int(best['best_round']),
        })
    if not merged:
        return

    output = Path(log_path)
    _write_csv(output / 'synthetic_performance_summary.csv', merged)

    severity = np.asarray([row['severity'] for row in merged])
    accuracy = np.asarray([row['test_accuracy'] for row in merged])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(severity, accuracy, marker='o')
    for row in merged:
        ax.annotate(
            str(row['client_id']),
            (row['severity'], row['test_accuracy']),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=8,
        )
    ax.set_xlabel('Ground-truth heterogeneity severity')
    ax.set_ylabel('Paired best-validation test accuracy')
    ax.set_title('Model performance vs synthetic heterogeneity')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / 'synthetic_accuracy_vs_severity.png', dpi=180)
    plt.close(fig)

    # A second plot exposes feature-side behavior directly when that component is active.
    feature_w2 = np.asarray([row['feature_w2'] for row in merged])
    if float(feature_w2.max()) > 0.0:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(feature_w2, accuracy)
        for row in merged:
            ax.annotate(
                str(row['client_id']),
                (row['feature_w2'], row['test_accuracy']),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=8,
            )
        ax.set_xlabel('Ground-truth class-conditional Gaussian W2 to client 0')
        ax.set_ylabel('Test accuracy')
        ax.set_title('Performance vs feature heterogeneity')
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / 'synthetic_accuracy_vs_feature_w2.png', dpi=180)
        plt.close(fig)
