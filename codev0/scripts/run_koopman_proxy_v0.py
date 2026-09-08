#!/usr/bin/env python3
"""Train and evaluate the local V0 graph Koopman proxy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from models.s0.model import build_s0_model
from proxy.graph_koopman_v0 import GraphKoopmanV0, compute_normalization


DATASET_META = {'Cora': (1433, 7), 'CiteSeer': (3703, 6)}
HORIZONS = (1, 4, 8, 16)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_backbone_args(dataset: str, base_path: Path) -> SimpleNamespace:
    n_feat, n_clss = DATASET_META[dataset]
    return SimpleNamespace(
        base_path=str(base_path),
        n_feat=n_feat,
        n_clss=n_clss,
        hidden_dim=64,
        ode_steps=16,
        adgn_step_size=0.1,
        ode_gamma=0.1,
        ode_activation='tanh',
    )


def load_graph(dataset: str, client_id: int, base_path: Path):
    path = base_path / 'datasets' / f'{dataset}_disjoint' / '10' / f'partition_{client_id}.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Client partition does not exist: {path}')
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    data = loaded['client_data']
    if getattr(data, 'edge_weight', None) is None:
        data.edge_weight = None
    return data, path


def find_checkpoint(dataset: str, client_id: int, explicit: str, base_path: Path) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint does not exist: {path}')
        return path
    root = base_path / 'checkpoints' / f'{dataset}_disjoint' / 'clients_10'
    candidates = sorted(root.glob(f'*/{client_id}_state.pt'), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f'No best checkpoint found under {root}. Run codev0/scripts/run_official_adgn_all11.sh {dataset} first.'
        )
    return candidates[-1]


def load_native(dataset: str, client_id: int, checkpoint: Path, base_path: Path, device: torch.device):
    data, partition_path = load_graph(dataset, client_id, base_path)
    args = build_backbone_args(dataset, base_path)
    model = build_s0_model(args).to(device)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict({key: torch.as_tensor(value) for key, value in payload['model'].items()}, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, data, partition_path


@torch.no_grad()
def native_trajectory(model, graph, features: Tensor, device: torch.device) -> Tuple[Tensor, Tensor]:
    sample = copy.copy(graph)
    sample.x = features.to(device)
    sample.edge_index = graph.edge_index.to(device)
    if getattr(graph, 'edge_weight', None) is not None:
        sample.edge_weight = graph.edge_weight.to(device)
    else:
        sample.edge_weight = None
    states = model.encode_native_dynamics_states(sample)
    logits = model.readout(states)
    return states.cpu(), logits.cpu()


@torch.no_grad()
def generate_bank(model, graph, count: int, seed: int, device: torch.device):
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    base_features = graph.x.detach().cpu().float()
    states = []
    logits = []
    for index in range(int(count)):
        alpha = torch.empty((), dtype=base_features.dtype).uniform_(0.01, 0.05, generator=generator).item()
        noise = torch.randn(base_features.shape, generator=generator)
        features = base_features * (1.0 + alpha * noise)
        state, output = native_trajectory(model, graph, features, device)
        states.append(state)
        logits.append(output)
    return torch.stack(states), torch.stack(logits)


def save_bank(path: Path, states: Tensor, logits: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'states': states, 'logits': logits}, temporary)
    temporary.replace(path)


def load_or_generate_bank(
    model,
    graph,
    cache_path: Path,
    count: int,
    seed: int,
    device: torch.device,
):
    if cache_path.is_file():
        saved = torch.load(cache_path, map_location='cpu', weights_only=False)
        if saved['states'].shape[0] == count:
            return saved['states'].float(), saved['logits'].float(), 'cache'
    states, logits = generate_bank(model, graph, count, seed, device)
    save_bank(cache_path, states, logits)
    return states, logits, 'generated'


def batch_indices(count: int, batch_size: int, generator: torch.Generator) -> Tensor:
    permutation = torch.randperm(count, generator=generator)
    return permutation[:min(int(batch_size), count)]


def select_horizon(step: int, warmup: int, total_steps: int, native_steps: int) -> int:
    if warmup > 0 and step < warmup:
        return min(1, native_steps)
    phase = max(0, step - warmup) / max(1, total_steps - warmup)
    if phase < 0.25:
        return min(1, native_steps)
    if phase < 0.5:
        return min(4, native_steps)
    if phase < 0.75:
        return min(8, native_steps)
    return native_steps


def metric_rows(
    proxy: GraphKoopmanV0,
    native_states: Tensor,
    native_logits: Tensor,
    base_states: Tensor,
    base_logits: Tensor,
    readout,
    condition: Tensor,
    neighborhood_condition: Tensor,
    edge_cache,
    device: torch.device,
    split: str,
) -> list:
    rows = []
    with torch.no_grad():
        mean, _ = proxy.encode_trajectory(native_states.to(device), condition, neighborhood_condition, edge_cache)
        base_mean, _ = proxy.encode_trajectory(base_states[None].to(device), condition, neighborhood_condition, edge_cache)
        for horizon in sorted(set(horizon for horizon in HORIZONS if horizon <= native_states.shape[1] - 1)):
            latent = proxy.rollout(mean[:, 0], horizon)
            base_latent = proxy.rollout(base_mean[:, 0], horizon)
            prediction = proxy.decode_rollout(latent, condition, neighborhood_condition)
            base_prediction = proxy.decode_rollout(base_latent, condition, neighborhood_condition)
            prediction_logits = readout(prediction.reshape(-1, prediction.shape[-2], prediction.shape[-1])).reshape(
                native_logits[:, :horizon + 1].shape
            )
            base_prediction_logits = readout(base_prediction.reshape(-1, base_prediction.shape[-2], base_prediction.shape[-1])).reshape(
                1, horizon + 1, native_logits.shape[-2], native_logits.shape[-1]
            )
            truth_states = native_states[:, horizon]
            truth_logits = native_logits[:, horizon]
            state_error = (prediction[:, -1].cpu() - truth_states).square().sum()
            state_scale = (truth_states - native_states[:, 0]).square().sum().clamp_min(1e-8)
            logit_error = (prediction_logits[:, -1].cpu() - truth_logits).square().sum()
            logit_scale = (truth_logits - native_logits[:, 0]).square().sum().clamp_min(1e-8)
            response_truth = truth_logits - base_logits[horizon]
            response_prediction = prediction_logits[:, -1].cpu() - base_prediction_logits[0, -1].cpu()
            response_error = (response_prediction - response_truth).square().sum()
            response_scale = response_truth.square().sum().clamp_min(1e-8)
            native_labels = truth_logits.argmax(dim=-1)
            proxy_labels = prediction_logits[:, -1].cpu().argmax(dim=-1)
            rows.append({
                'split': split,
                'horizon': int(horizon),
                'state_increment_nrmse': float(torch.sqrt(state_error / state_scale)),
                'logit_increment_nrmse': float(torch.sqrt(logit_error / logit_scale)),
                'response_nrmse': float(torch.sqrt(response_error / response_scale)),
                'argmax_agreement': float((native_labels == proxy_labels).float().mean()),
                'response_mae': float(response_prediction.sub(response_truth).abs().mean()),
            })
    return rows


def train_proxy(
    proxy: GraphKoopmanV0,
    train_states: Tensor,
    train_logits: Tensor,
    val_states: Tensor,
    val_logits: Tensor,
    base_states: Tensor,
    base_logits: Tensor,
    readout,
    reference_state: Tensor,
    edge_cache,
    scales: Dict[str, Tensor],
    args,
    device: torch.device,
    output_dir: Path,
):
    optimizer = torch.optim.Adam(proxy.parameters(), lr=args.proxy_lr)
    index_generator = torch.Generator(device='cpu').manual_seed(args.seed + 1000)
    history = []
    best_score = float('inf')
    best_state = None
    base_train = base_states.unsqueeze(0).expand(train_states.shape[0], -1, -1, -1)
    base_train_logits = base_logits.unsqueeze(0).expand(train_logits.shape[0], -1, -1, -1)
    native_steps = train_states.shape[1] - 1
    for step in range(args.steps):
        proxy.train()
        indices = batch_indices(train_states.shape[0], args.batch_size, index_generator)
        horizon = select_horizon(step, args.warmup, args.steps, native_steps)
        origin = int(torch.randint(0, native_steps - horizon + 1, (1,), generator=index_generator).item())
        condition, neighborhood_condition = proxy.conditions(reference_state, edge_cache)
        batch_states = train_states[indices].to(device)
        batch_logits = train_logits[indices].to(device)
        batch_base_states = base_train[indices].to(device)
        batch_base_logits = base_train_logits[indices].to(device)
        if step < args.warmup:
            losses = proxy.reconstruction_loss(
                batch_states,
                batch_logits,
                condition,
                neighborhood_condition,
                edge_cache,
                readout,
                scales,
                kl_weight=args.kl_weight,
            )
        else:
            losses = proxy.loss(
                batch_states,
                batch_logits,
                batch_base_states,
                batch_base_logits,
                origin,
                horizon,
                condition,
                neighborhood_condition,
                edge_cache,
                readout,
                scales,
                kl_weight=args.kl_weight,
                linear_weight=args.linear_weight,
            )
        optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(proxy.parameters(), args.grad_clip)
        optimizer.step()
        row = {'step': step + 1, 'horizon': horizon}
        row.update({key: float(value) for key, value in losses.items()})
        history.append(row)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            proxy.eval()
            with torch.no_grad():
                eval_condition, eval_neighborhood = proxy.conditions(reference_state, edge_cache)
            rows = metric_rows(
                proxy, val_states, val_logits, base_states, base_logits, readout,
                eval_condition, eval_neighborhood, edge_cache, device, 'val',
            )
            score = sum(
                row['state_increment_nrmse']
                + row['logit_increment_nrmse']
                + row['response_nrmse']
                for row in rows
            )
            if horizon == native_steps and score < best_score:
                best_score = score
                best_state = copy.deepcopy(proxy.state_dict())
            print(
                f"step={step + 1:04d} horizon={horizon} loss={row['total']:.5f} "
                f"val_score={score:.5f}",
                flush=True,
            )
    if best_state is not None:
        proxy.load_state_dict(best_state)
    with (output_dir / 'training.jsonl').open('w', encoding='utf-8') as stream:
        for row in history:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    torch.save({'model': proxy.state_dict()}, output_dir / 'proxy.pt')
    return history


def write_plots(history: list, metrics: list, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name in ('total', 'rec', 'pred', 'output', 'response', 'kl'):
        axis.plot([row['step'] for row in history], [row[name] for row in history], label=name)
    axis.set_xlabel('optimization step')
    axis.set_ylabel('normalized loss')
    axis.legend(ncol=3, fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / 'training_curves.png', dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    horizons = sorted({row['horizon'] for row in metrics})
    for axis, field, title in zip(
        axes,
        ('state_increment_nrmse', 'logit_increment_nrmse', 'response_nrmse'),
        ('state', 'logit', 'response'),
    ):
        values = [next(row[field] for row in metrics if row['horizon'] == horizon) for horizon in horizons]
        axis.plot(horizons, values, marker='o')
        axis.set_title(title)
        axis.set_xlabel('horizon')
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / 'horizon_metrics.png', dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description='V0 local graph Koopman proxy experiment')
    parser.add_argument('--dataset', choices=sorted(DATASET_META), required=True)
    parser.add_argument('--client-id', type=int, default=0)
    parser.add_argument('--base-path', type=Path, default=ROOT)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train-count', type=int, default=128)
    parser.add_argument('--val-count', type=int, default=32)
    parser.add_argument('--test-count', type=int, default=64)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--proxy-seed', type=int, default=11)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--condition-dim', type=int, default=16)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--queries', type=int, default=8)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--proxy-lr', type=float, default=1e-3)
    parser.add_argument('--kl-weight', type=float, default=1e-4)
    parser.add_argument('--linear-weight', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--cache-dir', type=Path, default=None)
    parser.add_argument('--no-cache', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.client_id < 0:
        raise ValueError('client-id must be non-negative.')
    seed_everything(args.seed)
    device = torch.device(args.device)
    base_path = args.base_path.resolve()
    checkpoint = find_checkpoint(args.dataset, args.client_id, args.checkpoint, base_path)
    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_dir = args.output or base_path / 'codev0' / 'run_logs' / f'{args.dataset}_client{args.client_id}_{run_stamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or output_dir / 'trajectory_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f'dataset={args.dataset} client={args.client_id} checkpoint={checkpoint}', flush=True)
    model, graph, partition_path = load_native(args.dataset, args.client_id, checkpoint, base_path, device)
    base_states, base_logits = native_trajectory(model, graph, graph.x.detach().cpu(), device)
    banks = {}
    for split, count, seed in (
        ('train', args.train_count, args.seed + 101),
        ('val', args.val_count, args.seed + 202),
        ('test', args.test_count, args.seed + 303),
    ):
        cache_path = cache_dir / f'{split}.pt'
        if args.no_cache and cache_path.exists():
            cache_path.unlink()
        bank_states, bank_logits, source = load_or_generate_bank(model, graph, cache_path, count, seed, device)
        banks[split] = (bank_states, bank_logits)
        print(f'{split}_trajectories={banks[split][0].shape[0]} source={source}', flush=True)

    train_states, train_logits = banks['train']
    val_states, val_logits = banks['val']
    test_states, test_logits = banks['test']
    seed_everything(args.proxy_seed)
    edge_index = graph.edge_index.to(device)
    edge_weight = graph.edge_weight.to(device) if getattr(graph, 'edge_weight', None) is not None else None
    proxy = GraphKoopmanV0(
        state_dim=train_states.shape[-1],
        output_dim=train_logits.shape[-1],
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        width=args.width,
        queries=args.queries,
        step_size=0.1,
    ).to(device)
    edge_cache = proxy.make_edge_cache(edge_index, edge_weight, train_states.shape[-2])
    scales = compute_normalization(train_states, train_logits, train_logits - base_logits.unsqueeze(0))
    scales = {key: value.to(device) for key, value in scales.items()}
    readout = model.readout
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    train_start = time.perf_counter()
    history = train_proxy(
        proxy, train_states, train_logits, val_states, val_logits,
        base_states, base_logits, readout,
        base_states[0].to(device), edge_cache, scales, args, device, output_dir,
    )
    proxy.eval()
    with torch.no_grad():
        condition, neighborhood_condition = proxy.conditions(base_states[0].to(device), edge_cache)
    metrics = []
    for split, states, logits in (
        ('train', train_states, train_logits),
        ('val', val_states, val_logits),
        ('test', test_states, test_logits),
    ):
        metrics.extend(metric_rows(
            proxy, states, logits, base_states, base_logits, readout,
            condition, neighborhood_condition, edge_cache, device, split,
        ))
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    write_plots(history, [row for row in metrics if row['split'] == 'test'], output_dir)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': args.dataset,
        'client_id': args.client_id,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': digest(checkpoint),
        'partition': str(partition_path),
        'partition_sha256': digest(partition_path),
        'device': str(device),
        'native_state_shape': list(base_states.shape),
        'native_logit_shape': list(base_logits.shape),
        'proxy_parameters': sum(parameter.numel() for parameter in proxy.parameters()),
        'training_seconds': time.perf_counter() - train_start,
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'cache_shapes': {split: list(states.shape) for split, (states, _logits) in banks.items()},
    }
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    print(f'output={output_dir}', flush=True)
    for row in metrics:
        if row['split'] == 'test':
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
