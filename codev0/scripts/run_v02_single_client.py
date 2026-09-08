#!/usr/bin/env python3
"""Train/ablate V0.2 on an existing partition, without federated aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import DATASET_META
from misc.utils import seed_everything
from models.v01.model import LatentLinearGraphDynamics
from models.v02.model import GraphKoopmanBackbone, graph_operator
from models.v02.training import LossWeights, checked_step, objective, task_loss, trajectory_diagnostics
from run_v01_single_client import BINARY_DATASETS, load_partition, task_metrics


class GCNControl(nn.Module):
    """Explicit two-layer GCN control; parameter counts are reported, not matched."""
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.first = GCNConv(input_dim, hidden_dim)
        self.second = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        weight = getattr(data, 'edge_weight', None)
        h = F.relu(self.first(data.x, data.edge_index, weight))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.second(h, data.edge_index, weight)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASET_META, default='Cora')
    parser.add_argument('--client-id', type=int, default=3)
    parser.add_argument('--n-clients', type=int, default=10)
    parser.add_argument('--base-path', default=str(ROOT.parent))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=101)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--encoder-width', type=int, default=64)
    parser.add_argument('--linear-steps', type=int, default=16)
    parser.add_argument('--linear-step-size', type=float, default=0.1)
    parser.add_argument('--linear-gamma', type=float, default=0.1)
    parser.add_argument('--generator-norm-bound', type=float, default=4.0)
    parser.add_argument('--correction-interval', type=int, default=0)
    parser.add_argument('--native-weight', type=float, default=1.0)
    parser.add_argument('--reconstruction-weight', type=float, default=1.0)
    parser.add_argument('--prediction-weight', type=float, default=1.0)
    parser.add_argument('--linearity-weight', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=5.0)
    parser.add_argument('--feature-noise', type=float, default=0.0)
    parser.add_argument('--variant', choices=['joint', 'identity', 'no_dynamics', 'no_aux', 'native', 'v01', 'gcn'], default='joint')
    parser.add_argument('--output-dir')
    return parser.parse_args()


def make_model(args):
    input_dim, output_dim = DATASET_META[args.dataset]
    if args.variant == 'gcn':
        return GCNControl(input_dim, output_dim, args.hidden_dim)
    if args.variant == 'v01':
        return LatentLinearGraphDynamics(
            input_dim, output_dim, latent_dim=args.latent_dim,
            encoder_width=args.encoder_width, num_steps=args.linear_steps,
            step_size=args.linear_step_size, gamma=args.linear_gamma,
        )
    return GraphKoopmanBackbone(
        input_dim, output_dim, state_dim=args.hidden_dim, latent_dim=args.latent_dim,
        width=args.encoder_width, num_steps=args.linear_steps,
        step_size=args.linear_step_size, damping=args.linear_gamma,
        generator_norm_bound=args.generator_norm_bound,
        correction_interval=args.correction_interval,
        identity_dynamics=args.variant == 'identity',
    )


@torch.no_grad()
def diagnostics(model, data, dataset, seed):
    model.eval()
    output = model.forward_with_aux(data, auxiliary=False)
    values = trajectory_diagnostics(model, output)
    values['native_at_proxy_checkpoint'] = task_metrics(
        dataset, output.native_logits, data.y, data.test_mask,
    )
    # Inference interventions are distinguished from separately retrained
    # identity/no_aux variants. None is used for checkpoint selection.
    operator = graph_operator(data)
    z0 = output.latent_trajectory[0]
    interventions = {
        'identity_at_inference': model.readout(model.decoder(z0)),
        'zero_latent_at_inference': model.readout(model.decoder(torch.zeros_like(z0))),
    }
    original_interval = model.correction_interval
    if not model.identity_dynamics:
        intervals = sorted({0, 4, original_interval})
        for interval in intervals:
            z = model.rollout(z0, operator, correction_interval=interval)[-1]
            interventions[f'correction_interval_{interval}'] = model.readout(model.decoder(z))
    values['inference_interventions'] = {
        name: task_metrics(dataset, logits, data.y, data.test_mask)
        for name, logits in interventions.items()
    }
    # Unseen feature perturbation probes the same frozen reference field.
    # A tiny trajectory NMSE alone could hide failure to predict responses.
    generator = torch.Generator(device=data.x.device).manual_seed(seed + 104729)
    noise = torch.randn(data.x.shape, device=data.x.device, dtype=data.x.dtype, generator=generator)
    perturbed = data.clone()
    perturbed.x = data.x * (1.0 + 0.02 * noise)
    changed = model.forward_with_aux(perturbed, auxiliary=False)
    actual_response = changed.native_states[-1] - output.native_states[-1]
    predicted_response = changed.decoded_trajectory[-1] - output.decoded_trajectory[-1]
    response_mse = actual_response.square().mean()
    error = (predicted_response - actual_response).square().mean()
    values['perturbation'] = {
        'multiplicative_std': 0.02,
        'response_rms': float(response_mse.sqrt()),
        'response_rmse': float(error.sqrt()),
        'response_nrmse': float((error / response_mse.clamp_min(1e-12)).sqrt()),
        'note': 'one held-out perturbation of the jointly learned reference; diagnostic only',
    }
    return values


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.n_clients) <= 0 or not 0 <= args.client_id < args.n_clients:
        raise ValueError('Invalid epoch count or partition index.')
    if args.lr <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0 or args.feature_noise < 0:
        raise ValueError('Invalid optimizer/augmentation setting.')
    if args.variant in {'native', 'v01', 'gcn'} and args.correction_interval:
        raise ValueError('Correction is available only on Koopman variants.')
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / 'run_logs' / 'v02_local' / (
        f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_{args.dataset}_client{args.client_id}_{args.variant}'
    )
    # Never silently replace an earlier experimental run.
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seed_everything(args.seed, torch_seed=True, cuda=torch.cuda.is_available())
    data, partition = load_partition(args, torch.device(args.device))
    if not data.val_mask.any() or not data.test_mask.any():
        raise ValueError('Local evaluation needs non-empty validation and test masks.')
    model = make_model(args).to(args.device)
    weights = LossWeights(args.native_weight, args.reconstruction_weight, args.prediction_weight, args.linearity_weight)
    if args.variant == 'no_aux':
        weights = LossWeights(native=0.0, reconstruction=0.0, prediction=0.0, linear=0.0)
    elif args.variant == 'no_dynamics':
        weights = LossWeights(native=args.native_weight, reconstruction=args.reconstruction_weight,
                              prediction=0.0, linear=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    joint = args.variant in {'joint', 'identity', 'no_dynamics'}
    predict = model.forward_native if args.variant == 'native' else model
    counts = model.parameter_counts() if isinstance(model, GraphKoopmanBackbone) else {
        'training_parameters': sum(p.numel() for p in model.parameters()),
        'deployment_parameters': sum(p.numel() for p in model.parameters()),
    }
    if args.variant == 'native':
        counts['deployment_parameters'] = sum(p.numel() for module in (
            model.stem, model.native_self, model.native_neighbor, model.native_readout,
        ) for p in module.parameters())
    counts['note'] = 'allocated training parameters; inactive ablation modules may have no gradients'
    config = {**vars(args), 'resolved_loss_weights': vars(weights), 'cost': counts}
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    best, best_state, history = None, None, []
    started = time.perf_counter()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = data
        if args.feature_noise:
            batch = data.clone()
            batch.x = data.x * (1.0 + args.feature_noise * torch.randn_like(data.x))
        if joint:
            output = model.forward_with_aux(batch, auxiliary=weights.auxiliary)
            loss, pieces = objective(model, output, batch.y, batch.train_mask, weights)
        else:
            loss = task_loss(predict(batch), batch.y, batch.train_mask)
            pieces = {'task_loss': loss}
        grad_norm = checked_step(loss, model, optimizer, args.max_grad_norm)
        model.eval()
        with torch.no_grad():
            validation = task_metrics(args.dataset, predict(data), data.y, data.val_mask)
        row = {'epoch': epoch, 'train_loss': float(loss.detach()), **{
            name: float(value.detach()) for name, value in pieces.items()
        }, 'grad_norm': grad_norm, 'val_loss': validation['loss'], 'val_metric': validation['metric'], 'val_f1': validation['f1']}
        history.append(row)
        # Same strict best-validation-accuracy rule used for V0.1. Test labels
        # are evaluated once, after training and checkpoint selection.
        if best is None or validation['metric'] > best['validation']['metric']:
            best = {'epoch': epoch, 'validation': validation}
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f'[v02:{args.variant}] epoch={epoch} loss={float(loss):.5f} val={validation["metric"]:.4f}', flush=True)
        if epoch - best['epoch'] >= args.patience:
            break
    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best['paired_test'] = task_metrics(args.dataset, predict(data), data.y, data.test_mask)
        majority_label = torch.bincount(data.y[data.train_mask].long()).argmax()
        majority_accuracy = float((data.y[data.test_mask] == majority_label).float().mean())
    dynamics = diagnostics(model, data, args.dataset, args.seed) if joint or args.variant == 'no_aux' else None
    if dynamics is not None:
        dynamics['reference_trained'] = joint and weights.native > 0
        if not dynamics['reference_trained']:
            dynamics['reference_warning'] = 'Reference field is untrained in this variant; its fit errors are not Koopman evidence.'
    result = {
        'dataset': args.dataset, 'client_id': args.client_id, 'partition': str(partition),
        'variant': args.variant, 'training': 'independent local; no aggregation',
        'selection': 'strict best validation metric; test evaluated after selection',
        'metric': 'AUC' if args.dataset in BINARY_DATASETS else 'ACC',
        'best': best, 'epochs_completed': epoch, 'train_seconds': train_seconds,
        'train_majority_label_test_accuracy': majority_accuracy,
        'initial_train_loss': history[0]['train_loss'], 'last_train_loss': history[-1]['train_loss'],
        'cost': counts, 'dynamics': dynamics, 'config': config,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device.startswith('cuda') else None,
    }
    with (output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save({'model': best_state, 'config': config, 'best': best}, output_dir / 'best_model.pt')
    (output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'[v02:{args.variant}] best_epoch={best["epoch"]} paired_test={best["paired_test"]["metric"]:.6f} output={output_dir}', flush=True)


if __name__ == '__main__':
    main()
