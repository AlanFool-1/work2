#!/usr/bin/env python3
"""Gate-0 training for Method V0.1 on one federated graph partition."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import DATASET_META
from misc.utils import seed_everything
from models.v01.model import LatentLinearGraphDynamics


BINARY_DATASETS = {'Minesweeper', 'Tolokers', 'Questions'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train V0.1 on one client partition without FedAvg.'
    )
    parser.add_argument('--dataset', required=True, choices=DATASET_META)
    parser.add_argument('--client-id', type=int, default=0)
    parser.add_argument('--n-clients', type=int, default=10)
    parser.add_argument(
        '--base-path', default='/opt/data/private/xzc/work2'
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--encoder-width', type=int, default=64)
    parser.add_argument('--linear-steps', type=int, default=16)
    parser.add_argument('--linear-step-size', type=float, default=0.1)
    parser.add_argument('--linear-gamma', type=float, default=0.1)
    parser.add_argument('--reconstruction-weight', type=float, default=0.0)
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def task_loss(dataset, logits, labels, mask):
    if dataset in BINARY_DATASETS:
        return F.binary_cross_entropy_with_logits(
            logits[mask].reshape(-1), labels[mask].float().reshape(-1)
        )
    return F.cross_entropy(logits[mask], labels[mask].long())


def task_metrics(dataset, logits, labels, mask):
    if int(mask.sum().item()) == 0:
        return {'loss': 0.0, 'metric': 0.0, 'f1': 0.0}
    loss = float(task_loss(dataset, logits, labels, mask).item())
    truth = labels[mask].detach().cpu().numpy().reshape(-1)
    if dataset in BINARY_DATASETS:
        scores = logits[mask].detach().cpu().numpy().reshape(-1)
        metric = (
            float(roc_auc_score(truth, scores))
            if np.unique(truth).size > 1 else 0.5
        )
        predictions = (scores > 0.0).astype(np.int64)
        f1 = float(f1_score(
            truth, predictions, average='binary', zero_division=0
        ))
    else:
        predictions = logits[mask].argmax(dim=-1).detach().cpu().numpy()
        metric = float(np.mean(predictions == truth))
        f1 = float(f1_score(
            truth, predictions, average='macro', zero_division=0
        ))
    return {'loss': loss, 'metric': metric, 'f1': f1}


@torch.no_grad()
def evaluate(model, data, dataset, mask):
    model.eval()
    return task_metrics(dataset, model(data), data.y, mask)


def load_partition(args, device):
    path = Path(args.base_path) / 'datasets' / (
        f'{args.dataset}_disjoint/{args.n_clients}/'
        f'partition_{args.client_id}.pt'
    )
    if not path.is_file():
        raise FileNotFoundError(f'Client partition does not exist: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return payload['client_data'].to(device), path


def latent_diagnostics(trajectory):
    flat = trajectory.reshape(trajectory.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=1)
    if len(norms) == 1:
        growth = 1.0
        delta = 0.0
    else:
        denominator = norms[:-1].clamp_min(1e-12)
        growth = float((norms[1:] / denominator).max().item())
        delta = float((torch.linalg.vector_norm(
            flat[1:] - flat[:-1], dim=1
        ) / denominator).max().item())
    return {
        'latent_norms': [float(value) for value in norms.cpu().tolist()],
        'max_latent_growth_ratio': growth,
        'max_latent_relative_delta': delta,
    }


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.n_clients) <= 0:
        raise ValueError('epochs, patience, and n_clients must be positive.')
    if not 0 <= args.client_id < args.n_clients:
        raise ValueError('client-id must be in [0, n-clients).')
    if args.reconstruction_weight < 0.0:
        raise ValueError('reconstruction-weight must be non-negative.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    seed_everything(args.seed, torch_seed=True, cuda=torch.cuda.is_available())
    device = torch.device(args.device)
    data, partition_path = load_partition(args, device)
    input_dim, output_dim = DATASET_META[args.dataset]
    model = LatentLinearGraphDynamics(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=args.latent_dim,
        encoder_width=args.encoder_width,
        num_steps=args.linear_steps,
        step_size=args.linear_step_size,
        gamma=args.linear_gamma,
        reconstruction=args.reconstruction_weight > 0.0,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best = None
    best_state = None
    initial_train_loss = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_with_aux(data)
        supervised = task_loss(
            args.dataset, output.logits, data.y, data.train_mask
        )
        reconstruction = (
            F.mse_loss(output.reconstruction, data.x)
            if output.reconstruction is not None
            else output.logits.new_zeros(())
        )
        loss = supervised + args.reconstruction_weight * reconstruction
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at epoch {epoch}.')
        if initial_train_loss is None:
            initial_train_loss = float(loss.item())
        loss.backward()
        if any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f'Non-finite gradient at epoch {epoch}.')
        optimizer.step()

        validation = evaluate(model, data, args.dataset, data.val_mask)
        test = evaluate(model, data, args.dataset, data.test_mask)
        history.append({
            'epoch': epoch,
            'train_loss': float(loss.item()),
            'task_loss': float(supervised.item()),
            'reconstruction_loss': float(reconstruction.item()),
            'val_loss': validation['loss'],
            'val_metric': validation['metric'],
            'val_f1': validation['f1'],
            'test_loss': test['loss'],
            'test_metric': test['metric'],
            'test_f1': test['f1'],
        })
        if best is None or validation['metric'] > best['validation']['metric']:
            best = {
                'epoch': epoch,
                'validation': validation,
                'paired_test': test,
            }
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(
                f'[gate0] epoch={epoch} train_loss={loss.item():.6f} '
                f'task={supervised.item():.6f} '
                f'rec={reconstruction.item():.6f} '
                f'val={validation["metric"]:.4f} '
                f'test={test["metric"]:.4f}'
            )
        if epoch - best['epoch'] >= args.patience:
            print(f'[gate0] early stop at epoch {epoch}')
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_output = model.forward_with_aux(data)
        dynamics = latent_diagnostics(final_output.latent_trajectory)
    result = {
        'dataset': args.dataset,
        'client_id': args.client_id,
        'partition': str(partition_path),
        'selection': 'best validation epoch paired test metric',
        'metric': 'AUC' if args.dataset in BINARY_DATASETS else 'ACC',
        'best': best,
        'initial_train_loss': initial_train_loss,
        'last_train_loss': float(loss.item()),
        'last_task_loss': float(supervised.item()),
        'last_reconstruction_loss': float(reconstruction.item()),
        'epochs_completed': epoch,
        'dynamics': dynamics,
        'config': vars(args),
    }

    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / 'run_logs' / 'v01_gate0' /
        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_'
        f'{args.dataset}_client{args.client_id}_seed{args.seed}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_fields = list(history[0])
    temporary_history = output_dir / 'history.csv.tmp'
    with open(temporary_history, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=history_fields)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary_history, output_dir / 'history.csv')
    temporary_checkpoint = output_dir / 'best_model.pt.tmp'
    torch.save(best_state, temporary_checkpoint)
    os.replace(temporary_checkpoint, output_dir / 'best_model.pt')
    temporary_result = output_dir / 'result.json.tmp'
    with open(temporary_result, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary_result, output_dir / 'result.json')
    print(
        f'[gate0] best_epoch={best["epoch"]} '
        f'val={best["validation"]["metric"]:.6f} '
        f'paired_test={best["paired_test"]["metric"]:.6f}'
    )
    print(f'[gate0] output_dir={output_dir}')


if __name__ == '__main__':
    main()
