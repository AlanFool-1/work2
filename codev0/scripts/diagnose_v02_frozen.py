#!/usr/bin/env python3
"""Separate representation and transition errors on one frozen V0.2 checkpoint.

No original run is overwritten. Newly generated train/validation/held-out feature
perturbations have disjoint deterministic RNG streams. Node test labels are never
used by this diagnostic; displayed classification scores use validation labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from models.v02.diagnostics import (
    RidgeGraphTransition, StableTransition, contraction_audit, encode_states,
    per_time_nmse, rollout,
)
from models.v02.model import graph_operator
from models.v02.training import relative_mse
from run_v01_single_client import load_partition, task_metrics
from run_v02_single_client import make_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=31415)
    parser.add_argument('--train-trajectories', type=int, default=24)
    parser.add_argument('--val-trajectories', type=int, default=12)
    parser.add_argument('--test-trajectories', type=int, default=24)
    parser.add_argument('--noise-std', type=float, default=0.05)
    parser.add_argument('--updates', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.003)
    return parser.parse_args()


@torch.no_grad()
def reference_bank(model, data, operator, count, noise_std, seed):
    generator = torch.Generator(device=data.x.device).manual_seed(seed)
    bank = []
    for _ in range(count):
        noise = torch.randn(data.x.shape, device=data.x.device, dtype=data.x.dtype, generator=generator)
        changed = data.clone()
        changed.x = data.x * (1 + noise_std * noise)
        initial = model.initial_state(changed, operator)
        bank.append(model.native_trajectory(initial, operator))
    return torch.stack(bank)


@torch.no_grad()
def trajectory_metrics(model, predicted_z, target_z, true_h, base_h, base_pred_z):
    predicted_h = model.decoder(predicted_z)
    delta = true_h[:, -1] - base_h[-1]
    predicted_delta = predicted_h[:, -1] - model.decoder(base_pred_z[:, -1])
    rms = delta.square().mean()
    error = (predicted_delta - delta).square().mean()
    return {
        'latent_nmse_by_horizon': per_time_nmse(predicted_z, target_z).cpu().tolist(),
        'state_nmse_by_horizon': per_time_nmse(predicted_h, true_h).cpu().tolist(),
        'state_final_nmse': float(relative_mse(predicted_h[:, -1], true_h[:, -1])),
        'latent_final_nmse': float(relative_mse(predicted_z[:, -1], target_z[:, -1])),
        'response_nrmse': float((error / rms.clamp_min(1e-12)).sqrt()),
        'response_rms': float(rms.sqrt()),
        'response_rmse': float(error.sqrt()),
        'predicted_latent_norm_ratio': float(predicted_z[:, -1].norm() / predicted_z[:, 0].norm().clamp_min(1e-12)),
    }


@torch.no_grad()
def validation_fit(transition, encoded, operator, steps):
    prediction = rollout(transition, encoded[:, 0], operator, steps)
    value = per_time_nmse(prediction, encoded)[1:].mean()
    return float(value) if torch.isfinite(value) else float('inf')


def refit_stable(model, banks, operator, args):
    transition = StableTransition(copy.deepcopy(model.generator), model.step_size)
    transition.requires_grad_(True)
    optimizer = torch.optim.Adam(transition.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    best_score = validation_fit(transition, banks['validation'], operator, model.num_steps)
    best = copy.deepcopy(transition.state_dict())
    selected_update, history = 0, []
    for update in range(1, args.updates + 1):
        indices = torch.randint(len(banks['train']), (args.batch_size,), generator=generator)
        target = banks['train'][indices]
        optimizer.zero_grad(set_to_none=True)
        predicted = rollout(transition, target[:, 0], operator, model.num_steps)
        loss = per_time_nmse(predicted, target)[1:].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Stable refit loss is non-finite.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transition.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == args.updates:
            value = validation_fit(transition, banks['validation'], operator, model.num_steps)
            history.append({'update': update, 'train_loss': float(loss.detach()), 'validation_latent_nmse': value})
            print(f'[fixed:stable] update={update} val={value:.6f}', flush=True)
            if value < best_score:
                best_score, best, selected_update = value, copy.deepcopy(transition.state_dict()), update
    transition.load_state_dict(best)
    transition.requires_grad_(False)
    return transition, {'selected_update': selected_update, 'validation_latent_nmse': best_score, 'history': history}


def refit_autoencoder(model, banks, operator, args, balanced):
    candidate = copy.deepcopy(model)
    candidate.requires_grad_(False)
    candidate.encoder.requires_grad_(True)
    candidate.decoder.requires_grad_(True)
    parameters = [p for p in candidate.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    def loss_for(states):
        decoded = candidate.decoder(encode_states(candidate, states, operator))
        return per_time_nmse(decoded, states).mean() if balanced else relative_mse(decoded, states)

    with torch.no_grad():
        best_score = float(loss_for(banks['validation']))
    best = copy.deepcopy(candidate.state_dict())
    selected_update, history = 0, []
    for update in range(1, args.updates + 1):
        indices = torch.randint(len(banks['train']), (args.batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(banks['train'][indices])
        if not torch.isfinite(loss):
            raise FloatingPointError('AE refit loss is non-finite.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == args.updates:
            with torch.no_grad():
                value = float(loss_for(banks['validation']))
            history.append({'update': update, 'train_loss': float(loss.detach()), 'validation_reconstruction': value})
            if value < best_score:
                best_score, best, selected_update = value, copy.deepcopy(candidate.state_dict()), update
    candidate.load_state_dict(best)
    candidate.requires_grad_(False)
    print(f'[fixed:ae balanced={balanced}] selected={selected_update} val={best_score:.6f}', flush=True)
    return candidate, {'selected_update': selected_update, 'validation_reconstruction': best_score, 'history': history}


@torch.no_grad()
def choose_ridge(encoded, operator, steps, affine):
    best, best_score, candidates, alpha = None, float('inf'), [], None
    for regularizer in (1e-6, 1e-4, 1e-2):
        candidate = RidgeGraphTransition.fit(encoded['train'], operator, regularizer, affine)
        score = validation_fit(candidate, encoded['validation'], operator, steps)
        candidates.append({'ridge': regularizer, 'validation_latent_nmse': score if score < float('inf') else None})
        if score < best_score:
            best, best_score, alpha = candidate, score, regularizer
    if best is None:
        raise FloatingPointError('All ridge candidates diverged on validation trajectories.')
    return best, {'ridge': alpha, 'validation_latent_nmse': best_score, 'candidates': candidates}


@torch.no_grad()
def representation_metrics(model, bank, base, data, dataset, operator):
    reconstructed = model.decoder(encode_states(model, bank, operator))
    base_reconstructed = model.decoder(model.encode(base, operator))
    return {
        'nmse_by_time': per_time_nmse(reconstructed, bank).cpu().tolist(),
        'pooled_nmse': float(relative_mse(reconstructed, bank)),
        'original_graph_native_head_validation': task_metrics(
            dataset, model.native_readout(base_reconstructed[-1]), data.y, data.val_mask,
        ),
    }


def main():
    args = parse_args()
    if min(args.train_trajectories, args.val_trajectories, args.test_trajectories, args.updates, args.batch_size) <= 0:
        raise ValueError('Trajectory/update counts must be positive.')
    if args.noise_std <= 0 or args.lr <= 0:
        raise ValueError('Noise standard deviation and learning rate must be positive.')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = SimpleNamespace(**checkpoint['config'])
    if config.variant not in {'joint', 'no_dynamics'} or config.correction_interval:
        raise ValueError('Use a trained V0.2 reference with uncorrected linear dynamics.')
    model = make_model(config).to(args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval().requires_grad_(False)
    data, partition = load_partition(config, torch.device(args.device))
    operator = graph_operator(data)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fingerprint = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    with torch.no_grad():
        base = model.native_trajectory(model.initial_state(data, operator), operator)
        banks = {
            name: reference_bank(model, data, operator, count, args.noise_std, args.seed + offset)
            for name, count, offset in (
                ('train', args.train_trajectories, 101),
                ('validation', args.val_trajectories, 202),
                ('held_out', args.test_trajectories, 303),
            )
        }
        encoded = {name: encode_states(model, bank, operator) for name, bank in banks.items()}
        base_z = model.encode(base, operator)[None]
    report = {
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': fingerprint,
        'partition': str(partition), 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'scope': 'fixed-checkpoint diagnostic; perturbation holdout is not a new node classification test set',
        'selection': 'perturbation validation reconstruction/rollout error only; node test labels never read',
        'contraction': contraction_audit(model, base[None], operator),
        'reference_native_head_validation': task_metrics(config.dataset, model.native_readout(base[-1]), data.y, data.val_mask),
        'representations': {}, 'transitions': {},
    }
    with torch.no_grad():
        training_h = banks['train'].flatten(0, 2)
        mean = training_h.mean(0)
        _, singular, vectors = torch.linalg.svd(training_h - mean, full_matrices=False)
        basis = vectors[:model.latent_dim].T
        reconstructed = (banks['held_out'] - mean) @ basis @ basis.T + mean
        report['pca_reconstruction_control'] = {
            'dim': model.latent_dim,
            'training_energy_fraction': float(singular[:model.latent_dim].square().sum() / singular.square().sum()),
            'held_out_nmse_by_time': per_time_nmse(reconstructed, banks['held_out']).cpu().tolist(),
            'note': 'Reconstruction of observed states only; not a forecasting or classification result.',
        }
    report['representations']['checkpoint'] = representation_metrics(model, banks['held_out'], base, data, config.dataset, operator)

    transitions = {'checkpoint_stable': (StableTransition(model.generator, model.step_size), {})}
    transitions['refit_stable'] = refit_stable(model, encoded, operator, args)
    for affine in (False, True):
        transitions['ridge_affine' if affine else 'ridge_linear'] = choose_ridge(encoded, operator, model.num_steps, affine)

    with torch.no_grad():
        for name, (transition, selection) in transitions.items():
            predicted = rollout(transition, encoded['held_out'][:, 0], operator, model.num_steps)
            predicted_base = rollout(transition, base_z[:, 0], operator, model.num_steps)
            row = trajectory_metrics(model, predicted, encoded['held_out'], banks['held_out'], base, predicted_base)
            row['selection'] = selection
            row['original_graph_native_head_validation'] = task_metrics(
                config.dataset, model.native_readout(model.decoder(predicted_base[0, -1])), data.y, data.val_mask,
            )
            report['transitions'][name] = row
            torch.save(transition.state_dict(), args.output_dir / f'{name}.pt')
            print(f'[fixed:{name}] latent={row["latent_final_nmse"]:.6f} state={row["state_final_nmse"]:.6f} response={row["response_nrmse"]:.6f}', flush=True)

    for balanced in (False, True):
        candidate, selection = refit_autoencoder(model, banks, operator, args, balanced)
        name = 'ae_time_balanced' if balanced else 'ae_pooled'
        row = representation_metrics(candidate, banks['held_out'], base, data, config.dataset, operator)
        row['selection'] = selection
        report['representations'][name] = row
        torch.save({'encoder': candidate.encoder.state_dict(), 'decoder': candidate.decoder.state_dict()}, args.output_dir / f'{name}.pt')
        # A changed encoder requires refitting K. Never score the old K in new coordinates.
        with torch.no_grad():
            new_encoded = {key: encode_states(candidate, bank, operator) for key, bank in banks.items()}
            transition, ridge_selection = choose_ridge(new_encoded, operator, model.num_steps, True)
            z0 = candidate.encode(base[0], operator)[None]
            predicted_base = rollout(transition, z0, operator, model.num_steps)
            predicted = rollout(transition, new_encoded['held_out'][:, 0], operator, model.num_steps)
            row = trajectory_metrics(candidate, predicted, new_encoded['held_out'], banks['held_out'], base, predicted_base)
            row['selection'] = ridge_selection
            row['original_graph_native_head_validation'] = task_metrics(
                config.dataset, candidate.native_readout(candidate.decoder(predicted_base[0, -1])), data.y, data.val_mask,
            )
            report['transitions'][f'{name}_ridge_affine'] = row
            torch.save(transition.state_dict(), args.output_dir / f'{name}_ridge_affine.pt')
    # Fingerprint confirms the source file was not changed during diagnostics.
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError('Source checkpoint changed while running diagnostics.')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = [
        '# Frozen V0.2 diagnosis', '',
        'One frozen Cora partition/checkpoint; no node test labels or federated training.', '',
        f'Target latent growth: {report["contraction"]["target_final_to_initial_norm"]:.4f}; '
        f'fixed-operator latent NMSE lower bound: {report["contraction"]["frozen_operator_nmse_lower_bound"]:.6f}.', '',
        '| Transition | Held-out final latent NMSE | Final state NMSE | Response NRMSE | Native-head node val ACC |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name, row in report['transitions'].items():
        lines.append(f'| {name} | {row["latent_final_nmse"]:.6f} | {row["state_final_nmse"]:.6f} | {row["response_nrmse"]:.6f} | {row["original_graph_native_head_validation"]["metric"]:.4f} |')
    lines += ['', '| Representation | H0 reconstruction NMSE | HT reconstruction NMSE | Pooled NMSE |', '| --- | ---: | ---: | ---: |']
    for name, row in report['representations'].items():
        lines.append(f'| {name} | {row["nmse_by_time"][0]:.6f} | {row["nmse_by_time"][-1]:.6f} | {row["pooled_nmse"]:.6f} |')
    lines += ['', 'Ridge candidates use one-step regression with validation-selected regularization; stable refitting uses multi-step gradient descent. '
              'This diagnoses feasibility in frozen coordinates, not an optimization-budget-matched architecture comparison. '
              'AE controls add training on a frozen reference and do not establish that historical target drift caused the original error.']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[fixed] report={args.output_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
