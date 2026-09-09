#!/usr/bin/env python3
"""Audit completed runs; distinguish validation-paired local and final-global scores.

Read-only with respect to training logs/checkpoints. Test scores and interventions
are post-hoc reports, never used for selecting a checkpoint or configuration.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from models.v01.model import build_v01_model
from models.v02.model import build_v02_model
from run_v01_single_client import load_partition, task_metrics
from run_v02_single_client import diagnostics


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def validate_completion(config, result, rounds, best, expected_rounds):
    if config['n_rnds'] != expected_rounds:
        raise ValueError('Configured round count does not match requested audit.')
    if [int(row['round']) for row in rounds] != list(range(1, expected_rounds + 1)):
        raise ValueError('Incomplete, duplicated, or unordered round history.')
    if sorted(int(row['client_id']) for row in best) != list(range(config['n_clients'])):
        raise ValueError('Missing or duplicated client result.')
    if any(not 1 <= int(row['best_round']) <= expected_rounds for row in best):
        raise ValueError('Invalid validation-selected round.')
    if len(result['client_test_metrics']) != config['n_clients']:
        raise ValueError('Incomplete final result.')
    reported = result['client_test_metrics']
    selected = [float(row['test_accuracy']) for row in sorted(best, key=lambda row: int(row['client_id']))]
    if any(abs(left - right) > 1e-10 for left, right in zip(reported, selected)):
        raise ValueError('Summary vector does not match client-level scores.')
    if abs(statistics.mean(float(row['test_accuracy']) for row in best) - result['mean']) > 1e-10:
        raise ValueError('Summary does not match client-level scores.')


def load_state(model, checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict({name: torch.as_tensor(value) for name, value in payload['model'].items()})
    model.eval()
    return payload


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def audit_run(label, path, expected_rounds):
    config = json.loads((path / 'config.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    rounds = read_csv(path / 'metrics.csv')
    best = read_csv(path / 'client_best.csv')
    validate_completion(config, result, rounds, best, expected_rounds)
    if config['mode'] != 'disjoint' or config['frac'] != 1:
        raise ValueError('This audit expects disjoint graphs and all-client participation.')
    args = argparse.Namespace(**config)
    builder = {'v01_linear': build_v01_model, 'v02_koopman': build_v02_model}[args.model]
    model = builder(args).cpu()
    checkpoint_root = Path(config['checkpt_path'])
    clients, global_clients = [], []
    for row in sorted(best, key=lambda row: int(row['client_id'])):
        args.client_id = int(row['client_id'])
        data, partition = load_partition(args, torch.device('cpu'))
        checkpoint = checkpoint_root / f'{args.client_id}_state.pt'
        payload = load_state(model, checkpoint)
        if int(payload['best']['round']) != int(row['best_round']):
            raise ValueError('Checkpoint and selected round disagree.')
        score = task_metrics(args.dataset, model(data), data.y, data.test_mask)
        if abs(score['metric'] - float(row['test_accuracy'])) > 1e-7:
            raise ValueError('Reloaded checkpoint does not reproduce reported accuracy.')
        entry = {
            'client_id': args.client_id, 'best_round': int(row['best_round']),
            'test': score, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'partition': str(partition), 'partition_sha256': sha256(partition),
        }
        if args.model == 'v02_koopman':
            entry['dynamics'] = diagnostics(model, data, args.dataset, args.seed + args.client_id)
        clients.append(entry)
    server_checkpoint = checkpoint_root / 'server_state.pt'
    server = load_state(model, server_checkpoint)
    if server['round'] != expected_rounds:
        raise ValueError('Server checkpoint does not contain the final aggregation.')
    for client_id in range(args.n_clients):
        args.client_id = client_id
        data, _ = load_partition(args, torch.device('cpu'))
        global_clients.append({
            'client_id': client_id,
            **task_metrics(args.dataset, model(data), data.y, data.test_mask),
        })
    output = {
        'label': label, 'source': str(path.resolve()), 'config': config,
        'completed_rounds': len(rounds), 'result': result, 'clients': clients,
        'round_metrics': rounds,
        'round100_post_local_mean': float(rounds[-1]['test_accuracy']),
        'final_global': {
            'checkpoint': str(server_checkpoint), 'checkpoint_sha256': sha256(server_checkpoint),
            'round': server['round'], 'clients': global_clients,
            'mean': statistics.mean(row['metric'] for row in global_clients),
        },
    }
    if args.model == 'v02_koopman':
        # These are different clients' jointly learned reference systems, not
        # one fixed common target. Means are diagnostics, not an accuracy proof.
        fields = ('reconstruction_nmse', 'linearity_nmse_h16', 'prediction_nmse_h16',
                  'latent_effective_rank', 'latent_norm_ratio')
        output['selected_checkpoint_diagnostics_mean'] = {
            key: statistics.mean(row['dynamics'][key] for row in clients) for key in fields
        }
        output['identity_at_inference_mean'] = statistics.mean(
            row['dynamics']['inference_interventions']['identity_at_inference']['metric']
            for row in clients
        )
        output['perturbation_response_nrmse_mean'] = statistics.mean(
            row['dynamics']['perturbation']['response_nrmse'] for row in clients
        )
    return output


def plot_round_history(reports, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    colors = ('#6b7280', '#2563eb', '#dc2626', '#059669')
    for index, report in enumerate(reports):
        history = report['round_metrics']
        for axis, field in zip(axes, ('val_accuracy', 'test_accuracy')):
            axis.plot([int(row['round']) for row in history],
                      [100 * float(row[field]) for row in history],
                      label=report['label'], color=colors[index % len(colors)], linewidth=1.6,
                      linestyle='--' if report['config']['base_lr'] == 0.015 else '-')
    for axis, title in zip(axes, ('Post-local validation ACC', 'Post-local test ACC (report only)')):
        axis.set(title=title, xlabel='Federated round', xlim=(1, 100), ylim=(0, 100))
        axis.grid(alpha=0.15)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Unweighted client mean (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Cora | 10 clients | 100 rounds | 1 local epoch | seed 42', fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={'Date': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=LOG_DIRECTORY')
    parser.add_argument('--expected-rounds', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plot-file', type=Path, help='Optional new SVG/PNG round-history figure.')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Refusing to overwrite an existing audit.')
    if args.plot_file is not None and args.plot_file.exists():
        raise FileExistsError('Refusing to overwrite an existing figure.')
    torch.set_num_threads(1)
    reports = []
    for value in args.run:
        label, path = value.split('=', 1)
        reports.append(audit_run(label, Path(path), args.expected_rounds))
        print(f'Audited {label}: {reports[-1]["completed_rounds"]} rounds', flush=True)
    shared_keys = ('dataset', 'n_clients', 'n_rnds', 'n_eps', 'frac', 'seed', 'aggregation',
                   'weight_decay', 'latent_dim', 'linear_steps', 'linear_step_size', 'max_grad_norm')
    shared = {key: reports[0]['config'][key] for key in shared_keys}
    if any(any(report['config'][key] != value for key, value in shared.items()) for report in reports):
        raise ValueError('Substantive protocol mismatch; do not pool these runs.')
    if any([row['partition_sha256'] for row in report['clients']] !=
           [row['partition_sha256'] for row in reports[0]['clients']] for report in reports):
        raise ValueError('Partition contents differ across runs.')
    lines = [
        '# Cora full federated audit (2026-09-09)', '',
        f'All runs completed {args.expected_rounds} rounds; 10 clients, local epoch 1, '
        'seed 42, all-client equal FedAvg, persistent per-client Adam. '
        'Client partitions and reloaded validation-selected checkpoint scores are verified.', '',
        '| Model | LR | Val-selected client Test ACC | Macro-F1 | Round 100 post-local ACC | Round 100 aggregated global ACC |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for report in reports:
        result = report['result']
        lines.append(f'| {report["label"]} | {report["config"]["base_lr"]} '
                     f'| {100 * result["mean"]:.2f}% | {100 * result["f1_mean"]:.2f}% '
                     f'| {100 * report["round100_post_local_mean"]:.2f}% '
                     f'| {100 * report["final_global"]["mean"]:.2f}% |')
    lines += [
        '', 'The historical 69.85% is the validation-selected *client* checkpoint mean, '
        'not the final global-model accuracy. All three columns must stay separate. '
        'Different learning rates are shown explicitly. One seed is not a significance estimate.',
        '', '## Operator diagnostics on validation-selected client checkpoints', '',
        '| Model | Reconstruction NMSE | Latent NMSE h16 | Decoded state NMSE h16 | Identity inference ACC | Perturbation response NRMSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for report in reports:
        if 'selected_checkpoint_diagnostics_mean' not in report:
            continue
        values = report['selected_checkpoint_diagnostics_mean']
        lines.append(f'| {report["label"]} | {values["reconstruction_nmse"]:.4f} '
                     f'| {values["linearity_nmse_h16"]:.4f} | {values["prediction_nmse_h16"]:.4f} '
                     f'| {100 * report["identity_at_inference_mean"]:.2f}% '
                     f'| {report["perturbation_response_nrmse_mean"]:.4f} |')
    lines += [
        '', 'Diagnostics are unweighted client means; reference systems differ across models/clients. '
        'Perturbations use one previously unseen 2% feature-noise draw per client. '
        'Identity intervention is inference-only, not a retrained ablation; '
        'it checks reliance on K, not Koopman closure. Test results never select an intervention.',
        '', '## Sources', '',
        *[f'- {report["label"]}: `{report["source"]}`' for report in reports],
    ]
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'report.json').write_text(json.dumps({
        'protocol': shared, 'evaluation_device': 'cpu', 'runs': reports,
    }, indent=2) + '\n')
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    if args.plot_file is not None:
        plot_round_history(reports, args.plot_file)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
