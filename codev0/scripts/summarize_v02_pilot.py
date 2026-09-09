#!/usr/bin/env python3
"""Summarize one matched-partition V0.2 pilot (not an all-client benchmark)."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference-result', type=Path, action='append', default=[],
                        help='Read an existing matched result without copying or rerunning it.')
    args = parser.parse_args()
    paths = sorted(args.root.glob('*/result.json')) + args.reference_result
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    results = [json.loads(path.read_text()) for path in paths]
    if not results:
        raise ValueError('No completed pilot runs found.')
    # Match the substantive protocol; actual compute and parameter counts vary.
    keys = ('dataset', 'client_id', 'n_clients', 'seed', 'epochs', 'patience', 'lr', 'weight_decay', 'feature_noise')
    reference = results[0]['config']
    for result in results:
        if any(result['config'][key] != reference[key] for key in keys):
            raise ValueError('Mixed pilot protocols; summarize matched runs in separate directories.')
    lines = [
        f'# V0.2 pilot: {reference["dataset"]}, client {reference["client_id"]}, seed {reference["seed"]}',
        '',
        f'{reference["epochs"]} local epochs; lr={reference["lr"]}; no aggregation. '
        'Checkpoint selection uses validation accuracy only. This is one diagnostic partition, '
        'not the 10-client average and not a multi-seed estimate.',
        '',
        '| Variant | Best epoch | Val ACC | Test ACC | Macro-F1 | Parameters (train / deploy) | Device | Train seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |',
    ]
    for result in results:
        best, cost = result['best'], result['cost']
        label = result['variant']
        if label == 'joint':
            label += f' ({result["config"].get("generator_mode", "dissipative")}, {result["config"].get("loss_normalization", "pooled")})'
        lines.append(
            f'| {label} | {best["epoch"]} | {100 * best["validation"]["metric"]:.2f}% '
            f'| {100 * best["paired_test"]["metric"]:.2f}% | {100 * best["paired_test"]["f1"]:.2f}% '
            f'| {cost["training_parameters"]:,} / {cost["deployment_parameters"]:,} '
            f'| {result["config"]["device"]} | {result["train_seconds"]:.2f} |'
        )
    lines += [
        '', 'Allocated parameter counts include inactive modules in ablations. '
        'Training-only modules are excluded from deployment counts. Epochs are matched; '
        'FLOPs, model size, and hardware time are not matched. The GCN is an explicit '
        'two-layer control, not a reproduction of a paper result.',
        '', '## Joint-model diagnostics at its selected checkpoint', '',
    ]
    for result in results:
        if result['variant'] != 'joint':
            continue
        values = result['dynamics']
        lines += [
            f'### {result["config"].get("generator_mode", "dissipative")} / {result["config"].get("loss_normalization", "pooled")}', '',
            f'- Reconstruction NMSE: {values["reconstruction_nmse"]:.6f}',
            f'- Latent norm final / initial: {values["latent_norm_ratio"]:.6f}',
            f'- Latent effective rank: {values["latent_effective_rank"]:.3f}',
        ]
        for key, value in values.items():
            if key.startswith(('prediction_nmse_h', 'increment_nrmse_h', 'linearity_nmse_h')):
                lines.append(f'- {key}: {value:.6f}')
        lines += [
            f'- Held-out perturbation response NRMSE: {values["perturbation"]["response_nrmse"]:.6f}',
            f'- Reference response RMS: {values["perturbation"]["response_rms"]:.6g}',
            '', 'Inference interventions (no retraining; not used to select the checkpoint):', '',
        ]
        for name, metric in values['inference_interventions'].items():
            lines.append(f'- {name}: ACC {100 * metric["metric"]:.2f}%, macro-F1 {100 * metric["f1"]:.2f}%')
    (args.root / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (args.root / 'summary.json').write_text(json.dumps({
        'protocol': {key: reference[key] for key in keys},
        'scope': 'single-partition pilot; not a federated or all-client result',
        'source_results': [str(path) for path in paths],
        'results': results,
    }, indent=2) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
