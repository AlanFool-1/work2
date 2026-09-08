#!/usr/bin/env python3
"""Summarize independent-client V0.1 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch


FIELDS = [
    'client_id', 'best_epoch', 'val_metric', 'test_metric', 'test_f1',
    'majority_test_metric', 'improvement_over_majority',
    'max_latent_growth_ratio', 'z_norm_ratio',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_root')
    return parser.parse_args()


def main():
    root = Path(parse_args().result_root).resolve()
    rows = []
    for path in sorted(
        root.glob('client_*/result.json'),
        key=lambda item: int(item.parent.name.split('_')[-1]),
    ):
        with open(path, encoding='utf-8') as stream:
            result = json.load(stream)
        partition = torch.load(
            result['partition'], map_location='cpu', weights_only=False
        )['client_data']
        labels = partition.y[partition.test_mask].long().reshape(-1)
        counts = torch.bincount(labels)
        majority = float(counts.max().item() / counts.sum().item())
        test_metric = float(result['best']['paired_test']['metric'])
        norms = result['dynamics']['latent_norms']
        rows.append({
            'client_id': int(result['client_id']),
            'best_epoch': int(result['best']['epoch']),
            'val_metric': float(result['best']['validation']['metric']),
            'test_metric': test_metric,
            'test_f1': float(result['best']['paired_test']['f1']),
            'majority_test_metric': majority,
            'improvement_over_majority': test_metric - majority,
            'max_latent_growth_ratio': float(result['dynamics'][
                'max_latent_growth_ratio'
            ]),
            'z_norm_ratio': float(norms[-1] / max(norms[0], 1e-12)),
        })
    if not rows:
        raise RuntimeError(f'No local result files found below {root}.')

    csv_temporary = root / 'summary.csv.tmp'
    with open(csv_temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temporary, root / 'summary.csv')

    metrics = np.asarray([row['test_metric'] for row in rows])
    f1_values = np.asarray([row['test_f1'] for row in rows])
    majorities = np.asarray([row['majority_test_metric'] for row in rows])
    aggregate = {
        'dataset': 'Cora',
        'model': 'v01_linear',
        'training': 'independent local, no broadcast or aggregation',
        'seed': 42,
        'local_epochs': 100,
        'n_clients': len(rows),
        'selection': 'per-client best validation epoch paired test metric',
        'test_accuracy_mean': float(metrics.mean()),
        'test_accuracy_client_std': float(metrics.std()),
        'test_macro_f1_mean': float(f1_values.mean()),
        'test_macro_f1_client_std': float(f1_values.std()),
        'majority_accuracy_mean': float(majorities.mean()),
        'client_results': rows,
    }
    json_temporary = root / 'summary.json.tmp'
    with open(json_temporary, 'w', encoding='utf-8') as stream:
        json.dump(aggregate, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(json_temporary, root / 'summary.json')

    lines = [
        '# Cora V0.1 independent-local summary',
        '',
        '| Client | Best epoch | Val ACC | Test ACC | Macro-F1 | Majority | '
        'Delta | ZL/Z0 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f'| {row["client_id"]} | {row["best_epoch"]} | '
            f'{row["val_metric"]:.4f} | {row["test_metric"]:.4f} | '
            f'{row["test_f1"]:.4f} | '
            f'{row["majority_test_metric"]:.4f} | '
            f'{row["improvement_over_majority"]:+.4f} | '
            f'{row["z_norm_ratio"]:.3f} |'
        )
    lines.extend([
        '',
        f'- Mean test ACC: {metrics.mean():.6f} ± {metrics.std():.6f}',
        f'- Mean Macro-F1: {f1_values.mean():.6f} ± {f1_values.std():.6f}',
        f'- Mean majority baseline: {majorities.mean():.6f}',
    ])
    markdown_temporary = root / 'summary.md.tmp'
    with open(markdown_temporary, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    os.replace(markdown_temporary, root / 'summary.md')
    print(
        f'[local-summary] ACC={metrics.mean():.6f} ± {metrics.std():.6f}; '
        f'F1={f1_values.mean():.6f} ± {f1_values.std():.6f}'
    )


if __name__ == '__main__':
    main()
