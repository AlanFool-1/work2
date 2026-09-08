#!/usr/bin/env python3
"""Collect Gate-0 result files into compact CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


FIELDS = [
    'dataset', 'variant', 'latent_dim', 'reconstruction_weight',
    'epochs_completed', 'best_epoch', 'val_metric', 'paired_test_metric',
    'paired_test_f1', 'initial_train_loss', 'last_train_loss',
    'last_task_loss', 'last_reconstruction_loss', 'z_norm_0', 'z_norm_last',
    'max_latent_growth_ratio', 'max_latent_relative_delta',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_root')
    return parser.parse_args()


def collect(root):
    rows = []
    for path in sorted(root.glob('*/*/result.json')):
        with open(path, encoding='utf-8') as stream:
            result = json.load(stream)
        norms = result['dynamics']['latent_norms']
        rows.append({
            'dataset': result['dataset'],
            'variant': path.parent.name,
            'latent_dim': result['config']['latent_dim'],
            'reconstruction_weight': result['config'][
                'reconstruction_weight'
            ],
            'epochs_completed': result['epochs_completed'],
            'best_epoch': result['best']['epoch'],
            'val_metric': result['best']['validation']['metric'],
            'paired_test_metric': result['best']['paired_test']['metric'],
            'paired_test_f1': result['best']['paired_test']['f1'],
            'initial_train_loss': result['initial_train_loss'],
            'last_train_loss': result['last_train_loss'],
            'last_task_loss': result['last_task_loss'],
            'last_reconstruction_loss': result[
                'last_reconstruction_loss'
            ],
            'z_norm_0': norms[0],
            'z_norm_last': norms[-1],
            'max_latent_growth_ratio': result['dynamics'][
                'max_latent_growth_ratio'
            ],
            'max_latent_relative_delta': result['dynamics'][
                'max_latent_relative_delta'
            ],
        })
    dataset_order = {'Cora': 0, 'CiteSeer': 1}
    variant_order = {
        'm1_r32': 0,
        'a1_r16': 1,
        'a1_r64': 2,
        'a2_r32_rec10': 3,
    }
    return sorted(rows, key=lambda row: (
        dataset_order.get(row['dataset'], 99),
        variant_order.get(row['variant'], 99),
    ))


def main():
    args = parse_args()
    root = Path(args.result_root).resolve()
    rows = collect(root)
    if not rows:
        raise RuntimeError(f'No Gate-0 result.json files found below {root}.')

    csv_temporary = root / 'summary.csv.tmp'
    with open(csv_temporary, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temporary, root / 'summary.csv')

    lines = [
        '# Method V0.1 Gate 0 summary',
        '',
        '| Dataset | Variant | r | lambda_rec | Best/stop epoch | Val | Paired test | '
        'F1 | Loss start -> end | Z norm 0 -> L | Max growth |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | '
        '--- | ---: |',
    ]
    for row in rows:
        lines.append(
            f'| {row["dataset"]} | {row["variant"]} | '
            f'{row["latent_dim"]} | {row["reconstruction_weight"]:g} | '
            f'{row["best_epoch"]}/{row["epochs_completed"]} | '
            f'{row["val_metric"]:.4f} | '
            f'{row["paired_test_metric"]:.4f} | '
            f'{row["paired_test_f1"]:.4f} | '
            f'{row["initial_train_loss"]:.4f} -> '
            f'{row["last_train_loss"]:.4f} | '
            f'{row["z_norm_0"]:.3f} -> {row["z_norm_last"]:.3f} | '
            f'{row["max_latent_growth_ratio"]:.4f} |'
        )
    markdown_temporary = root / 'summary.md.tmp'
    with open(markdown_temporary, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    os.replace(markdown_temporary, root / 'summary.md')
    print(f'[gate0-summary] collected {len(rows)} runs from {root}')


if __name__ == '__main__':
    main()
