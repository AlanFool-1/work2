"""Entry point for the official A-DGN + FedAvg baseline."""

from __future__ import annotations

import json
import os
from datetime import datetime

from myparser import Parser
from misc.utils import seed_everything
from models.s0 import Client, Server
from modules.multiprocs import ParentProcess


DATASET_META = {
    'Cora': (1433, 7),
    'CiteSeer': (3703, 6),
    'PubMed': (500, 3),
    'Computers': (767, 10),
    'Photo': (745, 8),
    'ogbn-arxiv': (128, 40),
    'Roman-empire': (300, 18),
    'Amazon-ratings': (300, 5),
    'Minesweeper': (7, 1),
    'Tolokers': (10, 1),
    'Questions': (301, 1),
}


def _safe_tag(value):
    return ''.join(
        character if character.isalnum() or character in '-_'
        else '_' for character in str(value)
    )


def _load_config(args):
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs',
            's0_ode_gnn.json',
        )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'S0 config does not exist: {config_path}')
    with open(config_path, encoding='utf-8') as stream:
        config = json.load(stream)
    explicit = set(args._explicit_args)
    sections = [config.get('_defaults', {})]
    dataset_config = config.get(args.dataset, {})
    sections.append(dataset_config)
    sections.append(dataset_config.get(f'clients_{args.n_clients}', {}))
    for section in sections:
        if not isinstance(section, dict):
            raise TypeError('Every S0 config section must be a JSON object.')
        for name, value in section.items():
            if name.startswith('_') or name.startswith('clients_'):
                continue
            if not hasattr(args, name):
                raise KeyError(f'Unknown S0 config field: {name}')
            if name not in explicit:
                setattr(args, name, value)
    args.config_source = os.path.abspath(config_path)


def set_config(args):
    if args.dataset not in DATASET_META:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    _load_config(args)
    args.base_path = os.path.abspath(os.path.expanduser(args.base_path))
    args.mode = 'disjoint'
    args.n_feat, args.n_clss = DATASET_META[args.dataset]
    args.base_lr = float(args.lr)
    if min(
        args.n_workers,
        args.n_clients,
        args.n_rnds,
        args.n_eps,
        args.hidden_dim,
        args.ode_steps,
    ) <= 0:
        raise ValueError('Worker/client/training/model counts must be positive.')
    if not 0.0 < float(args.frac) <= 1.0:
        raise ValueError('frac must be in (0,1].')
    if args.adgn_step_size <= 0.0:
        raise ValueError('Official A-DGN adgn_step_size must be positive.')
    if args.ode_gamma < 0.0:
        raise ValueError('Official A-DGN gamma must be non-negative.')

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'baseline_official_adgn'
    if str(args.run_tag).strip():
        tag = f'{tag}_{_safe_tag(args.run_tag)}'
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{tag}'
    args.data_path = os.path.join(args.base_path, 'datasets')
    args.checkpt_path = os.path.join(args.base_path, 'checkpoints', trial)
    args.log_path = os.path.join(args.base_path, 'logs', trial)
    return args


def main(args):
    args = set_config(args)
    seed_everything(args.seed, torch_seed=True)
    print(
        f'[Run] model={args.model}, dataset={args.dataset}, '
        f'clients={args.n_clients}, seed={args.seed}, rounds={args.n_rnds}, '
        f'local_epochs={args.n_eps}, workers={args.n_workers}, gpu={args.gpu}, '
        f'aggregation={args.aggregation}, persistent_adam=True, '
        f'config={args.config_source}'
    )
    ParentProcess(args, Server, Client).start()


if __name__ == '__main__':
    main(Parser().parse())
