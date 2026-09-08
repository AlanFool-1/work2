"""Entry point for S0, V0.1, and V0.2 federated graph models."""

from __future__ import annotations

import json
import os
from datetime import datetime

from myparser import Parser
from misc.utils import seed_everything
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
        config_name = {
            's0_ode': 's0_ode_gnn.json',
            'v01_linear': 'v01_linear_gnn.json',
            'v02_koopman': 'v02_koopman_gnn.json',
        }[args.model]
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs',
            config_name,
        )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Model config does not exist: {config_path}')
    with open(config_path, encoding='utf-8') as stream:
        config = json.load(stream)
    explicit = set(args._explicit_args)
    sections = [config.get('_defaults', {})]
    dataset_config = config.get(args.dataset, {})
    sections.append(dataset_config)
    sections.append(dataset_config.get(f'clients_{args.n_clients}', {}))
    for section in sections:
        if not isinstance(section, dict):
            raise TypeError('Every model config section must be a JSON object.')
        for name, value in section.items():
            if name.startswith('_') or name.startswith('clients_'):
                continue
            if not hasattr(args, name):
                raise KeyError(f'Unknown model config field: {name}')
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
    if min(args.n_workers, args.n_clients, args.n_rnds, args.n_eps) <= 0:
        raise ValueError('Worker/client/training counts must be positive.')
    if not 0.0 < float(args.frac) <= 1.0:
        raise ValueError('frac must be in (0,1].')
    if args.model == 's0_ode':
        if min(args.hidden_dim, args.ode_steps) <= 0:
            raise ValueError('Official A-DGN dimensions must be positive.')
        if args.adgn_step_size <= 0.0:
            raise ValueError('Official A-DGN adgn_step_size must be positive.')
        if args.ode_gamma < 0.0:
            raise ValueError('Official A-DGN gamma must be non-negative.')
    elif args.model in {'v01_linear', 'v02_koopman'}:
        if min(
            args.latent_dim,
            args.encoder_width,
            args.linear_steps,
        ) <= 0:
            raise ValueError('V0.1 dimensions and steps must be positive.')
        if args.linear_step_size <= 0.0:
            raise ValueError('V0.1 linear_step_size must be positive.')
        if args.linear_gamma < 0.0:
            raise ValueError('V0.1 linear_gamma must be non-negative.')
        if args.reconstruction_weight < 0.0:
            raise ValueError('V0.1 reconstruction_weight must be non-negative.')
        if args.model == 'v02_koopman':
            from models.v02.training import LossWeights
            LossWeights(args.native_weight, args.reconstruction_weight,
                        args.prediction_weight, args.linearity_weight)
            if min(args.hidden_dim, args.generator_norm_bound, args.max_grad_norm) <= 0:
                raise ValueError('V0.2 dimensions and norm limits must be positive.')
            if args.correction_interval < 0 or (args.identity_dynamics and args.correction_interval):
                raise ValueError('Invalid V0.2 correction/identity configuration.')
    else:
        raise ValueError(f'Unknown model: {args.model}')

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = {
        's0_ode': 'baseline_official_adgn',
        'v01_linear': 'method_v01_linear',
        'v02_koopman': 'method_v02_koopman',
    }[args.model]
    if str(args.run_tag).strip():
        tag = f'{tag}_{_safe_tag(args.run_tag)}'
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{tag}'
    args.data_path = os.path.join(args.base_path, 'datasets')
    args.checkpt_path = os.path.join(args.base_path, 'checkpoints', trial)
    args.log_path = os.path.join(args.base_path, 'logs', trial)
    return args


def _model_components(model_name):
    if model_name == 's0_ode':
        from models.s0 import Client, Server
    elif model_name == 'v01_linear':
        from models.v01 import Client, Server
    elif model_name == 'v02_koopman':
        from models.v02.client import Client
        from models.v02.server import Server
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return Client, Server


def main(args):
    args = set_config(args)
    Client, Server = _model_components(args.model)
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
