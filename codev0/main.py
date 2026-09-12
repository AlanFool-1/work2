"""Main entry point for the versioned federated graph experiments."""

from __future__ import annotations

import json
import os
from datetime import datetime

from misc.utils import seed_everything
from models.fedaux import Client as FedAuxClient, Server as FedAuxServer
from models.fedavg import Client as FedAvgClient, Server as FedAvgServer
from models.fedpub import Client as FedPubClient, Server as FedPubServer
from models.local import Client as LocalClient, Server as LocalServer
from models.v04 import Client as V04Client, Server as V04Server
from models.v11 import Client as V11Client, Server as V11Server
from modules.multiprocs import ParentProcess
from myparser import Parser


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

MODEL_REGISTRY = {
    'v04': (V04Server, V04Client),
    'v11': (V11Server, V11Client),
    'local': (LocalServer, LocalClient),
    'fedavg': (FedAvgServer, FedAvgClient),
    'fedpub': (FedPubServer, FedPubClient),
    'fedaux': (FedAuxServer, FedAuxClient),
}

V04_MODELS = {'v04'}
V11_MODELS = {'v11'}
REFERENCE_MODELS = {'local', 'fedavg', 'fedpub', 'fedaux'}


def _safe_tag(value):
    return ''.join(
        character if character.isalnum() or character in '-_'
        else '_' for character in str(value)
    )


def _load_config(args):
    path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'configs',
        f'{args.model}.json',
    )
    if (
        args.config is None
        and not os.path.isfile(path)
        and args.model in REFERENCE_MODELS
    ):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs',
            'baselines.json',
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{args.model} config does not exist: {path}')
    with open(path, encoding='utf-8') as stream:
        config = json.load(stream)

    sections = [config.get('_defaults', {})]
    dataset_config = config.get(args.dataset, {})
    sections.extend([
        dataset_config,
        dataset_config.get(f'clients_{args.n_clients}', {}),
    ])
    explicit = set(args._explicit_args)
    for section in sections:
        if not isinstance(section, dict):
            raise TypeError('Every model config section must be a JSON object.')
        for name, value in section.items():
            if name.startswith('_') or name.startswith('clients_'):
                continue
            if not hasattr(args, name):
                raise KeyError(f'Unknown {args.model} config field: {name}')
            if name not in explicit:
                setattr(args, name, value)
    args.config_source = os.path.abspath(path)


def set_config(args):
    if args.dataset not in DATASET_META:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    # These are protocol metadata, not user-selectable CLI switches. Seed the
    # attributes so legacy config files cannot add a new command-line branch.
    args.local_optimizer = 'adam'
    args.optimizer_state_mode = 'not_applicable'
    _load_config(args)
    args.base_path = os.path.abspath(os.path.expanduser(args.base_path))
    args.mode = 'disjoint'
    args.n_feat, args.n_clss = DATASET_META[args.dataset]
    args.base_lr = float(args.lr)

    if min(args.n_workers, args.n_clients, args.n_rnds, args.n_eps) <= 0:
        raise ValueError('Worker/client/training counts must be positive.')
    if not 0.0 < float(args.frac) <= 1.0:
        raise ValueError('frac must lie in (0, 1].')
    if args.model in V11_MODELS:
        # Local-only backbone: the shared FedAdamW-v protocol does not apply.
        # Method1LocalConfig.validate() owns the remaining field checks.
        args.local_optimizer = 'adamw'
        args.optimizer_state_mode = 'local_only'
        if args.v11_operator_rank > args.v11_hidden_dim + args.v11_aux_dim:
            raise ValueError('operator_rank cannot exceed the observable width.')
    elif args.model in V04_MODELS:
        # O4 block-v is the default V0.4 optimizer protocol; the plain variants
        # exist only to locate how much of it the federated path needs.
        args.local_optimizer = 'adam'
        if args.local_only:
            args.optimizer_state_mode = 'local_persistent_adam'
        elif args.fed_optimizer == 'block_v':
            args.optimizer_state_mode = 'fedadamw_v_block'
        else:
            args.optimizer_state_mode = args.fed_optimizer
        dimensions = (
            args.state_dim,
            args.encoder_width,
            args.observable_width,
            args.field_hidden_dim,
            args.classifier_width,
        )
        if min(dimensions) <= 0 or args.observable_aux_dim < 0:
            raise ValueError('V0.4 dimensions are invalid.')
        if args.spectral_order != 1:
            raise ValueError('V0.4 Gate 0-4 require spectral_order=1.')
        if not 0 < args.field_scale_min < args.field_scale_init:
            raise ValueError('V0.4 field scales are invalid.')
        if args.generator_gamma <= 0:
            raise ValueError('generator_gamma must be positive.')
        if min(
            args.generator_low_damping_init,
            args.generator_high_damping_init,
        ) < 0:
            raise ValueError('Generator damping initialization must be non-negative.')
        if not 0.0 <= args.dropout < 1.0:
            raise ValueError('dropout must lie in [0, 1).')
        if min(args.lr_main, args.lr_field, args.lr_probe) <= 0:
            raise ValueError('V0.4 learning rates must be positive.')
        if args.lie_warmup_rounds < 0:
            raise ValueError('lie_warmup_rounds must be non-negative.')
        if args.lie_ramp_end_round <= args.lie_warmup_rounds:
            raise ValueError('Lie ramp must end after warmup.')
        if args.lie_weight_max < 0 or args.lie_speed_weight != 0:
            raise ValueError(
                'V0.4 Gate 0-4 require non-negative Lie direction weight '
                'and lie_speed_weight=0.'
            )
        if min(
            args.integration_time,
            args.solver_step_size,
            args.max_grad_norm,
        ) <= 0:
            raise ValueError('Time and gradient limits must be positive.')
    elif args.model in REFERENCE_MODELS:
        if args.n_dims <= 0:
            raise ValueError('Reference GCN width must be positive.')
        if not 0.0 <= args.dropout < 1.0:
            raise ValueError('Reference GCN dropout must lie in [0, 1).')
        if min(args.base_lr, args.sigma) <= 0:
            raise ValueError('Reference optimizer and kernel scales must be positive.')
        if min(args.l1, args.loc_l2, args.weight_decay) < 0:
            raise ValueError('Reference regularization weights must be non-negative.')
        if args.n_proxy <= 0 or args.proxy_nodes <= 1:
            raise ValueError('FedPub proxy dimensions are invalid.')
        # Protocol metadata only; it is recorded in the server checkpoint and
        # the structured log, and never selects a code path. Values describe
        # what each client actually does, not what the paper specifies.
        if args.model == 'local':
            args.optimizer_state_mode = 'local_persistent_adam'
        elif args.model == 'fedavg':
            # Deliberate deviation: O4 block-v instead of the reference's plain
            # persistent Adam. See models/fedavg/client.py.
            args.optimizer_state_mode = 'fedadamw_v_block'
        else:
            # FedPub/FedAux keep the reference's Adam, whose moments persist
            # across rounds via save_state/load_state.
            args.optimizer_state_mode = 'persistent_adam'
    else:
        raise ValueError(f'Unknown model: {args.model}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = args.model
    if str(args.run_tag).strip():
        tag = f'{tag}_{_safe_tag(args.run_tag)}'
    trial = (
        f'{args.dataset}_{args.mode}/clients_{args.n_clients}/'
        f'{timestamp}_{tag}'
    )
    args.data_path = os.path.join(args.base_path, 'datasets')
    args.checkpt_path = os.path.join(args.base_path, 'checkpoints', trial)
    args.log_path = os.path.join(args.base_path, 'logs', trial)
    return args


def main(args):
    args = set_config(args)
    seed_everything(args.seed, torch_seed=True)
    server_class, client_class = MODEL_REGISTRY[args.model]
    print(
        f'[Run] model={args.model}, dataset={args.dataset}, clients={args.n_clients}, '
        f'seed={args.seed}, rounds={args.n_rnds}, local_epochs={args.n_eps}, '
        f'workers={args.n_workers}, gpu={args.gpu}, '
        f'aggregation={args.aggregation}, '
        f'local_optimizer={args.local_optimizer}, '
        f'optimizer_state_mode={args.optimizer_state_mode}, '
        f'config={args.config_source}'
    )
    ParentProcess(args, server_class, client_class).start()


if __name__ == '__main__':
    main(Parser().parse())
