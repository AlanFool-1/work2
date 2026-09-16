"""Command-line arguments for the versioned federated experiment."""

import argparse
import sys

MODELS = [
    'v04', 'feddgn', 'feddgn_local', 'feddgn_avg',
    'local', 'fedavg', 'fedpub', 'fedaux',
]


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Versioned methods on the Fedrated-compatible runtime.'
        )
        self._set_arguments()

    @staticmethod
    def _add(group, name, default, **kwargs):
        group.add_argument(f'--{name.replace("_", "-")}', default=default,
                           **kwargs)

    @classmethod
    def _int(cls, group, name, default, **kwargs):
        cls._add(group, name, default, type=int, **kwargs)

    @classmethod
    def _float(cls, group, name, default, **kwargs):
        cls._add(group, name, default, type=float, **kwargs)

    @classmethod
    def _str(cls, group, name, default, **kwargs):
        cls._add(group, name, default, type=str, **kwargs)

    @classmethod
    def _flag(cls, group, name, default=False, **kwargs):
        cls._add(group, name, default, action='store_true', **kwargs)

    def _set_arguments(self):
        run = self.parser.add_argument_group('run')
        self._str(run, 'model', 'v04', choices=MODELS)
        self._str(run, 'gpu', '0,1')
        self._int(run, 'seed', 42)
        self._str(run, 'dataset', None, required=True)
        self._str(run, 'base_path', '/opt/data/private/xzc/work2')
        self._str(run, 'config', None)
        self._str(run, 'run_tag', '')
        self._str(run, 'log_profile', 'minimal',
                  choices=['minimal', 'diagnostic'])

        fed = self.parser.add_argument_group('federated')
        self._int(fed, 'n_workers', 10)
        self._int(fed, 'loader_workers', 0)
        self._int(fed, 'n_clients', 10)
        self._int(fed, 'n_rnds', 100)
        self._int(fed, 'n_eps', 1)
        self._float(fed, 'frac', 1.0)
        self._str(fed, 'aggregation', 'equal', choices=['equal', 'weighted'])
        self._flag(fed, 'local_only', help=(
            'V0.4/FedDGN local mode: ignore the broadcast model and optimizer '
            'state, keeping one continuous persistent-Adam local trajectory. '
            'The server still aggregates, so this is not a federated result.'
        ))
        self._str(fed, 'fed_optimizer', 'block_v', help=(
            'V0.4 only: federated optimizer protocol. block_v is O4; the '
            'others replace it with a plain optimizer that is either reset '
            'or kept persistent across global receives.'
        ), choices=['block_v', 'reset_adamw', 'reset_adam', 'persistent_adam'])

        ref = self.parser.add_argument_group('reference models')
        self._str(ref, 'agg_norm', 'exp', choices=['cosine', 'exp'],
                  help='FedPub/FedAux client-similarity normalization.')
        self._float(ref, 'norm_scale', 5.0)
        self._int(ref, 'n_proxy', 5)
        self._int(ref, 'proxy_nodes', 100)
        self._int(ref, 'n_dims', 128)
        self._float(ref, 'l1', 1e-3)
        self._float(ref, 'loc_l2', 1e-3)
        self._float(ref, 'sigma', 0.1)
        self._flag(ref, 'laye_mask_one', default=True)
        self._flag(ref, 'clsf_mask_one', default=True)

        opt = self.parser.add_argument_group('shared optimizer')
        self._float(opt, 'lr', 0.003)
        self._float(opt, 'weight_decay', 1e-4)
        self._float(opt, 'max_grad_norm', 5.0)

        v04 = self.parser.add_argument_group('V0.4')
        self._int(v04, 'encoder_width', 64)
        self._float(v04, 'integration_time', 1.6)
        self._float(v04, 'solver_step_size', 0.1)
        self._str(v04, 'solver', 'rk4', choices=['rk4', 'dopri5'])
        self._int(v04, 'state_dim', 64)
        self._int(v04, 'observable_aux_dim', 32)
        self._int(v04, 'observable_width', 64)
        self._int(v04, 'field_hidden_dim', 64)
        self._float(v04, 'field_scale_init', 0.1)
        self._float(v04, 'field_scale_min', 0.001)
        self._int(v04, 'spectral_order', 1)
        self._float(v04, 'generator_gamma', 1e-4)
        self._float(v04, 'generator_low_damping_init', 1e-4)
        self._float(v04, 'generator_high_damping_init', 2.0)
        self._int(v04, 'classifier_width', 64)
        self._float(v04, 'dropout', 0.3)
        self._str(v04, 'reference_mode', 'anchor', choices=['off', 'anchor'])
        self._int(v04, 'lie_warmup_rounds', 10)
        self._int(v04, 'lie_ramp_end_round', 30)
        self._float(v04, 'lie_weight_max', 0.05)
        self._float(v04, 'lr_main', 0.003)
        self._float(v04, 'lr_field', 0.001)
        self._float(v04, 'lr_probe', 0.003)

        # FedDGN: official-style A-DGN dynamics with an inactive force port.
        dgn = self.parser.add_argument_group('FedDGN backbone')
        self._int(dgn, 'feddgn_hidden_dim', 64)
        self._int(dgn, 'feddgn_steps', 16)
        self._float(dgn, 'feddgn_step_size', 0.1)
        self._float(dgn, 'feddgn_gamma', 0.1)
        self._str(dgn, 'feddgn_activation', 'tanh',
                  choices=['tanh', 'relu', 'gelu'])
        self._float(dgn, 'feddgn_input_dropout', 0.0)
        self._float(dgn, 'feddgn_classifier_dropout', 0.0)
        self._str(dgn, 'feddgn_head', 'linear', choices=['mlp', 'linear'])
        self._float(dgn, 'feddgn_lr', 0.015)
        self._float(dgn, 'feddgn_label_smoothing', 0.0)
        self._float(dgn, 'feddgn_max_grad_norm', 0.0,
                    help='FedDGN gradient clipping threshold; 0 disables.')
        self._str(dgn, 'feddgn_fed_optimizer', 'reset_adam', choices=[
            'persistent_adam', 'reset_adam', 'block_v',
        ])

        # FedDGN local Koopman fit: a private K per client, with only the
        # observable dictionary aggregated across clients.
        koo = self.parser.add_argument_group('FedDGN Koopman')
        self._int(koo, 'feddgn_koopman_aux_dim', 16)
        self._int(koo, 'feddgn_koopman_hidden_dim', 64)
        self._int(koo, 'feddgn_koopman_hidden_layers', 2)
        self._float(koo, 'feddgn_koopman_ridge', 1e-3)
        self._int(koo, 'feddgn_koopman_max_snapshots', 8192)
        self._int(koo, 'feddgn_koopman_outer_steps', 1)
        self._int(koo, 'feddgn_koopman_inner_steps', 4)
        self._int(koo, 'feddgn_koopman_batch_size', 1024)
        self._float(koo, 'feddgn_koopman_lr', 1e-3)
        self._float(koo, 'feddgn_koopman_weight_decay', 1e-5)
        self._float(koo, 'feddgn_koopman_grad_clip', 5.0)
        self._int(koo, 'feddgn_koopman_save_every', 10)

    def parse(self):
        explicit = set()
        for token in sys.argv[1:]:
            if token.startswith('--'):
                name = token[2:].split('=', 1)[0].replace('-', '_')
                explicit.add(name)
                # argparse.BooleanOptionalAction spells the negative switch
                # as --no-<name>, but configuration precedence is tracked by
                # the positive destination name.
                if name.startswith('no_'):
                    explicit.add(name[3:])
        args, unparsed = self.parser.parse_known_args()
        args._explicit_args = sorted(explicit)
        if unparsed:
            raise SystemExit(f'Unknown argument: {unparsed}')
        return args
