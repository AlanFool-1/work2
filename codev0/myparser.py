"""Command-line arguments for the versioned federated experiment."""

import argparse
import sys


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Versioned methods on the Fedrated-compatible runtime.'
        )
        self._set_arguments()

    def _set_arguments(self):
        parser = self.parser
        parser.add_argument(
            '--model',
            choices=[
                'v04', 'v11', 'local', 'fedavg', 'fedpub', 'fedaux',
            ],
            default='v04',
        )
        parser.add_argument('--gpu', type=str, default='0,1')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument(
            '--base-path', type=str, default='/opt/data/private/xzc/work2'
        )
        parser.add_argument('--config', type=str, default=None)
        parser.add_argument('--run-tag', dest='run_tag', type=str, default='')
        parser.add_argument(
            '--log-profile',
            choices=['minimal', 'diagnostic'],
            default='minimal',
        )

        parser.add_argument('--n-workers', type=int, default=10)
        parser.add_argument('--loader-workers', type=int, default=0)
        parser.add_argument('--n-clients', type=int, default=10)
        parser.add_argument('--n-rnds', type=int, default=100)
        parser.add_argument('--n-eps', type=int, default=1)
        parser.add_argument('--frac', type=float, default=1.0)
        parser.add_argument(
            '--aggregation', choices=['equal', 'weighted'], default='equal'
        )
        parser.add_argument(
            '--agg-norm',
            choices=['cosine', 'exp'],
            default='exp',
            help='FedPub/FedAux client-similarity normalization.',
        )
        parser.add_argument('--norm-scale', type=float, default=5.0)
        parser.add_argument('--n-proxy', type=int, default=5)
        parser.add_argument('--proxy-nodes', type=int, default=100)
        parser.add_argument('--n-dims', type=int, default=128)
        parser.add_argument('--l1', type=float, default=1e-3)
        parser.add_argument('--loc-l2', type=float, default=1e-3)
        parser.add_argument('--sigma', type=float, default=0.1)
        parser.add_argument('--laye-mask-one', action='store_true', default=True)
        parser.add_argument('--clsf-mask-one', action='store_true', default=True)

        parser.add_argument('--encoder-width', type=int, default=64)
        parser.add_argument('--integration-time', type=float, default=1.6)
        parser.add_argument('--solver-step-size', type=float, default=0.1)
        parser.add_argument('--solver', choices=['rk4', 'dopri5'], default='rk4')
        parser.add_argument('--lr', type=float, default=0.003)
        parser.add_argument('--weight-decay', type=float, default=1e-4)
        parser.add_argument('--max-grad-norm', type=float, default=5.0)

        # Method V0.4. Defaults are repeated in configs/v04.json so every run
        # records one fully resolved and CLI-overridable configuration.
        parser.add_argument('--state-dim', type=int, default=64)
        parser.add_argument('--observable-aux-dim', type=int, default=32)
        parser.add_argument('--observable-width', type=int, default=64)
        parser.add_argument('--field-hidden-dim', type=int, default=64)
        parser.add_argument('--field-scale-init', type=float, default=0.1)
        parser.add_argument('--field-scale-min', type=float, default=0.001)
        parser.add_argument('--spectral-order', type=int, default=1)
        parser.add_argument('--generator-gamma', type=float, default=1e-4)
        parser.add_argument(
            '--generator-low-damping-init', type=float, default=1e-4
        )
        parser.add_argument(
            '--generator-high-damping-init', type=float, default=2.0
        )
        parser.add_argument('--classifier-width', type=int, default=64)
        parser.add_argument('--dropout', type=float, default=0.3)
        parser.add_argument(
            '--reference-mode', choices=['off', 'anchor'], default='anchor'
        )
        parser.add_argument('--lie-warmup-rounds', type=int, default=10)
        parser.add_argument('--lie-ramp-end-round', type=int, default=30)
        parser.add_argument('--lie-weight-max', type=float, default=0.05)
        parser.add_argument('--lie-speed-weight', type=float, default=0.0)
        parser.add_argument('--lr-main', type=float, default=0.003)
        parser.add_argument('--lr-field', type=float, default=0.001)
        parser.add_argument('--lr-probe', type=float, default=0.003)

        # Method 1.1 local Port-Hamiltonian Koopman-GNN. Defaults mirror
        # Method1LocalConfig "首轮配置" so a bare `--model v11` run reproduces
        # the settings recommended by the reference design document.
        parser.add_argument('--v11-hidden-dim', type=int, default=64)
        parser.add_argument('--v11-aux-dim', type=int, default=32)
        parser.add_argument('--v11-encoder-hidden', type=int, default=128)
        parser.add_argument('--v11-encoder-layers', type=int, default=2)
        parser.add_argument('--v11-input-dropout', type=float, default=0.20)
        parser.add_argument('--v11-encoder-dropout', type=float, default=0.35)
        parser.add_argument('--v11-classifier-dropout', type=float, default=0.50)
        parser.add_argument(
            '--v11-gauge-fix', action='store_true', default=True
        )
        parser.add_argument('--v11-observable-hidden', type=int, default=64)
        parser.add_argument(
            '--v11-observable-activation',
            choices=['tanh', 'gelu'],
            default='tanh',
        )
        parser.add_argument('--v11-aux-scale', type=float, default=1.0)
        parser.add_argument('--v11-basis-degree', type=int, default=2)
        parser.add_argument('--v11-operator-rank', type=int, default=8)
        parser.add_argument(
            '--v11-generator-type',
            choices=['low_rank_ph', 'full_rank_ph'],
            default='low_rank_ph',
        )
        parser.add_argument(
            '--v11-band-parameterization',
            choices=['shared_core', 'per_band'],
            default='shared_core',
        )
        parser.add_argument('--v11-gamma-init', type=float, default=1e-4)
        parser.add_argument('--v11-min-gamma', type=float, default=0.0)
        parser.add_argument(
            '--v11-learnable-gamma', action='store_true', default=True
        )
        parser.add_argument(
            '--v11-dissipation-basis-init', type=float, default=0.05
        )
        parser.add_argument(
            '--v11-dissipation-gain-init', type=float, default=1e-3
        )
        parser.add_argument(
            '--v11-conservative-init', type=float, default=0.05
        )
        parser.add_argument('--v11-ode-horizon', type=float, default=1.0)
        parser.add_argument('--v11-ode-steps', type=int, default=8)
        parser.add_argument(
            '--v11-solver', choices=['euler', 'heun', 'rk4'], default='rk4'
        )
        parser.add_argument(
            '--v11-symmetrize-edges', action='store_true', default=False
        )
        parser.add_argument(
            '--v11-remove-self-loops', action='store_true', default=True
        )
        parser.add_argument(
            '--v11-classifier-on', choices=['h', 'z'], default='h'
        )
        parser.add_argument('--v11-manifold-weight', type=float, default=1e-3)
        parser.add_argument('--v11-label-smoothing', type=float, default=0.05)
        parser.add_argument(
            '--v11-operator-speed-weight', type=float, default=1e-4
        )

        parser.add_argument(
            '--local-only',
            action='store_true',
            help=(
                'V0.4 only: ignore the broadcast model and optimizer state, '
                'keeping one continuous persistent-Adam local trajectory. The '
                'server still aggregates, so this is not a federated result.'
            ),
        )
        parser.add_argument(
            '--fed-optimizer',
            choices=['block_v', 'reset_adamw', 'reset_adam', 'persistent_adam'],
            default='block_v',
            help=(
                'V0.4 only: federated optimizer protocol. block_v is O4; the '
                'others replace it with a plain optimizer that is either reset '
                'or kept persistent across global receives.'
            ),
        )

    def parse(self):
        explicit = set()
        for token in sys.argv[1:]:
            if token.startswith('--'):
                explicit.add(token[2:].split('=', 1)[0].replace('-', '_'))
        args, unparsed = self.parser.parse_known_args()
        args._explicit_args = sorted(explicit)
        if unparsed:
            raise SystemExit(f'Unknown argument: {unparsed}')
        return args
