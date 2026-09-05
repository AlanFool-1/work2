"""CLI for the official A-DGN + FedAvg baseline."""

import argparse
import sys


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Fedrated-compatible official A-DGN baseline runner.'
        )
        self.set_arguments()

    def set_arguments(self):
        parser = self.parser
        parser.add_argument('--model', choices=['s0_ode'], default='s0_ode')
        parser.add_argument('--gpu', type=str, default='0,1')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--dataset', type=str, default='Synthetic')
        parser.add_argument(
            '--base-path', type=str, default='/opt/data/private/xzc/work2'
        )
        parser.add_argument('--config', type=str, default=None)
        parser.add_argument('--run-tag', dest='run_tag', type=str, default='')
        parser.add_argument(
            '--log-profile',
            choices=['minimal', 'diagnostic'],
            default='minimal',
            help='Diagnostic mode additionally records client/update metrics.',
        )

        parser.add_argument('--n-workers', type=int, default=10)
        parser.add_argument('--loader-workers', type=int, default=0)
        parser.add_argument('--n-clients', type=int, default=10)
        parser.add_argument('--n-rnds', type=int, default=100)
        parser.add_argument('--n-eps', type=int, default=2)
        parser.add_argument('--frac', type=float, default=1.0)
        parser.add_argument(
            '--aggregation', choices=['equal', 'weighted'], default='equal'
        )

        parser.add_argument('--hidden-dim', type=int, default=64)
        parser.add_argument(
            '--ode-steps', type=int, default=16,
            help='Number of shared-field Euler iterations in official A-DGN.',
        )
        parser.add_argument(
            '--adgn-step-size', type=float, default=0.1,
            help='Official A-DGN Forward-Euler step size epsilon.',
        )
        parser.add_argument('--ode-gamma', type=float, default=0.1)
        parser.add_argument(
            '--ode-activation', choices=['tanh', 'relu'], default='tanh'
        )
        parser.add_argument('--lr', type=float, default=0.015)
        parser.add_argument('--weight-decay', type=float, default=1e-4)
        parser.add_argument(
            '--synthetic-scenario',
            choices=[
                'iid', 'label_shift', 'feature_shift', 'feature_mixed',
                'structure_homophily', 'structure_degree',
                'structure_mixed', 'mixed',
            ],
            default='feature_shift',
            help='Controlled synthetic heterogeneity regime. Client 0 is always the reference.',
        )
        parser.add_argument('--synthetic-seed', type=int, default=2026)
        parser.add_argument('--synthetic-nodes-per-client', type=int, default=1000)
        parser.add_argument('--synthetic-train-ratio', type=float, default=0.2)
        parser.add_argument('--synthetic-val-ratio', type=float, default=0.2)
        parser.add_argument('--synthetic-class-separation', type=float, default=2.3)
        parser.add_argument('--synthetic-feature-std', type=float, default=1.7)
        parser.add_argument('--synthetic-feature-mean-shift', type=float, default=2.5)
        parser.add_argument('--synthetic-feature-std-shift', type=float, default=0.35)
        parser.add_argument('--synthetic-base-degree', type=float, default=12.0)
        parser.add_argument('--synthetic-degree-delta', type=float, default=12.0)
        parser.add_argument('--synthetic-base-homophily', type=float, default=0.65)
        parser.add_argument('--synthetic-homophily-delta', type=float, default=0.45)
        parser.add_argument(
            '--synthetic-regenerate', action='store_true',
            help='Force regeneration even if a matching synthetic benchmark already exists.',
        )
        parser.add_argument(
            '--synthetic-no-viz', action='store_true',
            help='Skip dataset-level synthetic heterogeneity plots.',
        )
        parser.add_argument(
            '--plot-initial-state', dest='plot_initial_state',
            action='store_true', default=None,
            help='Save and plot Graph ODE initial states H0.',
        )
        parser.add_argument(
            '--no-plot-initial-state', dest='plot_initial_state',
            action='store_false',
            help='Disable Graph ODE initial-state diagnostics.',
        )
        parser.add_argument(
            '--initial-state-snapshot-round', type=int, default=0,
            help='Zero-based round used to snapshot H0 before local training.',
        )
        parser.add_argument(
            '--initial-state-pca-max-points', type=int, default=1000,
            help='Maximum PCA fit points sampled from each client.',
        )
        parser.add_argument(
            '--initial-state-tsne-max-points', type=int, default=300,
            help='Maximum class-aware t-SNE points sampled from each client.',
        )
        parser.add_argument(
            '--initial-state-tsne-perplexity', type=float, default=30.0,
            help='Perplexity for the joint initial-state t-SNE embedding.',
        )
        parser.add_argument(
            '--plot-dynamics-html', action='store_true',
            help='Create a last-snapshot-round interactive H0...HT t-SNE HTML.',
        )
        parser.add_argument(
            '--dynamics-html-max-points-per-client', type=int, default=60,
            help='Class-aware nodes sampled per client and reused at every dynamics step.',
        )

    def parse(self):
        explicit_args = set()
        for token in sys.argv[1:]:
            if token.startswith('--'):
                explicit_args.add(
                    token[2:].split('=', 1)[0].replace('-', '_')
                )
        args, unparsed = self.parser.parse_known_args()
        args._explicit_args = sorted(explicit_args)
        if unparsed:
            raise SystemExit(f'Unknown argument: {unparsed}')
        return args
