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
