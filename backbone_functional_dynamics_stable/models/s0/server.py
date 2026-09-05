"""S0 full-model ODE-GNN FedAvg server on the Fedrated runtime."""

from __future__ import annotations

import numpy as np

from misc.utils import get_state_dict, set_state_dict, torch_save
from models.s0.model import build_s0_model
from modules.federated import ServerModule
from modules.structured_logger import StructuredLogger
from modules.functional_dynamics_visualizer import (
    plot_client_validation_curves,
    plot_cluster_assignments,
    plot_functional_dynamics_diagnostics,
)
from federation import FunctionalDynamicsAggregator


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)
        self.model = build_s0_model(args).cuda(self.gpu_id)
        self.structured = StructuredLogger(args)
        self.functional_aggregator = FunctionalDynamicsAggregator(
            ema_a=args.fd_ema_a, ema_b=args.fd_ema_b,
            num_prototypes=args.fd_num_prototypes,
            min_cluster_size=args.fd_min_cluster_size,
        )

    def on_round_begin(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        self.sd['global'] = self.get_weights()
        if self.args.enable_functional_dynamics:
            self.sd['functional_global'] = self.functional_aggregator.state_dict()
        self.structured.begin_round()

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def _aggregation_ratio(self, messages):
        if self.args.aggregation == 'equal':
            return None
        sizes = np.asarray(
            [item['train_size'] for item in messages], dtype=np.float64
        )
        if not np.isfinite(sizes).all() or float(sizes.sum()) <= 0.0:
            raise ValueError('Weighted FedAvg requires positive train sizes.')
        return (sizes / sizes.sum()).tolist()

    def update(self, updated):
        messages = [self.sd[client_id] for client_id in sorted(updated)]
        if not messages:
            raise RuntimeError('S0 server received no client updates.')
        reference = self.get_weights()
        self.structured.record_round(
            self.curr_rnd,
            messages,
            reference,
            self.args.aggregation,
        )
        ratio = self._aggregation_ratio(messages)
        aggregated = self.aggregate(
            [item['model'] for item in messages], ratio=ratio
        )
        set_state_dict(self.model, aggregated, self.gpu_id)
        if self.args.enable_functional_dynamics:
            if any('fd_A_transport' not in item for item in messages):
                raise RuntimeError('Missing client functional-dynamics payload.')
            weights = ratio
            if weights is None:
                weights = [1.0 / len(messages)] * len(messages)
            self.functional_aggregator.update(
                [item['fd_A_transport'] for item in messages],
                [item['fd_B_transport'] for item in messages],
                weights=weights,
                cluster_ids=[item.get('fd_cluster_id', -1) for item in messages],
                client_ids=[item['client_id'] for item in messages],
            )
        for client_id in updated:
            del self.sd[client_id]

    def get_weights(self):
        return get_state_dict(self.model)

    def save_state(self):
        torch_save(
            self.args.checkpt_path,
            'server_state.pt',
            {
                'round': self.curr_rnd + 1,
                'model': self.get_weights(),
                'aggregation': self.args.aggregation,
                **self.functional_aggregator.state_dict(),
                'secure_aggregation': 'simulated',
            },
        )

    def finalize(self):
        result = self.structured.finalize()
        validation_plot = plot_client_validation_curves(self.args.log_path)
        if validation_plot is not None:
            print(f'[validation] visualization={validation_plot}')
        if self.args.enable_functional_dynamics:
            cluster_plot = plot_cluster_assignments(self.args.log_path)
            if cluster_plot is not None:
                print(f'[functional-dynamics] visualization={cluster_plot}')
            state = self.functional_aggregator.state_dict()
            a_stars = state['A_star'] if self.args.fd_num_prototypes > 1 else [state['A_star']]
            b_stars = state['B_star'] if self.args.fd_num_prototypes > 1 else [state['B_star']]
            outputs = plot_functional_dynamics_diagnostics(self.args.log_path)
            for cluster, (a_star, b_star) in enumerate(zip(a_stars, b_stars)):
                outputs.extend(plot_functional_dynamics_diagnostics(
                    self.args.log_path, a_star, b_star,
                    canonical_suffix=(f'_cluster_{cluster}' if self.args.fd_num_prototypes > 1 else ''),
                    plot_curves=False,
                ))
            for output in outputs:
                print(f'[functional-dynamics] visualization={output}')
        return result
