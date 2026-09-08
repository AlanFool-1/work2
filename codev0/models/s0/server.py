"""S0 full-model ODE-GNN FedAvg server on the Fedrated runtime."""

from __future__ import annotations

import numpy as np

from misc.utils import get_state_dict, set_state_dict, torch_save
from models.s0.model import build_s0_model
from modules.federated import ServerModule
from modules.structured_logger import StructuredLogger


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)
        self.model = build_s0_model(args).cuda(self.gpu_id)
        self.structured = StructuredLogger(args)

    def on_round_begin(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        self.sd['global'] = self.get_weights()
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
            },
        )

    def finalize(self):
        return self.structured.finalize()
