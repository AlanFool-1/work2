"""Shared full-model server lifecycles for reset and O4 block-v training."""

from __future__ import annotations

import numpy as np

from misc.utils import get_state_dict, set_state_dict, torch_save
from modules.fed_optimizer import aggregate_block_second_moments
from modules.federated import ServerModule
from modules.structured_logger import StructuredLogger


class FullModelServer(ServerModule):
    """Common model aggregation without a federated optimizer state."""

    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)
        self.model = self.build_model().cuda(self.gpu_id)
        self.structured = StructuredLogger(args)

    def build_model(self):
        raise NotImplementedError('A full-model server must provide its model.')

    def on_round_begin(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        self.sd['global'] = self.get_weights()
        self.broadcast_optimizer_state()
        self.structured.begin_round()

    def broadcast_optimizer_state(self):
        if 'global_optimizer' in self.sd:
            del self.sd['global_optimizer']

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def _aggregation_ratio(self, messages):
        if self.args.aggregation == 'equal':
            return None
        sizes = np.asarray(
            [message['train_size'] for message in messages], dtype=np.float64
        )
        if not np.isfinite(sizes).all() or float(sizes.sum()) <= 0.0:
            raise ValueError('Weighted aggregation requires positive train sizes.')
        return (sizes / sizes.sum()).tolist()

    def _aggregation_weights(self, messages):
        ratio = self._aggregation_ratio(messages)
        if ratio is None:
            return [1.0 / len(messages)] * len(messages)
        return ratio

    def update_optimizer_state(self, messages, weights):
        return None

    def update(self, updated):
        messages = [dict(self.sd[c_id]) for c_id in sorted(updated)]
        if not messages:
            raise RuntimeError('Federated server received no client updates.')
        reference = self.get_weights()
        self.structured.record_round(
            self.curr_rnd, messages, reference, self.args.aggregation
        )
        weights = self._aggregation_weights(messages)
        aggregated = self.aggregate(
            [message['model'] for message in messages], ratio=weights
        )
        self.update_optimizer_state(messages, weights)
        set_state_dict(self.model, aggregated, self.gpu_id)
        for client_id in updated:
            del self.sd[client_id]

    def get_weights(self):
        return get_state_dict(self.model)

    def checkpoint_optimizer_state(self):
        return {}

    def checkpoint_extra_state(self):
        return {}

    def save_state(self):
        torch_save(
            self.args.checkpt_path,
            'server_state.pt',
            {
                'round': self.curr_rnd + 1,
                'model': self.get_weights(),
                'aggregation': self.args.aggregation,
                'optimizer_state_mode': self.args.optimizer_state_mode,
                **self.checkpoint_optimizer_state(),
                **self.checkpoint_extra_state(),
            },
        )

    def finalize(self):
        return self.structured.finalize()


class BlockVServer(FullModelServer):
    """Full-model aggregation with the O4 block-v optimizer protocol."""

    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)
        self.global_main_v_blocks = {}
        self.global_v_step = 0

    def broadcast_optimizer_state(self):
        self.sd['global_optimizer'] = {
            'main_v_blocks': self.global_main_v_blocks,
            'global_v_step': int(self.global_v_step),
        }

    def update_optimizer_state(self, messages, weights):
        self.global_main_v_blocks = aggregate_block_second_moments(
            [message['main_v_blocks'] for message in messages], weights
        )
        self.global_v_step += int(self.args.n_eps)

    def checkpoint_optimizer_state(self):
        return {
            'global_main_v_blocks': self.global_main_v_blocks,
            'global_v_step': int(self.global_v_step),
        }
