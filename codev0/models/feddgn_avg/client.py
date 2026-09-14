"""FedAvg client with selectable optimizer-state semantics."""

from misc.utils import set_state_dict
from models.feddgn_local.client import Client as LocalClient
from modules.fed_optimizer import (
    FederatedAdamW,
    export_block_second_moment,
    initialize_second_moment_blocks,
    mapping_mean,
    optimizer_state_diagnostics,
)


class Client(LocalClient):
    """Install each global average and apply the selected optimizer protocol."""

    def _mode(self):
        return str(self.args.feddgn_fed_optimizer)

    def _build_block_v_optimizer(self, global_v_step, block_v):
        named_parameters = list(self.model.named_parameters())
        self.optimizer = FederatedAdamW(
            named_parameters,
            lr=self._lr(),
            weight_decay=float(self.args.weight_decay),
            global_v_step=int(global_v_step),
        )
        initialize_second_moment_blocks(
            self.optimizer, named_parameters, block_v
        )

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        set_state_dict(self.model, self.sd['global'], self.gpu_id)
        mode = self._mode()
        global_v_step = 0
        global_v_mean = 0.0
        if mode == 'persistent_adam':
            # Deliberate ablation: keep the old client moments while replacing
            # the model underneath them with the global average.
            pass
        elif mode == 'reset_adam':
            self._build_optimizer()
        elif mode == 'block_v':
            optimizer_meta = dict(self.sd.get('global_optimizer', {}))
            global_v_step = int(optimizer_meta.get('global_v_step', 0))
            block_v = optimizer_meta.get('main_v_blocks', {})
            self._build_block_v_optimizer(global_v_step, block_v)
            global_v_mean = mapping_mean(block_v)
        else:
            raise ValueError(f'Unsupported FedDGN optimizer mode: {mode}')
        self._global_optimizer_diagnostics = {
            'global_v_step': float(global_v_step),
            'global_v_block_mean': float(global_v_mean),
        }

    def train(self):
        super().train()
        self._round_result.update(optimizer_state_diagnostics(self.optimizer))
        self._round_result.update(self._global_optimizer_diagnostics)

    def transfer_to_server(self):
        super().transfer_to_server()
        payload = dict(self.sd[self.client_id])
        block_v = {}
        if self._mode() == 'block_v':
            block_v = export_block_second_moment(
                self.optimizer, list(self.model.named_parameters())
            )
        payload['main_v_blocks'] = block_v
        payload['upload_bytes'] += 8 * len(block_v)
        self.sd[self.client_id] = payload


__all__ = ['Client']
