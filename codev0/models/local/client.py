"""Local ordinary-GCN client that deliberately ignores server broadcasts."""

from models.fedavg.client import Client as FedAvgClient


class Client(FedAvgClient):
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = int(curr_rnd)
        self._global_optimizer_diagnostics = {
            'global_v_step': 0.0,
            'global_v_block_mean': 0.0,
        }

    def optimizer_state_upload(self):
        return {}, 0
