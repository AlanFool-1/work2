"""FedAvg server with optional O4 block-v aggregation for FedDGN."""

from models.feddgn_local.model import build_feddgn_model
from modules.fed_server import BlockVServer, FullModelServer


class Server(BlockVServer):
    def build_model(self):
        return build_feddgn_model(self.args)

    def _uses_block_v(self):
        return str(self.args.feddgn_fed_optimizer) == 'block_v'

    def broadcast_optimizer_state(self):
        if self._uses_block_v():
            return super().broadcast_optimizer_state()
        return FullModelServer.broadcast_optimizer_state(self)

    def update_optimizer_state(self, messages, weights):
        if self._uses_block_v():
            return super().update_optimizer_state(messages, weights)
        return None

    def checkpoint_optimizer_state(self):
        if self._uses_block_v():
            return super().checkpoint_optimizer_state()
        return {}

__all__ = ['Server']
