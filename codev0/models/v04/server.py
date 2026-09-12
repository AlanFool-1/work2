"""Full-model equal/weighted FedAvg server for Method V0.4."""

from __future__ import annotations

from models.v04.model import build_v04_model
from modules.fed_server import BlockVServer


class Server(BlockVServer):
    def build_model(self):
        return build_v04_model(self.args)


__all__ = ['Server']
