"""Local-baseline server; aggregation runs but clients do not consume it."""

from models.nets import GCN
from modules.fed_server import FullModelServer


class Server(FullModelServer):
    def build_model(self):
        return GCN(
            self.args.n_feat,
            self.args.n_dims,
            self.args.n_clss,
            self.args,
        )

__all__ = ['Server']
