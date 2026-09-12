"""FedAvg server for the reference two-layer GCN and O4 block-v.

``BlockVServer`` aggregates and re-broadcasts the block second moments, which
the reference FedAvg server never did (it only aggregated model weights). See
the deviation note in ``models/fedavg/client.py`` for the full rationale.
"""

from models.nets import GCN
from modules.fed_server import BlockVServer


class Server(BlockVServer):
    def build_model(self):
        return GCN(
            self.args.n_feat,
            self.args.n_dims,
            self.args.n_clss,
            self.args,
        )


__all__ = ['Server']
