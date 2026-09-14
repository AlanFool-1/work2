"""Normal full-model aggregation server for FedDGN.

The server is deliberately *not* converted into a local-only server.  It keeps
receiving client uploads and computing the ordinary global aggregate.  During
this stage the client-side broadcast installation line is commented out, so
that aggregate is simply not consumed by local models yet.
"""

from models.feddgn_local.model import build_feddgn_model
from modules.fed_server import FullModelServer


class Server(FullModelServer):
    def build_model(self):
        return build_feddgn_model(self.args)


__all__ = ['Server']
