"""Method 1.1 server.

The backbone is local-only, so aggregation runs but is never consumed by the
clients. This mirrors `models/local/`; the entry exists so the normal lifecycle,
logging and result reporting stay identical across methods.
"""

from __future__ import annotations

from models.v11.builder import build_v11_model
from modules.fed_server import FullModelServer


class Server(FullModelServer):
    def build_model(self):
        return build_v11_model(self.args)


__all__ = ['Server']
