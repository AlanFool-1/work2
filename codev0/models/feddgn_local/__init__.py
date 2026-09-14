"""FedDGN: clean local A-DGN-style dynamics with a future force port."""

from models.feddgn_local.model import (
    FedDGNBackbone,
    FedDGNOutput,
    build_feddgn_model,
)

__all__ = [
    'Client',
    'Server',
    'FedDGNBackbone',
    'FedDGNOutput',
    'build_feddgn_model',
]


def __getattr__(name):
    if name == 'Client':
        from models.feddgn_local.client import Client
        return Client
    if name == 'Server':
        from models.feddgn_local.server import Server
        return Server
    raise AttributeError(name)
