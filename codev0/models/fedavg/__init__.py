"""Reference two-layer GCN FedAvg with O4 block-v.

Note: runs the proposed O4 optimizer rather than the reference's plain Adam.
See ``models/fedavg/client.py`` for the deviation note.
"""

from models.fedavg.client import Client
from models.fedavg.server import Server

__all__ = ['Client', 'Server']
