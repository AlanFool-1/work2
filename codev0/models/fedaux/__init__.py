"""Reference auxiliary MaskedGCN FedAux with round-reset Adam."""

from models.fedaux.client import Client
from models.fedaux.server import Server

__all__ = ['Client', 'Server']
