"""Ordinary-GCN local baseline in the full federated lifecycle."""

from models.local.client import Client
from models.local.server import Server

__all__ = ['Client', 'Server']
