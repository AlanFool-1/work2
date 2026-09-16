"""Local FedDGN backbone with a private ResKoop K fit and dictionary FedAvg."""

from models.feddgn.client import Client
from models.feddgn.reskoop import PassiveResKoopNet, ResKoopDictionary
from models.feddgn.server import Server

__all__ = [
    'Client',
    'PassiveResKoopNet',
    'ResKoopDictionary',
    'Server',
]
