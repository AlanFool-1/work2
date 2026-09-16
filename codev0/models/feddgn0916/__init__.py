"""Local FedDGN with dictionary FedAvg and passive Koopman candidates."""

from models.feddgn0916.client import Client
from models.feddgn0916.functional_map import solve_functional_map
from models.feddgn0916.candidate_pipeline import ConflictCandidateBuilder
from models.feddgn0916.personalized_koopman import fit_personalized_koopman
from models.feddgn0916.reskoop import PassiveResKoopNet, ResKoopDictionary
from models.feddgn0916.server import Server

__all__ = [
    'Client',
    'ConflictCandidateBuilder',
    'fit_personalized_koopman',
    'solve_functional_map',
    'PassiveResKoopNet',
    'ResKoopDictionary',
    'Server',
]
