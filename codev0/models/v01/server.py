"""Full-model FedAvg server for Method V0.1."""

from __future__ import annotations

from models.s0.server import Server as S0Server
from models.v01.logger import V01StructuredLogger
from models.v01.model import build_v01_model
from modules.federated import ServerModule


class Server(S0Server):
    """Reuse S0 aggregation/checkpoint semantics with the V0.1 model."""

    def __init__(self, args, sd, gpu_server):
        ServerModule.__init__(self, args, sd, gpu_server)
        self.model = build_v01_model(args).cuda(self.gpu_id)
        self.structured = V01StructuredLogger(args, self.model)
