"""Full coupled FedAvg for V0.2; Bayesian operator adaptation is not enabled."""

from models.s0.server import Server as S0Server
from modules.federated import ServerModule
from models.v02.model import build_v02_model
from models.v02.logger import V02StructuredLogger


class Server(S0Server):
    def __init__(self, args, sd, gpu_server):
        ServerModule.__init__(self, args, sd, gpu_server)
        self.model = build_v02_model(args).cuda(self.gpu_id)
        self.structured = V02StructuredLogger(args, self.model)
