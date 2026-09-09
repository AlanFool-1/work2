"""V0.2 client using the existing persistent-Adam FedAvg lifecycle."""

import torch

from misc.utils import get_state_dict
from models.s0.client import Client as S0Client
from modules.federated import ClientModule
from models.v02.model import build_v02_model
from models.v02.training import LossWeights, checked_step, objective, trajectory_diagnostics


class Client(S0Client):
    def __init__(self, args, w_id, g_id, sd):
        ClientModule.__init__(self, args, w_id, g_id, sd)
        self.model = build_v02_model(args).cuda(g_id)
        self.loss_weights = LossWeights(
            args.native_weight, args.reconstruction_weight,
            args.prediction_weight, args.linearity_weight,
        )
        self._build_optimizer()

    def _train_step(self):
        batch = self._first_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model.forward_with_aux(batch, auxiliary=self.loss_weights.auxiliary)
        loss, values = objective(self.model, output, batch.y, batch.train_mask, self.loss_weights,
                                 normalization=getattr(self.args, 'loss_normalization', 'pooled'))
        grad_norm = checked_step(loss, self.model, self.optimizer, self.args.max_grad_norm)
        self._last_train_lss = loss.detach()
        self._last_v02 = {key: float(value.detach()) for key, value in values.items()}
        self._last_v02['grad_norm'] = grad_norm

    def train(self):
        torch.cuda.reset_peak_memory_stats(self.gpu_id)
        super().train()
        with torch.no_grad():
            self.model.eval()
            output = self.model.forward_with_aux(self._first_batch(), auxiliary=False)
            self._last_v02.update(trajectory_diagnostics(self.model, output))
        self._round_result.update(self._last_v02)
        self._round_result['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(self.gpu_id)

    def transfer_to_server(self):
        state = get_state_dict(self.model)
        self.sd[self.client_id] = {
            'client_id': int(self.client_id), 'model': state,
            'train_size': int(self._first_batch().train_mask.sum()),
            'upload_bytes': sum(value.nbytes for value in state.values()),
            **self._round_result,
        }
