"""FedPub client with personalized model mixing and cross-round Adam."""

from __future__ import annotations

import numpy as np
import torch

from models.fedavg.client import Client as FedAvgClient
from models.nets import MaskedGCN


class Client(FedAvgClient):
    def build_model(self):
        return MaskedGCN(
            self.args.n_feat,
            self.args.n_dims,
            self.args.n_clss,
            self.args.l1,
            self.args,
        )

    def _model_state_for_round(self):
        key = f'personalized_{self.client_id}'
        return self.sd[key] if key in self.sd else self.sd['global']

    def on_receive_message(self, curr_rnd):
        model_state = self._model_state_for_round()
        self.prev_w = {
            name: torch.as_tensor(value, device=f'cuda:{self.gpu_id}')
            for name, value in model_state.items()
        }
        self._install_model_state(curr_rnd, model_state, preserve_masks=True)
        # The optimizer is intentionally NOT rebuilt here. The reference
        # implementation (Fedrated/models/fedpub/client.py) creates Adam once in
        # init_state() and its moments persist across rounds: the worker
        # lifecycle is switch_state -> on_receive_message -> on_round_begin ->
        # save_state, so load_state() restores the moments each round while only
        # the model weights are overwritten by the broadcast above.
        self._global_optimizer_diagnostics = {
            'global_v_step': 0.0,
            'global_v_block_mean': 0.0,
        }

    def regularization_loss(self):
        regularization = 0.0
        for name, parameter in self.model.state_dict().items():
            if 'mask' in name:
                regularization = regularization + (
                    torch.norm(parameter.float(), 1) * self.args.l1
                )
            elif self.curr_rnd > 0 and ('conv' in name or 'clsif' in name):
                regularization = regularization + (
                    torch.norm(parameter.float() - self.prev_w[name], 2)
                    * self.args.loc_l2
                )
        return regularization

    @torch.no_grad()
    def get_functional_embedding(self):
        self.model.eval()
        proxy = self.sd['proxy'].cuda(self.gpu_id)
        embedding = self.model(proxy, is_proxy=True).mean(dim=0)
        return embedding.detach().cpu().numpy().astype(np.float64, copy=True)

    def extra_upload(self):
        embedding = self.get_functional_embedding()
        return {'functional_embedding': embedding}, int(embedding.nbytes)

    def optimizer_state_upload(self):
        return {}, 0
