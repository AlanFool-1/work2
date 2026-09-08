"""Federated client for Method V0.1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from misc.utils import get_state_dict
from models.s0.client import Client as S0Client
from models.v01.model import build_v01_model
from modules.federated import ClientModule


class Client(S0Client):
    """Reuse S0's checkpoint/metric lifecycle with a V0.1 train step."""

    def __init__(self, args, w_id, g_id, sd):
        ClientModule.__init__(self, args, w_id, g_id, sd)
        self.model = build_v01_model(args).cuda(g_id)
        self._last_v01_diagnostics = {}
        self._build_optimizer()

    @staticmethod
    def _spectral_norm(matrix: torch.Tensor) -> float:
        return float(torch.linalg.svdvals(matrix.detach()).max().item())

    @torch.no_grad()
    def _dynamics_diagnostics(self, trajectory: torch.Tensor) -> dict:
        flat = trajectory.detach().reshape(trajectory.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=1)
        if flat.shape[0] == 1:
            max_growth = 1.0
            max_relative_delta = 0.0
        else:
            denominator = norms[:-1].clamp_min(1e-12)
            growth = norms[1:] / denominator
            relative_delta = torch.linalg.vector_norm(
                flat[1:] - flat[:-1], dim=1
            ) / denominator
            max_growth = float(growth.max().item())
            max_relative_delta = float(relative_delta.max().item())
        return {
            'latent_norms': [float(value) for value in norms.cpu().tolist()],
            'max_latent_growth_ratio': max_growth,
            'max_latent_relative_delta': max_relative_delta,
            'a0_spectral_norm': self._spectral_norm(
                self.model.self_generator()
            ),
            'a1_spectral_norm': self._spectral_norm(
                self.model.neighbor_generator
            ),
        }

    def _train_step(self):
        batch = self._first_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model.forward_with_aux(batch)
        task_loss = self._loss(
            output.logits, batch.y, batch.train_mask
        )
        if output.reconstruction is None:
            reconstruction_loss = output.logits.new_zeros(())
        else:
            reconstruction_loss = F.mse_loss(
                output.reconstruction, batch.x
            )
        self._last_train_lss = task_loss + float(
            self.args.reconstruction_weight
        ) * reconstruction_loss
        if not torch.isfinite(self._last_train_lss):
            raise FloatingPointError(
                f'Non-finite V0.1 loss on client {self.client_id}.'
            )
        self._last_train_lss.backward()
        if any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ):
            raise FloatingPointError(
                f'Non-finite V0.1 gradient on client {self.client_id}.'
            )
        self.optimizer.step()

        self._last_v01_diagnostics = {
            'task_loss': float(task_loss.detach().item()),
            'reconstruction_loss': float(
                reconstruction_loss.detach().item()
            ),
            **self._dynamics_diagnostics(output.latent_trajectory),
        }

    def train(self):
        torch.cuda.reset_peak_memory_stats(self.gpu_id)
        super().train()
        self._round_result.update({
            **self._last_v01_diagnostics,
            'peak_cuda_memory_bytes': int(
                torch.cuda.max_memory_allocated(self.gpu_id)
            ),
        })

    def transfer_to_server(self):
        batch = self._first_batch()
        weights = get_state_dict(self.model)
        upload_bytes = sum(value.nbytes for value in weights.values())
        self.sd[self.client_id] = {
            'client_id': int(self.client_id),
            'model': weights,
            'train_size': int(batch.train_mask.sum().item()),
            'upload_bytes': int(upload_bytes),
            **self._round_result,
        }
