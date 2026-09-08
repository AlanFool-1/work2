"""V0.1-specific dynamics and cost logging."""

from __future__ import annotations

import numpy as np

from modules.structured_logger import StructuredLogger


class V01StructuredLogger(StructuredLogger):
    def __init__(self, args, model):
        self.dynamics_fields = [
            'round',
            'task_loss',
            'reconstruction_loss',
            'a0_spectral_norm',
            'a1_spectral_norm',
            'max_latent_growth_ratio',
            'max_latent_relative_delta',
            'mean_client_elapsed_seconds',
            'max_client_peak_cuda_memory_bytes',
            'round_upload_bytes',
            'round_download_bytes',
            'cumulative_upload_bytes',
            'cumulative_download_bytes',
        ] + [f'z_norm_{step}' for step in range(args.linear_steps + 1)]
        self.cumulative_upload_bytes = 0
        self.cumulative_download_bytes = 0
        super().__init__(args)
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.state_dict().values()
        )
        self._write_json(
            'model_cost.json',
            {
                'trainable_parameters': int(trainable_parameters),
                'state_dict_bytes': int(state_bytes),
                'latent_dim': int(args.latent_dim),
                'linear_steps': int(args.linear_steps),
                'reconstruction_enabled': bool(
                    float(args.reconstruction_weight) > 0.0
                ),
            },
        )

    def record_round(self, round_id, messages, reference_state, aggregation):
        super().record_round(round_id, messages, reference_state, aggregation)
        ordered = sorted(messages, key=lambda item: int(item['client_id']))
        norms = np.asarray(
            [item['latent_norms'] for item in ordered], dtype=np.float64
        )
        round_upload_bytes = int(sum(
            item['upload_bytes'] for item in ordered
        ))
        # Every selected client receives one copy of the same full state that
        # it later uploads. V0.1 has no optimizer buffers in either transfer.
        round_download_bytes = round_upload_bytes
        self.cumulative_upload_bytes += round_upload_bytes
        self.cumulative_download_bytes += round_download_bytes
        row = {
            'round': int(round_id) + 1,
            'task_loss': float(np.mean([
                item['task_loss'] for item in ordered
            ])),
            'reconstruction_loss': float(np.mean([
                item['reconstruction_loss'] for item in ordered
            ])),
            'a0_spectral_norm': float(np.mean([
                item['a0_spectral_norm'] for item in ordered
            ])),
            'a1_spectral_norm': float(np.mean([
                item['a1_spectral_norm'] for item in ordered
            ])),
            'max_latent_growth_ratio': float(max(
                item['max_latent_growth_ratio'] for item in ordered
            )),
            'max_latent_relative_delta': float(max(
                item['max_latent_relative_delta'] for item in ordered
            )),
            'mean_client_elapsed_seconds': float(np.mean([
                item['elapsed_seconds'] for item in ordered
            ])),
            'max_client_peak_cuda_memory_bytes': int(max(
                item['peak_cuda_memory_bytes'] for item in ordered
            )),
            'round_upload_bytes': round_upload_bytes,
            'round_download_bytes': round_download_bytes,
            'cumulative_upload_bytes': self.cumulative_upload_bytes,
            'cumulative_download_bytes': self.cumulative_download_bytes,
        }
        for step in range(norms.shape[1]):
            row[f'z_norm_{step}'] = float(norms[:, step].mean())
        self._append_csv(
            'dynamics.csv', [row], self.dynamics_fields
        )
