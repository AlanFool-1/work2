"""V0.2 trajectory, component-loss, and full-model cost reporting."""

import numpy as np

from modules.structured_logger import StructuredLogger


class V02StructuredLogger(StructuredLogger):
    def __init__(self, args, model):
        super().__init__(args)
        self.total_bytes = 0
        self.scalar_fields = [
            'task_loss', 'native_task_loss', 'reconstruction_loss',
            'prediction_loss', 'linearity_loss', 'grad_norm',
            'latent_norm_ratio', 'max_latent_growth_ratio',
            'latent_effective_rank', 'native_relative_change', 'reconstruction_nmse',
        ] + [f'{metric}_h{horizon}' for horizon in model.horizons for metric in (
            'prediction_nmse', 'increment_nrmse', 'linearity_nmse',
        )]
        self.norm_fields = [f'z_norm_{step}' for step in range(model.num_steps + 1)]
        self.fields = ['round', *self.scalar_fields, *self.norm_fields,
                       'round_upload_bytes', 'cumulative_upload_bytes',
                       'round_download_bytes', 'cumulative_download_bytes',
                       'max_client_peak_cuda_memory_bytes']
        self._write_json('model_cost.json', {
            **model.parameter_counts(),
            'state_dict_bytes': sum(v.numel() * v.element_size() for v in model.state_dict().values()),
            'decoder_in_classification_path': True,
            'reference_transitions_needed_for_inference': False,
            'federated_payload': 'entire model including training-only reference modules',
            'correction_interval': model.correction_interval,
            'generator_mode': model.generator_mode,
            'loss_normalization': getattr(args, 'loss_normalization', 'pooled'),
            'loss_diagnostics': 'last local step, before update; trajectory diagnostics after update',
        })

    def record_round(self, round_id, messages, reference_state, aggregation):
        super().record_round(round_id, messages, reference_state, aggregation)
        row = {'round': int(round_id) + 1}
        for key in self.scalar_fields:
            row[key] = float(np.mean([item[key] for item in messages]))
        norms = np.mean([item['latent_norms'] for item in messages], axis=0)
        row.update(dict(zip(self.norm_fields, norms.tolist())))
        byte_count = int(sum(item['upload_bytes'] for item in messages))
        self.total_bytes += byte_count
        row.update({
            'round_upload_bytes': byte_count, 'round_download_bytes': byte_count,
            'cumulative_upload_bytes': self.total_bytes, 'cumulative_download_bytes': self.total_bytes,
            'max_client_peak_cuda_memory_bytes': max(item['peak_cuda_memory_bytes'] for item in messages),
        })
        self._append_csv('dynamics.csv', [row], self.fields)
