import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.v01.logger import V01StructuredLogger
from models.v01.model import LatentLinearGraphDynamics


def message(client_id, upload_bytes, latent_norms):
    return {
        'client_id': client_id,
        'model': {'weight': np.asarray([1.0 + 0.1 * client_id])},
        'train_size': 10 + client_id,
        'train_loss': 1.0,
        'val_loss': 0.8,
        'val_accuracy': 0.6 + 0.1 * client_id,
        'test_loss': 0.7,
        'test_accuracy': 0.55 + 0.1 * client_id,
        'test_f1': 0.5 + 0.1 * client_id,
        'task_loss': 0.9,
        'reconstruction_loss': 0.0,
        'latent_norms': latent_norms,
        'max_latent_growth_ratio': 1.01,
        'max_latent_relative_delta': 0.02,
        'a0_spectral_norm': 0.2,
        'a1_spectral_norm': 0.1,
        'upload_bytes': upload_bytes,
        'elapsed_seconds': 0.5 + 0.1 * client_id,
        'peak_cuda_memory_bytes': 1024 + client_id,
    }


class V01StructuredLoggerTest(unittest.TestCase):
    def test_model_cost_dynamics_and_cumulative_communication(self):
        with tempfile.TemporaryDirectory() as path:
            args = SimpleNamespace(
                log_path=path,
                log_profile='minimal',
                n_clients=2,
                dataset='Cora',
                model='v01_linear',
                seed=42,
                latent_dim=4,
                linear_steps=2,
                reconstruction_weight=0.0,
            )
            model = LatentLinearGraphDynamics(
                input_dim=3,
                output_dim=2,
                latent_dim=4,
                encoder_width=5,
                num_steps=2,
            )
            logger = V01StructuredLogger(args, model)
            messages = [
                message(0, 100, [2.0, 2.1, 2.2]),
                message(1, 120, [4.0, 4.2, 4.4]),
            ]
            reference = {'weight': np.asarray([1.0])}
            for round_id in range(2):
                logger.begin_round()
                logger.record_round(round_id, messages, reference, 'equal')

            with open(
                os.path.join(path, 'dynamics.csv'), newline=''
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[0]['round_upload_bytes']), 220)
            self.assertEqual(int(rows[1]['cumulative_upload_bytes']), 440)
            self.assertAlmostEqual(float(rows[0]['z_norm_1']), 3.15)

            with open(os.path.join(path, 'model_cost.json')) as stream:
                cost = json.load(stream)
            self.assertGreater(cost['trainable_parameters'], 0)
            self.assertGreater(cost['state_dict_bytes'], 0)
            self.assertFalse(cost['reconstruction_enabled'])


if __name__ == '__main__':
    unittest.main()
