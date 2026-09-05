import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from modules.structured_logger import StructuredLogger


def _message(client_id, val_accuracy, test_accuracy, offset):
    return {
        'client_id': client_id,
        'model': {'weight': np.asarray([1.0 + offset, 2.0 - offset])},
        'train_size': 10 + client_id,
        'train_loss': 1.0,
        'val_loss': 0.8,
        'val_accuracy': val_accuracy,
        'test_loss': 0.7,
        'test_accuracy': test_accuracy,
        'test_f1': test_accuracy - 0.05,
    }


class StructuredLoggerTest(unittest.TestCase):
    def _args(self, path, profile):
        return SimpleNamespace(
            log_path=path,
            log_profile=profile,
            n_clients=2,
            dataset='Cora',
            model='s0_ode',
            seed=42,
        )

    def test_minimal_profile_and_best_validation_pairing(self):
        with tempfile.TemporaryDirectory() as path:
            logger = StructuredLogger(self._args(path, 'minimal'))
            reference = {'weight': np.asarray([1.0, 2.0])}
            rounds = [
                [_message(0, 0.8, 0.7, 0.1), _message(1, 0.7, 0.6, -0.1)],
                [_message(0, 0.7, 0.9, 0.2), _message(1, 0.9, 0.8, -0.2)],
            ]
            for round_id, messages in enumerate(rounds):
                logger.begin_round()
                logger.record_round(round_id, messages, reference, 'equal')
            result = logger.finalize()

            self.assertEqual(
                set(os.listdir(path)),
                {'config.json', 'metrics.csv', 'client_best.csv', 'result.json'},
            )
            self.assertEqual(result['client_test_metrics'], [0.7, 0.8])
            with open(os.path.join(path, 'metrics.csv'), newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            with open(os.path.join(path, 'result.json')) as stream:
                self.assertEqual(json.load(stream)['selection'], result['selection'])

    def test_diagnostic_profile_is_explicit(self):
        with tempfile.TemporaryDirectory() as path:
            logger = StructuredLogger(self._args(path, 'diagnostic'))
            messages = [
                _message(0, 0.8, 0.7, 0.1),
                _message(1, 0.7, 0.6, -0.1),
            ]
            logger.begin_round()
            logger.record_round(
                0,
                messages,
                {'weight': np.asarray([1.0, 2.0])},
                'equal',
            )
            logger.finalize()
            self.assertTrue(os.path.exists(os.path.join(path, 'client_metrics.csv')))
            self.assertTrue(os.path.exists(os.path.join(path, 'server_diagnostics.csv')))
            self.assertTrue(
                os.path.exists(os.path.join(path, 'pairwise_update_diagnostics.csv'))
            )


if __name__ == '__main__':
    unittest.main()
