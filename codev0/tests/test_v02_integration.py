import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import _model_components, set_config
from myparser import Parser
from models.v02.client import Client
from models.v02.logger import V02StructuredLogger
from models.v02.model import build_v02_model
from models.v02.training import LossWeights, trajectory_diagnostics


class V02IntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def args(self, extra=()):
        with patch.object(sys, 'argv', ['main.py', '--model', 'v02_koopman', '--dataset', 'Cora', *extra]):
            return set_config(Parser().parse())

    def test_config_respects_epoch_one_and_explicit_overrides(self):
        args = self.args()
        self.assertEqual(args.n_eps, 1)
        self.assertEqual(args.reconstruction_weight, 1.0)
        self.assertEqual(args.lr, 0.003)
        explicit = self.args(['--n-eps', '3', '--reconstruction-weight', '0', '--lr', '0.01'])
        self.assertEqual(explicit.n_eps, 3)
        self.assertEqual(explicit.reconstruction_weight, 0)
        self.assertEqual(explicit.lr, 0.01)
        self.assertIs(_model_components('v02_koopman')[0], Client)

    def test_client_step_and_logger_share_local_objective(self):
        args = self.args(['--latent-dim', '4', '--hidden-dim', '8', '--encoder-width', '9', '--linear-steps', '2'])
        args.n_feat, args.n_clss, args.n_clients = 5, 3, 1
        data = Data(
            x=torch.randn(5, 5), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            y=torch.arange(5) % 3, train_mask=torch.tensor([True, True, True, False, False]),
        )
        # Exercise the actual federated train step on CPU; process spawning and
        # CUDA transport are intentionally not part of this local integration test.
        client = Client.__new__(Client)
        client.args, client._active_batch = args, data
        client.model = build_v02_model(args)
        client.loss_weights = LossWeights()
        client.optimizer = torch.optim.Adam(client.model.parameters(), lr=args.lr)
        client.client_id, client.sd = 0, {}
        before = client.model.decoder[0].weight.detach().clone()
        client._train_step()
        self.assertFalse(torch.equal(before, client.model.decoder[0].weight))
        values = trajectory_diagnostics(client.model, client.model.forward_with_aux(data, auxiliary=False))
        client._round_result = {
            'train_loss': float(client._last_train_lss), 'val_loss': 1.0,
            'val_accuracy': 0.5, 'test_loss': 1.0, 'test_accuracy': 0.4,
            'test_f1': 0.3, 'elapsed_seconds': 0.1, 'peak_cuda_memory_bytes': 0,
            **client._last_v02, **values,
        }
        client.transfer_to_server()
        message = client.sd[0]
        with tempfile.TemporaryDirectory() as directory:
            args.log_path = directory
            logger = V02StructuredLogger(args, client.model)
            for step in range(2):
                logger.begin_round()
                logger.record_round(step, [message], message['model'], 'equal')
            with (Path(directory) / 'dynamics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[-1]['cumulative_upload_bytes']), 2 * message['upload_bytes'])
            cost = json.loads((Path(directory) / 'model_cost.json').read_text())
            self.assertTrue(cost['decoder_in_classification_path'])
            self.assertLess(cost['deployment_parameters'], cost['training_parameters'])


if __name__ == '__main__':
    unittest.main()
