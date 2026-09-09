import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_v02_federated import validate_completion


class FederatedAuditTest(unittest.TestCase):
    def setUp(self):
        self.config = {'n_rnds': 100, 'n_clients': 2}
        self.result = {'client_test_metrics': [0.4, 0.6], 'mean': 0.5}
        self.rounds = [{'round': index} for index in range(1, 101)]
        self.best = [
            {'client_id': 0, 'best_round': 50, 'test_accuracy': 0.4},
            {'client_id': 1, 'best_round': 70, 'test_accuracy': 0.6},
        ]

    def check(self):
        validate_completion(self.config, self.result, self.rounds, self.best, 100)

    def test_completed_history(self):
        self.check()

    def test_reject_incomplete_history_even_if_config_says_100(self):
        self.rounds = self.rounds[:81]
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.check()

    def test_reject_duplicate_round(self):
        self.rounds[-1]['round'] = 99
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.check()

    def test_reject_missing_client(self):
        self.best.pop()
        with self.assertRaisesRegex(ValueError, 'client'):
            self.check()

    def test_reject_inconsistent_summary(self):
        self.result['mean'] = 0.6
        with self.assertRaisesRegex(ValueError, 'Summary'):
            self.check()

    def test_reject_misassigned_client_vector_even_with_same_mean(self):
        self.result['client_test_metrics'] = [0.6, 0.4]
        with self.assertRaisesRegex(ValueError, 'Summary vector'):
            self.check()


if __name__ == '__main__':
    unittest.main()
