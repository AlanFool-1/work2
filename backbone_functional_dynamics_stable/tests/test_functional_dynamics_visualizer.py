import csv
import os
import tempfile
import unittest

import numpy as np

from modules.functional_dynamics_visualizer import (
    plot_client_validation_curves,
    plot_cluster_assignments,
    plot_functional_dynamics_diagnostics,
)


class FunctionalDynamicsVisualizerTest(unittest.TestCase):
    def test_creates_cluster_assignment_plot(self):
        fields = ['round', 'client_id', 'cluster_id']
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'functional_dynamics.csv')
            with open(path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for round_id in range(1, 4):
                    for client_id in range(4):
                        writer.writerow({
                            'round': round_id, 'client_id': client_id,
                            'cluster_id': client_id % 2,
                        })
            output = plot_cluster_assignments(directory)
            self.assertTrue(os.path.getsize(output) > 0)

    def test_creates_per_client_validation_plot(self):
        fields = ['round', 'client_id', 'val_accuracy']
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'client_metrics.csv')
            with open(path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for round_id in range(1, 4):
                    for client_id in range(2):
                        writer.writerow({
                            'round': round_id,
                            'client_id': client_id,
                            'val_accuracy': 0.1 * round_id + 0.01 * client_id,
                        })
            output = plot_client_validation_curves(directory)
            self.assertTrue(os.path.getsize(output) > 0)

    def test_creates_curve_and_canonical_plots(self):
        fields = [
            'round', 'client_id', 'dyn_fit_error', 'fm_desc_error',
            'fm_dyn_error', 'basis_staleness', 'injection_native_ratio',
            'delta_safe_max_sym_eig',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'functional_dynamics.csv')
            with open(path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({field: (1 if field == 'round' else 0) for field in fields})
            outputs = plot_functional_dynamics_diagnostics(
                directory, np.eye(3), np.ones((3, 5))
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(os.path.getsize(item) > 0 for item in outputs))


if __name__ == '__main__':
    unittest.main()
