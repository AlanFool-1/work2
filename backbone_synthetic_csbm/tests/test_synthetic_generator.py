import numpy as np

import data.synthetic as synthetic


def test_label_shift_is_monotonic_from_client_zero():
    base = np.full(8, 1.0 / 8.0)
    values = []
    for client_id in range(10):
        proportions = synthetic._proportions_for_client(
            client_id, 10, 8, 'label_shift'
        )
        values.append(synthetic._jensen_shannon(proportions, base))
    assert values[0] == 0.0
    assert all(values[index] <= values[index + 1] + 1e-12 for index in range(9))


def test_feature_shift_gaussian_w2_is_monotonic():
    base_means = synthetic._base_class_means(7, 8, 128, 3.0)
    distances = []
    for client_id in range(10):
        means, std = synthetic._feature_parameters_for_client(
            client_id=client_id,
            n_clients=10,
            scenario='feature_shift',
            base_means=base_means,
            base_std=1.5,
            mean_shift=3.0,
            std_shift=0.35,
            seed=11,
        )
        distances.append(
            synthetic._gaussian_w2_isotropic(
                base_means, means, 1.5, std
            )
        )
    assert distances[0] == 0.0
    assert all(distances[index] <= distances[index + 1] + 1e-9 for index in range(9))


def test_sbm_solver_preserves_requested_expected_edge_budget():
    counts = synthetic._integer_class_counts(np.full(8, 1 / 8), 1000)
    p_in, p_out = synthetic._solve_sbm_probabilities(counts, 12.0, 0.75)
    same_pairs = np.sum(counts * (counts - 1) / 2)
    all_pairs = 1000 * 999 / 2
    cross_pairs = all_pairs - same_pairs
    expected_same = p_in * same_pairs
    expected_cross = p_out * cross_pairs
    expected_edges = expected_same + expected_cross
    assert abs(expected_edges - 6000.0) < 1e-8
    assert abs(expected_same / expected_edges - 0.75) < 1e-10
