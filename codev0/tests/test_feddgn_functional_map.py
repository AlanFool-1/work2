import json
import os

import numpy as np
import torch

from models.feddgn.functional_map import solve_functional_map
from models.feddgn.client import (
    Client as FunctionalMapClient,
    _state_relative_delta,
)
from models.feddgn.reskoop import KOOPMAN_DICTIONARY_PROTOCOL
from models.feddgn.server import (
    Server as FunctionalMapServer,
    _aggregate_dictionary_states,
)
from models.feddgn_avg.client import Client as FedAvgClient
from models.feddgn_avg.server import Server as FedAvgServer
from models.feddgn.mode_graph import (
    MODE_GRAPH_PROTOCOL,
    ModeGraphBuilder,
    make_mode_upload,
    save_mode_graph_artifacts,
)
from models.feddgn.mode_graph_visualize import (
    generate_mode_graph_visualizations,
)


def _upload(client_id, operator, reliability):
    values, vectors = np.linalg.eig(operator)
    dimension = operator.shape[0]
    return {
        'protocol': MODE_GRAPH_PROTOCOL,
        'client_id': int(client_id),
        'operator': np.asarray(operator, dtype=np.float32),
        'spectral_residual': float(reliability),
        'local_mode_indices': np.arange(dimension, dtype=np.int64),
        'eigenvalues': values.astype(np.complex64),
        'eigenvectors': vectors.T.astype(np.complex64),
        'residuals': np.full(dimension, reliability, dtype=np.float32),
        'state_modes': np.eye(dimension, dtype=np.complex64),
        'bytes': 0,
    }


def test_functional_map_model_uses_fedavg_task_lifecycle():
    assert issubclass(FunctionalMapClient, FedAvgClient)
    assert issubclass(FunctionalMapServer, FedAvgServer)


def test_koopman_dictionary_fedavg_is_separate_from_local_operator():
    first = {
        'network.0.weight': np.asarray([[1.0, 3.0]], dtype=np.float32),
        'network.0.bias': np.asarray([2.0], dtype=np.float32),
    }
    second = {
        'network.0.weight': np.asarray([[5.0, 7.0]], dtype=np.float32),
        'network.0.bias': np.asarray([6.0], dtype=np.float32),
    }
    averaged = _aggregate_dictionary_states(
        [first, second], [0.25, 0.75]
    )
    np.testing.assert_allclose(averaged['network.0.weight'], [[4.0, 6.0]])
    np.testing.assert_allclose(averaged['network.0.bias'], [5.0])
    assert _state_relative_delta(averaged, averaged) == 0.0
    assert KOOPMAN_DICTIONARY_PROTOCOL == 'reskoop_dictionary_fedavg_v1'


def test_functional_map_recovers_a_permuted_koopman_space():
    reference = np.diag(np.asarray([0.9, 0.7, 0.5], dtype=np.float32))
    permutation = np.asarray(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
    )
    local = permutation.T @ reference @ permutation
    functional_map, report = solve_functional_map(
        local, reference, refinement_steps=8
    )

    np.testing.assert_allclose(
        functional_map @ local,
        reference @ functional_map,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        functional_map.conj().T @ functional_map,
        np.eye(3),
        atol=1e-6,
    )
    assert report['intertwining_residual'] < 1e-6
    assert report['orthogonality_error'] < 1e-6


def test_mode_upload_includes_operator_and_reliability():
    result = {
        'operator': torch.diag(torch.tensor([1.0, 0.8, 0.9])),
        'eigenvalues': torch.tensor([1.0, 0.8, 0.9], dtype=torch.complex64),
        'eigenvectors': torch.eye(3, dtype=torch.complex64),
        'mode_residuals': torch.tensor([0.0, 0.2, 0.1]),
        'modes': torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.complex64,
        ),
        'diagnostics': {'koopman_spectral_residual': torch.tensor(0.125)},
    }
    upload = make_mode_upload(result, client_id=4, top_modes=1)

    assert upload['local_mode_indices'].tolist() == [2]
    assert upload['operator'].shape == (3, 3)
    assert upload['spectral_residual'] == 0.125
    assert upload['bytes'] > upload['operator'].nbytes


def test_builder_aligns_modes_builds_global_bank_and_quality_metrics():
    reference = np.diag(np.asarray([0.9, 0.7, 0.5], dtype=np.float32))
    permutation = np.asarray(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
    )
    local = permutation.T @ reference @ permutation
    builder = ModeGraphBuilder(
        neighbors=1,
        personalized_modes=2,
        resolution=1.0,
        functional_map_steps=8,
        seed=7,
    )
    graph, views = builder.build(
        [_upload(0, reference, 0.01), _upload(1, local, 0.02)],
        round_number=3,
    )

    assert graph['reference_client_id'] == 0
    assert graph['comparison_space'] == (
        'functional_map_shared_koopman_observable'
    )
    assert graph['community_count'] == 3
    assert graph['cluster_quality']['silhouette'] > 0.99
    assert graph['global_mode_bank']['representative_modes'].shape == (3, 3)
    assert graph['similarity'][:3, :3].sum() == 0.0
    for left in range(3):
        for right in range(3, 6):
            if abs(graph['eigenvalues'][left] - graph['eigenvalues'][right]) < 1e-6:
                assert graph['spatial_similarity'][left, right] > 0.999
    assert views[1]['functional_map'].shape == (3, 3)
    assert views[1]['global_mode_bank']['representative_modes'].shape == (3, 3)
    assert views[1]['functional_map_diagnostics'][
        'intertwining_residual'
    ] < 1e-6


def test_functional_map_artifacts_and_dashboard_are_readable(tmp_path):
    reference = np.diag(np.asarray([0.9, 0.7], dtype=np.float32))
    graph, views = ModeGraphBuilder(
        neighbors=1,
        personalized_modes=1,
        functional_map_steps=2,
    ).build(
        [_upload(0, reference, 0.01), _upload(1, reference, 0.02)],
        round_number=10,
    )
    root = save_mode_graph_artifacts(
        str(tmp_path), graph, views, archive=True
    )
    with open(os.path.join(root, 'latest.json'), encoding='utf-8') as stream:
        public = json.load(stream)
    assert public['reference_client_id'] == 0
    assert 'cluster_quality' in public
    assert len(public['global_mode_bank_summary']) == 2

    report = generate_mode_graph_visualizations(
        str(tmp_path),
        dpi=40,
        graph_path=os.path.join(root, 'round_0010', 'mode_graph.pt'),
        output_directory=os.path.join(root, 'round_0010', 'figures'),
    )
    assert os.path.isfile(report['dashboard'])
    assert len(report['figures']) == 6
    assert os.path.isfile(report['figures']['functional_map_quality'])
    assert 'community_spectrum' not in report['figures']
    assert not os.path.exists(
        os.path.join(root, 'round_0010', 'figures', 'community_spectrum.png')
    )
