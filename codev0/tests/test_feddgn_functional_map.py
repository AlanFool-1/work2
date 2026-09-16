"""Passive Functional Map → conflict candidates → private need → K-star."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.feddgn0916.candidate_pipeline import (
    ConflictCandidateBuilder, _conflict, make_koopman_packet,
    save_candidate_artifacts,
)
from models.feddgn0916.client import Client, _state_relative_delta
from models.feddgn0916.functional_map import solve_functional_map
from models.feddgn0916.kstar_visualize import (
    append_personalized_history, operator_comparison, render_client_history,
    render_kstar_client,
)
from models.feddgn0916.personalized_koopman import fit_personalized_koopman
from models.feddgn0916.server import Server, _aggregate_dictionary_states
from models.feddgn0916.server_visualize import render_server_diagnostics
from models.feddgn_local.client import Client as LocalClient
from modules.federated import ServerModule
from modules.fed_server import FullModelServer


def packet(client_id, operator, round_number=1):
    row = np.asarray(operator, dtype=np.float32).T
    result = {
        'operator': torch.as_tensor(row),
        'eigenvalues': torch.as_tensor(
            np.linalg.eigvals(row), dtype=torch.complex64
        ),
        'mode_residuals': torch.full((row.shape[0],), 0.01),
        'diagnostics': {'koopman_spectral_residual': torch.tensor(0.01)},
    }
    return make_koopman_packet(
        result, client_id=client_id, round_number=round_number
    )


def candidate(u, eigenvalue, source_client=1):
    return {
        'mode_id': f'client_{source_client}:mode_0',
        'source_client': source_client, 'source_mode': 0,
        'U_local': np.asarray(u, dtype=np.complex64).reshape(-1, 1),
        'T': np.asarray([[eigenvalue]], dtype=np.complex64),
        'overlap': 0.0, 'disagreement': 0.0, 'conflict': 0.0,
        'compatibility': 1.0, 'reliability': 0.01,
    }


def package(client_id, candidates, source_round=1):
    return {
        'protocol': 'koopman_conflict_candidates_v1',
        'client_id': client_id, 'source_round': source_round,
        'reference_client_id': 0, 'candidates': candidates,
    }


def test_task_is_local_and_dictionary_fedavg_is_separate():
    assert issubclass(Client, LocalClient)
    assert issubclass(Server, ServerModule)
    assert not issubclass(Server, FullModelServer)
    left = {'weight': np.array([[1, 3]], dtype=np.float32)}
    right = {'weight': np.array([[5, 7]], dtype=np.float32)}
    average = _aggregate_dictionary_states([left, right], [0.25, 0.75])
    np.testing.assert_allclose(average['weight'], [[4, 6]])
    assert _state_relative_delta(average, average) == 0


def test_server_only_aggregates_dictionary_and_has_no_task_model(tmp_path):
    args = SimpleNamespace(
        log_path=str(tmp_path / 'logs'), checkpt_path=str(tmp_path / 'checkpoints'),
        log_profile='minimal', model='feddgn', dataset='Cora', n_clients=2,
        n_rnds=2, n_eps=1, aggregation='equal',
        optimizer_state_mode='local_persistent_adam',
        feddgn_candidate_top_modes=1, feddgn_candidate_max_per_client=1,
        feddgn_candidate_mode_residual_max=1.0,
        feddgn_candidate_conflict_threshold=0.15,
        feddgn_candidate_compat_temperature=0.1,
        feddgn_functional_map_steps=1,
        feddgn_functional_map_tolerance=1e-7,
    )
    shared = {}
    server = Server(args, shared, 0)
    server.on_round_begin(0)
    assert 'global' not in shared and 'global_optimizer' not in shared
    operator = np.diag([0.95, 0.8]).astype(np.float32)
    for client_id, weight in ((0, 1.0), (1, 3.0)):
        shared[client_id] = {
            'client_id': client_id, 'train_size': 10,
            'train_loss': 1.0, 'val_loss': 1.0,
            'val_accuracy': 0.5, 'test_loss': 1.0,
            'test_accuracy': 0.5, 'test_f1': 0.5,
            'koopman_dictionary_protocol': 'reskoop_dictionary_fedavg_v1',
            'koopman_dictionary': {'weight': np.array([weight], dtype=np.float32)},
            'koopman_packet': packet(client_id, operator),
            'task_seconds': 0.01,
            'reskoop_seconds': 0.02,
            'kstar_seconds': 0.0,
            'kstar_compute_seconds': 0.0,
            'kstar_visualization_seconds': 0.0,
            'dictionary_receive_seconds': 0.0,
            'client_total_seconds': 0.04,
        }
    server.on_round_complete({0, 1})
    np.testing.assert_allclose(
        shared['global_koopman_dictionary']['state_dict']['weight'], [2.0]
    )
    checkpoint = torch.load(tmp_path / 'checkpoints' / 'server_state.pt',
                            weights_only=False)
    assert 'model' not in checkpoint and 'global_v_step' not in checkpoint
    assert checkpoint['task_parameter_aggregation'] is False
    assert checkpoint['candidate_K_star_task_feedback'] is False
    with open(tmp_path / 'logs' / 'timing.csv') as stream:
        assert len(stream.readlines()) == 2
    server.on_round_begin(1)
    assert 'global' not in shared and 'global_optimizer' not in shared


def test_packet_convention_and_privacy():
    column = np.array([[0.8, 0.3], [0, 0.6]], dtype=np.float32)
    upload = packet(7, column)
    np.testing.assert_allclose(upload['operator'], column)
    assert upload['operator_convention'] == 'column_action_transpose_of_row_edmd'
    assert upload['bytes'] > column.nbytes
    assert 'snapshots' not in upload and 'trajectory' not in upload


def test_functional_map_recovers_permuted_column_operator():
    reference = np.diag([0.9, 0.7, 0.5]).astype(np.float32)
    p = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    local = p.T @ reference @ p
    c, report = solve_functional_map(local, reference)
    np.testing.assert_allclose(c @ local, reference @ c, atol=1e-6)
    assert report['orthogonality_error'] < 1e-6


def test_conflict_requires_overlap_and_dynamical_disagreement():
    local = [{'mode_id': 'local', 'U': np.array([[1], [0]]),
              'T': np.array([[0.2]])}]
    shared_conflicting = {'U': local[0]['U'], 'T': np.array([[0.9]])}
    complementary = {'U': np.array([[0], [1]]), 'T': np.array([[0.9]])}
    duplicate = {'U': local[0]['U'], 'T': local[0]['T']}
    chi, overlap, disagreement, _ = _conflict(local, shared_conflicting)
    assert chi > 0.5 and overlap == 1 and disagreement > 0.5
    assert _conflict(local, complementary)[0] == pytest.approx(0)
    assert _conflict(local, duplicate)[0] == pytest.approx(0)


def test_builder_offers_client_specific_packages_without_graph():
    reference = np.diag([0.95, 0.8, 0.6]).astype(np.float32)
    p = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    analysis = ConflictCandidateBuilder(top_modes=2, max_candidates=2).build(
        [packet(0, reference), packet(1, p.T @ reference @ p),
         packet(2, reference)], round_number=1
    )
    assert analysis['reference_client_id'] == 0
    assert set(analysis['packages']) == {0, 1, 2}
    assert np.diag(analysis['source_matrix']).sum() == 0
    assert not any('graph' in key or 'community' in key for key in analysis)
    for client_id, view in analysis['packages'].items():
        assert view['source_round'] == 1 and len(view['candidates']) <= 2
        assert all(c['source_client'] != client_id for c in view['candidates'])
    with pytest.raises(ValueError, match='Stale'):
        ConflictCandidateBuilder().build(
            [packet(0, reference)], round_number=2
        )


def test_builder_vetoes_a_shared_direction_with_incompatible_evolution():
    left = np.diag([0.95, 0.2]).astype(np.float32)
    right = np.diag([0.95, 0.9]).astype(np.float32)
    analysis = ConflictCandidateBuilder(
        top_modes=1, max_candidates=1, conflict_threshold=0.15
    ).build([packet(0, left), packet(1, right)], round_number=1)
    assert analysis['summary']['conflict_rejected_count'] == 2
    assert analysis['summary']['retained_count'] == 0
    assert all(not view['candidates'] for view in analysis['packages'].values())


def test_private_need_and_joint_right_solve():
    k = np.diag([0.5, 0.7]).astype(np.float32)
    z = np.array([[1, 2], [0, 0]])
    z_next = np.array([[0.9, 1.8], [0, 0]])
    view = package(0, [candidate([1, 0], 0.9)])
    result = fit_personalized_koopman(k, z, z_next, view, eta=1)
    np.testing.assert_allclose(result['K_star'], np.diag([0.7, 0.7]), atol=1e-6)
    assert result['stats']['candidate_active'] == 1
    assert result['stats']['candidate_prediction_residual_after'] < (
        result['stats']['candidate_prediction_residual_before']
    )
    assert result['candidate_rows'][0]['mode_residual_after'] < (
        result['candidate_rows'][0]['mode_residual_before']
    )
    assert result['task_feedback'] is False
    zero_eta = fit_personalized_koopman(k, z, z_next, view, eta=0)
    np.testing.assert_allclose(zero_eta['K_star'], k)


def test_negative_need_and_empty_candidates_leave_k_unchanged():
    k = np.diag([0.5, 0.7]).astype(np.float32)
    z = np.array([[1], [0]])
    z_next = np.array([[0.1], [0]])
    for modes in ([candidate([1, 0], 0.9)], []):
        result = fit_personalized_koopman(
            k, z, z_next, package(0, modes), eta=10
        )
        np.testing.assert_allclose(result['K_star'], k)
        assert result['stats']['candidate_active'] == 0


def test_numeric_artifacts_have_no_visualization_directory(tmp_path):
    k = np.diag([0.95, 0.8, 0.6]).astype(np.float32)
    analysis = ConflictCandidateBuilder(top_modes=2).build(
        [packet(0, k, 2), packet(1, k, 2)], round_number=2
    )
    root = save_candidate_artifacts(str(tmp_path), analysis)
    assert os.path.exists(os.path.join(root, 'latest.pt'))
    assert not os.path.exists(tmp_path / 'koopman' / 'mode_graph')
    assert not os.path.exists(tmp_path / 'koopman' / 'candidates' / 'round_0002')
    assert not os.path.exists(tmp_path / 'koopman' / 'candidates' / 'figures')
    with open(tmp_path / 'koopman' / 'candidates' / 'latest.json') as stream:
        assert json.load(stream)['round'] == 2
    assert analysis['summary']['task_feedback'] is False


def test_only_one_clients_same_frame_k_and_kstar_are_plotted(tmp_path):
    result = {
        'client_id': 3, 'source_round': 7,
        'K': np.diag([0.8, 0.6]).astype(np.complex64),
        'K_star': np.diag([0.85, 0.6]).astype(np.complex64),
        'task_feedback': False,
    }
    comparison = operator_comparison(result)
    assert comparison['summary']['relative_delta_K'] > 0
    assert comparison['summary']['client_id'] == 3
    report = render_kstar_client(result, tmp_path / 'figures', dpi=40)
    assert os.path.exists(report['dashboard'])
    assert os.path.exists(report['summary']['figures']['matrix'])
    for name in (
        'compatibility_need', 'private_residual',
        'active_mode_residuals', 'candidate_funnel',
    ):
        assert os.path.exists(report['summary']['figures'][name])
    assert 'spectrum' not in report['summary']['figures']
    assert not os.path.exists(
        tmp_path / 'figures' / 'source_round_0001'
        / 'K_vs_Kstar_spectrum.png'
    )
    assert os.path.exists(tmp_path / 'figures' / 'latest.html')
    with open(report['dashboard'], encoding='utf-8') as stream:
        document = stream.read()
    assert '客户端 3' in document
    assert '客户端间' not in document


def test_client_history_and_full_candidate_diagnostics(tmp_path):
    k = np.diag([0.5, 0.7]).astype(np.float32)
    z = np.array([[1, 2], [0, 0]])
    z_next = np.array([[0.9, 1.8], [0, 0]])
    result = fit_personalized_koopman(
        k, z, z_next, package(0, [candidate([1, 0], 0.9)]), eta=1
    )
    history = tmp_path / 'history.csv'
    append_personalized_history(result, history)
    assert history.exists()
    flow = render_client_history(history, tmp_path / 'figures', dpi=30)
    assert os.path.exists(flow)
    report = render_kstar_client(result, tmp_path / 'figures', dpi=30)
    assert os.path.exists(report['summary']['figures']['compatibility_need'])
    assert os.path.exists(report['summary']['figures']['active_mode_residuals'])


def test_server_diagnostic_renderer_uses_numeric_analysis(tmp_path):
    k = np.diag([0.95, 0.8, 0.6]).astype(np.float32)
    analysis = ConflictCandidateBuilder(top_modes=2).build(
        [packet(0, k, 2), packet(1, k, 2)], round_number=2
    )
    save_candidate_artifacts(str(tmp_path), analysis)
    history = tmp_path / 'koopman' / 'candidates' / 'server_rounds.csv'
    report = render_server_diagnostics(
        analysis, tmp_path / 'koopman' / 'visualization' / 'server',
        dpi=30, history_path=history,
    )
    assert os.path.exists(report['dashboard'])
    assert os.path.exists(report['figures']['functional_map_residuals'])
    assert os.path.exists(report['figures']['candidate_funnel'])
    assert os.path.exists(report['figures']['knowledge_flow'])
    for client in ('0', '1'):
        assert os.path.exists(report['clients'][client]['conflict'])
        assert os.path.exists(report['clients'][client]['correspondence'])
