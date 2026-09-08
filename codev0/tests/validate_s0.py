#!/usr/bin/env python3
"""Fast invariants for the official A-DGN + FedAvg baseline."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.s0.model import build_s0_model, resolve_official_adgn_source


def ring_graph(nodes=12, features=5, classes=3):
    source = torch.arange(nodes)
    target = torch.roll(source, -1)
    return Data(
        x=torch.randn(nodes, features),
        edge_index=torch.stack([
            torch.cat([source, target]),
            torch.cat([target, source]),
        ]),
        edge_weight=None,
        y=torch.arange(nodes) % classes,
    )


def build_model(steps=4):
    return build_s0_model(SimpleNamespace(
        base_path=str(ROOT.parent),
        n_feat=5,
        n_clss=3,
        hidden_dim=8,
        ode_steps=steps,
        adgn_step_size=0.1,
        ode_gamma=0.1,
        ode_activation='tanh',
    ))


def validate_official_source_and_gradients():
    model = build_model()
    expected_source = resolve_official_adgn_source(str(ROOT.parent))
    assert Path(inspect.getsourcefile(type(model))).resolve() == expected_source
    data = ring_graph()
    logits = model(data)
    assert logits.shape == (12, 3)
    F.cross_entropy(logits, data.y).backward()
    used = [
        model.emb.weight,
        model.conv.W,
        model.conv.gcn_conv.lin.weight,
        model.readout.weight,
    ]
    assert all(parameter.grad is not None for parameter in used)
    assert all(torch.isfinite(parameter.grad).all() for parameter in used)
    # The official class always constructs ``conv.lin`` for its alternate
    # plain-aggregation path. With gcn_norm=True that branch is intentionally
    # inactive and its parameter therefore has no gradient.
    assert model.conv.lin.weight.grad is None
    print('[ok] official A-DGN source and finite gradients')


def validate_shared_field_euler_iterations():
    model = build_model(steps=4)
    calls = []
    handle = model.conv.gcn_conv.register_forward_hook(
        lambda *_: calls.append(1)
    )
    try:
        model(ring_graph())
    finally:
        handle.remove()
    assert len(calls) == 4
    assert model.conv.num_iters == 4
    assert abs(model.conv.epsilon - 0.1) < 1e-12
    print('[ok] one shared vector field is evaluated by four Euler iterations')


def validate_multiple_client_topologies():
    model = build_model(steps=2)
    assert model(ring_graph(nodes=12)).shape[0] == 12
    assert model(ring_graph(nodes=9)).shape[0] == 9
    print('[ok] model accepts different client graph topologies')


def validate_adam_state():
    model = build_model(steps=2)
    data = ring_graph()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(data), data.y).backward()
    optimizer.step()
    state = optimizer.state_dict()['state']
    assert state
    assert all('exp_avg' in item and 'exp_avg_sq' in item
               for item in state.values())
    print('[ok] Adam first/second moments are present')


if __name__ == '__main__':
    torch.set_num_threads(1)
    validate_official_source_and_gradients()
    validate_shared_field_euler_iterations()
    validate_multiple_client_topologies()
    validate_adam_state()
    print('[done] baseline validations passed')
