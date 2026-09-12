"""Sparse graph normalization and the K=1 Bernstein graph basis."""

from __future__ import annotations

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def graph_operator(data) -> torch.Tensor:
    """Return symmetric normalized adjacency without adding self-loops.

    For an undirected non-negative graph this gives ``spec(A_hat)`` inside
    ``[-1, 1]``, which is the assumption used by the Bernstein stability proof.
    """

    edge_weight = getattr(data, 'edge_weight', None)
    if edge_weight is not None:
        if not torch.isfinite(edge_weight).all() or (edge_weight < 0).any():
            raise ValueError(
                'V0.4 requires finite non-negative edge weights.'
            )
    edge_index, normalized_weight = gcn_norm(
        data.edge_index,
        edge_weight,
        num_nodes=data.x.shape[0],
        improved=False,
        add_self_loops=False,
        flow='source_to_target',
        dtype=data.x.dtype,
    )
    if normalized_weight is None:
        raise RuntimeError('Graph normalization did not return edge weights.')
    operator = torch.sparse_coo_tensor(
        edge_index.flip(0),
        normalized_weight,
        (data.x.shape[0], data.x.shape[0]),
        device=data.x.device,
        dtype=data.x.dtype,
    ).coalesce()
    difference = (operator - operator.transpose(0, 1)).coalesce()
    if difference._nnz():
        maximum = float(difference.values().abs().max().detach().cpu())
        if maximum > 1e-6:
            raise ValueError(
                'V0.4 stability requires an undirected symmetric graph.'
            )
    return operator


def aggregate(operator: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply a sparse node operator to tensors whose node axis is ``-2``."""

    node_first = value.movedim(-2, 0)
    flat = node_first.reshape(node_first.shape[0], -1)
    result = torch.sparse.mm(operator, flat)
    return result.reshape(node_first.shape).movedim(0, -2)


def bernstein_k1(value: torch.Tensor, operator: torch.Tensor):
    """Return low/high K=1 Bernstein responses that sum to ``value``."""

    neighbor = aggregate(operator, value)
    return 0.5 * (value + neighbor), 0.5 * (value - neighbor)
