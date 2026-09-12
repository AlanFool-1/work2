from dataclasses import dataclass
from math import comb
from typing import Optional

import torch


@dataclass
class GraphCache:
    """Normalized adjacency cache used by the Bernstein graph basis.

    The sparse matrix follows A[dst, src], so sparse.mm(A, X) performs
    source-to-destination message aggregation.
    """

    norm_adj: torch.Tensor
    num_nodes: int

    def adj_mm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.norm_adj, x)

    def scaled_laplacian_mm(self, x: torch.Tensor) -> torch.Tensor:
        # S = L/2 = (I - A_norm)/2. Spectrum lies in [0, 1] for a
        # symmetric normalized Laplacian.
        return 0.5 * (x - self.adj_mm(x))

    def one_minus_scaled_laplacian_mm(self, x: torch.Tensor) -> torch.Tensor:
        # (I - S) = (I + A_norm)/2.
        return 0.5 * (x + self.adj_mm(x))


def _coalesce_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Build a temporary sparse tensor to coalesce duplicate entries.
    sp = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes),
        device=edge_weight.device,
        dtype=edge_weight.dtype,
    ).coalesce()
    return sp.indices(), sp.values()


def build_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    symmetrize: bool = False,
    remove_self_loops: bool = True,
) -> GraphCache:
    """Create symmetric-normalized adjacency without requiring PyG.

    Expected edge_index convention is [src, dst], matching PyG.
    For an undirected dataset that already stores both directions, keep
    symmetrize=False. If edges are stored once, symmetrize=True is convenient.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")

    if device is None:
        device = edge_index.device
    edge_index = edge_index.to(device=device, dtype=torch.long)

    if dtype is None:
        dtype = torch.float32
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=device, dtype=dtype)
    else:
        edge_weight = edge_weight.to(device=device, dtype=dtype)

    src, dst = edge_index[0], edge_index[1]

    if remove_self_loops:
        keep = src != dst
        src, dst, edge_weight = src[keep], dst[keep], edge_weight[keep]

    if symmetrize:
        # Half-weight each copy. If both directions already exist, coalescing
        # restores the original unit weight; if only one exists, both new
        # directions receive the same half-scale, which cancels in D^-1/2AD^-1/2.
        src_old, dst_old, w_old = src, dst, edge_weight
        src = torch.cat([src_old, dst_old], dim=0)
        dst = torch.cat([dst_old, src_old], dim=0)
        edge_weight = 0.5 * torch.cat([w_old, w_old], dim=0)

    # Sparse adjacency uses rows=dst, cols=src.
    sparse_idx = torch.stack([dst, src], dim=0)
    sparse_idx, edge_weight = _coalesce_edges(sparse_idx, edge_weight, num_nodes)
    dst_c, src_c = sparse_idx[0], sparse_idx[1]

    degree = torch.zeros(num_nodes, device=device, dtype=dtype)
    degree.index_add_(0, dst_c, edge_weight)
    inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    norm_weight = edge_weight * inv_sqrt[dst_c] * inv_sqrt[src_c]

    norm_adj = torch.sparse_coo_tensor(
        sparse_idx,
        norm_weight,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()
    return GraphCache(norm_adj=norm_adj, num_nodes=num_nodes)


class BernsteinGraphBasis:
    """Bernstein polynomial graph basis on S=L/2.

    B_k^Q(S) = C(Q,k) S^k (I-S)^(Q-k), k=0..Q.
    No dense Laplacian is formed.
    """

    def __init__(self, degree: int):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = degree

    def apply(self, k: int, x: torch.Tensor, graph: GraphCache) -> torch.Tensor:
        q = self.degree
        if not 0 <= k <= q:
            raise ValueError(f"basis index k={k} outside [0,{q}]")
        out = x
        # S and I-S commute because both are polynomials in L; order is arbitrary.
        for _ in range(k):
            out = graph.scaled_laplacian_mm(out)
        for _ in range(q - k):
            out = graph.one_minus_scaled_laplacian_mm(out)
        return float(comb(q, k)) * out
