"""Masked layers used by the reference FedPub implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor


class MaskedGCNConv(GCNConv):
    """GCNConv whose linear weight is multiplied by a learnable mask."""

    def __init__(self, in_channels, out_channels, l1=1e-3, args=None, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.l1 = float(l1)
        self.mask_one_init = bool(args.laye_mask_one)
        self.mask = Parameter(torch.ones_like(self.lin.weight))
        if not self.mask_one_init:
            torch.nn.init.xavier_uniform_(self.mask)

    def _masked_weight(self):
        mask = self.mask
        if not self.training:
            mask = mask.masked_fill(torch.abs(mask) < self.l1, 0)
        return self.lin.weight * mask

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        if isinstance(x, (tuple, list)):
            raise ValueError('MaskedGCNConv does not support bipartite inputs.')
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        x = F.linear(x, self._masked_weight(), self.lin.bias)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class MaskedLinear(torch.nn.Module):
    """Linear layer with the same train/eval mask behavior as Fedrated."""

    def __init__(self, d_i, d_o, l1=1e-3, args=None):
        super().__init__()
        self.l1 = float(l1)
        self.weight = Parameter(torch.empty(d_o, d_i))
        self.mask = Parameter(torch.ones(d_o, d_i))
        self.bias = Parameter(torch.zeros(1, d_o))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.bias)
        if not bool(args.clsf_mask_one):
            torch.nn.init.xavier_uniform_(self.mask)

    def forward(self, inputs):
        mask = self.mask
        if not self.training:
            mask = mask.masked_fill(torch.abs(mask) < self.l1, 0)
        return F.linear(inputs, self.weight * mask, self.bias)
