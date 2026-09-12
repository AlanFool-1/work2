"""Reference GCN backbones used by FedAvg and FedPub."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from models.layers import MaskedGCNConv, MaskedLinear


class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
        super().__init__()
        self.dropout = float(args.dropout)
        self.conv1 = GCNConv(n_feat, n_dims, cached=False)
        self.conv2 = GCNConv(n_dims, n_dims, cached=False)
        self.clsif = nn.Linear(n_dims, n_clss)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy:
            return x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.clsif(x)


class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.dropout = float(args.dropout)
        self.conv1 = MaskedGCNConv(
            n_feat, n_dims, cached=False, l1=l1, args=args
        )
        self.conv2 = MaskedGCNConv(
            n_dims, n_dims, cached=False, l1=l1, args=args
        )
        self.clsif = MaskedLinear(n_dims, n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy:
            return x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.clsif(x)
