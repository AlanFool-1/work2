"""FedAux MaskedGCN ported from Fedrated's fedpub_aux implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.checkpoint import checkpoint

from models.layers import MaskedGCNConv, MaskedLinear


class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.sigma = float(args.sigma)
        self.dropout = float(args.dropout)
        self.conv1 = MaskedGCNConv(
            n_feat, n_dims, cached=False, l1=l1, args=args
        )
        self.conv2 = MaskedGCNConv(
            n_dims, n_dims, cached=False, l1=l1, args=args
        )
        self.clsif = MaskedLinear(2 * n_dims, n_clss, l1=l1, args=args)
        self.aux = nn.Parameter(torch.empty(n_dims))
        init.uniform_(self.aux)

    def _kernel_chunk(self, query_score, score, features):
        difference = query_score.unsqueeze(1) - score.unsqueeze(0)
        kernel = torch.exp(-(difference ** 2) / (self.sigma ** 2))
        return (kernel @ features) / kernel.sum(dim=1, keepdim=True).clamp(
            min=1e-12
        )

    def _kernel_aggregate(self, features):
        score = F.cosine_similarity(features, self.aux.unsqueeze(0), dim=-1)
        if features.size(0) <= 2048:
            return self._kernel_chunk(score, score, features)
        outputs = []
        for start in range(0, features.size(0), 1024):
            query = score[start:start + 1024]
            if self.training:
                output = checkpoint(
                    self._kernel_chunk,
                    query,
                    score,
                    features,
                    use_reentrant=False,
                )
            else:
                output = self._kernel_chunk(query, score, features)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy:
            return x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self._kernel_aggregate(x)
        return self.clsif(torch.cat([x, z], dim=-1))
