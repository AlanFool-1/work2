"""FedPub personalized server without federated optimizer-state mixing."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from models.nets import MaskedGCN
from modules.fed_server import FullModelServer


def similarity_matrix(embeddings, mode, scale):
    vectors = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    if mode == 'exp':
        similarities = np.exp(float(scale) * similarities)
    elif mode != 'cosine':
        raise ValueError(f'Unknown FedPub aggregation normalization: {mode}')
    row_sums = similarities.sum(axis=1, keepdims=True)
    if not np.isfinite(similarities).all() or np.any(row_sums <= 0.0):
        raise FloatingPointError('FedPub produced an invalid similarity matrix.')
    return similarities / row_sums


class Server(FullModelServer):
    embedding_key = 'functional_embedding'
    uses_proxy = True

    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)
        self.sim_matrices = []
        self.update_lists = []
        if self.uses_proxy:
            self.sd['proxy'] = self.get_proxy_data()

    def build_model(self):
        return MaskedGCN(
            self.args.n_feat,
            self.args.n_dims,
            self.args.n_clss,
            self.args.l1,
            self.args,
        )

    def get_proxy_data(self):
        num_graphs = int(self.args.n_proxy)
        num_nodes = int(self.args.proxy_nodes)
        graph = nx.random_partition_graph(
            [num_nodes] * num_graphs,
            p_in=0.1,
            p_out=0.0,
            seed=int(self.args.seed),
        )
        proxy = from_networkx(graph)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(self.args.seed))
        proxy.x = torch.normal(
            mean=0.0,
            std=1.0,
            size=(num_nodes * num_graphs, self.args.n_feat),
            generator=generator,
        )
        proxy.edge_attr = None
        return proxy

    def personalization_embedding(self, message):
        return message[self.embedding_key]

    def update(self, updated):
        client_ids = sorted(int(client_id) for client_id in updated)
        messages = [dict(self.sd[client_id]) for client_id in client_ids]
        embeddings = [self.personalization_embedding(message) for message in messages]
        local_weights = [message['model'] for message in messages]

        super().update(client_ids)

        similarities = similarity_matrix(
            embeddings, self.args.agg_norm, self.args.norm_scale
        )
        for index, client_id in enumerate(client_ids):
            self.sd[f'personalized_{client_id}'] = self.aggregate(
                local_weights, ratio=similarities[index].tolist()
            )
        self.sim_matrices.append(similarities)
        self.update_lists.append(client_ids)

    def checkpoint_extra_state(self):
        return {
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists,
            'personalization_embedding': self.embedding_key,
        }
