"""Prepare non-overlapping federated graph partitions for FedTFS."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import metispy as metis
import numpy as np
import torch
import torch_geometric.datasets as pyg_datasets
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph, to_networkx, to_scipy_sparse_matrix


DATASET_NAMES = [
    "Cora",
    "CiteSeer",
    "PubMed",
    "Roman-empire",
    "Computers",
    "Photo",
    "Amazon-ratings",
    "Minesweeper",
    "Tolokers",
    "Questions",
    "ogbn-arxiv",
]
BINARY_DATASETS = {"Minesweeper", "Tolokers", "Questions"}


class LargestConnectedComponents(BaseTransform):
    def __init__(self, num_components: int = 1):
        self.num_components = int(num_components)

    def __call__(self, data: Data) -> Data:
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj)
        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components :])
        return data.subgraph(torch.from_numpy(subset).to(torch.bool))


def save_pt(base_dir: Path, relative_path: str, payload: dict) -> None:
    path = base_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_graph(dataset: str, raw_root: Path) -> Data:
    lcc = LargestConnectedComponents()
    if dataset in {"Cora", "CiteSeer", "PubMed"}:
        transform = T.Compose([lcc, T.NormalizeFeatures()])
        data = pyg_datasets.Planetoid(str(raw_root), dataset, transform=transform)[0]
    elif dataset in {"Computers", "Photo"}:
        transform = T.Compose([lcc, T.NormalizeFeatures()])
        data = pyg_datasets.Amazon(str(raw_root), dataset, transform=transform)[0]
    elif dataset == "ogbn-arxiv":
        transform = T.Compose([T.ToUndirected(), lcc])
        data = PygNodePropPredDataset(dataset, root=str(raw_root), transform=transform)[0]
        data.y = data.y.view(-1)
    elif dataset in {"Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"}:
        transform = T.Compose([T.ToUndirected(), lcc])
        data = pyg_datasets.HeterophilousGraphDataset(str(raw_root), dataset, transform=transform)[0]
        if dataset in BINARY_DATASETS:
            data.y = data.y.float()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    return data


def set_random_split(data: Data, train_ratio: float) -> None:
    n_nodes = int(data.num_nodes)
    n_train = round(n_nodes * float(train_ratio))
    n_test = round(n_nodes * (1.0 - float(train_ratio)) / 2.0)
    indices = torch.randperm(n_nodes)

    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)
    data.train_mask[indices[:n_train]] = True
    data.test_mask[indices[n_train : n_train + n_test]] = True
    data.val_mask[indices[n_train + n_test :]] = True


def metis_membership(data: Data, n_clients: int) -> list[int]:
    graph = to_networkx(data, to_undirected=True)
    n_cuts, membership = metis.part_graph(graph, int(n_clients))
    if len(set(membership)) != int(n_clients):
        raise RuntimeError(f"METIS returned {len(set(membership))} partitions for {n_clients} clients.")
    print(f"[partition] clients={n_clients}, cut_edges={n_cuts}")
    return list(membership)


def all_clients_have_train_nodes(membership: list[int], train_mask: torch.Tensor, n_clients: int) -> bool:
    membership_array = np.asarray(membership)
    train_array = train_mask.cpu().numpy().astype(bool)
    for client_id in range(int(n_clients)):
        if np.sum(train_array[membership_array == client_id]) == 0:
            return False
    return True


def save_global_split(data: Data, dataset: str, n_clients: int, output_root: Path) -> None:
    prefix = f"{dataset}_disjoint/{n_clients}"
    save_pt(output_root, f"{prefix}/train.pt", {"data": data})
    save_pt(output_root, f"{prefix}/val.pt", {"data": data})
    save_pt(output_root, f"{prefix}/test.pt", {"data": data})


def save_client_partitions(data: Data, membership: list[int], dataset: str, n_clients: int, output_root: Path) -> None:
    membership_array = np.asarray(membership)
    prefix = f"{dataset}_disjoint/{n_clients}"
    for client_id in range(int(n_clients)):
        node_ids = np.flatnonzero(membership_array == client_id)
        node_index = torch.as_tensor(node_ids, dtype=torch.long)
        edge_index, _ = subgraph(node_index, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        client_data = Data(
            x=data.x[node_index],
            y=data.y[node_index],
            edge_index=edge_index.contiguous(),
            train_mask=data.train_mask[node_index],
            val_mask=data.val_mask[node_index],
            test_mask=data.test_mask[node_index],
        )
        if int(client_data.train_mask.sum().item()) == 0:
            raise RuntimeError(f"{dataset}/{n_clients}/client_{client_id} has no training node.")
        save_pt(output_root, f"{prefix}/partition_{client_id}.pt", {"client_data": client_data, "client_id": client_id})
        print(
            f"[save] dataset={dataset}, clients={n_clients}, client={client_id}, "
            f"nodes={client_data.num_nodes}, edges={client_data.num_edges}"
        )


def prepare_dataset(
    dataset: str,
    n_clients: int,
    raw_root: Path,
    output_root: Path,
    train_ratio: float,
    seed: int,
    max_split_attempts: int,
) -> None:
    print(f"[dataset] {dataset}, clients={n_clients}")
    data = load_graph(dataset, raw_root)
    membership = metis_membership(data, n_clients)

    for attempt in range(int(max_split_attempts)):
        torch.manual_seed(int(seed) + attempt)
        set_random_split(data, train_ratio)
        if all_clients_have_train_nodes(membership, data.train_mask, n_clients):
            save_global_split(data, dataset, n_clients, output_root)
            save_client_partitions(data, membership, dataset, n_clients, output_root)
            return

    raise RuntimeError(
        f"Could not produce a split with at least one training node per client "
        f"for {dataset}/{n_clients} after {max_split_attempts} attempts."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare disjoint federated graph partitions.")
    parser.add_argument("--datasets", nargs="+", default=DATASET_NAMES)
    parser.add_argument("--clients", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--raw-root", type=Path, default=Path("raw_datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-split-attempts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    for dataset in args.datasets:
        for n_clients in args.clients:
            prepare_dataset(
                dataset=dataset,
                n_clients=int(n_clients),
                raw_root=args.raw_root,
                output_root=args.output_root,
                train_ratio=float(args.train_ratio),
                seed=int(args.seed),
                max_split_attempts=int(args.max_split_attempts),
            )


if __name__ == "__main__":
    main()
