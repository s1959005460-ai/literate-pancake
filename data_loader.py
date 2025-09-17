"""
Data loader for real graph datasets (Cora, Citeseer, PubMed).
"""

import os
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from typing import Tuple, List, Optional

from . import constants

def load_dataset(name: str = "Cora", root: str = "/data") -> Data:
    """
    Load a graph dataset from PyTorch Geometric.
    """
    dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
    return dataset[0]  # Return the graph data

def split_data_for_clients(data: Data, num_clients: int,
                          split_method: str = "random") -> List[Data]:
    """
    Split graph data for federated learning clients.
    """
    if split_method == "random":
        return _random_split(data, num_clients)
    elif split_method == "community":
        return _community_based_split(data, num_clients)
    else:
        raise ValueError(f"Unknown split method: {split_method}")

def _random_split(data: Data, num_clients: int) -> List[Data]:
    """
    Randomly split node indices among clients.
    """
    # Get all node indices
    num_nodes = data.num_nodes
    all_indices = torch.randperm(num_nodes)

    # Split indices among clients
    client_indices = torch.chunk(all_indices, num_clients)

    # Create client datasets
    client_data = []
    for indices in client_indices:
        client_data.append(Data(
            x=data.x[indices],
            edge_index=_extract_subgraph_edges(data.edge_index, indices),
            y=data.y[indices],
            train_mask=data.train_mask[indices] if hasattr(data, 'train_mask') else None,
            val_mask=data.val_mask[indices] if hasattr(data, 'val_mask') else None,
            test_mask=data.test_mask[indices] if hasattr(data, 'test_mask') else None
        ))

    return client_data

def _community_based_split(data: Data, num_clients: int) -> List[Data]:
    """
    Split graph based on community structure for non-IID setting.
    """
    # Use Louvain community detection
    from torch_geometric.utils import to_networkx
    import networkx as nx
    import community as community_louvain

    # Convert to NetworkX graph
    g = to_networkx(data, to_undirected=True)

    # Detect communities
    partition = community_louvain.best_partition(g)

    # Group nodes by community
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    # Assign communities to clients
    client_communities = [[] for _ in range(num_clients)]
    for i, comm_id in enumerate(communities.keys()):
        client_idx = i % num_clients
        client_communities[client_idx].extend(communities[comm_id])

    # Create client datasets
    client_data = []
    for indices in client_communities:
        indices = torch.tensor(indices, dtype=torch.long)
        client_data.append(Data(
            x=data.x[indices],
            edge_index=_extract_subgraph_edges(data.edge_index, indices),
            y=data.y[indices],
            train_mask=data.train_mask[indices] if hasattr(data, 'train_mask') else None,
            val_mask=data.val_mask[indices] if hasattr(data, 'val_mask') else None,
            test_mask=data.test_mask[indices] if hasattr(data, 'test_mask') else None
        ))

    return client_data

def _extract_subgraph_edges(edge_index: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    """
    Extract edges for a subgraph containing only the given nodes.
    """
    # Create mapping from original node indices to subgraph indices
    node_mapping = torch.full((node_indices.max().item() + 1,), -1, dtype=torch.long)
    node_mapping[node_indices] = torch.arange(len(node_indices))

    # Filter edges where both endpoints are in the subgraph
    mask = torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices)
    subgraph_edges = edge_index[:, mask]

    # Map original indices to subgraph indices
    subgraph_edges[0] = node_mapping[subgraph_edges[0]]
    subgraph_edges[1] = node_mapping[subgraph_edges[1]]

    return subgraph_edges
