"""
Student implementation file — implement all TODO sections below.

This is the only Section B file you should submit.
"""

from __future__ import annotations

import torch
from torch_geometric.nn import SAGEConv


def get_feature_vectors(nodes_df):
    """Return node feature matrix x as a float torch.Tensor."""
    # TODO: Complete this function
    raise NotImplementedError


def get_edges(edges_df, inverse_node_id_mapping):
    """Return edge_index as a long torch.Tensor of shape [2, num_edges]."""
    # TODO: Complete this function
    raise NotImplementedError


def get_labels(nodes_df, subject_mapping):
    """Return node labels y as a long torch.Tensor."""
    # TODO: Complete this function
    raise NotImplementedError


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim, seed):
        super().__init__()
        torch.cuda.manual_seed(seed)
        # TODO: Complete this function

    def forward(self, x, edge_index):
        # TODO: Complete this function
        raise NotImplementedError


def train(data, model, optimizer, epochs, evaluate_fn):
    """
    Train the model for the given number of epochs.

    Use evaluate_fn(model, data.valid_mask) to track validation accuracy.
    Save the best checkpoint to 'best_model.pt' in the current working directory.
    """
    # TODO: Complete this function
    raise NotImplementedError
