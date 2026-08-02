"""
Student implementation file — implement all TODO sections below.

This is the only Section B file you should submit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def get_feature_vectors(nodes_df):
    """Return node feature matrix x as a float torch.Tensor."""
    features = np.array(
        [
            np.fromstring(row.strip("[]"), dtype=np.float32, sep=",")
            for row in nodes_df["features"]
        ]
    )
    row_sums = features.sum(axis=1, keepdims=True)
    features = features / np.maximum(row_sums, 1.0)
    return torch.tensor(features, dtype=torch.float)


def get_edges(edges_df, inverse_node_id_mapping):
    """Return edge_index as a long torch.Tensor of shape [2, num_edges]."""
    sources = edges_df["sourceNodeId"].map(inverse_node_id_mapping).to_numpy()
    targets = edges_df["targetNodeId"].map(inverse_node_id_mapping).to_numpy()
    return torch.tensor(np.stack([sources, targets]), dtype=torch.long)


def get_labels(nodes_df, subject_mapping):
    """Return node labels y as a long torch.Tensor."""
    labels = nodes_df["subject"].map(subject_mapping).to_numpy()
    return torch.tensor(labels, dtype=torch.long)


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim, seed):
        super().__init__()
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        self.conv1 = SAGEConv(hidden_channels, 256)
        self.conv2 = SAGEConv(256, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


def train(data, model, optimizer, epochs, evaluate_fn):
    """
    Train the model for the given number of epochs.

    Use evaluate_fn(model, data.valid_mask) to track validation accuracy.
    Save the best checkpoint to 'best_model.pt' in the current working directory.
    """
    best_valid_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        valid_accuracy = evaluate_fn(model, data, data.valid_mask)
        if valid_accuracy >= best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model, "best_model.pt")
