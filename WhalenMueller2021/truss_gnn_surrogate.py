"""
WhalenMueller2021: Graph-Based Surrogate Model for Truss Displacement Prediction
Detailed implementation using PyTorch and PyTorch Geometric
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, input_features, hidden_dim, num_layers=2):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

class DisplacementDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(DisplacementDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_embeddings, loads):
        # Optionally concatenate loads to node embeddings
        x = torch.cat([node_embeddings, loads], dim=1)
        x = F.relu(self.fc1(x))
        displacements = self.fc2(x)
        return displacements

class TrussGraphSurrogate(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim, num_layers=2):
        super(TrussGraphSurrogate, self).__init__()
        self.encoder = GraphEncoder(input_features, hidden_dim, num_layers)
        self.decoder = DisplacementDecoder(hidden_dim + loads_dim, output_dim)

    def forward(self, data):
        # data: PyTorch Geometric Data object
        node_embeddings = self.encoder(data.x, data.edge_index)
        displacements = self.decoder(node_embeddings, data.loads)
        return displacements

# Example: Constructing a graph for a truss
# Node features: coordinates, boundary conditions (e.g., [x, y, fixed_x, fixed_y])
# Edge features: material properties, cross-sectional area (not used in GCNConv, but can be used in more advanced models)
# Loads: applied loads at each node

# Example data (dummy values)
import numpy as np
num_nodes = 5
input_features = 4  # [x, y, fixed_x, fixed_y]
hidden_dim = 16
output_dim = 2      # [dx, dy] per node
data_x = torch.tensor(np.random.rand(num_nodes, input_features), dtype=torch.float)
data_edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]], dtype=torch.long)  # shape [2, num_edges]
data_loads = torch.tensor(np.random.rand(num_nodes, 2), dtype=torch.float)  # [Fx, Fy] per node
loads_dim = data_loads.shape[1]

data = Data(x=data_x, edge_index=data_edge_index, loads=data_loads)

# Instantiate and run the model
model = TrussGraphSurrogate(input_features, hidden_dim, output_dim)
predicted_displacements = model(data)
print("Predicted displacements:", predicted_displacements)

# Training loop, loss, optimizer, and data loading would be added for a full implementation.
# For real use, replace dummy data with actual truss geometry, loads, and reference FEA results.
