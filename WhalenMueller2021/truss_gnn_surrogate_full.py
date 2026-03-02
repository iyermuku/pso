"""
WhalenMueller2021: Full Working Graph-Based Surrogate Model for Truss Displacement Prediction
Includes data loading, training loop, and evaluation using PyTorch Geometric
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np

# Model Components
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
    def __init__(self, hidden_dim, loads_dim, output_dim):
        super(DisplacementDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + loads_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_embeddings, loads):
        x = torch.cat([node_embeddings, loads], dim=1)
        x = F.relu(self.fc1(x))
        displacements = self.fc2(x)
        return displacements

class TrussGraphSurrogate(nn.Module):
    def __init__(self, input_features, hidden_dim, loads_dim, output_dim, num_layers=2):
        super(TrussGraphSurrogate, self).__init__()
        self.encoder = GraphEncoder(input_features, hidden_dim, num_layers)
        self.decoder = DisplacementDecoder(hidden_dim, loads_dim, output_dim)

    def forward(self, data):
        node_embeddings = self.encoder(data.x, data.edge_index)
        displacements = self.decoder(node_embeddings, data.loads)
        return displacements

# Synthetic Data Generation for Demo
# Replace with real truss data for actual use

def generate_synthetic_truss_data(num_graphs=100, num_nodes=5):
    data_list = []
    for _ in range(num_graphs):
        x = torch.tensor(np.random.rand(num_nodes, 4), dtype=torch.float)  # [x, y, fixed_x, fixed_y]
        edge_index = torch.tensor([[i for i in range(num_nodes)], [(i+1)%num_nodes for i in range(num_nodes)]], dtype=torch.long)
        loads = torch.tensor(np.random.rand(num_nodes, 2), dtype=torch.float)  # [Fx, Fy]
        # Target: synthetic displacements (for demo)
        y = torch.tensor(np.random.rand(num_nodes, 2), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, loads=loads, y=y)
        data_list.append(data)
    return data_list

# Hyperparameters
input_features = 4
hidden_dim = 16
loads_dim = 2
output_dim = 2
num_layers = 2
batch_size = 8
epochs = 20
learning_rate = 0.001

# Data Preparation
train_data = generate_synthetic_truss_data(num_graphs=80, num_nodes=5)
test_data = generate_synthetic_truss_data(num_graphs=20, num_nodes=5)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Model, Loss, Optimizer
model = TrussGraphSurrogate(input_features, hidden_dim, loads_dim, output_dim, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
total_test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        total_test_loss += loss.item()
avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# Example prediction
example = test_data[0]
pred_disp = model(example)
print("Predicted displacements:", pred_disp)
print("True displacements:", example.y)

# For real use, replace synthetic data generation with actual truss geometry, loads, and FEA results.
