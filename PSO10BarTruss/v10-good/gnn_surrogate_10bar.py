"""
Graph Neural Network (GNN) Surrogate Model for 10-bar Truss

This module builds a GNN-based surrogate model that predicts nodal displacements
and member stresses for the 10-bar truss given cross-sectional areas and loads.

The model uses:
- Graph Convolutional Networks (GCN) for encoding truss topology
- MLP decoders for predicting displacements and stresses
- Training on FEA-generated data with Latin Hypercube Sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger("gnn_surrogate")

# ============================================================================
# GNN Architecture Components
# ============================================================================

class GraphEncoder(nn.Module):
    """Graph Convolutional encoder for truss topology"""
    
    def __init__(self, input_features: int, hidden_dim: int, num_layers: int = 3):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class DisplacementDecoder(nn.Module):
    """Decoder MLP for displacement prediction"""
    
    def __init__(self, hidden_dim: int, output_dim: int = 2):
        super(DisplacementDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_embeddings):
        x = F.relu(self.fc1(node_embeddings))
        x = F.relu(self.fc2(x))
        displacements = self.fc3(x)
        return displacements


class StressDecoder(nn.Module):
    """Decoder MLP for stress prediction at members"""
    
    def __init__(self, hidden_dim: int, num_members: int = 10):
        super(StressDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, node_embeddings, edge_index):
        # Aggregate embeddings at edge endpoints
        source, target = edge_index
        edge_features = torch.cat([
            node_embeddings[source],
            node_embeddings[target]
        ], dim=1)
        
        edge_features = F.relu(self.fc1(edge_features))
        edge_features = F.relu(self.fc2(edge_features))
        return self.fc3(edge_features)


class TrussGNNSurrogate(nn.Module):
    """End-to-end GNN surrogate model for 10-bar truss"""
    
    def __init__(
        self,
        input_features: int = 12,  # node features
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_members: int = 10
    ):
        super(TrussGNNSurrogate, self).__init__()
        self.encoder = GraphEncoder(input_features, hidden_dim, num_layers)
        self.disp_decoder = DisplacementDecoder(hidden_dim, output_dim=2)
        self.stress_decoder = StressDecoder(hidden_dim, num_members=num_members)
        
        self.hidden_dim = hidden_dim
        self.num_members = num_members
    
    def forward(self, data):
        """
        Forward pass
        Args:
            data: torch_geometric Data object with:
                - x: node features (num_nodes, input_features)
                - edge_index: edge connectivity (2, num_edges)
        Returns:
            displacements: (num_nodes, 2)
            stresses: (num_members, 1)
        """
        # Encode topology
        node_embeddings = self.encoder(data.x, data.edge_index)
        
        # Decode displacements
        displacements = self.disp_decoder(node_embeddings)
        
        # Decode stresses
        stresses = self.stress_decoder(node_embeddings, data.edge_index)
        
        return displacements, stresses


# ============================================================================
# Data Generation and Training Utils
# ============================================================================

def create_truss_graph(member_dof_indices: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create edge index for 10-bar truss
    Args:
        member_dof_indices: mapping from member_id to DOF indices
    Returns:
        edge_index: (2, num_edges) connectivity
        member_to_edge: mapping from member ID to edge index
    """
    edges = []
    member_to_edge = {}
    
    for m_id, dof_idx in sorted(member_dof_indices.items()):
        # Extract node IDs from DOF indices
        # DOF indices are [iux, iuy, jux, juy]
        node_i = dof_idx[0] // 2
        node_j = dof_idx[2] // 2
        edges.append([node_i, node_j])
        member_to_edge[m_id] = len(edges) - 1
    
    edge_index = np.array(edges).T
    return edge_index, member_to_edge


def create_graph_data(
    areas: np.ndarray,
    displacements: np.ndarray,
    stresses: np.ndarray,
    load_scale: float,
    node_coords: np.ndarray,
    fixed_dofs: List[int],
    edge_index: np.ndarray,
    ndof: int = 12
) -> Data:
    """
    Create PyTorch Geometric Data object for a single sample
    Args:
        areas: cross-sectional areas (10,)
        displacements: nodal displacements (ndof,)
        stresses: member stresses (10,)
        node_coords: node coordinates (num_nodes, 2)
        fixed_dofs: indices of fixed DOFs
        edge_index: connectivity (2, num_edges)
        ndof: total degrees of freedom
    Returns:
        Data object for GNN
    """
    num_nodes = node_coords.shape[0]
    
    # Node features: [x, y, bc_x, bc_y, areas..., load_scale]
    # Normalize coordinates to [0,1]
    x_norm = node_coords / np.max(np.abs(node_coords))
    
    # Boundary condition features
    bc_feat = np.zeros((num_nodes, 2))
    for dof_idx in fixed_dofs:
        node_id = dof_idx // 2
        dof_type = dof_idx % 2
        bc_feat[node_id, dof_type] = 1.0
    
    # Area features (replicate for each node, then take mean)
    area_features = np.tile(areas / 35.0, (num_nodes, 1))    # Normalize areas
    load_feat = np.full((num_nodes, 1), float(load_scale), dtype=float)
    
    node_features = np.hstack([
        x_norm,                              # coordinates (2)
        bc_feat,                             # boundary conditions (2)
        area_features,                       # area features (10)
        load_feat,                           # load scale (1)
    ])  # Total: 15 features
    
    # Convert to torch tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    y_disp = torch.tensor(displacements.reshape(num_nodes, 2), dtype=torch.float32)
    y_stress = torch.tensor(stresses, dtype=torch.float32)
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index_t,
        y_disp=y_disp,
        y_stress=y_stress
    )
    
    return data


# ============================================================================
# Training
# ============================================================================

def train_gnn_surrogate(
    train_data_list: List[Data],
    val_data_list: List[Data],
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Tuple[TrussGNNSurrogate, Dict]:
    """Train GNN surrogate model"""
    
    model = TrussGNNSurrogate(
        input_features=15,
        hidden_dim=64,
        num_layers=3,
        num_members=10
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn_disp = nn.MSELoss()
    loss_fn_stress = nn.MSELoss()
    
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)
    
    train_losses = []
    val_losses = []
    
    logger.info(f"Training GNN with {len(train_data_list)} samples...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            disp_pred, stress_pred = model(data)
            
            loss_disp = loss_fn_disp(disp_pred, data.y_disp)
            loss_stress = loss_fn_stress(stress_pred, data.y_stress.unsqueeze(1))
            loss = loss_disp + 0.1 * loss_stress  # Weight stress loss lower
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                disp_pred, stress_pred = model(data)
                
                loss_disp = loss_fn_disp(disp_pred, data.y_disp)
                loss_stress = loss_fn_stress(stress_pred, data.y_stress.unsqueeze(1))
                loss = loss_disp + 0.1 * loss_stress
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % max(1, epochs // 10) == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    logger.info("Training complete!")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def predict_with_gnn(
    model: TrussGNNSurrogate,
    areas: np.ndarray,
    node_coords: np.ndarray,
    fixed_dofs: List[int],
    edge_index: np.ndarray,
    load_scale: float = 1.0,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use trained GNN to predict displacements and stresses
    Returns:
        displacements: (ndof,)
        stresses: (10,)
    """
    # Create feature tensor
    num_nodes = node_coords.shape[0]
    x_norm = node_coords / np.max(np.abs(node_coords))
    
    bc_feat = np.zeros((num_nodes, 2))
    for dof_idx in fixed_dofs:
        node_id = dof_idx // 2
        dof_type = dof_idx % 2
        bc_feat[node_id, dof_type] = 1.0
    
    area_features = np.tile(areas / 35.0, (num_nodes, 1))
    load_feat = np.full((num_nodes, 1), float(load_scale), dtype=float)

    node_features = np.hstack([x_norm, bc_feat, area_features, load_feat])
    
    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(device)
    
    # Single data point (batch_size=1)
    data = Data(x=x, edge_index=edge_index_t).to(device)
    
    model.eval()
    with torch.no_grad():
        disp_pred, stress_pred = model(data)
    
    # Reshape and return
    displacements = disp_pred.cpu().numpy().flatten()

    stress_np = stress_pred.cpu().numpy()
    if stress_np.ndim == 2:
        if stress_np.shape[1] == 1:
            stresses = stress_np[:, 0]
        elif stress_np.shape[0] == stress_np.shape[1]:
            stresses = np.diag(stress_np)
        else:
            stresses = np.mean(stress_np, axis=1)
    else:
        stresses = stress_np.flatten()

    if len(stresses) >= 10:
        stresses = stresses[:10]
    else:
        stresses = np.pad(stresses, (0, 10 - len(stresses)), mode='constant')
    
    return displacements, stresses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GNN Surrogate Model for 10-bar Truss - Module loaded successfully")
