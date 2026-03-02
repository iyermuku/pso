"""
15-Node Truss Bridge Model - Graph-Based Surrogate for Displacement Prediction
Based on the truss structure: 15.24m span, 3.81m height, with distributed loads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt

# Model Components
class GraphEncoder(nn.Module):
    def __init__(self, input_features, hidden_dim, num_layers=3):
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
    def __init__(self, hidden_dim, loads_dim, output_dim):
        super(DisplacementDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + loads_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_embeddings, loads):
        x = torch.cat([node_embeddings, loads], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        displacements = self.fc3(x)
        return displacements

class TrussGraphSurrogate(nn.Module):
    def __init__(self, input_features, hidden_dim, loads_dim, output_dim, num_layers=3):
        super(TrussGraphSurrogate, self).__init__()
        self.encoder = GraphEncoder(input_features, hidden_dim, num_layers)
        self.decoder = DisplacementDecoder(hidden_dim, loads_dim, output_dim)

    def forward(self, data):
        node_embeddings = self.encoder(data.x, data.edge_index)
        displacements = self.decoder(node_embeddings, data.loads)
        return displacements

# 15-Node Truss Geometry Definition
def create_15_node_truss_geometry():
    """
    Create the 15-node truss bridge geometry
    Span: 15.24 m, Height: 3.81 m
    """
    # Node coordinates (x, y) in meters
    # Bottom chord (nodes 0-7)
    span = 15.24
    height = 3.81
    num_bottom_nodes = 8
    num_top_nodes = 7
    
    bottom_spacing = span / (num_bottom_nodes - 1)
    top_spacing = span / (num_top_nodes - 1)
    
    nodes = []
    # Bottom chord
    for i in range(num_bottom_nodes):
        nodes.append([i * bottom_spacing, 0.0])
    
    # Top chord (nodes 8-14)
    for i in range(num_top_nodes):
        nodes.append([i * top_spacing, height])
    
    nodes = np.array(nodes)
    
    # Define connectivity (edges) - member connections
    edges = []
    
    # Bottom chord
    for i in range(num_bottom_nodes - 1):
        edges.append([i, i + 1])
    
    # Top chord
    for i in range(num_top_nodes - 1):
        edges.append([8 + i, 8 + i + 1])
    
    # Vertical and diagonal members
    # Left to right pattern
    edges.extend([
        [0, 8],      # Vertical
        [1, 8],      # Diagonal
        [1, 9],      # Vertical
        [2, 9],      # Diagonal
        [2, 10],     # Vertical
        [3, 10],     # Diagonal
        [3, 11],     # Vertical
        [4, 11],     # Diagonal
        [4, 12],     # Vertical
        [5, 12],     # Diagonal
        [5, 13],     # Vertical
        [6, 13],     # Diagonal
        [6, 14],     # Vertical
        [7, 14],     # Diagonal
    ])
    
    return nodes, edges

def create_boundary_conditions():
    """
    Define boundary conditions
    Node 0: Pinned support (fixed in x and y)
    Node 7: Roller support (fixed in y only)
    """
    num_nodes = 15
    bc = np.zeros((num_nodes, 2))  # [fixed_x, fixed_y]
    bc[0] = [1, 1]  # Pinned support at left
    bc[7] = [0, 1]  # Roller support at right
    return bc

def create_loads(load_magnitude=11.12):
    """
    Create load vector
    Loads applied at top nodes (downward)
    load_magnitude in kN
    """
    num_nodes = 15
    loads = np.zeros((num_nodes, 2))  # [Fx, Fy]
    
    # Apply downward loads at top chord nodes (8-14)
    for i in range(8, 15):
        loads[i] = [0.0, -load_magnitude]
    
    return loads

def simple_fea_solver(nodes, edges, bc, loads, E=210.39e9, A=0.29):
    """
    Simplified FEA solver for truss displacements
    E: Young's modulus (Pa) - 30.5 Msi = 210.39 GPa for steel
    A: Cross-sectional area (m^2) - 0.29 m^2
    I: Moment of inertia (m^4) - 2.3 × 10^-3 m^4 (for reference, not used in pure truss analysis)
    """
    num_nodes = len(nodes)
    num_dofs = num_nodes * 2
    
    # Initialize global stiffness matrix
    K_global = np.zeros((num_dofs, num_dofs))
    
    # Assemble stiffness matrix
    for edge in edges:
        i, j = edge
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        c = (xj - xi) / L
        s = (yj - yi) / L
        
        k = (E * A / L) * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])
        
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        for m in range(4):
            for n in range(4):
                K_global[dofs[m], dofs[n]] += k[m, n]
    
    # Apply boundary conditions and loads
    F = loads.flatten() * 1000  # Convert kN to N
    fixed_dofs = []
    for i in range(num_nodes):
        if bc[i, 0] == 1:
            fixed_dofs.append(2*i)
        if bc[i, 1] == 1:
            fixed_dofs.append(2*i + 1)
    
    free_dofs = [i for i in range(num_dofs) if i not in fixed_dofs]
    
    # Solve for displacements
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]
    
    try:
        U_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        # If singular, add small regularization
        K_reduced += np.eye(len(free_dofs)) * 1e-6 * np.mean(np.abs(K_reduced))
        U_reduced = np.linalg.solve(K_reduced, F_reduced)
    
    U = np.zeros(num_dofs)
    U[free_dofs] = U_reduced
    
    return U.reshape((num_nodes, 2))

def generate_truss_dataset(num_samples=200):
    """
    Generate dataset with varying load conditions
    """
    nodes, edges = create_15_node_truss_geometry()
    bc = create_boundary_conditions()
    
    # Convert edges to edge_index format
    edge_list = edges + [[j, i] for i, j in edges]  # Undirected
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data_list = []
    
    for _ in range(num_samples):
        # Vary load magnitude (random between 5 and 20 kN)
        load_magnitude = np.random.uniform(5.0, 20.0)
        loads = create_loads(load_magnitude)
        
        # Solve using FEA
        displacements = simple_fea_solver(nodes, edges, bc, loads)
        
        # Normalize coordinates for neural network
        node_features = np.hstack([nodes / 15.24, bc])  # Normalize by span
        
        # Create data object with normalized loads for better NN convergence
        x = torch.tensor(node_features, dtype=torch.float)
        loads_tensor = torch.tensor(loads / 20.0, dtype=torch.float)  # Normalize loads to [-1, 1] range
        # Scale displacements for better NN training (multiply by 1e4 to get values in useful range)
        y = torch.tensor(displacements * 1e4, dtype=torch.float)  # Scale to mm × 100
        
        data = Data(x=x, edge_index=edge_index, loads=loads_tensor, y=y)
        data_list.append(data)
    
    return data_list

def visualize_truss(nodes, edges, displacements=None, scale=100):
    """
    Visualize the truss structure
    """
    plt.figure(figsize=(12, 6))
    
    # Plot original structure
    for edge in edges:
        i, j = edge
        plt.plot([nodes[i, 0], nodes[j, 0]], 
                [nodes[i, 1], nodes[j, 1]], 'b-', linewidth=1, alpha=0.5)
    
    plt.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=6, label='Nodes')
    
    # Plot deformed structure if provided
    if displacements is not None:
        deformed = nodes + displacements * scale
        for edge in edges:
            i, j = edge
            plt.plot([deformed[i, 0], deformed[j, 0]], 
                    [deformed[i, 1], deformed[j, 1]], 'r--', linewidth=1.5, alpha=0.7)
        plt.plot(deformed[:, 0], deformed[:, 1], 'ro', markersize=6, label='Deformed')
        plt.title(f'15-Node Truss (Deformation scale: {scale}x)')
    else:
        plt.title('15-Node Truss Structure')
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('truss_15_node_structure.png', dpi=150)
    print("Saved visualization to truss_15_node_structure.png")
    plt.close()

# Main Training Script
if __name__ == "__main__":
    print("="*60)
    print("15-Node Truss Bridge - GNN Surrogate Model Training")
    print("="*60)
    
    # Visualize the truss structure
    nodes, edges = create_15_node_truss_geometry()
    bc = create_boundary_conditions()
    loads = create_loads(11.12)
    
    print(f"\nTruss Configuration:")
    print(f"  Number of nodes: {len(nodes)}")
    print(f"  Number of members: {len(edges)}")
    print(f"  Span: 15.24 m")
    print(f"  Height: 3.81 m")
    print(f"  Load per top node: 11.12 kN (downward)")
    print(f"\nMaterial Properties:")
    print(f"  Material: Steel")
    print(f"  Young's modulus (E): 30.5 Msi = 210.39 GPa")
    print(f"  Cross-sectional area (A): 0.29 m²")
    print(f"  Moment of inertia (I): 2.3 × 10⁻³ m⁴")
    
    # Generate sample displacement for visualization
    sample_disp = simple_fea_solver(nodes, edges, bc, loads)
    visualize_truss(nodes, edges, sample_disp, scale=500)
    
    print(f"\nGenerating training dataset...")
    # Generate dataset
    dataset = generate_truss_dataset(num_samples=500)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model parameters
    input_features = 4  # [x, y, fixed_x, fixed_y]
    hidden_dim = 64
    loads_dim = 2
    output_dim = 2
    num_layers = 3
    
    # Initialize model
    model = TrussGraphSurrogate(input_features, hidden_dim, loads_dim, output_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print(f"\nModel Architecture:")
    print(f"  Input features: {input_features}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  GNN layers: {num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    epochs = 50
    print(f"\nTraining for {epochs} epochs...")
    print("-"*60)
    
    train_losses = []
    test_losses = []
    
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
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    print("-"*60)
    print(f"Training completed!")
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    
    # Test on specific example
    print("\n" + "="*60)
    print("Testing on Example with 11.12 kN Load")
    print("="*60)
    
    test_example = test_data[0]
    model.eval()
    with torch.no_grad():
        pred_disp_scaled = model(test_example).numpy()
    
    # Descale predictions back to meters
    pred_disp = pred_disp_scaled / 1e4
    true_disp = test_example.y.numpy() / 1e4
    
    print("\nDisplacement Predictions (meters):")
    print(f"{'Node':>4} | {'Pred_dx':>10} | {'True_dx':>10} | {'Pred_dy':>10} | {'True_dy':>10} | {'Error_dx':>10} | {'Error_dy':>10}")
    print("-"*90)
    for i in range(len(pred_disp)):
        error_dx = abs(pred_disp[i, 0] - true_disp[i, 0])
        error_dy = abs(pred_disp[i, 1] - true_disp[i, 1])
        print(f"{i:4d} | {pred_disp[i, 0]:10.6f} | {true_disp[i, 0]:10.6f} | {pred_disp[i, 1]:10.6f} | {true_disp[i, 1]:10.6f} | {error_dx:10.6f} | {error_dy:10.6f}")
    
    # Calculate max displacements
    max_pred_disp = np.max(np.abs(pred_disp))
    max_true_disp = np.max(np.abs(true_disp))
    
    print(f"\nMaximum Displacement:")
    print(f"  Predicted: {max_pred_disp:.6f} m")
    print(f"  True (FEA): {max_true_disp:.6f} m")
    if max_true_disp > 0:
        print(f"  Relative Error: {abs(max_pred_disp - max_true_disp) / max_true_disp * 100:.2f}%")
    else:
        print(f"  Relative Error: N/A (true displacement ~0)")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History - 15-Node Truss GNN Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history_15_node.png', dpi=150)
    print("\nSaved training history to training_history_15_node.png")
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'truss_15_node_model.pth')
    print("Saved model to truss_15_node_model.pth")
    
    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)
