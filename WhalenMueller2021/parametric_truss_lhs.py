"""
Parametric 15-Node Truss Bridge with 5 Design Variables
Implements Latin Hypercube Sampling (LHS) for design space exploration
Latin Hypercube Sampling (LHS) with filtering of top 10% worst designs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Model Components (same as before)
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

# Base Truss Geometry
def create_15_node_truss_geometry():
    """
    Create the 15-node truss bridge geometry baseline
    Span: 15.24 m, Height: 3.81 m
    """
    span = 15.24
    height = 3.81
    num_bottom_nodes = 8
    num_top_nodes = 7
    
    bottom_spacing = span / (num_bottom_nodes - 1)
    top_spacing = span / (num_top_nodes - 1)
    
    nodes = []
    # Bottom chord (nodes 0-7)
    for i in range(num_bottom_nodes):
        nodes.append([i * bottom_spacing, 0.0])
    # Top chord (nodes 8-14)
    for i in range(num_top_nodes):
        nodes.append([i * top_spacing, height])
    
    return np.array(nodes)

def create_parametric_truss(p_values):
    """
    Create parametric truss with 5 design variables
    p_values: [p1, p2, p3, p4, p5] each in range [0, 1]
    
    p1: Internal diagonal pattern (0=minimal, 1=maximal)
    p2: Cross-diagonal inclusion (0=none, 1=all)
    p3: Side diagonal pattern (0=minimal, 1=full)
    p4: Vertical member density (0=none, 1=all)
    p5: Secondary diagonals (0=none, 1=all)
    """
    p1, p2, p3, p4, p5 = p_values
    
    # Base geometry
    nodes = create_15_node_truss_geometry()
    
    # Create edges based on parameters
    edges = []
    
    # Chord members (always included - essential for stability)
    # Bottom chord
    for i in range(7):
        edges.append([i, i + 1])
    # Top chord
    for i in range(6):
        edges.append([8 + i, 8 + i + 1])
    
    # Ensure minimum connectivity with vertical members
    # At least corner supports
    edges.extend([
        [0, 8],      # Left vertical
        [7, 14],     # Right vertical
    ])
    
    # Additional vertical members (controlled by p4)
    # At least one middle vertical to improve stability
    edges.append([3, 11])  # Middle vertical (always)
    
    if p4 > 0.3:
        edges.extend([[1, 9], [6, 13]])
    if p4 > 0.6:
        edges.extend([[2, 10], [5, 12]])
    if p4 > 0.85:
        edges.append([4, 12])
    
    # Main diagonals - ensure at least one pair for stability
    diagonal_patterns = []
    
    # Core diagonals (bottom heavy for better performance)
    diagonal_patterns.extend([
        [1, 8],   # Left main diagonal
        [6, 14],  # Right main diagonal
    ])
    
    # Additional diagonals based on p1 and p3
    if p1 > 0.3:
        diagonal_patterns.extend([
            [2, 9],
            [5, 13],
        ])
    
    if p1 > 0.6:
        diagonal_patterns.extend([
            [3, 10],
            [4, 12],
        ])
    
    if p3 > 0.4:
        diagonal_patterns.extend([
            [2, 8],
            [7, 13],
        ])
    
    if p3 > 0.7:
        diagonal_patterns.extend([
            [1, 10],
            [5, 14],
        ])
    
    edges.extend(diagonal_patterns)
    
    # Cross diagonals (p2) - less critical for stability
    if p2 > 0.4:
        edges.extend([
            [1, 9],
            [2, 8],
        ])
    
    if p2 > 0.7:
        edges.extend([
            [5, 13],
            [6, 12],
        ])
    
    # Secondary diagonals (p5)
    if p5 > 0.5:
        edges.extend([
            [0, 10],
            [1, 11],
        ])
    
    if p5 > 0.8:
        edges.extend([
            [6, 10],
            [7, 11],
        ])
    
    # Remove duplicates
    edges = list(set([tuple(sorted(e)) for e in edges]))
    edges = [list(e) for e in edges]
    
    return nodes, edges

def create_boundary_conditions():
    """Define boundary conditions"""
    num_nodes = 15
    bc = np.zeros((num_nodes, 2))
    bc[0] = [1, 1]  # Pinned support at left
    bc[7] = [0, 1]  # Roller support at right
    return bc

def create_loads(load_magnitude=11.12):
    """Create load vector"""
    num_nodes = 15
    loads = np.zeros((num_nodes, 2))
    for i in range(8, 15):
        loads[i] = [0.0, -load_magnitude]
    return loads

def simple_fea_solver(nodes, edges, bc, loads, E=210.39e9, A=0.29):
    """Simplified FEA solver for truss displacements"""
    num_nodes = len(nodes)
    num_dofs = num_nodes * 2
    
    # Check for minimum connectivity (at least 12 members for stability)
    if len(edges) < 12:
        return np.full((num_nodes, 2), 1e3)  # Return large displacements for poor designs
    
    K_global = np.zeros((num_dofs, num_dofs))
    
    for edge in edges:
        i, j = edge
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        if L < 1e-10:
            continue
        
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
    
    F = loads.flatten() * 1000
    fixed_dofs = []
    for i in range(num_nodes):
        if bc[i, 0] == 1:
            fixed_dofs.append(2*i)
        if bc[i, 1] == 1:
            fixed_dofs.append(2*i + 1)
    
    free_dofs = [i for i in range(num_dofs) if i not in fixed_dofs]
    
    if len(free_dofs) == 0:
        return np.full((num_nodes, 2), 0.0)
    
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]
    
    try:
        # Check conditioning of matrix
        cond = np.linalg.cond(K_reduced)
        if cond > 1e12:  # Ill-conditioned matrix
            return np.full((num_nodes, 2), 1e3)
        
        U_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        # Check for unrealistic displacements
        if np.any(np.abs(U_reduced) > 1.0):  # More than 1m displacement
            return np.full((num_nodes, 2), 1e3)
        
    except np.linalg.LinAlgError:
        # Singular matrix - add regularization
        K_reduced_reg = K_reduced + np.eye(len(free_dofs)) * 1e-6 * np.mean(np.abs(K_reduced.diagonal()))
        try:
            U_reduced = np.linalg.solve(K_reduced_reg, F_reduced)
            if np.any(np.abs(U_reduced) > 1.0):
                return np.full((num_nodes, 2), 1e3)
        except:
            return np.full((num_nodes, 2), 1e3)
    
    U = np.zeros(num_dofs)
    U[free_dofs] = U_reduced
    
    return U.reshape((num_nodes, 2))

def get_edge_index_from_edges(edges):
    """Convert edge list to PyTorch edge_index format"""
    edge_list = edges + [[j, i] for i, j in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

# LHS Sampling
def latin_hypercube_sampling(n_samples, n_dims, seed=42):
    """
    Generate Latin Hypercube samples
    Returns samples in [0, 1] range
    """
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n_samples)
    return samples

# Main Training Script
if __name__ == "__main__":
    print("="*70)
    print("Parametric 15-Node Truss Bridge with LHS Design Space Exploration")
    print("="*70)
    
    # Generate LHS samples
    n_samples = 1000
    n_dims = 5
    
    print(f"\nGenerating {n_samples} LHS samples with {n_dims} design variables...")
    design_samples = latin_hypercube_sampling(n_samples, n_dims, seed=42)
    print(f"Design space: {n_samples} x {n_dims}")
    print(f"  p1 (internal diagonals): [0, 1]")
    print(f"  p2 (cross diagonals): [0, 1]")
    print(f"  p3 (side diagonals): [0, 1]")
    print(f"  p4 (vertical members): [0, 1]")
    print(f"  p5 (secondary diagonals): [0, 1]")
    
    # Evaluate displacement for each design
    print(f"\nEvaluating {n_samples} designs...")
    max_displacements = []
    valid_designs = []
    design_geometries = []
    
    for idx, p_values in enumerate(design_samples):
        if (idx + 1) % 100 == 0:
            print(f"  Evaluated {idx + 1}/{n_samples} designs...")
        
        try:
            # Create truss with these parameters
            nodes, edges = create_parametric_truss(p_values)
            
            # Skip if no valid edges or insufficient connectivity
            if len(edges) < 6:
                max_displacements.append(np.nan)
                continue
            
            bc = create_boundary_conditions()
            loads = create_loads(11.12)
            
            # Get displacements
            displacements = simple_fea_solver(nodes, edges, bc, loads)
            max_disp = np.max(np.abs(displacements))
            
            # Filter unrealistic displacements (more than 10cm is suspect for this structure)
            if max_disp > 0.1:  # More than 10cm displacement
                max_displacements.append(np.nan)
            else:
                max_displacements.append(max_disp)
                design_geometries.append((nodes, edges))
                valid_designs.append(idx)
            
        except Exception as e:
            max_displacements.append(np.nan)
    
    max_displacements = np.array(max_displacements)
    
    # Filter based on displacement
    print(f"\nFiltering designs...")
    print(f"  Total designs evaluated: {len(max_displacements)}")
    
    # Remove NaN/invalid displacements
    valid_mask = ~np.isnan(max_displacements)
    valid_indices = np.where(valid_mask)[0]
    valid_displacements = max_displacements[valid_mask]
    valid_samples = design_samples[valid_mask]
    
    print(f"  Max displacement range: [{np.min(valid_displacements):.6f}, {np.max(valid_displacements):.6f}] m")
    print(f"  Valid designs: {len(valid_indices)}")
    print(f"  Invalid designs: {n_samples - len(valid_indices)}")
    
    # Sort by displacement and discard top 10%
    sorted_indices = np.argsort(valid_displacements)
    n_to_discard = max(1, int(0.1 * len(sorted_indices)))
    n_to_keep = len(sorted_indices) - n_to_discard
    
    # Keep bottom 90%
    keep_indices = sorted_indices[:n_to_keep]
    discard_indices = sorted_indices[n_to_keep:]
    
    kept_indices = valid_indices[keep_indices]
    kept_samples = valid_samples[keep_indices]
    kept_displacements = valid_displacements[keep_indices]
    
    discarded_indices = valid_indices[discard_indices]
    discarded_displacements = valid_displacements[discard_indices]
    
    print(f"  Discarded (top 10%): {len(discard_indices)} designs")
    print(f"  Kept (bottom 90%): {len(keep_indices)} designs")
    print(f"  Displacement threshold: {kept_displacements[-1]:.6f} m")
    
    # Generate training data from kept designs
    print(f"\nGenerating training dataset from {len(kept_samples)} kept designs...")
    
    bc = create_boundary_conditions()
    data_list = []
    
    for idx, kept_idx in enumerate(kept_indices):
        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx + 1}/{len(keep_indices)} training samples...")
        
        p_values = design_samples[kept_idx]
        nodes, edges = create_parametric_truss(p_values)
        
        # Vary loads for each design
        for load_mag in np.linspace(8.0, 15.0, 3):
            loads = create_loads(load_mag)
            
            try:
                displacements = simple_fea_solver(nodes, edges, bc, loads)
                
                # Create edge_index
                edge_list = edges + [[j, i] for i, j in edges]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                
                # Node features with design parameters
                node_features = np.hstack([
                    nodes / 15.24,  # Normalized coordinates
                    bc,              # Boundary conditions
                    np.tile(p_values, (len(nodes), 1))  # Design parameters repeated
                ])
                
                x = torch.tensor(node_features, dtype=torch.float)
                loads_tensor = torch.tensor(loads / 20.0, dtype=torch.float)
                y = torch.tensor(displacements * 1e4, dtype=torch.float)
                
                data = Data(x=x, edge_index=edge_index, loads=loads_tensor, y=y)
                data_list.append(data)
            except:
                pass
    
    print(f"Total training samples generated: {len(data_list)}")
    
    # Split into train/test
    train_size = int(0.8 * len(data_list))
    test_size = len(data_list) - train_size
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model parameters
    input_features = 4 + 5  # Coordinates + BC + design params
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
    print("-"*70)
    
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
        
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch)
                loss = loss_fn(pred, batch.y)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0
        test_losses.append(avg_test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    print("-"*70)
    print(f"Training completed!")
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History - Parametric 15-Node Truss with LHS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history_parametric_lhs.png', dpi=150)
    print("Saved training history to training_history_parametric_lhs.png")
    plt.close()
    
    # Plot displacement distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of all displacements
    axes[0].hist(valid_displacements, bins=50, alpha=0.7, color='blue', label='All valid designs')
    axes[0].axvline(kept_displacements[-1], color='red', linestyle='--', linewidth=2, label='Threshold (90th percentile)')
    axes[0].set_xlabel('Maximum Displacement (m)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Maximum Displacements (1000 LHS samples)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1].boxplot([kept_displacements, discarded_displacements], tick_labels=['Kept (90%)', 'Discarded (10%)'])
    axes[1].set_ylabel('Maximum Displacement (m)')
    axes[1].set_title('Displacement Distribution: Kept vs Discarded Designs')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('displacement_distribution_lhs.png', dpi=150)
    print("Saved displacement distribution to displacement_distribution_lhs.png")
    plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'parametric_truss_model_lhs.pth')
    print("Saved model to parametric_truss_model_lhs.pth")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Design Space Exploration Summary")
    print("="*70)
    print(f"Total designs sampled: {n_samples}")
    print(f"Valid designs: {len(valid_indices)}")
    print(f"Invalid designs: {n_samples - len(valid_indices)}")
    print(f"\nDisplacement Statistics (Valid Designs):")
    print(f"  Minimum: {np.min(valid_displacements):.6f} m")
    print(f"  Maximum: {np.max(valid_displacements):.6f} m")
    print(f"  Mean: {np.mean(valid_displacements):.6f} m")
    print(f"  Median: {np.median(valid_displacements):.6f} m")
    print(f"  90th percentile: {np.percentile(valid_displacements, 90):.6f} m")
    print(f"\nDesign Filtering:")
    print(f"  Kept designs: {len(kept_indices)} ({100*len(kept_indices)/len(valid_indices):.1f}%)")
    print(f"  Discarded designs: {len(discard_indices)} ({100*len(discard_indices)/len(valid_indices):.1f}%)")
    print(f"\nTraining Dataset:")
    print(f"  Total training samples: {len(data_list)}")
    print(f"  Training set: {len(train_data)}")
    print(f"  Test set: {len(test_data)}")
    print("="*70)
