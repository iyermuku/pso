"""
Model Deployment and Evaluation
Computes MAE and MSE metrics and visualizes predictions for 4 loading cases
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Model Components (same as training)
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
    span = 15.24
    height = 3.81
    num_bottom_nodes = 8
    num_top_nodes = 7
    
    bottom_spacing = span / (num_bottom_nodes - 1)
    top_spacing = span / (num_top_nodes - 1)
    
    nodes = []
    for i in range(num_bottom_nodes):
        nodes.append([i * bottom_spacing, 0.0])
    for i in range(num_top_nodes):
        nodes.append([i * top_spacing, height])
    
    return np.array(nodes)

def create_parametric_truss(p_values):
    """Create parametric truss with 5 design variables"""
    p1, p2, p3, p4, p5 = p_values
    
    nodes = create_15_node_truss_geometry()
    edges = []
    
    # Chord members (always included)
    for i in range(7):
        edges.append([i, i + 1])
    for i in range(6):
        edges.append([8 + i, 8 + i + 1])
    
    # Ensure minimum connectivity with vertical members
    edges.extend([[0, 8], [7, 14], [3, 11]])
    
    if p4 > 0.3:
        edges.extend([[1, 9], [6, 13]])
    if p4 > 0.6:
        edges.extend([[2, 10], [5, 12]])
    if p4 > 0.85:
        edges.append([4, 12])
    
    # Main diagonals
    diagonal_patterns = [[1, 8], [6, 14]]
    
    if p1 > 0.3:
        diagonal_patterns.extend([[2, 9], [5, 13]])
    if p1 > 0.6:
        diagonal_patterns.extend([[3, 10], [4, 12]])
    if p3 > 0.4:
        diagonal_patterns.extend([[2, 8], [7, 13]])
    if p3 > 0.7:
        diagonal_patterns.extend([[1, 10], [5, 14]])
    
    edges.extend(diagonal_patterns)
    
    if p2 > 0.4:
        edges.extend([[1, 9], [2, 8]])
    if p2 > 0.7:
        edges.extend([[5, 13], [6, 12]])
    
    if p5 > 0.5:
        edges.extend([[0, 10], [1, 11]])
    if p5 > 0.8:
        edges.extend([[6, 10], [7, 11]])
    
    edges = list(set([tuple(sorted(e)) for e in edges]))
    edges = [list(e) for e in edges]
    
    return nodes, edges

def create_boundary_conditions():
    num_nodes = 15
    bc = np.zeros((num_nodes, 2))
    bc[0] = [1, 1]
    bc[7] = [0, 1]
    return bc

def create_loads(load_magnitude=11.12):
    num_nodes = 15
    loads = np.zeros((num_nodes, 2))
    for i in range(8, 15):
        loads[i] = [0.0, -load_magnitude]
    return loads

def simple_fea_solver(nodes, edges, bc, loads, E=210.39e9, A=0.29):
    """Simplified FEA solver for truss displacements"""
    num_nodes = len(nodes)
    num_dofs = num_nodes * 2
    
    if len(edges) < 12:  # Minimum connectivity requirement
        return np.full((num_nodes, 2), 1e3)
    
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
        cond = np.linalg.cond(K_reduced)
        if cond > 1e12:  # Ill-conditioned matrix check
            return np.full((num_nodes, 2), 1e3)
        
        U_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        if np.any(np.abs(U_reduced) > 1.0):
            return np.full((num_nodes, 2), 1e3)
        
    except:
        return np.full((num_nodes, 2), 1e3)
    
    U = np.zeros(num_dofs)
    U[free_dofs] = U_reduced
    
    return U.reshape((num_nodes, 2))

def compute_metrics(pred, true):
    """Compute MAE and MSE"""
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true)**2)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def visualize_truss_comparison(nodes, edges, undeformed, fea_def, gsm_def, title, filename, loads=None, bc=None):
    """
    Visualize truss with undeformed, FEA-deformed, and GSM-predicted deformed overlaid
    Includes loading arrows and boundary condition symbols
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Adaptive scale factor for deformation visibility
    max_displacement = max(np.max(np.abs(fea_def)), np.max(np.abs(gsm_def)))
    span = np.max(nodes[:, 0]) - np.min(nodes[:, 0])
    if max_displacement > 0:
        scale_factor = span * 0.15 / max_displacement  # Scale to ~15% of span
    else:
        scale_factor = 1000
    
    # Plot undeformed structure (black, solid, thicker)
    for edge in edges:
        i, j = edge
        ax.plot([nodes[i, 0], nodes[j, 0]], 
               [nodes[i, 1], nodes[j, 1]], 'k-', linewidth=2.5, alpha=0.6, 
               label='Undeformed' if edge == edges[0] else '')
    ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=8, label='Undeformed Nodes', zorder=5)
    
    # Plot FEA deformed structure (blue, dashed)
    fea_nodes = nodes + fea_def * scale_factor
    for edge in edges:
        i, j = edge
        ax.plot([fea_nodes[i, 0], fea_nodes[j, 0]], 
               [fea_nodes[i, 1], fea_nodes[j, 1]], 'b--', linewidth=2.5, alpha=0.8,
               label='FEA Deformed' if edge == edges[0] else '')
    ax.plot(fea_nodes[:, 0], fea_nodes[:, 1], 'bs', markersize=8, label='FEA Nodes', zorder=5)
    
    # Plot GSM deformed structure (red, dash-dot)
    gsm_nodes = nodes + gsm_def * scale_factor
    for edge in edges:
        i, j = edge
        ax.plot([gsm_nodes[i, 0], gsm_nodes[j, 0]], 
               [gsm_nodes[i, 1], gsm_nodes[j, 1]], 'r-.', linewidth=2.5, alpha=0.8,
               label='GSM Deformed' if edge == edges[0] else '')
    ax.plot(gsm_nodes[:, 0], gsm_nodes[:, 1], 'r^', markersize=8, label='GSM Nodes', zorder=5)
    
    # Add boundary conditions
    if bc is not None:
        for i, bc_node in enumerate(bc):
            if bc_node[0] == 1 and bc_node[1] == 1:  # Pinned support
                # Draw triangle for pinned support
                support_size = 0.3
                triangle = plt.Polygon([[nodes[i, 0] - support_size/2, nodes[i, 1] - support_size],
                                       [nodes[i, 0] + support_size/2, nodes[i, 1] - support_size],
                                       [nodes[i, 0], nodes[i, 1]]], 
                                      color='green', alpha=0.7, zorder=10,
                                      label='Pinned Support' if i == 0 else '')
                ax.add_patch(triangle)
            elif bc_node[0] == 0 and bc_node[1] == 1:  # Roller support
                # Draw circle and triangle for roller support
                support_size = 0.3
                circle = plt.Circle((nodes[i, 0], nodes[i, 1] - support_size*0.7), 
                                   support_size/3, color='green', alpha=0.7, zorder=10)
                ax.add_patch(circle)
                triangle = plt.Polygon([[nodes[i, 0] - support_size/2, nodes[i, 1] - support_size*1.3],
                                       [nodes[i, 0] + support_size/2, nodes[i, 1] - support_size*1.3],
                                       [nodes[i, 0], nodes[i, 1] - support_size*0.9]], 
                                      color='green', alpha=0.7, zorder=10,
                                      label='Roller Support' if i == 7 else '')
                ax.add_patch(triangle)
    
    # Add loads as arrows
    if loads is not None:
        arrow_scale = span * 0.08  # Arrow length proportional to span
        for i, load in enumerate(loads):
            if np.any(load != 0):
                # Normalize load magnitude for arrow length
                load_mag = np.linalg.norm(load)
                if load_mag > 0:
                    arrow_length = arrow_scale * (load_mag / np.max(np.abs(loads)))
                    ax.arrow(nodes[i, 0], nodes[i, 1], 
                           load[0]/load_mag * arrow_length, 
                           load[1]/load_mag * arrow_length,
                           head_width=0.3, head_length=0.2, 
                           fc='darkorange', ec='darkorange', 
                           linewidth=2.5, alpha=0.9, zorder=10,
                           label='Applied Load' if i == 8 else '')
    
    ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\n(Deformation scaled by {scale_factor:.0f}× for visibility)", 
                fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {filename}")
    plt.close()

# Main Deployment Script
if __name__ == "__main__":
    print("="*80)
    print("Model Deployment and Evaluation - 4 Loading Cases")
    print("="*80)
    
    # Load trained model
    model = TrussGraphSurrogate(input_features=9, hidden_dim=64, loads_dim=2, output_dim=2, num_layers=3)
    model.load_state_dict(torch.load('parametric_truss_model_lhs.pth'))
    model.eval()
    
    print("\n✓ Model loaded successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define 4 different loading cases
    loading_cases = [
        {"name": "Light Load (5 kN)", "magnitude": 5.0},
        {"name": "Standard Load (11.12 kN)", "magnitude": 11.12},
        {"name": "Heavy Load (15 kN)", "magnitude": 15.0},
        {"name": "Very Heavy Load (20 kN)", "magnitude": 20.0},
    ]
    
    # Define 4 different design cases (using p-values that generate 35+ member topologies)
    design_cases = [
        {"name": "Design 1: Full Bracing A", "p_values": np.array([0.85, 0.85, 0.85, 0.85, 0.85])},
        {"name": "Design 2: Full Bracing B", "p_values": np.array([0.87, 0.87, 0.87, 0.87, 0.87])},
        {"name": "Design 3: Maximum Bracing", "p_values": np.array([0.9, 0.9, 0.9, 0.9, 0.9])},
        {"name": "Design 4: Extreme Bracing", "p_values": np.array([0.95, 0.95, 0.95, 0.95, 0.95])},
    ]
    
    # Evaluation results storage
    all_metrics = []
    
    # Test each design with corresponding loading case
    for design_idx, (design_case, load_case) in enumerate(zip(design_cases, loading_cases)):
        print(f"\n{'='*80}")
        print(f"Case {design_idx + 1}: {design_case['name']} + {load_case['name']}")
        print(f"{'='*80}")
        
        # Create truss geometry
        nodes, edges = create_parametric_truss(design_case['p_values'])
        bc = create_boundary_conditions()
        loads = create_loads(load_case['magnitude'])
        
        print(f"  Truss Configuration:")
        print(f"    Nodes: {len(nodes)}, Members: {len(edges)}")
        print(f"    Design params: {design_case['p_values']}")
        print(f"    Load magnitude: {load_case['magnitude']} kN")
        
        # Compute FEA reference solution
        fea_displacements = simple_fea_solver(nodes, edges, bc, loads)
        
        # Create graph data for model
        edge_list = edges + [[j, i] for i, j in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        node_features = np.hstack([
            nodes / 15.24,
            bc,
            np.tile(design_case['p_values'], (len(nodes), 1))
        ])
        
        x = torch.tensor(node_features, dtype=torch.float)
        loads_tensor = torch.tensor(loads / 20.0, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, loads=loads_tensor, y=None)
        
        # Get model predictions
        with torch.no_grad():
            gsm_pred_scaled = model(data).numpy()
        
        # Descale predictions
        gsm_displacements = gsm_pred_scaled / 1e4
        
        # Compute metrics
        mae, mse, rmse = compute_metrics(gsm_displacements, fea_displacements)
        all_metrics.append({
            'case': f"Case {design_idx + 1}",
            'design': design_case['name'],
            'load': load_case['name'],
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'max_fea': np.max(np.abs(fea_displacements)),
            'max_gsm': np.max(np.abs(gsm_displacements))
        })
        
        print(f"\n  Metrics:")
        print(f"    MAE (Mean Absolute Error):  {mae:.8f} m")
        print(f"    MSE (Mean Squared Error):   {mse:.10f} m²")
        print(f"    RMSE (Root Mean Squared):   {rmse:.8f} m")
        print(f"    Max FEA displacement:        {np.max(np.abs(fea_displacements)):.8f} m")
        print(f"    Max GSM displacement:        {np.max(np.abs(gsm_displacements)):.8f} m")
        print(f"    Relative error (max):        {abs(np.max(np.abs(gsm_displacements)) - np.max(np.abs(fea_displacements))) / np.max(np.abs(fea_displacements)) * 100:.2f}%")
        
        # Visualize
        undeformed = np.zeros_like(fea_displacements)
        title = f"{design_case['name']} - {load_case['name']}"
        filename = f"truss_comparison_case{design_idx + 1}.png"
        visualize_truss_comparison(nodes, edges, undeformed, fea_displacements, 
                                  gsm_displacements, title, filename, loads=loads, bc=bc)
        
        # Print node-level details
        print(f"\n  Node-level Displacements (top 5 nodes with max displacement):")
        print(f"  {'Node':>4} | {'FEA_dx':>10} | {'GSM_dx':>10} | {'FEA_dy':>10} | {'GSM_dy':>10} | {'Error':>10}")
        print(f"  {'-'*76}")
        
        max_disp_indices = np.argsort(np.max(np.abs(fea_displacements), axis=1))[-5:][::-1]
        for node_idx in max_disp_indices:
            error = np.linalg.norm(gsm_displacements[node_idx] - fea_displacements[node_idx])
            print(f"  {node_idx:4d} | {fea_displacements[node_idx, 0]:10.8f} | {gsm_displacements[node_idx, 0]:10.8f} | "
                  f"{fea_displacements[node_idx, 1]:10.8f} | {gsm_displacements[node_idx, 1]:10.8f} | {error:10.8f}")
    
    # Summary comparison table
    print(f"\n\n{'='*80}")
    print("SUMMARY - Performance Metrics Across All Test Cases")
    print(f"{'='*80}")
    print(f"{'Case':>8} | {'MAE':>12} | {'MSE':>14} | {'RMSE':>12} | {'FEA Max':>12} | {'GSM Max':>12} | {'Rel Err %':>10}")
    print(f"{'-'*110}")
    
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    
    for metrics in all_metrics:
        print(f"{metrics['case']:>8} | {metrics['MAE']:>12.8f} | {metrics['MSE']:>14.10f} | {metrics['RMSE']:>12.8f} | "
              f"{metrics['max_fea']:>12.8f} | {metrics['max_gsm']:>12.8f} | {abs(metrics['max_gsm'] - metrics['max_fea'])/metrics['max_fea']*100:>10.2f}")
        total_mae += metrics['MAE']
        total_mse += metrics['MSE']
        total_rmse += metrics['RMSE']
    
    print(f"{'-'*110}")
    print(f"{'Average':>8} | {total_mae/len(all_metrics):>12.8f} | {total_mse/len(all_metrics):>14.10f} | {total_rmse/len(all_metrics):>12.8f}")
    
    # Create combined metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cases = [m['case'] for m in all_metrics]
    mae_vals = [m['MAE'] for m in all_metrics]
    mse_vals = [m['MSE'] for m in all_metrics]
    rmse_vals = [m['RMSE'] for m in all_metrics]
    rel_errs = [abs(m['max_gsm'] - m['max_fea'])/m['max_fea']*100 for m in all_metrics]
    
    # MAE plot
    axes[0, 0].bar(cases, mae_vals, color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('MAE (m)')
    axes[0, 0].set_title('Mean Absolute Error')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # MSE plot
    axes[0, 1].bar(cases, mse_vals, color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('MSE (m²)')
    axes[0, 1].set_title('Mean Squared Error')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # RMSE plot
    axes[1, 0].bar(cases, rmse_vals, color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_ylabel('RMSE (m)')
    axes[1, 0].set_title('Root Mean Squared Error')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Relative error plot
    axes[1, 1].bar(cases, rel_errs, color='mediumpurple', alpha=0.7)
    axes[1, 1].set_ylabel('Relative Error (%)')
    axes[1, 1].set_title('Relative Error in Max Displacement')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for ax in axes.flat:
        ax.set_xticklabels(cases, rotation=0)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved metrics comparison plot to metrics_comparison.png")
    plt.close()
    
    print(f"\n{'='*80}")
    print("Deployment and Evaluation Complete!")
    print(f"{'='*80}\n")
