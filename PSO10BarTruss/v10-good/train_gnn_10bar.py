"""
Train GNN surrogate model for 10-bar truss

Generates training data using FEA+LHS sampling, then trains GNN
"""

import numpy as np
import torch
from scipy.stats import qmc
import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple

from truss_model import (
    nodes, members, Amin, Amax, U_ALLOW, S_ALLOW, fixed_dofs, free_dofs,
    solve_displacements, member_stresses, mass_from_A, dof_index,
    member_lengths, member_cs, member_dof_idx
)
from constraints import constraint_vector
from gnn_surrogate_10bar import (
    TrussGNNSurrogate, create_graph_data, train_gnn_surrogate,
    create_truss_graph
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_gnn")

def generate_training_data(
    num_samples: int = 2000,
    num_load_cases: int = 1,
    seed: int = 42
) -> List:
    """
    Generate training data using Latin Hypercube Sampling + FEA
    
    Returns:
        data_list: List of torch_geometric Data objects
    """
    logger.info(f"Generating {num_samples} LHS samples for training...")
    
    # LHS sampling in [0,1]^10
    sampler = qmc.LatinHypercube(d=10, seed=seed)
    samples_01 = sampler.random(n=num_samples)
    
    # Scale to [Amin, Amax]
    areas_samples = Amin + samples_01 * (Amax - Amin)
    
    # FEA evaluation
    logger.info("Evaluating FEA for all samples...")
    valid_count = 0
    data_list = []
    
    # Get graph structure (once)
    edge_index, _ = create_truss_graph(member_dof_idx)
    
    for i, A in enumerate(areas_samples):
        try:
            # Compute FEA
            U = solve_displacements(A)
            stresses = member_stresses(U)
            
            # Check constraints
            g = constraint_vector(U)
            cv = np.sum(np.maximum(g, 0.0))
            
            # Focus on feasible and near-feasible regions used during PSO at nominal load.
            if np.isfinite(cv) and cv < 50.0:
                load_scales = np.full(num_load_cases, 1.0)
                for load_scale in load_scales:
                    U_scaled = U * load_scale
                    stresses_scaled = stresses * load_scale

                    # Get node coordinates
                    node_coords = np.array([nodes[m] for m in sorted(nodes.keys())])

                    # Create Data object
                    data = create_graph_data(
                        areas=A,
                        displacements=U_scaled,
                        stresses=stresses_scaled,
                        load_scale=float(load_scale),
                        node_coords=node_coords,
                        fixed_dofs=fixed_dofs,
                        edge_index=edge_index,
                        ndof=12
                    )
                    data_list.append(data)
                    valid_count += 1
        except:
            pass
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{num_samples} samples, {valid_count} valid")
    
    logger.info(f"Generated {len(data_list)} data points from {valid_count} valid designs")
    return data_list


def main():
    # ========================================================================
    # Generate Training Data
    # ========================================================================
    logger.info("="*80)
    logger.info("GNN SURROGATE MODEL TRAINING FOR 10-BAR TRUSS")
    logger.info("="*80)
    
    start_gen = time.time()
    data_list = generate_training_data(num_samples=2000, num_load_cases=1, seed=42)
    gen_time = time.time() - start_gen
    logger.info(f"Data generation time: {gen_time:.2f} seconds\n")
    
    if len(data_list) < 10:
        logger.error("Not enough training data generated!")
        return
    
    # Split into train/val
    split_idx = int(0.8 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    
    logger.info(f"Train set: {len(train_data)}, Val set: {len(val_data)}\n")
    
    # ========================================================================
    # Train GNN Model
    # ========================================================================
    logger.info("Training GNN model...")
    start_train = time.time()
    
    model, history = train_gnn_surrogate(
        train_data_list=train_data,
        val_data_list=val_data,
        epochs=150,
        batch_size=16,
        learning_rate=0.001,
        device='cpu'
    )
    
    train_time = time.time() - start_train
    logger.info(f"Training time: {train_time:.2f} seconds\n")
    
    # Save model + metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'data_generation_time_s': float(gen_time),
            'training_time_s': float(train_time),
            'gnn_creation_time_s': float(gen_time + train_time),
            'num_samples_total': int(len(data_list)),
            'num_train': int(len(train_data)),
            'num_val': int(len(val_data)),
            'final_train_loss': float(history['train_losses'][-1]),
            'final_val_loss': float(history['val_losses'][-1]),
        }
    }
    torch.save(checkpoint, 'gnn_10bar_model.pth')
    logger.info("Saved model to 'gnn_10bar_model.pth'\n")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total data generated: {len(data_list)}")
    logger.info(f"Data generation time: {gen_time:.2f}s")
    logger.info(f"GNN training time: {train_time:.2f}s")
    logger.info(f"Final train loss: {history['train_losses'][-1]:.6f}")
    logger.info(f"Final val loss: {history['val_losses'][-1]:.6f}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
