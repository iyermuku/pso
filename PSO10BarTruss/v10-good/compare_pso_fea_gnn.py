"""
Compare PSO with FEA vs PSO with GNN Surrogate

Runs both approaches and compares:
- Execution time
- Number of FEA/GNN evaluations
- Final results (mass, displacements, stresses)
"""

import numpy as np
import torch
import time
import logging
import os
from datetime import datetime

from pso import pso_single_run_robust
from pso_gnn import pso_single_run_gnn
from gnn_surrogate_10bar import TrussGNNSurrogate
from truss_model import solve_displacements, member_stresses, mass_from_A

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("comparison")

def load_gnn_model(model_path: str = 'gnn_10bar_model.pth', device: str = 'cpu'):
    """Load pre-trained GNN model"""
    model = TrussGNNSurrogate(
        input_features=15,
        hidden_dim=64,
        num_layers=3,
        num_members=10
    ).to(device)
    
    try:
        ckpt = torch.load(model_path, map_location=device)
        metadata = {}
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            metadata = ckpt.get('metadata', {}) if isinstance(ckpt.get('metadata', {}), dict) else {}
        else:
            model.load_state_dict(ckpt)

        if 'gnn_creation_time_s' not in metadata and os.path.exists(model_path):
            metadata['created_at'] = datetime.fromtimestamp(os.path.getctime(model_path)).isoformat()
            metadata['gnn_creation_time_s'] = float('nan')

        logger.info(f"Loaded GNN model from {model_path}")
        return model, metadata
    except FileNotFoundError:
        logger.warning(f"GNN model not found at {model_path}. Will skip GNN comparison.")
        return None, {}


def _diff_percent(base: float, other: float) -> float:
    if abs(base) < 1e-12:
        return float('nan')
    return (other - base) / base * 100.0


def _fmt_num(v: float, dec: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{dec}f}"


def _print_row(metric: str, fea_v: float, gnn_v: float, dec: int = 4, unit: str = ""):
    diff = _diff_percent(fea_v, gnn_v)
    fea_s = _fmt_num(fea_v, dec)
    gnn_s = _fmt_num(gnn_v, dec)
    diff_s = "N/A" if np.isnan(diff) else f"{diff:.2f}%"
    unit_s = f" {unit}" if unit else ""
    logger.info(f"  {metric:<24} {fea_s:<14}{gnn_s:<14}{diff_s:<10}{unit_s}")


def verify_solution(A: np.ndarray):
    """Verify solution quality by running FEA"""
    try:
        U = solve_displacements(A)
        stresses = member_stresses(U)
        m = mass_from_A(A)
        max_disp = np.max(np.abs(U))
        max_stress = np.max(np.abs(stresses))
        return {
            'mass': m,
            'max_disp': max_disp,
            'max_stress': max_stress,
            'success': True
        }
    except:
        return {
            'mass': float('inf'),
            'max_disp': float('inf'),
            'max_stress': float('inf'),
            'success': False
        }


def main():
    logger.info("="*80)
    logger.info("PSO COMPARISON: FEA vs GNN SURROGATE")
    logger.info("="*80)
    
    # Parameters
    swarm_size = 60
    iters = 100  # Shorter for quick comparison
    seed = 2026
    
    logger.info(f"\nParameters:")
    logger.info(f"  Swarm size: {swarm_size}")
    logger.info(f"  Iterations: {iters}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Total evaluations expected: {swarm_size * iters}")
    
    # ========================================================================
    # Run 1: PSO with FEA (baseline)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RUN 1: PSO WITH FEA (BASELINE)")
    logger.info("="*80)
    
    start_fea = time.time()
    result_fea = pso_single_run_robust(
        swarm_size=swarm_size,
        iters=iters,
        seed=seed,
        stall_window=20,
        max_restarts=0
    )
    time_fea = time.time() - start_fea
    
    logger.info(f"PSO-FEA completed in {time_fea:.2f} seconds")
    best_mass_fea = float(result_fea.get('gbest_mass', 0))
    logger.info(f"  Best mass: {best_mass_fea:.2f} lbm")
    
    # Verify FEA solution
    verify_fea = verify_solution(result_fea['gbest_A'])
    logger.info(f"  Verification (FEA):")
    logger.info(f"    Max displacement: {verify_fea['max_disp']:.6f} in")
    logger.info(f"    Max stress: {verify_fea['max_stress']:.2f} ksi")
    
    # ========================================================================
    # Run 2: PSO with GNN (if model available)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RUN 2: PSO WITH GNN SURROGATE")
    logger.info("="*80)
    
    gnn_model, gnn_meta = load_gnn_model('gnn_10bar_model.pth', device='cpu')
    
    if gnn_model is None:
        logger.warning("GNN model not available. Skipping GNN comparison.")
        logger.info("To train the model, run: python train_gnn_10bar.py")
    else:
        start_gnn = time.time()
        result_gnn = pso_single_run_gnn(
            gnn_model,
            swarm_size=swarm_size,
            iters=iters,
            seed=seed,
            device='cpu'
        )
        time_gnn = time.time() - start_gnn
        
        logger.info(f"PSO-GNN completed in {time_gnn:.2f} seconds")
        best_mass_gnn = float(result_gnn.get('gbest_mass', 0))
        logger.info(f"  Best mass: {best_mass_gnn:.2f} lbm")
        time_gnn_evals = float(result_gnn.get('time_gnn_evals', 0))
        logger.info(f"  GNN eval time: {time_gnn_evals:.2f} seconds")
        
        # Verify GNN solution with FEA
        verify_gnn = verify_solution(result_gnn['gbest_A'])
        logger.info(f"  Verification (FEA on GNN result):")
        logger.info(f"    Max displacement: {verify_gnn['max_disp']:.6f} in")
        logger.info(f"    Max stress: {verify_gnn['max_stress']:.2f} ksi")
        
        # ========================================================================
        # Comparison
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nTiming:")
        logger.info(f"  PSO-FEA:  {time_fea:8.2f} seconds")
        logger.info(f"  PSO-GNN:  {time_gnn:8.2f} seconds")
        logger.info(f"  Speedup:  {time_fea/time_gnn:8.2f}x faster")
        
        time_per_eval_fea = time_fea / (swarm_size * iters)
        if hasattr(result_gnn, '__getitem__') and result_gnn.get('time_gnn_evals'):
            time_per_eval_gnn = result_gnn['time_gnn_evals'] / (swarm_size * iters)
        else:
            time_per_eval_gnn = time_gnn / (swarm_size * iters)
        
        logger.info(f"\nPer-evaluation time:")
        logger.info(f"  FEA:  {time_per_eval_fea*1000:8.4f} ms")
        logger.info(f"  GNN:  {time_per_eval_gnn*1000:8.4f} ms")
        logger.info(f"  Speedup: {time_per_eval_fea/time_per_eval_gnn:8.2f}x")
        
        logger.info(f"\nDETAILED FEA vs GNN COMPARISON")
        logger.info(f"  {'Metric':<24} {'FEA':<14}{'GNN':<14}{'Diff %':<10}")
        logger.info(f"  {'-'*70}")

        areas_fea = np.asarray(result_fea['gbest_A'], dtype=float)
        areas_gnn = np.asarray(result_gnn['gbest_A'], dtype=float)
        for i in range(10):
            _print_row(f"Area {i+1}", areas_fea[i], areas_gnn[i], dec=4, unit='in^2')

        mass_fea = float(verify_fea['mass'])
        mass_gnn = float(verify_gnn['mass'])
        _print_row("Mass", mass_fea, mass_gnn, dec=2, unit='lbm')

        disp_fea = float(verify_fea['max_disp'])
        disp_gnn = float(verify_gnn['max_disp'])
        _print_row("Maximum displacement", disp_fea, disp_gnn, dec=6, unit='in')

        stress_fea = float(verify_fea['max_stress'])
        stress_gnn = float(verify_gnn['max_stress'])
        _print_row("Maximum stress", stress_fea, stress_gnn, dec=2, unit='ksi')

        _print_row("Full time", float(time_fea), float(time_gnn), dec=3, unit='s')

        fea_eval_time = float(time_fea) / float(swarm_size * iters)
        gnn_eval_time = float(result_gnn.get('time_gnn_evals', time_gnn)) / float(swarm_size * iters)
        _print_row("FEA eval time", fea_eval_time, gnn_eval_time, dec=6, unit='s/eval')

        gnn_creation_time = gnn_meta.get('gnn_creation_time_s', float('nan'))
        gnn_creation_str = _fmt_num(float(gnn_creation_time), 3)
        logger.info(f"  {'GNN creation time':<24} {'N/A':<14}{gnn_creation_str:<14}{'N/A':<10} s")
        
        logger.info("\n" + "="*80)
        logger.info("CONCLUSIONS")
        logger.info("="*80)
        logger.info(f"✓ PSO-GNN is {time_fea/time_gnn:.1f}x faster than PSO-FEA")
        logger.info(f"✓ Per-eval: {time_per_eval_fea/time_per_eval_gnn:.1f}x speedup")
        mass_diff = abs(_diff_percent(mass_fea, mass_gnn)) if mass_fea > 0 else float('nan')
        stress_diff = abs(_diff_percent(stress_fea, stress_gnn)) if stress_fea > 0 else float('nan')

        logger.info(f"✓ GNN solution quality: mass within {mass_diff:.1f}%")
        
        if mass_diff < 5 and stress_diff < 10:
            logger.info("✓ GNN provides good approximation for design space exploration")
        else:
            logger.warning("⚠ GNN predictions have higher error - may need more training data")


if __name__ == "__main__":
    main()
