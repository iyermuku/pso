"""
Timing analysis for 10-bar truss PSO
Compares FEA evaluation time vs PSO optimization time
"""
import numpy as np
import time
import logging
from datetime import datetime

from truss_model import (
    solve_displacements, solve_stresses, mass_from_A, member_stresses,
    Amin, Amax
)
from pso import pso_single_run_robust
from constraints import constraint_vector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("timing_analysis")

def time_single_fea_evaluation(num_evals: int = 1000):
    """Time how long it takes to evaluate FEA for random designs"""
    logger.info(f"Timing FEA evaluations ({num_evals} random designs)...")
    
    rng = np.random.default_rng(42)
    areas_list = []
    for _ in range(num_evals):
        A = rng.uniform(Amin, Amax, 10)
        areas_list.append(A)
    
    start = time.time()
    for A in areas_list:
        U = solve_displacements(A)
        stresses = member_stresses(U)
        mass = mass_from_A(A)
        g = constraint_vector(U)
    end = time.time()
    
    total_time = end - start
    time_per_eval = total_time / num_evals
    
    logger.info(f"  Total FEA time: {total_time:.4f} seconds")
    logger.info(f"  Per-evaluation: {time_per_eval*1000:.4f} ms")
    logger.info(f"  Evaluations/sec: {num_evals/total_time:.2f}")
    
    return total_time, time_per_eval

def time_pso_optimization(swarm_size: int = 60, iters: int = 200, seed: int = 2026):
    """Time the entire PSO optimization"""
    logger.info(f"\nTiming PSO optimization...")
    logger.info(f"  Swarm size: {swarm_size}")
    logger.info(f"  Iterations: {iters}")
    
    start = time.time()
    result = pso_single_run_robust(
        swarm_size=swarm_size,
        iters=iters,
        seed=seed,
        stall_window=20,
        max_restarts=0
    )
    end = time.time()
    
    pso_time = end - start
    total_evals = swarm_size * iters  # Approximate (restarts add more)
    time_per_eval = pso_time / total_evals
    
    logger.info(f"  Total PSO time: {pso_time:.4f} seconds")
    logger.info(f"  Approximate evals: {total_evals}")
    logger.info(f"  Avg time per eval: {time_per_eval*1000:.4f} ms")
    best_mass = float(result.get('gbest_mass', 0))
    logger.info(f"  Best mass found: {best_mass:.4f} lbm")
    
    return pso_time, time_per_eval, result

def breakdown_pso_time(swarm_size: int = 60, iters: int = 10, seed: int = 2026):
    """Detailed timing breakdown of PSO components"""
    logger.info(f"\nDetailed PSO timing breakdown ({swarm_size} swarm, {iters} iters)...")
    
    from truss_model import ndof
    from constraints import project_params, constraint_vector
    from objectives import penalized_objective
    
    rng = np.random.default_rng(seed)
    D = 10
    
    # Initialize
    lo = np.full(D, Amin)
    hi = np.full(D, Amax)
    span = hi - lo
    
    X = lo + rng.random((swarm_size, D)) * span
    V = rng.uniform(-0.2*span, 0.2*span, size=(swarm_size, D))
    
    # Time FEA evaluations
    fea_start = time.time()
    for i in range(swarm_size):
        U = solve_displacements(X[i])
        g = constraint_vector(U)
        m = mass_from_A(X[i])
    fea_time = time.time() - fea_start
    per_fea = fea_time / swarm_size
    
    # Time constraint computations
    constraint_start = time.time()
    for i in range(swarm_size):
        U = solve_displacements(X[i])
        g = constraint_vector(U)
    constraint_time = time.time() - constraint_start
    
    # Time PSO updates
    update_start = time.time()
    for it in range(iters):
        r1 = rng.random((swarm_size, D))
        r2 = rng.random((swarm_size, D))
        V = 0.7298 * (V + 2.05 * r1 * 0.1 + 2.05 * r2 * 0.1)
        X = X + V
    update_time = time.time() - update_start
    
    logger.info(f"  FEA solve time (per eval): {per_fea*1000:.4f} ms")
    logger.info(f"  Total FEA + constraints: {(fea_time + constraint_time)*1000:.2f} ms ({swarm_size} particles)")
    logger.info(f"  PSO updates ({iters} iters): {update_time*1000:.2f} ms")
    
    total_cycle = fea_time + constraint_time + update_time
    fea_fraction = (fea_time + constraint_time) / total_cycle * 100
    logger.info(f"  FEA/constraint fraction of iteration: {fea_fraction:.1f}%")

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("10-BAR TRUSS PSO - TIMING ANALYSIS")
    logger.info("="*80)
    
    # Part 1: Time FEA evaluations
    fea_total, fea_per = time_single_fea_evaluation(num_evals=1000)
    
    # Part 2: Detailed breakdown
    breakdown_pso_time(swarm_size=60, iters=10, seed=2026)
    
    # Part 3: Full PSO timing
    pso_total, pso_per, pso_result = time_pso_optimization(swarm_size=60, iters=200, seed=2026)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"FEA evaluation time:     {fea_per*1000:.4f} ms per design")
    logger.info(f"PSO avg time per eval:   {pso_per*1000:.4f} ms per design")
    logger.info(f"Speedup potential (GNN): {fea_per/pso_per:.1f}x")
    logger.info(f"Full PSO run:            {pso_total:.2f} seconds")
    gbest_mass = float(pso_result.get('gbest_mass', 0))
    logger.info(f"Best mass found:         {gbest_mass:.4f} lbm")
    logger.info("="*80)
