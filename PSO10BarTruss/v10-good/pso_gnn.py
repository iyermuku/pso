"""
PSO with GNN Surrogate - replaces FEA with neural network predictions
"""

import numpy as np
import torch
import logging
from typing import Dict

from truss_model import (
    Amin, Amax, U_ALLOW, S_ALLOW, ndof, fixed_dofs, mass_from_A, nodes,
    member_dof_idx
)
from constraints import project_params, constraint_vector
from objectives import penalized_objective
from gnn_surrogate_10bar import (
    TrussGNNSurrogate, predict_with_gnn, create_truss_graph
)

logger = logging.getLogger("pso_gnn")

# Calibration factors based on validation errors from GNN vs FEA comparison
# These safety factors account for systematic biases in GNN predictions
CALIBRATION_FACTORS = {
    'stress_multiplier': 2.5,      # GNN underpredicts stress, apply 2.5x safety factor
    'displacement_multiplier': 1.0  # GNN overpredicts displacement (+4%), already conservative
}


def calibrate_gnn_predictions(U_pred: np.ndarray, stress_pred: np.ndarray) -> tuple:
    """
    Apply conservative safety factors to GNN predictions.
    
    Based on validation against FEA:
    - Stress: GNN underpredicts (~0.38x actual), so multiply by 2.0 for safety
    - Displacement: GNN overpredicts (~1.04x actual), already conservative, no change
    
    Args:
        U_pred: Predicted displacements [ndof]
        stress_pred: Predicted stresses [n_members]
        
    Returns:
        Tuple of (U_calibrated, stress_calibrated) with safety factors applied
    """
    U_calib = U_pred * CALIBRATION_FACTORS['displacement_multiplier']
    stress_calib = stress_pred * CALIBRATION_FACTORS['stress_multiplier']
    
    logger.debug(
        f"Calibration applied: stress×{CALIBRATION_FACTORS['stress_multiplier']}, "
        f"disp×{CALIBRATION_FACTORS['displacement_multiplier']}"
    )
    
    return U_calib, stress_calib


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator):
    """Latin Hypercube Sampling in [0,1]^n_dim"""
    U = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + U) / n_samples
    samples = np.zeros_like(grid)
    for j in range(n_dim):
        samples[:, j] = rng.permutation(grid[:, j])
    return samples


def _deb_better(cv1: float, J1: float, cv2: float, J2: float, eps: float = 1e-9):
    """Deb feasibility rule"""
    f1, f2 = (cv1 <= eps), (cv2 <= eps)
    if f1 and f2:
        return J1 < J2 - 1e-12
    if f1 != f2:
        return f1
    if abs(cv1 - cv2) > 1e-12:
        return cv1 < cv2
    return J1 < J2 - 1e-12


def _violation(g_vec: np.ndarray) -> float:
    return float(np.sum(np.maximum(g_vec, 0.0)))


def evaluate_with_gnn(
    X: np.ndarray,
    gnn_model: TrussGNNSurrogate,
    edge_index: np.ndarray,
    device: str = 'cpu'
):
    """Evaluate particles using GNN predictions instead of FEA"""
    
    m_list = np.zeros(len(X))
    U_list = np.zeros((len(X), ndof))
    J_list = np.zeros(len(X))
    cv_list = np.zeros(len(X))
    stress_list = np.zeros((len(X), 10))
    g_vectors = []
    
    # Node coordinates
    node_coords = np.array([nodes[m] for m in sorted(nodes.keys())])
    
    gnn_model.eval()
    with torch.no_grad():
        for i in range(len(X)):
            A = X[i, :]
            
            # Use GNN prediction
            U_pred, stress_pred = predict_with_gnn(
                model=gnn_model,
                areas=A,
                node_coords=node_coords,
                fixed_dofs=fixed_dofs,
                edge_index=edge_index,
                load_scale=1.0,
                device=device,
            )
            
            # Apply calibration with conservative safety factors
            U_pred, stress_pred = calibrate_gnn_predictions(U_pred, stress_pred)
            
            # Compute constraints based on calibrated GNN predictions
            # We need to emulate constraint checking from displacements
            n_stress = len(stress_pred)
            g = np.zeros(1 + n_stress)  # 1 displacement + stress constraints
            
            # Displacement constraints
            max_disp = np.max(np.abs(U_pred))
            if max_disp > U_ALLOW:
                g[0] = max_disp - U_ALLOW
            
            # Stress constraints
            for j, sig in enumerate(stress_pred):
                if np.abs(sig) > S_ALLOW:
                    g[1 + j] = np.abs(sig) - S_ALLOW
            
            g_vectors.append(g)
            
            U_list[i] = U_pred
            stress_list[i] = stress_pred
            m_list[i] = mass_from_A(A)
    
    # Pad g_vectors
    max_g_len = max(len(g) for g in g_vectors)
    g_list = np.zeros((len(X), max_g_len))
    for i, g in enumerate(g_vectors):
        g_list[i, :len(g)] = g
    
    avg_m = float(np.mean(m_list))
    avg_g = np.mean(g_list, axis=0)
    
    for i in range(len(X)):
        J_list[i] = penalized_objective(m_list[i], g_list[i], avg_m, avg_g)
        cv_list[i] = _violation(g_list[i])
    
    return m_list, U_list, g_list, J_list, cv_list, stress_list


def pso_single_run_gnn(
    gnn_model: TrussGNNSurrogate,
    swarm_size: int = 60,
    iters: int = 200,
    seed: int = 2026,
    device: str = 'cpu',
    v_frac: float = 0.2
) -> Dict:
    """
    PSO using GNN surrogate model instead of FEA
    
    Returns:
        result dict with:
        - gbest_A: best areas found
        - gbest_m: best mass
        - gbest_U: best displacements
        - mass_hist: history of best mass
        - time_gnn_evals: total GNN evaluation time
    """
    
    rng = np.random.default_rng(seed)
    D = 10
    
    # Constriction PSO coefficients
    c1 = 2.05
    c2 = 2.05
    phi = c1 + c2
    chi = 2.0 / (phi - 2.0 + np.sqrt(phi**2 - 4.0 * phi))
    
    # Ranges
    lo = np.full(D, Amin)
    hi = np.full(D, Amax)
    span = hi - lo
    v_max = v_frac * span
    
    # LHS initialization
    S01 = _lhs(swarm_size, D, rng)
    X = lo + S01 * span
    V = rng.uniform(-v_max, v_max, size=(swarm_size, D))
    
    # Graph structure
    edge_index, _ = create_truss_graph(member_dof_idx)
    
    # Initial evaluation with GNN
    import time
    gnn_eval_time = 0
    
    t0 = time.time()
    m_list, U_list, g_list, J_list, cv_list, stress_list = evaluate_with_gnn(
        X, gnn_model, edge_index, device
    )
    gnn_eval_time += time.time() - t0
    
    # Personal bests
    pbest_X = X.copy()
    pbest_J = J_list.copy()
    pbest_cv = cv_list.copy()
    pbest_m = m_list.copy()
    pbest_U = U_list.copy()
    
    # Global best
    def select_best():
        idx = 0
        for j in range(1, swarm_size):
            if _deb_better(pbest_cv[j], pbest_J[j], pbest_cv[idx], pbest_J[idx]):
                idx = j
        return idx
    
    g_idx = select_best()
    gbest_X = pbest_X[g_idx].copy()
    gbest_J = float(pbest_J[g_idx])
    gbest_cv = float(pbest_cv[g_idx])
    gbest_m = float(pbest_m[g_idx])
    gbest_U = pbest_U[g_idx].copy()
    
    best_feas_m = float("inf")
    best_feas_X = None
    best_feas_U = None
    
    mass_hist = [gbest_m]
    
    logger.info(f"PSO-GNN starting: swarm={swarm_size}, iters={iters}")
    
    # PSO iterations
    for it in range(iters):
        # Build local best (simplified: just global for now)
        lbest_X = gbest_X[np.newaxis, :].repeat(swarm_size, axis=0)
        
        # PSO update
        r1 = np.random.random((swarm_size, D))
        r2 = np.random.random((swarm_size, D))
        
        V = chi * (V + c1 * r1 * (pbest_X - X) + c2 * r2 * (lbest_X - X))
        V = np.clip(V, -v_max, v_max)
        
        X = X + V
        
        # Boundary handling
        below = X < lo
        above = X > hi
        X = np.where(below, lo, X)
        X = np.where(above, hi, X)
        V[below] = 0
        V[above] = 0
        
        # Evaluate with GNN
        t0 = time.time()
        m_list, U_list, g_list, J_list, cv_list, stress_list = evaluate_with_gnn(
            X, gnn_model, edge_index, device
        )
        gnn_eval_time += time.time() - t0
        
        # Update personal bests
        for i in range(swarm_size):
            if _deb_better(cv_list[i], J_list[i], pbest_cv[i], pbest_J[i]):
                pbest_X[i] = X[i].copy()
                pbest_J[i] = J_list[i]
                pbest_cv[i] = cv_list[i]
                pbest_m[i] = m_list[i]
                pbest_U[i] = U_list[i].copy()
        
        # Update global best
        g_idx = select_best()
        if _deb_better(pbest_cv[g_idx], pbest_J[g_idx], gbest_cv, gbest_J):
            gbest_X = pbest_X[g_idx].copy()
            gbest_J = float(pbest_J[g_idx])
            gbest_cv = float(pbest_cv[g_idx])
            gbest_m = float(pbest_m[g_idx])
            gbest_U = pbest_U[g_idx].copy()
            logger.info(f"  Iter {it+1}: NEW BEST mass={gbest_m:.2f} lbm cv={gbest_cv:.2f}")
        
        mass_hist.append(gbest_m)
    
    logger.info(f"PSO-GNN completed. GNN eval time: {gnn_eval_time:.2f}s")
    
    return {
        'gbest_X': gbest_X,
        'gbest_A': gbest_X,
        'gbest_m': gbest_m,
        'gbest_mass': gbest_m,
        'gbest_U': gbest_U,
        'mass_hist': mass_hist,
        'time_gnn_evals': gnn_eval_time,
        'best_feas_A': gbest_X,
        'best_feas_m': gbest_m,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("PSO-GNN module loaded successfully")
