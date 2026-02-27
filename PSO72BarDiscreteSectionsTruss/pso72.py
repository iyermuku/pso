#!/usr/bin/env python3
"""
PSO for 72-bar truss (16 area groups) with two modes:
- Robust constriction mode: optimize only 16 areas using Clerc-Kennedy constriction coefficient.
- Single-run meta-PSO mode: optimize 19 variables (16 areas + w, c1, c2) where PSO parameters are part of the search.
This module exposes helper functions used by pso72_main.py.
"""
import math
import logging
import numpy as np
import truss72 as t72
# Tolerances matching constraint-check script
TOL_CONSTRAINT = 0.00 # 1% slack on allowables (effective allow = (1+tol)*allow)
# Bounds for decision variables
A_MIN = float(t72.A_MIN)
A_MAX = float(t72.A_MAX)
W_BOUNDS = (0.40, 0.99)
C_BOUNDS = (1.0, 3.0)

# === Discrete area handling ===
AVAILABLE_A = np.asarray(t72.available_A, dtype=float)

# def _snap_areas_to_available(A16: np.ndarray) -> np.ndarray:
    # """Snap a 16-vector of areas to the nearest values in t72.available_A.
    # Returns a new numpy array (float64) of length 16.
    # """
    # A16 = np.asarray(A16, dtype=float).reshape(-1)
    # #Compute distance matrix [16 x nAvail] and take argmin along axis=1
    # diffs = np.abs(A16[:, None] - AVAILABLE_A[None, :])
    # idx = np.argmin(diffs, axis=1)
    # return AVAILABLE_A[idx]


def _snap_areas_to_available(A16: np.ndarray) -> np.ndarray:
    """
    Snap a length-16 vector of (continuous) areas to the nearest values from
    the discrete set AVAILABLE_A (one dimension at a time).

    Rules:
      - For each dimension j, pick argmin_k |A16[j] - AVAILABLE_A[k]|.
      - Ties are broken toward the lower available value (stable / conservative).
      - Returns a 1D float64 array of length 16.
    """
    A = np.asarray(A16, dtype=float).reshape(-1)
    if A.shape[0] != 16:
        raise ValueError("A16 must be a 16-vector")

    avail = np.asarray(AVAILABLE_A, dtype=float)
    if avail.ndim != 1 or avail.size == 0:
        raise ValueError("AVAILABLE_A must be a non-empty 1D array")

    # Work with a sorted copy for fast, stable nearest-neighbor via searchsorted
    order = np.argsort(avail)
    avail_sorted = avail[order]

    # For each A[j], find the insertion point in the sorted list
    idx_hi = np.searchsorted(avail_sorted, A, side="left")
    idx_lo = np.clip(idx_hi - 1, 0, avail_sorted.size - 1)
    idx_hi = np.clip(idx_hi,     0, avail_sorted.size - 1)

    # Compare distances to lower and higher neighbors; on ties, choose lower
    dist_lo = np.abs(A - avail_sorted[idx_lo])
    dist_hi = np.abs(A - avail_sorted[idx_hi])
    use_lo = dist_lo <= dist_hi
    nearest_idx = np.where(use_lo, idx_lo, idx_hi)

    snapped = avail_sorted[nearest_idx]
    # (Optional) map back through 'order' not required since we return values, not indices
    return snapped.astype(float)
    
    

# Utility: evaluate truss and return objective and metrics
def evaluate_truss(A16: np.ndarray):
    """Return dict with mass, disp metrics, stress metrics, and violations.
    Displacement considers nodes 1..4 X/Y components across load cases.
    Stress considers max member stress across load cases.
    Violations compare against effective allowables with 1% slack.
    """
    A16 = np.clip(np.asarray(A16, dtype=float), A_MIN, A_MAX)
    res = t72.evaluate(A16)
    mass = float(res['mass'])
    # Displacement metrics
    max_disp = 0.0
    for U in res['U']:
        for nid in [1, 2, 3, 4]:
            ux = abs(U[3*(nid-1)+0])
            uy = abs(U[3*(nid-1)+1])
            max_disp = max(max_disp, ux, uy)
    disp_allow_nom = float(t72.U_ALLOW)
    disp_allow_eff = (1.0 + TOL_CONSTRAINT) * disp_allow_nom
    disp_violation = max(0.0, max_disp - disp_allow_eff)
    # Stress metrics
    max_stress = 0.0
    A_members = t72.areas_from_groups(A16)
    for U in res['U']:
        sig = t72.member_stresses(U, A_members)
        max_stress = max(max_stress, float(np.max(np.abs(sig))))
    stress_allow_nom = float(t72.S_ALLOW)
    stress_allow_eff = (1.0 + TOL_CONSTRAINT) * stress_allow_nom
    stress_violation = max(0.0, max_stress - stress_allow_eff)
    return {
        'mass': mass,
        'max_disp': max_disp,
        'disp_allow_nom': disp_allow_nom,
        'disp_allow_eff': disp_allow_eff,
        'disp_violation': disp_violation,
        'max_stress': max_stress,
        'stress_allow_nom': stress_allow_nom,
        'stress_allow_eff': stress_allow_eff,
        'stress_violation': stress_violation,
        'A16': A16.copy(),
    }

def objective_with_penalty(A16: np.ndarray, alpha_mass=1.0, alpha_disp=1e5, alpha_stress=1e5):
    """Weighted penalty objective; minimize mass plus penalties when constraints exceed.
    Penalties are scaled by allowables to be unit-consistent.
    """
    m = evaluate_truss(A16)
    # Normalized violations
    disp_norm = m['disp_violation'] / (m['disp_allow_eff'] + 1e-12)
    stress_norm = m['stress_violation'] / (m['stress_allow_eff'] + 1e-12)
    J = alpha_mass*m['mass'] + alpha_disp*disp_norm + alpha_stress*stress_norm
    return J, m

# Initialization helpers
def init_positions(n_particles: int, n_vars: int, bounds):
    """bounds: list of (lo,hi) tuples length n_vars"""
    X = np.zeros((n_particles, n_vars))
    for j, (lo, hi) in enumerate(bounds):
        X[:, j] = np.random.uniform(lo, hi, size=n_particles)
    return X

def init_velocities(n_particles: int, n_vars: int, bounds, frac=0.2):
    V = np.zeros((n_particles, n_vars))
    for j, (lo, hi) in enumerate(bounds):
        span = hi - lo
        V[:, j] = np.random.uniform(-frac*span, frac*span, size=n_particles)
    return V

# Robust constriction PSO (Clerc-Kennedy)
def robust_constriction_pso(seed: int, iters: int, swarm: int,
                            c1: float = 2.05, c2: float = 2.05,
                            alpha_mass=1.0, alpha_disp=1e5, alpha_stress=1e5,
                            logger: logging.Logger = None):
    np.random.seed(seed)
    if logger:
        logger.info(f"[robust] seed={seed}, iters={iters}, swarm={swarm}, c1={c1}, c2={c2}")
    # Constriction coefficient
    phi = c1 + c2
    if phi <= 4.0:
        phi = 4.0001
    chi = 2.0 / (abs(2.0 - phi - math.sqrt(phi*phi - 4.0*phi)))
    if logger:
        logger.debug(f"[robust] constriction chi={chi:.6f} (phi={phi:.6f})")

    n_vars = 16
    bounds = [(A_MIN, A_MAX)] * n_vars
    X = init_positions(swarm, n_vars, bounds)
    V = init_velocities(swarm, n_vars, bounds)

    # Personal and global bests
    J = np.zeros(swarm)
    pbest_X = X.copy()
    pbest_J = np.full(swarm, np.inf)
    gbest_X = None
    gbest_J = np.inf
    gbest_metrics = None

    # Bookkeeping caches for evaluations (keyed by snapped A16)
    feas_cache = {}   # key -> (J, full_metrics)
    infeas_cache = {} # key -> minimal_metrics (includes J)

    mass_hist = []  # gbest mass per iteration

    for i in range(iters):
        for p in range(swarm):
            # === Snap to nearest available area set before evaluating ===
            A_snapped = _snap_areas_to_available(X[p])
            key = tuple(np.round(A_snapped, 6))

            if key in feas_cache:
                J[p], metrics = feas_cache[key]
            elif key in infeas_cache:
                metrics = infeas_cache[key]
                J[p] = metrics['J']
            else:
                Jp, m = objective_with_penalty(A_snapped, alpha_mass, alpha_disp, alpha_stress)
                # Determine feasibility
                feasible = (m['disp_violation'] <= 0.0) and (m['stress_violation'] <= 0.0)
                if feasible:
                    feas_cache[key] = (Jp, m)
                    metrics = m
                else:
                    metrics_min = {
                        'A16': A_snapped.copy(),
                        'mass': m['mass'],
                        'max_disp': m['max_disp'],
                        'max_stress': m['max_stress'],
                        'disp_violation': m['disp_violation'],
                        'stress_violation': m['stress_violation'],
                        'J': Jp,
                    }
                    infeas_cache[key] = metrics_min
                    metrics = m  # use full metrics for internal comparisons/logging
                J[p] = Jp

            if J[p] < pbest_J[p]:
                pbest_J[p] = J[p]
                pbest_X[p] = X[p].copy()
                pbest_X[p, :16] = A_snapped
            if J[p] < gbest_J:
                gbest_J = J[p]
                gbest_X = X[p].copy()
                gbest_X[:16] = A_snapped
                gbest_metrics = metrics

        # Velocity & position update with constriction
        r1 = np.random.rand(swarm, n_vars)
        r2 = np.random.rand(swarm, n_vars)
        cognitive = c1 * r1 * (pbest_X - X)
        social = c2 * r2 * (gbest_X - X)
        V = chi * (V + cognitive + social)
        X = X + V

        # Enforce bounds (reflect)
        for j, (lo, hi) in enumerate(bounds):
            out_lo = X[:, j] < lo
            out_hi = X[:, j] > hi
            if np.any(out_lo):
                X[out_lo, j] = lo + (lo - X[out_lo, j])  # reflect
                V[out_lo, j] *= -0.5
            if np.any(out_hi):
                X[out_hi, j] = hi - (X[out_hi, j] - hi)  # reflect
                V[out_hi, j] *= -0.5

        # record current gbest mass
        if gbest_metrics is not None:
            mass_hist.append(float(gbest_metrics['mass']))
        if logger and (i % max(1, iters//10) == 0 or i == iters-1):
            logger.info(f"[robust] iter={i+1}/{iters} J_best={gbest_J:.6f} mass={gbest_metrics['mass']:.4f} "
                        f"disp={gbest_metrics['max_disp']:.6f} stress={gbest_metrics['max_stress']:.6f}")

    return {
        'gbest_J': gbest_J,
        'gbest_X': gbest_X,
        'metrics': gbest_metrics,
        'chi': chi,
        'c1': c1,
        'c2': c2,
        'mass_hist': mass_hist,
        'feasible_cache': feas_cache,
        'infeasible_cache': infeas_cache,
    }

# Single-run meta-optimization: 19 vars (16 areas + w, c1, c2)
def single_run_pso(seed: int, iters: int, swarm: int,
                   alpha_mass=1.0, alpha_disp=1e5, alpha_stress=1e5,
                   logger: logging.Logger = None):
    np.random.seed(seed)
    if logger:
        logger.info(f"[single] seed={seed}, iters={iters}, swarm={swarm}, search w,c1,c2")
    # Bounds: first 16 are areas, then w, c1, c2
    bounds = [(A_MIN, A_MAX)] * 16 + [W_BOUNDS, C_BOUNDS, C_BOUNDS]
    n_vars = len(bounds)
    X = init_positions(swarm, n_vars, bounds)
    V = init_velocities(swarm, n_vars, bounds)

    pbest_X = X.copy()
    pbest_J = np.full(swarm, np.inf)
    gbest_X = None
    gbest_J = np.inf
    gbest_metrics = None

    # Track best particle's parameter and mass history
    w_hist = []
    c1_hist = []
    c2_hist = []
    mass_hist = []

    # Bookkeeping caches for evaluations (keyed by snapped A16)
    feas_cache = {}
    infeas_cache = {}

    for i in range(iters):
        # Extract parameters per particle
        w = np.clip(X[:, 16], W_BOUNDS[0], W_BOUNDS[1])
        c1p = np.clip(X[:, 17], C_BOUNDS[0], C_BOUNDS[1])
        c2p = np.clip(X[:, 18], C_BOUNDS[0], C_BOUNDS[1])

        # Evaluate objective for area components only
        for p in range(swarm):
            areas_cont = X[p, :16]
            areas = _snap_areas_to_available(areas_cont)  # snap to discrete set
            key = tuple(np.round(areas, 6))
            if key in feas_cache:
                Jp, metrics = feas_cache[key]
            elif key in infeas_cache:
                metrics = infeas_cache[key]
                Jp = metrics['J']
            else:
                Jp, m = objective_with_penalty(areas, alpha_mass, alpha_disp, alpha_stress)
                feasible = (m['disp_violation'] <= 0.0) and (m['stress_violation'] <= 0.0)
                if feasible:
                    feas_cache[key] = (Jp, m)
                    metrics = m
                else:
                    metrics_min = {
                        'A16': areas.copy(),
                        'mass': m['mass'],
                        'max_disp': m['max_disp'],
                        'max_stress': m['max_stress'],
                        'disp_violation': m['disp_violation'],
                        'stress_violation': m['stress_violation'],
                        'J': Jp,
                    }
                    infeas_cache[key] = metrics_min
                    metrics = m
            if Jp < pbest_J[p]:
                pbest_J[p] = Jp
                pbest_X[p] = X[p].copy()
                pbest_X[p, :16] = areas
            if Jp < gbest_J:
                gbest_J = Jp
                gbest_X = X[p].copy()
                gbest_X[:16] = areas
                gbest_metrics = metrics

        # Record best params and mass for history
        if gbest_X is not None:
            w_hist.append(float(np.clip(gbest_X[16], *W_BOUNDS)))
            c1_hist.append(float(np.clip(gbest_X[17], *C_BOUNDS)))
            c2_hist.append(float(np.clip(gbest_X[18], *C_BOUNDS)))
            mass_hist.append(float(gbest_metrics['mass']))
        else:
            w_hist.append(float(w.mean()))
            c1_hist.append(float(c1p.mean()))
            c2_hist.append(float(c2p.mean()))
            # do not append mass if no gbest yet

        # Velocity & position update using each particle's own parameters
        r1 = np.random.rand(swarm, 16)
        r2 = np.random.rand(swarm, 16)
        cognitive = (c1p[:, None]) * r1 * (pbest_X[:, :16] - X[:, :16])
        social = (c2p[:, None]) * r2 * (gbest_X[:16] - X[:, :16]) if gbest_X is not None else 0.0
        V[:, :16] = (w[:, None]) * V[:, :16] + cognitive + social
        X[:, :16] = X[:, :16] + V[:, :16]

        # Update parameter dims with random walk toward pbest/gbest
        r1p = np.random.rand(swarm, 3)
        r2p = np.random.rand(swarm, 3)
        if gbest_X is not None:
            X[:, 16:19] += 0.5 * r1p * (pbest_X[:, 16:19] - X[:, 16:19]) + 0.5 * r2p * (gbest_X[16:19] - X[:, 16:19])

        # Enforce bounds and reflect
        for j, (lo, hi) in enumerate(bounds):
            out_lo = X[:, j] < lo
            out_hi = X[:, j] > hi
            if np.any(out_lo):
                X[out_lo, j] = lo + (lo - X[out_lo, j])
                V[out_lo, j] *= -0.5 if j < 16 else 0.0
            if np.any(out_hi):
                X[out_hi, j] = hi - (X[out_hi, j] - hi)
                V[out_hi, j] *= -0.5 if j < 16 else 0.0

        if logger and (i % max(1, iters//10) == 0 or i == iters-1):
            logger.info(f"[single] iter={i+1}/{iters} J_best={gbest_J:.6f} mass={gbest_metrics['mass']:.4f} "
                        f"disp={gbest_metrics['max_disp']:.6f} stress={gbest_metrics['max_stress']:.6f} "
                        f"w={w_hist[-1]:.3f} c1={c1_hist[-1]:.3f} c2={c2_hist[-1]:.3f}")

    return {
        'gbest_J': gbest_J,
        'gbest_X': gbest_X,
        'metrics': gbest_metrics,
        'w_hist': w_hist,
        'c1_hist': c1_hist,
        'c2_hist': c2_hist,
        'mass_hist': mass_hist,
        'feasible_cache': feas_cache,
        'infeasible_cache': infeas_cache,
    }
