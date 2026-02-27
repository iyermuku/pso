
"""
Particle Swarm Optimization (PSO) core for the 10-bar truss.
Includes:
- pso_single_run: original exploratory run (13D with w, c1, c2 in particle)
- pso_single_run_robust: robust run (areas-only lbest PSO with constriction)
- pso_best_of_seeds: wrapper to run multiple seeds and pick best, with a mode switch
"""
import logging
import numpy as np
from typing import Dict, List

from truss_model import ndof, Amin, Amax, U_ALLOW, solve_displacements, mass_from_A, member_stresses
from constraints import project_params, constraint_vector
from objectives import penalized_objective

logger = logging.getLogger("pso")

# ------------------------------
# Helpers for robust PSO
# ------------------------------
def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator):
    """Latin Hypercube Sampling in [0,1]^n_dim; shuffle per dimension."""
    U = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + U) / n_samples
    samples = np.zeros_like(grid)
    for j in range(n_dim):
        samples[:, j] = rng.permutation(grid[:, j])
    return samples


def _deb_better(cv1: float, J1: float, cv2: float, J2: float, eps: float = 1e-9):
    """Deb feasibility rule: prefer feasible; else smaller violation; then smaller objective."""
    f1, f2 = (cv1 <= eps), (cv2 <= eps)
    if f1 and f2:
        return J1 < J2 - 1e-12
    if f1 != f2:
        return f1  # feasible wins
    # both infeasible
    if abs(cv1 - cv2) > 1e-12:
        return cv1 < cv2
    return J1 < J2 - 1e-12


def _violation(g_vec: np.ndarray) -> float:
    # Sum of positive parts (g<=0 satisfied)
    return float(np.sum(np.maximum(g_vec, 0.0)))


# ------------------------------
# Robust PSO (areas-only, lbest ring, constriction)
# ------------------------------
def pso_single_run_robust(
    swarm_size: int = 80,
    iters: int = 300,
    seed: int = 2026,
    stall_window: int = 30,
    max_restarts: int = 1,
    lbest_span: int = 2,      # ring topology: neighbors i-2..i+2
    v_frac: float = 0.2       # velocity clamp as fraction of range
) -> Dict:
    """
    Robust PSO for 10-bar truss (areas only, D=10).
    - fixed PSO coefficients (Clerc-Kennedy constriction; c1=c2=2.05, chi≈0.7298)
    - local-best ring topology (span configurable)
    - Latin Hypercube Sampling (LHS) initialization for areas
    - velocity clamping + reflective boundaries
    - Deb feasibility rule for selection
    - soft diversity injection when stalled

    Returns keys compatible with the original interface so that main.py remains unchanged.
    """
    rng = np.random.default_rng(seed)
    D = 10  # areas only

    # Constriction PSO coefficients (Clerc-Kennedy)
    c1 = 2.05
    c2 = 2.05
    phi = c1 + c2
    chi = 2.0 / (phi - 2.0 + np.sqrt(phi**2 - 4.0 * phi))  # ~0.7298

    # Ranges per dimension
    lo = np.full(D, Amin)
    hi = np.full(D, Amax)
    span = hi - lo
    v_max = v_frac * span

    # LHS initialization in [lo, hi]
    S01 = _lhs(swarm_size, D, rng)
    X = lo + S01 * span
    V = rng.uniform(-v_max, v_max, size=(swarm_size, D))

    def evaluate(X: np.ndarray):
        m_list = np.zeros(len(X))
        U_list = np.zeros((len(X), ndof))
        J_list = np.zeros(len(X))
        cv_list = np.zeros(len(X))
        g_vectors: List[np.ndarray] = []
        # First pass to compute U, g, m
        for i in range(len(X)):
            A = X[i, :]
            U = solve_displacements(A)
            g = constraint_vector(U)  # combined disp+stress violations
            m = mass_from_A(A)
            U_list[i] = U
            g_vectors.append(g)
            m_list[i] = m
        # Stack g_list dynamically (pad shorter vectors with zeros on the right)
        max_g_len = max(len(g) for g in g_vectors)
        g_list = np.zeros((len(X), max_g_len))
        for i, g in enumerate(g_vectors):
            g_list[i, :len(g)] = g
        avg_m = float(np.mean(m_list))
        avg_g = np.mean(g_list, axis=0)
        # Second pass for penalized objective and constraint violation
        for i in range(len(X)):
            J_list[i] = penalized_objective(m_list[i], g_list[i], avg_m, avg_g)
            cv_list[i] = _violation(g_list[i])
        return m_list, U_list, g_list, J_list, cv_list

    # Personal bests
    m_list, U_list, g_list, J_list, cv_list = evaluate(X)
    pbest_X = X.copy()
    pbest_J = J_list.copy()
    pbest_cv = cv_list.copy()
    pbest_m = m_list.copy()
    pbest_U = U_list.copy()

    # Select global best by Deb rule among pbest
    def select_best_from_pbest():
        idx = 0
        for j in range(1, swarm_size):
            if _deb_better(pbest_cv[j], pbest_J[j], pbest_cv[idx], pbest_J[idx]):
                idx = j
        return idx

    g_idx = select_best_from_pbest()
    gbest_X = pbest_X[g_idx].copy()
    gbest_J = float(pbest_J[g_idx])
    gbest_cv = float(pbest_cv[g_idx])
    gbest_m = float(pbest_m[g_idx])
    gbest_U = pbest_U[g_idx].copy()

    # Best-ever objective (for continuity with outputs)
    best_obj_J = gbest_J
    best_obj_X = gbest_X.copy()
    best_obj_m = gbest_m
    best_obj_U = gbest_U.copy()

    # Best feasible global solution
    best_feas_J = float("inf")
    best_feas_X = None
    best_feas_m = None
    best_feas_U = None

    # Histories
    mass_hist: List[float] = []
    disp_hist: List[float] = []
    feas_frac_hist: List[float] = []
    gbest_feas_hist: List[float] = []

    stall_counter = 0
    restarts_used = 0

    total_iters = iters * (max_restarts + 1)
    logger.info(
        "Starting robust PSO run: seed=%d, total_iters=%d, max_restarts=%d",
        seed, total_iters, max_restarts
    )

    # Precompute neighbor indices for ring topology
    nbrs = []
    for i in range(swarm_size):
        idxs = [((i + d) % swarm_size) for d in range(-lbest_span, lbest_span + 1)]
        nbrs.append(idxs)

    for it in range(total_iters):
        # Build local best for each particle under Deb rule from neighbor pbests
        lbest_X = np.zeros_like(X)
        for i in range(swarm_size):
            idx = nbrs[i][0]
            for j in nbrs[i][1:]:
                if _deb_better(pbest_cv[j], pbest_J[j], pbest_cv[idx], pbest_J[idx]):
                    idx = j
            lbest_X[i] = pbest_X[idx]

        # PSO velocity & position update (constriction)
        r1 = np.random.random((swarm_size, D))
        r2 = np.random.random((swarm_size, D))

        V = chi * (V + c1 * r1 * (pbest_X - X) + c2 * r2 * (lbest_X - X))

        # Clamp velocities
        V = np.clip(V, -v_max, v_max)

        # Update positions
        X = X + V

        # Reflective boundaries (element-wise)
        below = X < lo
        above = X > hi
        X = np.where(below, lo + (lo - X), X)
        X = np.where(above, hi - (X - hi), X)
        V[below | above] *= -0.5  # damped reflection

        # Evaluate
        m_list, U_list, g_list, J_list, cv_list = evaluate(X)

        # Update pbests under Deb rule
        for i in range(swarm_size):
            if _deb_better(cv_list[i], J_list[i], pbest_cv[i], pbest_J[i]):
                pbest_X[i] = X[i]
                pbest_J[i] = J_list[i]
                pbest_cv[i] = cv_list[i]
                pbest_m[i] = m_list[i]
                pbest_U[i] = U_list[i]

        # Global best selection from pbests (Deb rule)
        prev_gbest_J, prev_gbest_cv = gbest_J, gbest_cv
        g_idx = select_best_from_pbest()
        gbest_X = pbest_X[g_idx].copy()
        gbest_J = float(pbest_J[g_idx])
        gbest_cv = float(pbest_cv[g_idx])
        gbest_m = float(pbest_m[g_idx])
        gbest_U = pbest_U[g_idx].copy()

        # Stall counter: reset only if we strictly improve under Deb
        improved_global = _deb_better(gbest_cv, gbest_J, prev_gbest_cv, prev_gbest_J)
        stall_counter = 0 if improved_global else (stall_counter + 1)

        # Best-ever objective
        if gbest_J < best_obj_J - 1e-12:
            best_obj_J = gbest_J
            best_obj_X = gbest_X.copy()
            best_obj_m = gbest_m
            best_obj_U = gbest_U.copy()

        # Feasibility stats
        feas_mask = (cv_list <= 1e-9)
        feas_frac = float(np.mean(feas_mask))
        gbest_feas = 1.0 if gbest_cv <= 1e-9 else 0.0

        feas_frac_hist.append(feas_frac)
        gbest_feas_hist.append(gbest_feas)

        # Update best feasible solution
        if gbest_feas == 1.0 and (gbest_J < best_feas_J - 1e-12):
            best_feas_J = gbest_J
            best_feas_X = gbest_X.copy()
            best_feas_m = gbest_m
            best_feas_U = gbest_U.copy()
            max_stress = float(np.max(np.abs(member_stresses(best_feas_U))))
        logger.info(
            "NEW BEST FEASIBLE iter=%d J=%.6f mass=%.6f max_disp=%.6f max_stress=%.6f ksi",
            it + 1, best_feas_J, best_feas_m, float(np.max(np.abs(best_feas_U))), max_stress
        )

        # Histories (best feasible mass in swarm this iteration; gbest disp)
        if np.any(feas_mask):
            best_swarm_feas_mass = float(np.min(m_list[feas_mask]))
        else:
            best_swarm_feas_mass = np.nan
        mass_hist.append(best_swarm_feas_mass)
        disp_hist.append(float(np.max(np.abs(gbest_U))))

        # Diversity injection on stall
        if stall_counter >= stall_window and restarts_used < max_restarts:
            logger.warning(
                "Stall detected (>= %d iters). Injecting diversity into worst 30%%. restarts_used=%d",
                stall_window, restarts_used
            )
            #breakpoint()
            # Identify worst by (cv, J) lexicographic (largest first)
            order = np.lexsort((J_list, cv_list))
            worst_idx = order[::-1][:max(1, swarm_size // 3)]
            # Reinitialize those with LHS jitter
            S01w = _lhs(len(worst_idx), D, rng)
            X[worst_idx] = lo + S01w * span
            V[worst_idx] = rng.uniform(-v_max, v_max, size=(len(worst_idx), D))
            # Re-evaluate & overwrite pbests for those
            m_list2, U_list2, g_list2, J_list2, cv_list2 = evaluate(X[worst_idx])
            pbest_X[worst_idx] = X[worst_idx]
            pbest_J[worst_idx] = J_list2
            pbest_cv[worst_idx] = cv_list2
            pbest_m[worst_idx] = m_list2
            pbest_U[worst_idx] = U_list2
            stall_counter = 0
            restarts_used += 1

    logger.info("Finished PSO robust run: seed=%d, restarts_used=%d", seed, restarts_used)

    # Prepare outputs (keep original keys)
    return {
        'gbest_A': gbest_X[:10],
        'gbest_w': 0.7298,            # constant for compatibility (chi)
        'gbest_c1': 2.05,
        'gbest_c2': 2.05,
        'gbest_mass': float(gbest_m),
        'gbest_max_disp': float(np.max(np.abs(gbest_U))),
        'mass_hist': np.array(mass_hist),
        'disp_hist': np.array(disp_hist),
        'feas_frac_hist': np.array(feas_frac_hist),
        'gbest_feas_hist': np.array(gbest_feas_hist),

        # Best-ever objective solution (may be infeasible)
        'best_obj_A': best_obj_X[:10],
        'best_obj_w': 0.7298,
        'best_obj_c1': 2.05,
        'best_obj_c2': 2.05,
        'best_obj_mass': float(best_obj_m),
        'best_obj_max_disp': float(np.max(np.abs(best_obj_U))),
        'best_obj_J': float(best_obj_J),

        # Best feasible global solution by objective
        'best_feas_A': (best_feas_X[:10] if best_feas_X is not None else np.array([])),
        'best_feas_w': 0.7298,
        'best_feas_c1': 2.05,
        'best_feas_c2': 2.05,
        'best_feas_mass': (float(best_feas_m) if best_feas_m is not None else float('nan')),
        'best_feas_max_disp': (float(np.max(np.abs(best_feas_U))) if best_feas_U is not None else float('nan')),
        'best_feas_J': (float(best_feas_J) if best_feas_X is not None else float('inf')),
    }


# ------------------------------
# Original PSO (kept intact)
# ------------------------------
def pso_single_run(swarm_size: int = 60, iters: int = 250, seed: int = 2026,
                   stall_window: int = 25, stall_tol: float = 1e-4, max_restarts: int = 2) -> Dict:
    rng = np.random.default_rng(seed)
    D = 13  # 10 areas + w + c1 + c2

    def init_swarm():
        # latin hypercube sampling over all dimensions
        U = _lhs(swarm_size, D, rng)
        X = np.zeros((swarm_size, D))
        V = rng.uniform(-1.0, 1.0, size=(swarm_size, D))
        # Areas (first 10 dims): map to [Amin,Amax]
        X[:, :10] = Amin + U[:, :10] * (Amax - Amin)
        # Parameters c1,c2 (random positive), w in [0.5, 0.95]
        X[:, 10] = 0.5 + U[:, 10] * (0.95 - 0.5)
        X[:, 11] = 0.3 + U[:, 11] * (2.7 - 0.3)
        X[:, 12] = 0.3 + U[:, 12] * (2.7 - 0.3)
        # Project each particle's params to ensure valid
        for i in range(swarm_size):
            w, c1p, c2p = project_params(X[i, 10], X[i, 11], X[i, 12])
            X[i, 10], X[i, 11], X[i, 12] = w, c1p, c2p
        logger.debug("Initialized swarm with LHS: size=%d, D=%d", swarm_size, D)
        return X, V

    def evaluate_swarm(X: np.ndarray):
        m_list = np.zeros(len(X))
        U_list = np.zeros((len(X), ndof))
        J_list = np.zeros(len(X))
        cv_list = np.zeros(len(X))
        g_vectors: List[np.ndarray] = []
        # First pass to compute U, g, m
        for i in range(len(X)):
            A = X[i, :10]
            U = solve_displacements(A)
            g = constraint_vector(U)  # combined disp+stress violations
            m = mass_from_A(A)
            U_list[i] = U
            g_vectors.append(g)
            m_list[i] = m
        # Stack g_list dynamically (pad shorter vectors with zeros on the right)
        max_g_len = max(len(g) for g in g_vectors)
        g_list = np.zeros((len(X), max_g_len))
        for i, g in enumerate(g_vectors):
            g_list[i, :len(g)] = g
        avg_m = float(np.mean(m_list))
        avg_g = np.mean(g_list, axis=0)
        # Second pass for penalized objective and constraint violation
        for i in range(len(X)):
            J_list[i] = penalized_objective(m_list[i], g_list[i], avg_m, avg_g)
            cv_list[i] = _violation(g_list[i])
        #breakpoint()
        return m_list, U_list, g_list, J_list, cv_list
        
        # m_list = np.zeros(len(X))
        # g_list = np.zeros(len(X), ndof)
        # U_list = np.zeros(len(X), ndof)
        # for i in range(len(X)):
            # A = X[i, :10]
            # U = solve_displacements(A)
            # U_list[i] = U
            # g_list[i] = constraint_vector(U)
            # m_list[i] = mass_from_A(A)
        # avg_m = float(np.mean(m_list))
        # avg_g = np.mean(g_list, axis=0)
        # J_list = np.array([
            # penalized_objective(m_list[i], g_list[i], avg_m, avg_g) for i in range(len(X))
        # ])
        # return m_list, g_list, U_list, J_list

    # Initialize
    X, V = init_swarm()
    m_list, U_list, g_list, J_list, cv_list = evaluate_swarm(X)
    pbest_X = X.copy()
    pbest_J = J_list.copy()
    pbest_cv = cv_list.copy()
    pbest_m = m_list.copy()
    pbest_U = U_list.copy()

    # Select global best by Deb rule among pbest
    def select_best_from_pbest():
        idx = 0
        for j in range(1, swarm_size):
            if _deb_better(pbest_cv[j], pbest_J[j], pbest_cv[idx], pbest_J[idx]):
                idx = j
        return idx

    g_idx = select_best_from_pbest()
    gbest_X = pbest_X[g_idx].copy()
    gbest_J = float(pbest_J[g_idx])
    gbest_cv = float(pbest_cv[g_idx])
    gbest_m = float(pbest_m[g_idx])
    gbest_U = pbest_U[g_idx].copy()
    # logger.info(
        # 'index=%d found global best after initialization',
        # g_idx)
    #breakpoint()
    # pbest_X = X.copy(); pbest_J = J_list.copy(); pbest_m = m_list.copy(); pbest_U = U_list.copy()
    # g_idx = int(np.argmin(pbest_J))
    # gbest_X = pbest_X[g_idx].copy(); gbest_J = float(pbest_J[g_idx]); gbest_m = float(pbest_m[g_idx]); gbest_U = pbest_U[g_idx].copy()

    # Track best-ever objective solution (over entire run)
    best_obj_J = gbest_J
    best_obj_X = gbest_X.copy()
    best_obj_m = float(gbest_m)
    best_obj_U = gbest_U.copy()

    # Track best FEASIBLE global solution by objective (mass when feasible)
    best_feas_J = float('inf')
    best_feas_X = None
    best_feas_m = None
    best_feas_U = None

    # Histories (across restarts)
    mass_hist: List[float] = []
    disp_hist: List[float] = []
    feas_frac_hist: List[float] = []  # fraction of feasible particles per iteration
    gbest_feas_hist: List[float] = []  # 1 if gbest feasible else 0
    w_hist: List[float] = []
    c1_hist: List[float] = []
    c2_hist: List[float] = []

    # Track best feasible mass seen (for plotting)
    best_feasible_mass = float('nan')

    best_J_global = gbest_J
    stall_counter = 0
    restarts_used = 0
    total_iters = iters * (max_restarts + 1)

    logger.info("Starting PSO run: seed=%d, total_iters=%d, max_restarts=%d", seed, total_iters, max_restarts)

    for it in range(total_iters):
        w_vec = X[:, 10]; c1_vec = X[:, 11]; c2_vec = X[:, 12]
        r1 = np.random.random((swarm_size, D)); r2 = np.random.random((swarm_size, D))
        V = (
            w_vec[:, None] * V
            + c1_vec[:, None] * r1 * (pbest_X - X)
            + c2_vec[:, None] * r2 * (gbest_X - X)
        )
        X = X + V

        # Clamp areas and project params
        X[:, :10] = np.clip(X[:, :10], Amin, Amax)
        for i in range(swarm_size):
            w, c1p, c2p = project_params(X[i, 10], X[i, 11], X[i, 12])
            X[i, 10], X[i, 11], X[i, 12] = w, c1p, c2p

        # Evaluate
        m_list, U_list, g_list, J_list, cv_list = evaluate_swarm(X)
        # Update pbest
        for i in range(swarm_size):
            if _deb_better(cv_list[i], J_list[i], pbest_cv[i], pbest_J[i]):
                pbest_X[i] = X[i]
                pbest_J[i] = J_list[i]
                pbest_cv[i] = cv_list[i]
                pbest_m[i] = m_list[i]
                pbest_U[i] = U_list[i]

        # improved = J_list < pbest_J
        # if np.any(improved):
            # pbest_J[improved] = J_list[improved]
            # pbest_X[improved] = X[improved]
            # pbest_m[improved] = m_list[improved]
            # pbest_U[improved] = U_list[improved]

        # Update gbest (global across restarts)
        # Global best selection from pbests (Deb rule)
        prev_gbest_J, prev_gbest_cv = gbest_J, gbest_cv
        g_idx = select_best_from_pbest()
        gbest_X = pbest_X[g_idx].copy()
        gbest_J = float(pbest_J[g_idx])
        gbest_cv = float(pbest_cv[g_idx])
        gbest_m = float(pbest_m[g_idx])
        gbest_U = pbest_U[g_idx].copy()
        # logger.info(
            # 'index=%d in iter=%d found best',
            # g_idx,it + 1)

        # Stall counter: reset only if we strictly improve under Deb
        improved_global = _deb_better(gbest_cv, gbest_J, prev_gbest_cv, prev_gbest_J)
        stall_counter = 0 if improved_global else (stall_counter + 1)
        
        # g_idx = int(np.argmin(pbest_J))
        # if pbest_J[g_idx] < best_J_global - 1e-12:
            # best_J_global = float(pbest_J[g_idx])
            # stall_counter = 0
        # else:
            # stall_counter += 1
        # gbest_X = pbest_X[g_idx].copy(); gbest_J = float(pbest_J[g_idx]); gbest_m = float(pbest_m[g_idx]); gbest_U = pbest_U[g_idx].copy()
        w_hist.append(gbest_X[10]); c1_hist.append(gbest_X[11]); c2_hist.append(gbest_X[12]);
        # Best-ever objective tracking
        if gbest_J < best_obj_J - 1e-12:
            best_obj_J = float(gbest_J)
            best_obj_X = gbest_X.copy()
            best_obj_m = float(gbest_m)
            best_obj_U = gbest_U.copy()
        # Feasibility stats
        feas_mask = np.all(g_list <= 1e-9, axis=1)
        feas_frac = float(np.mean(feas_mask))
        gbest_feas = 1.0 if np.all(constraint_vector(gbest_U) <= 1e-9) else 0.0
        feas_frac_hist.append(feas_frac)
        gbest_feas_hist.append(gbest_feas)

        # Update best FEASIBLE solution by objective
        #breakpoint()
        if gbest_feas == 1.0 and (gbest_J < best_feas_J - 1e-12):
            best_feas_J = float(gbest_J)
            best_feas_X = gbest_X.copy()
            best_feas_m = float(gbest_m)
            best_feas_U = gbest_U.copy()
            mem_stres = member_stresses(best_feas_U)
            max_stress = float(np.max(np.abs(mem_stres)))
            #breakpoint()
            logger.info(
                'NEW BEST FEASIBLE iter=%d J=%.6f mass=%.6f max_disp=%.6f max_stress=%.6f ksi',
                it + 1, best_feas_J, best_feas_m, float(np.max(np.abs(best_feas_U))), max_stress
            )
            #breakpoint()
        # Best feasible mass tracking (for plotting)
        if np.any(feas_mask):
            best_swarm_feas_mass = float(np.min(m_list[feas_mask]))
            if np.isnan(best_feasible_mass) or best_swarm_feas_mass < best_feasible_mass:
                best_feasible_mass = best_swarm_feas_mass
        mass_hist.append(best_feasible_mass)
        disp_hist.append(float(np.max(np.abs(gbest_U))))

        # Iteration-level logging (verbose)
        logger.debug(
            "iter=%4d feas_frac=%.3f gbest_feas=%d gbest_mass=%.4f gbest_maxU=%.6f stall_ctr=%d",
            it + 1, feas_frac, int(gbest_feas), gbest_m, float(np.max(np.abs(gbest_U))), stall_counter
        )

        # Restart if stalled
        if stall_counter >= stall_window and restarts_used < max_restarts:
            logger.warning(
                "Stall detected (>= %d iters). Restarting swarm around current gbest. restarts_used=%d",
                stall_window, restarts_used
            )
            #breakpoint()
            X_new = np.zeros_like(X)
            jitter_A = np.clip(
                gbest_X[:10] + 0.10 * (Amax - Amin) * rng.uniform(-1, 1, size=(swarm_size, 10)),
                Amin, Amax
            )
            X_new[:, :10] = jitter_A
            # Reinit params near current best
            X_new[:, 10] = gbest_X[10] + 0.05 * rng.uniform(-1, 1, size=swarm_size)
            X_new[:, 11] = gbest_X[11] + 0.2 * rng.uniform(-1, 1, size=swarm_size)
            X_new[:, 12] = gbest_X[12] + 0.2 * rng.uniform(-1, 1, size=swarm_size)
            for i in range(swarm_size):
                w, c1p, c2p = project_params(X_new[i, 10], X_new[i, 11], X_new[i, 12])
                X_new[i, 10], X_new[i, 11], X_new[i, 12] = w, c1p, c2p
            V = rng.uniform(-0.5, 0.5, size=V.shape)
            X = X_new
            # Reset personal bests to new swarm (keep global best)
            m_list, U_list, g_list , J_list, cv_list = evaluate_swarm(X)
            pbest_X = X.copy(); pbest_J = J_list.copy(); pbest_m = m_list.copy(); pbest_U = U_list.copy()
            stall_counter = 0
            restarts_used += 1

    logger.info("Finished PSO single run: seed=%d, restarts_used=%d", seed, restarts_used)

    return {
        'gbest_A': gbest_X[:10],
        'gbest_w': float(gbest_X[10]),
        'gbest_c1': float(gbest_X[11]),
        'gbest_c2': float(gbest_X[12]),
        'gbest_mass': float(gbest_m),
        'gbest_max_disp': float(np.max(np.abs(gbest_U))),
        'mass_hist': np.array(mass_hist),
        'disp_hist': np.array(disp_hist),
        'feas_frac_hist': np.array(feas_frac_hist),
        'gbest_feas_hist': np.array(gbest_feas_hist),
        'w_hist': np.array(w_hist), 'c1_hist': np.array(c1_hist), 'c2_hist': np.array(c2_hist),
        #'w_mean_hist': np.array(w_mean_hist), 'c1_mean_hist': np.array(c1_mean_hist), 'c2_mean_hist': np.array(c2_mean_hist),
        # Best-ever objective solution (may be infeasible)
        'best_obj_A': best_obj_X[:10],
        'best_obj_w': float(best_obj_X[10]) if best_obj_X.shape[0] > 10 else float('nan'),
        'best_obj_c1': float(best_obj_X[11]) if best_obj_X.shape[0] > 11 else float('nan'),
        'best_obj_c2': float(best_obj_X[12]) if best_obj_X.shape[0] > 12 else float('nan'),
        'best_obj_mass': float(best_obj_m),
        'best_obj_max_disp': float(np.max(np.abs(best_obj_U))),
        'best_obj_J': float(best_obj_J),
        # Best feasible global solution by objective
        'best_feas_A': (best_feas_X[:10] if best_feas_X is not None else np.array([])),
        'best_feas_w': (float(best_feas_X[10]) if best_feas_X is not None else float('nan')),
        'best_feas_c1': (float(best_feas_X[11]) if best_feas_X is not None else float('nan')),
        'best_feas_c2': (float(best_feas_X[12]) if best_feas_X is not None else float('nan')),
        'best_feas_mass': (float(best_feas_m) if best_feas_m is not None else float('nan')),
        'best_feas_max_disp': (float(np.max(np.abs(best_feas_U))) if best_feas_U is not None else float('nan')),
        'best_feas_J': (float(best_feas_J) if best_feas_X is not None else float('inf')),
    }


# ------------------------------
# Seed wrapper with mode switch
# ------------------------------
def pso_best_of_seeds(num_runs: int = 5, swarm_size: int = 60, iters: int = 250, max_restarts: int = 2,
                      stall_window: int = 25, base_seed: int = 2026,robust: bool = True):
    best = None
    all_runs = []
    for k in range(num_runs):
        seed = base_seed + 97 * k
        logger.info("Running seed %d/%d: seed_val=%d", k + 1, num_runs, seed)
        if robust:
            res = pso_single_run_robust(
                swarm_size=swarm_size, iters=iters, seed=seed,
                stall_window=stall_window, max_restarts=max_restarts
            )
        else:
            res = pso_single_run(
                swarm_size=swarm_size, iters=iters, seed=seed,
                stall_window=stall_window, stall_tol=1e-4, max_restarts=max_restarts
            )
        all_runs.append(res)
        feasible = res['gbest_max_disp'] <= U_ALLOW + 1e-9
        score = (0, res['gbest_mass']) if feasible else (1, res['gbest_max_disp'], res['gbest_mass'])
        logger.info(
            "Seed result: feasible=%s, mass=%.4f, max_disp=%.6f, score=%s",
            str(feasible), res['gbest_mass'], res['gbest_max_disp'], score
        )
        if best is None or score < best['score']:
            best = {'score': score, 'res': res, 'seed': seed}
            logger.info("New best selected (seed=%d)", seed)
    return best, all_runs
