from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from problem_adapters import TrussProblem


@dataclass
class ParticleState:
    x: np.ndarray
    v: np.ndarray
    objective: float
    mass: float
    cv: float
    feasible: bool


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + u) / n_samples
    samples = np.zeros_like(grid)
    for dim in range(n_dim):
        samples[:, dim] = rng.permutation(grid[:, dim])
    return samples


def _deb_better(cv1: float, j1: float, cv2: float, j2: float, eps: float = 1e-9) -> bool:
    f1, f2 = (cv1 <= eps), (cv2 <= eps)
    if f1 and f2:
        return j1 < j2 - 1e-12
    if f1 != f2:
        return f1
    if abs(cv1 - cv2) > 1e-12:
        return cv1 < cv2
    return j1 < j2 - 1e-12


def _evaluate(problem: TrussProblem, x: np.ndarray) -> ParticleState:
    res = problem.evaluate(x)
    x_eval = np.asarray(res["x_eval"], dtype=float)
    return ParticleState(
        x=x_eval,
        v=np.zeros_like(x_eval),
        objective=float(res["objective"]),
        mass=float(res["mass"]),
        cv=float(res["constraint_violation"]),
        feasible=bool(res["constraint_violation"] <= 1e-9),
    )


def _reflect_bounds(x: np.ndarray, v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_new = x.copy()
    v_new = v.copy()
    below = x_new < lo
    above = x_new > hi
    if np.any(below):
        x_new = np.where(below, lo + (lo - x_new), x_new)
        v_new[below] *= -0.5
    if np.any(above):
        x_new = np.where(above, hi - (x_new - hi), x_new)
        v_new[above] *= -0.5
    x_new = np.clip(x_new, lo, hi)
    return x_new, v_new


def run_fixed_coeff_pso(
    problem: TrussProblem,
    swarm_size: int | None = None,
    iters: int | None = None,
    seed: int = 2026,
    v_frac: float = 0.20,
    reflection_on_violation: bool = True,
    coeff_mode: str = "fixed",
    seed_optima_pct: float = 0.0,
) -> Dict[str, object]:
    # Use landscape-recommended defaults when caller does not override.
    if swarm_size is None:
        swarm_size = problem.recommended_swarm_size
    if iters is None:
        iters = problem.recommended_iters
    rng = np.random.default_rng(seed)
    d = problem.dim
    lo = problem.lo.astype(float)
    hi = problem.hi.astype(float)
    span = hi - lo
    v_max = v_frac * span

    if coeff_mode not in {"fixed", "two-phase"}:
        raise ValueError("coeff_mode must be one of: fixed, two-phase")
    if seed_optima_pct < 0.0:
        raise ValueError("seed_optima_pct must be >= 0")

    def coeffs_for_iter(iter_idx: int) -> Tuple[float, float, float]:
        if coeff_mode == "fixed":
            return float(problem.recommended_w), float(problem.recommended_c1), float(problem.recommended_c2)

        schedule = problem.recommended_schedule
        switch_frac = float(schedule.get("switch_fraction_of_iters", 0.60))
        switch_iter = int(max(1, np.floor(switch_frac * max(iters, 1))))
        if iter_idx < switch_iter:
            phase = schedule["phase_1_explore"]
        else:
            phase = schedule["phase_2_refine"]
        return float(phase["w"]), float(phase["c1"]), float(phase["c2"])

    w0, c10, c20 = coeffs_for_iter(0)

    max_seed_slots = int(np.floor((seed_optima_pct / 100.0) * swarm_size))
    available_optima = list(problem.detected_optima or [])
    n_seed_target = int(min(max_seed_slots, len(available_optima), swarm_size))

    particle_names = [f"lhs_particle_{i:03d}" for i in range(swarm_size)]
    seeded_particles: List[Dict[str, object]] = []

    x = np.zeros((swarm_size, d), dtype=float)
    n_seed = 0
    if n_seed_target > 0:
        for i in range(n_seed_target):
            opt = available_optima[i]
            x_seed = np.asarray(opt["design_variables"], dtype=float)
            if x_seed.shape[0] != d:
                continue
            x[n_seed] = np.clip(x_seed, lo, hi)
            seed_name = f"seed_optimum_{int(opt.get('rank', i + 1)):03d}"
            particle_names[n_seed] = seed_name
            seeded_particles.append(
                {
                    "particle_index": n_seed,
                    "particle_name": seed_name,
                    "optimum_rank": int(opt.get("rank", i + 1)),
                    "seed_objective": float(opt.get("objective", np.nan)),
                    "seed_basin_size": int(opt.get("basin_size", 0)),
                    "design_variables": x[n_seed].tolist(),
                }
            )
            n_seed += 1

    lhs_count = swarm_size - n_seed
    if lhs_count > 0:
        s01 = _lhs(lhs_count, d, rng)
        x[n_seed:] = lo + s01 * span

    v = rng.uniform(-v_max, v_max, size=(swarm_size, d))

    pbest_x = np.zeros_like(x)
    pbest_obj = np.zeros(swarm_size)
    pbest_cv = np.zeros(swarm_size)
    pbest_mass = np.zeros(swarm_size)

    current_obj = np.zeros(swarm_size)
    current_cv = np.zeros(swarm_size)
    current_mass = np.zeros(swarm_size)

    for i in range(swarm_size):
        state = _evaluate(problem, x[i])
        x[i] = state.x
        current_obj[i] = state.objective
        current_cv[i] = state.cv
        current_mass[i] = state.mass
        pbest_x[i] = state.x.copy()
        pbest_obj[i] = state.objective
        pbest_cv[i] = state.cv
        pbest_mass[i] = state.mass

    g_idx = 0
    for i in range(1, swarm_size):
        if _deb_better(pbest_cv[i], pbest_obj[i], pbest_cv[g_idx], pbest_obj[g_idx]):
            g_idx = i

    gbest_x = pbest_x[g_idx].copy()
    gbest_obj = float(pbest_obj[g_idx])
    gbest_cv = float(pbest_cv[g_idx])
    gbest_mass = float(pbest_mass[g_idx])
    gbest_particle_name = particle_names[g_idx]

    obj_hist: List[float] = [gbest_obj]
    mass_hist: List[float] = [gbest_mass]
    cv_hist: List[float] = [gbest_cv]
    feas_frac_hist: List[float] = [float(np.mean(current_cv <= 1e-9))]

    for _ in range(iters):
        r1 = rng.random((swarm_size, d))
        r2 = rng.random((swarm_size, d))
        w, c1, c2 = coeffs_for_iter(_)
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -v_max, v_max)

        prev_x = x.copy()
        x_trial = x + v
        x_trial, v_trial = _reflect_bounds(x_trial, v, lo, hi)

        new_x = x_trial.copy()
        new_v = v_trial.copy()
        new_obj = np.zeros(swarm_size)
        new_cv = np.zeros(swarm_size)
        new_mass = np.zeros(swarm_size)

        for i in range(swarm_size):
            trial_state = _evaluate(problem, x_trial[i])
            chosen_x = trial_state.x
            chosen_v = v_trial[i].copy()
            chosen_obj = trial_state.objective
            chosen_cv = trial_state.cv
            chosen_mass = trial_state.mass

            if reflection_on_violation and trial_state.cv > 1e-9:
                reflected_x = prev_x[i] - 0.5 * v_trial[i]
                reflected_x, reflected_v = _reflect_bounds(reflected_x, -0.5 * v_trial[i], lo, hi)
                reflected_state = _evaluate(problem, reflected_x)
                if _deb_better(reflected_state.cv, reflected_state.objective, trial_state.cv, trial_state.objective):
                    chosen_x = reflected_state.x
                    chosen_v = reflected_v
                    chosen_obj = reflected_state.objective
                    chosen_cv = reflected_state.cv
                    chosen_mass = reflected_state.mass

            new_x[i] = chosen_x
            new_v[i] = chosen_v
            new_obj[i] = chosen_obj
            new_cv[i] = chosen_cv
            new_mass[i] = chosen_mass

            if _deb_better(new_cv[i], new_obj[i], pbest_cv[i], pbest_obj[i]):
                pbest_x[i] = new_x[i].copy()
                pbest_obj[i] = new_obj[i]
                pbest_cv[i] = new_cv[i]
                pbest_mass[i] = new_mass[i]

        x = new_x
        v = new_v
        current_obj = new_obj
        current_cv = new_cv
        current_mass = new_mass

        g_idx = 0
        for i in range(1, swarm_size):
            if _deb_better(pbest_cv[i], pbest_obj[i], pbest_cv[g_idx], pbest_obj[g_idx]):
                g_idx = i
        gbest_x = pbest_x[g_idx].copy()
        gbest_obj = float(pbest_obj[g_idx])
        gbest_cv = float(pbest_cv[g_idx])
        gbest_mass = float(pbest_mass[g_idx])
        gbest_particle_name = particle_names[g_idx]

        obj_hist.append(gbest_obj)
        mass_hist.append(gbest_mass)
        cv_hist.append(gbest_cv)
        feas_frac_hist.append(float(np.mean(current_cv <= 1e-9)))

    final_eval = problem.evaluate(gbest_x)
    seeded_name_set = {p["particle_name"] for p in seeded_particles}
    seeded_particle_reached_gbest = gbest_particle_name in seeded_name_set

    return {
        "problem_id": problem.problem_id,
        "label": problem.label,
        "coefficient_mode": coeff_mode,
        "recommended_coefficients": {"w": w0, "c1": c10, "c2": c20},
        "recommended_schedule": problem.recommended_schedule,
        "swarm_size": swarm_size,
        "iters": iters,
        "gbest_x": gbest_x,
        "gbest_objective": gbest_obj,
        "gbest_mass": gbest_mass,
        "gbest_cv": gbest_cv,
        "gbest_particle_name": gbest_particle_name,
        "best_design_variables": gbest_x.copy(),
        "seed_optima_pct": float(seed_optima_pct),
        "seeded_particles_requested_max": int(max_seed_slots),
        "seeded_particles_count": int(n_seed),
        "seeded_particles": seeded_particles,
        "seeded_particle_reached_gbest": bool(seeded_particle_reached_gbest),
        "best_max_disp": float(final_eval.get("max_disp", np.nan)),
        "best_max_stress": float(final_eval.get("max_stress", np.nan)),
        "objective_history": np.asarray(obj_hist),
        "mass_history": np.asarray(mass_hist),
        "cv_history": np.asarray(cv_hist),
        "feasible_fraction_history": np.asarray(feas_frac_hist),
        "final_feasible_fraction": float(feas_frac_hist[-1]),
    }
