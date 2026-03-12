from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LANDSCAPE_RESULTS = ROOT / "TrussLandscapeAnalysis" / "results"


@dataclass
class TrussProblem:
    problem_id: str
    label: str
    lo: np.ndarray
    hi: np.ndarray
    recommended_w: float
    recommended_c1: float
    recommended_c2: float
    recommended_schedule: Dict[str, Any]
    recommended_swarm_size: int
    recommended_iters: int
    evaluate: Callable[[np.ndarray], Dict[str, float]]

    @property
    def dim(self) -> int:
        return len(self.lo)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=(n, self.dim))


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _snap_to_available(a: np.ndarray, available: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    avail = np.asarray(available, dtype=float)
    order = np.argsort(avail)
    avail_sorted = avail[order]
    idx_hi = np.searchsorted(avail_sorted, a, side="left")
    idx_lo = np.clip(idx_hi - 1, 0, avail_sorted.size - 1)
    idx_hi = np.clip(idx_hi, 0, avail_sorted.size - 1)
    dist_lo = np.abs(a - avail_sorted[idx_lo])
    dist_hi = np.abs(a - avail_sorted[idx_hi])
    choose_lo = dist_lo <= dist_hi
    idx = np.where(choose_lo, idx_lo, idx_hi)
    return avail_sorted[idx]


def _build_fallback_schedule(w: float, c1: float, c2: float) -> Dict[str, Any]:
    return {
        "phase_1_explore": {
            "w": float(min(0.75, w + 0.05)),
            "c1": float(max(0.95, c1 - 0.10)),
            "c2": float(min(2.30, c2 + 0.10)),
        },
        "phase_2_refine": {
            "w": float(max(0.52, w - 0.08)),
            "c1": float(min(2.00, c1 + 0.10)),
            "c2": float(max(1.20, c2 - 0.10)),
        },
        "switch_fraction_of_iters": 0.60,
    }


def _load_recommendation(problem_id: str) -> Dict[str, Any]:
    metrics_path = LANDSCAPE_RESULTS / problem_id / f"{problem_id}_landscape_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        pso_rec = payload["pso_recommendation"]
        rec = pso_rec["recommended"]
        schedule = pso_rec.get("schedule")
        if schedule is None:
            schedule = _build_fallback_schedule(float(rec["w"]), float(rec["c1"]), float(rec["c2"]))
        return {
            "w": float(rec["w"]),
            "c1": float(rec["c1"]),
            "c2": float(rec["c2"]),
            "schedule": schedule,
            "swarm_size": int(pso_rec.get("recommended_swarm_size", 60)),
            "iters": int(pso_rec.get("recommended_iters", 250)),
        }

    # Fallback values when landscape metrics JSON is not available.
    fallback = {
        "truss10_continuous": {"w": 0.540, "c1": 1.292, "c2": 1.708, "swarm_size": 140, "iters": 400},
        "truss10_discrete":  {"w": 0.680, "c1": 1.210, "c2": 1.790, "swarm_size": 120, "iters": 350},
        "truss72_continuous": {"w": 0.600, "c1": 1.306, "c2": 1.694, "swarm_size": 200, "iters": 300},
        "truss72_discrete":  {"w": 0.730, "c1": 1.250, "c2": 1.750, "swarm_size": 200, "iters": 350},
    }
    if problem_id not in fallback:
        raise KeyError(f"No recommendation found for {problem_id}")
    base = fallback[problem_id]
    return {
        "w": float(base["w"]),
        "c1": float(base["c1"]),
        "c2": float(base["c2"]),
        "schedule": _build_fallback_schedule(float(base["w"]), float(base["c1"]), float(base["c2"])),
        "swarm_size": base["swarm_size"],
        "iters": base["iters"],
    }


def _make_10bar_continuous() -> TrussProblem:
    base = ROOT / "PSO10BarTruss" / "v10-good"
    truss = _load_module("pso_fea_truss10_cont_model", base / "truss_model.py")
    rec = _load_recommendation("truss10_continuous")

    state = {"avg_m": None, "avg_g": None}

    def g_vector(u: np.ndarray) -> np.ndarray:
        disp_viol = np.maximum(0.0, np.abs(u) - truss.U_ALLOW)
        sigma = truss.member_stresses(u)
        stress_viol = np.maximum(0.0, np.abs(sigma) - truss.S_ALLOW)
        return np.concatenate([disp_viol, stress_viol])

    def calibrate(n_ref: int = 400, seed: int = 2026) -> None:
        rng = np.random.default_rng(seed)
        x_ref = rng.uniform(truss.Amin, truss.Amax, size=(n_ref, 10))
        m_list = np.zeros(n_ref)
        g_mat = np.zeros((n_ref, truss.ndof + 10))
        for i in range(n_ref):
            u = truss.solve_displacements(x_ref[i])
            g = g_vector(u)
            m_list[i] = truss.mass_from_A(x_ref[i])
            g_mat[i] = g
        state["avg_m"] = float(np.mean(m_list))
        state["avg_g"] = np.mean(g_mat, axis=0)

    def evaluate(x: np.ndarray) -> Dict[str, float]:
        if state["avg_m"] is None:
            calibrate()
        a = np.clip(np.asarray(x, dtype=float), truss.Amin, truss.Amax)
        u = truss.solve_displacements(a)
        sigma = truss.member_stresses(u)
        g = g_vector(u)
        m = float(truss.mass_from_A(a))
        max_disp = float(np.max(np.abs(u)))
        max_stress = float(np.max(np.abs(sigma)))
        if np.all(g <= 1e-12):
            j = m
        else:
            avg_g = state["avg_g"]
            avg_m = state["avg_m"]
            denom = np.sum(avg_g ** 2)
            if denom <= 1e-16:
                j = m + 1e3 * float(np.sum(g))
            else:
                k = abs(avg_m) * (avg_g / denom)
                j = float(m + np.dot(k, g))
        cv = float(np.sum(np.maximum(g, 0.0)))
        return {
            "objective": float(j),
            "mass": float(m),
            "constraint_violation": float(cv),
            "feasible": float(cv <= 1e-9),
            "max_disp": max_disp,
            "max_stress": max_stress,
            "x_eval": a.copy(),
        }

    lo = np.full(10, truss.Amin, dtype=float)
    hi = np.full(10, truss.Amax, dtype=float)
    return TrussProblem(
        problem_id="truss10_continuous",
        label="10-Bar Truss (Continuous)",
        lo=lo,
        hi=hi,
        recommended_w=rec["w"],
        recommended_c1=rec["c1"],
        recommended_c2=rec["c2"],
        recommended_schedule=rec["schedule"],
        recommended_swarm_size=rec["swarm_size"],
        recommended_iters=rec["iters"],
        evaluate=evaluate,
    )


def _make_10bar_discrete() -> TrussProblem:
    base = ROOT / "PSO10BarDiscreeteSectionTruss" / "v1"
    truss = _load_module("pso_fea_truss10_disc_model", base / "truss_model.py")
    rec = _load_recommendation("truss10_discrete")

    state = {"avg_m": None, "avg_g": None}

    def g_vector(u: np.ndarray) -> np.ndarray:
        disp_viol = np.maximum(0.0, np.abs(u) - truss.U_ALLOW)
        sigma = truss.member_stresses(u)
        stress_viol = np.maximum(0.0, np.abs(sigma) - truss.S_ALLOW)
        return np.concatenate([disp_viol, stress_viol])

    def calibrate(n_ref: int = 400, seed: int = 2026) -> None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(truss.Amin, truss.Amax, size=(n_ref, 10))
        x_ref = np.array([_snap_to_available(v, truss.available_A) for v in raw])
        m_list = np.zeros(n_ref)
        g_mat = np.zeros((n_ref, truss.ndof + 10))
        for i in range(n_ref):
            u = truss.solve_displacements(x_ref[i])
            g = g_vector(u)
            m_list[i] = truss.mass_from_A(x_ref[i])
            g_mat[i] = g
        state["avg_m"] = float(np.mean(m_list))
        state["avg_g"] = np.mean(g_mat, axis=0)

    def evaluate(x: np.ndarray) -> Dict[str, float]:
        if state["avg_m"] is None:
            calibrate()
        a = np.clip(np.asarray(x, dtype=float), truss.Amin, truss.Amax)
        a = _snap_to_available(a, truss.available_A)
        u = truss.solve_displacements(a)
        sigma = truss.member_stresses(u)
        g = g_vector(u)
        m = float(truss.mass_from_A(a))
        max_disp = float(np.max(np.abs(u)))
        max_stress = float(np.max(np.abs(sigma)))
        if np.all(g <= 1e-12):
            j = m
        else:
            avg_g = state["avg_g"]
            avg_m = state["avg_m"]
            denom = np.sum(avg_g ** 2)
            if denom <= 1e-16:
                j = m + 1e3 * float(np.sum(g))
            else:
                k = abs(avg_m) * (avg_g / denom)
                j = float(m + np.dot(k, g))
        cv = float(np.sum(np.maximum(g, 0.0)))
        return {
            "objective": float(j),
            "mass": float(m),
            "constraint_violation": float(cv),
            "feasible": float(cv <= 1e-9),
            "max_disp": max_disp,
            "max_stress": max_stress,
            "x_eval": a.copy(),
        }

    lo = np.full(10, truss.Amin, dtype=float)
    hi = np.full(10, truss.Amax, dtype=float)
    return TrussProblem(
        problem_id="truss10_discrete",
        label="10-Bar Truss (Discrete Sections)",
        lo=lo,
        hi=hi,
        recommended_w=rec["w"],
        recommended_c1=rec["c1"],
        recommended_c2=rec["c2"],
        recommended_schedule=rec["schedule"],
        recommended_swarm_size=rec["swarm_size"],
        recommended_iters=rec["iters"],
        evaluate=evaluate,
    )


def _make_72_evaluator(truss_mod, rec: Dict[str, float], problem_id: str, label: str, snap: bool) -> TrussProblem:
    a_min = float(truss_mod.A_MIN)
    a_max = float(truss_mod.A_MAX)
    u_allow = float(truss_mod.U_ALLOW)
    s_allow = float(truss_mod.S_ALLOW)
    available = np.asarray(getattr(truss_mod, "available_A", []), dtype=float)

    def evaluate(x: np.ndarray) -> Dict[str, float]:
        a16 = np.clip(np.asarray(x, dtype=float), a_min, a_max)
        if snap:
            a16 = _snap_to_available(a16, available)
        try:
            res = truss_mod.evaluate(a16)
        except Exception:
            return {
                "objective": 1.0e12,
                "mass": 1.0e12,
                "constraint_violation": 1.0e6,
                "feasible": 0.0,
                "max_disp": float("inf"),
                "max_stress": float("inf"),
                "x_eval": a16.copy(),
            }

        mass = float(res["mass"])
        max_disp = 0.0
        for u in res["U"]:
            for nid in [1, 2, 3, 4]:
                ux = abs(u[3 * (nid - 1) + 0])
                uy = abs(u[3 * (nid - 1) + 1])
                max_disp = max(max_disp, ux, uy)

        a_members = truss_mod.areas_from_groups(a16)
        max_stress = 0.0
        for u in res["U"]:
            sig = truss_mod.member_stresses(u, a_members)
            max_stress = max(max_stress, float(np.max(np.abs(sig))))

        disp_violation = max(0.0, max_disp - u_allow)
        stress_violation = max(0.0, max_stress - s_allow)
        disp_norm = disp_violation / (u_allow + 1e-12)
        stress_norm = stress_violation / (s_allow + 1e-12)
        j = float(mass + 1e5 * disp_norm + 1e5 * stress_norm)
        cv = float(disp_violation + stress_violation)

        return {
            "objective": float(j),
            "mass": float(mass),
            "constraint_violation": float(cv),
            "feasible": float(cv <= 1e-9),
            "max_disp": float(max_disp),
            "max_stress": float(max_stress),
            "x_eval": a16.copy(),
        }

    lo = np.full(16, a_min, dtype=float)
    hi = np.full(16, a_max, dtype=float)
    return TrussProblem(
        problem_id=problem_id,
        label=label,
        lo=lo,
        hi=hi,
        recommended_w=rec["w"],
        recommended_c1=rec["c1"],
        recommended_c2=rec["c2"],
        recommended_schedule=rec["schedule"],
        recommended_swarm_size=rec["swarm_size"],
        recommended_iters=rec["iters"],
        evaluate=evaluate,
    )


def _make_72bar_continuous() -> TrussProblem:
    base = ROOT / "PSO72BarTruss" / "v2"
    truss = _load_module("pso_fea_truss72_cont", base / "truss72.py")
    rec = _load_recommendation("truss72_continuous")
    return _make_72_evaluator(truss, rec, "truss72_continuous", "72-Bar Truss (Continuous)", snap=False)


def _make_72bar_discrete() -> TrussProblem:
    base = ROOT / "PSO72BarDiscreteSectionsTruss"
    truss = _load_module("pso_fea_truss72_disc", base / "truss72.py")
    rec = _load_recommendation("truss72_discrete")
    return _make_72_evaluator(truss, rec, "truss72_discrete", "72-Bar Truss (Discrete Sections)", snap=True)


def get_problem(problem_id: str) -> TrussProblem:
    builders = {
        "truss10_continuous": _make_10bar_continuous,
        "truss10_discrete": _make_10bar_discrete,
        "truss72_continuous": _make_72bar_continuous,
        "truss72_discrete": _make_72bar_discrete,
    }
    if problem_id not in builders:
        raise KeyError(f"Unknown problem_id: {problem_id}")
    return builders[problem_id]()


def list_problem_ids() -> list[str]:
    return [
        "truss10_continuous",
        "truss10_discrete",
        "truss72_continuous",
        "truss72_discrete",
    ]
