from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable
import sys

import numpy as np

from landscape_core import LandscapeProblem


ROOT = Path(__file__).resolve().parents[1]


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


def make_10bar_continuous() -> LandscapeProblem:
    base = ROOT / "PSO10BarTruss" / "v10-good"
    truss = _load_module("truss10_cont", base / "truss_model.py")

    state = {"avg_m": None, "avg_g": None}

    def g_vector(u: np.ndarray) -> np.ndarray:
        disp_viol = np.maximum(0.0, np.abs(u) - truss.U_ALLOW)
        sigma = truss.member_stresses(u)
        stress_viol = np.maximum(0.0, np.abs(sigma) - truss.S_ALLOW)
        return np.concatenate([disp_viol, stress_viol])

    def calibrate(n_ref: int, seed: int) -> None:
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

    def evaluate(x: np.ndarray):
        a = np.clip(np.asarray(x, dtype=float), truss.Amin, truss.Amax)
        u = truss.solve_displacements(a)
        g = g_vector(u)
        m = float(truss.mass_from_A(a))
        if np.all(g <= 1e-12):
            j = m
        else:
            avg_g = state["avg_g"]
            avg_m = state["avg_m"]
            denom = np.sum(avg_g**2)
            if denom <= 1e-16:
                j = m + 1e3 * float(np.sum(g))
            else:
                k = abs(avg_m) * (avg_g / denom)
                j = float(m + np.dot(k, g))
        cv = float(np.sum(np.maximum(g, 0.0)))
        return j, m, cv

    lo = np.full(10, truss.Amin, dtype=float)
    hi = np.full(10, truss.Amax, dtype=float)
    return LandscapeProblem(
        problem_id="truss10_continuous",
        label="10-Bar Truss (Continuous)",
        lo=lo,
        hi=hi,
        evaluate=evaluate,
        calibrate=calibrate,
    )


def make_10bar_discrete() -> LandscapeProblem:
    base = ROOT / "PSO10BarDiscreeteSectionTruss" / "v1"
    truss = _load_module("truss10_disc", base / "truss_model.py")

    state = {"avg_m": None, "avg_g": None}

    def g_vector(u: np.ndarray) -> np.ndarray:
        disp_viol = np.maximum(0.0, np.abs(u) - truss.U_ALLOW)
        sigma = truss.member_stresses(u)
        stress_viol = np.maximum(0.0, np.abs(sigma) - truss.S_ALLOW)
        return np.concatenate([disp_viol, stress_viol])

    def calibrate(n_ref: int, seed: int) -> None:
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

    def evaluate(x: np.ndarray):
        a = np.clip(np.asarray(x, dtype=float), truss.Amin, truss.Amax)
        a = _snap_to_available(a, truss.available_A)
        u = truss.solve_displacements(a)
        g = g_vector(u)
        m = float(truss.mass_from_A(a))
        if np.all(g <= 1e-12):
            j = m
        else:
            avg_g = state["avg_g"]
            avg_m = state["avg_m"]
            denom = np.sum(avg_g**2)
            if denom <= 1e-16:
                j = m + 1e3 * float(np.sum(g))
            else:
                k = abs(avg_m) * (avg_g / denom)
                j = float(m + np.dot(k, g))
        cv = float(np.sum(np.maximum(g, 0.0)))
        return j, m, cv

    lo = np.full(10, truss.Amin, dtype=float)
    hi = np.full(10, truss.Amax, dtype=float)
    return LandscapeProblem(
        problem_id="truss10_discrete",
        label="10-Bar Truss (Discrete Sections)",
        lo=lo,
        hi=hi,
        evaluate=evaluate,
        calibrate=calibrate,
    )


def _make_72_eval(truss_mod, snap: bool = False) -> Callable[[np.ndarray], tuple[float, float, float]]:
    a_min = float(truss_mod.A_MIN)
    a_max = float(truss_mod.A_MAX)
    u_allow = float(truss_mod.U_ALLOW)
    s_allow = float(truss_mod.S_ALLOW)
    avail = np.asarray(getattr(truss_mod, "available_A", []), dtype=float)

    def evaluate(x: np.ndarray):
        a16 = np.clip(np.asarray(x, dtype=float), a_min, a_max)
        if snap:
            if avail.size == 0:
                raise ValueError("Discrete 72-bar adapter requested, but available_A is missing.")
            a16 = _snap_to_available(a16, avail)
        try:
            res = truss_mod.evaluate(a16)
        except Exception:
            # Ill-conditioned/singular FE state -> treat as heavily infeasible.
            return 1.0e12, 1.0e12, 1.0e6
        mass = float(res["mass"])

        max_disp = 0.0
        for u in res["U"]:
            for nid in [1, 2, 3, 4]:
                ux = abs(u[3 * (nid - 1) + 0])
                uy = abs(u[3 * (nid - 1) + 1])
                max_disp = max(max_disp, ux, uy)
        disp_violation = max(0.0, max_disp - u_allow)

        a_members = truss_mod.areas_from_groups(a16)
        max_stress = 0.0
        for u in res["U"]:
            sig = truss_mod.member_stresses(u, a_members)
            max_stress = max(max_stress, float(np.max(np.abs(sig))))
        stress_violation = max(0.0, max_stress - s_allow)

        disp_norm = disp_violation / (u_allow + 1e-12)
        stress_norm = stress_violation / (s_allow + 1e-12)
        j = float(mass + 1e5 * disp_norm + 1e5 * stress_norm)
        cv = float(disp_violation + stress_violation)
        return j, mass, cv

    return evaluate


def make_72bar_continuous_v2() -> LandscapeProblem:
    base = ROOT / "PSO72BarTruss" / "v2"
    truss = _load_module("truss72_cont", base / "truss72.py")
    lo = np.full(16, float(truss.A_MIN), dtype=float)
    hi = np.full(16, float(truss.A_MAX), dtype=float)
    return LandscapeProblem(
        problem_id="truss72_continuous",
        label="72-Bar Truss (Continuous)",
        lo=lo,
        hi=hi,
        evaluate=_make_72_eval(truss, snap=False),
        calibrate=None,
    )


def make_72bar_discrete() -> LandscapeProblem:
    base = ROOT / "PSO72BarDiscreteSectionsTruss"
    truss = _load_module("truss72_disc", base / "truss72.py")
    lo = np.full(16, float(truss.A_MIN), dtype=float)
    hi = np.full(16, float(truss.A_MAX), dtype=float)
    return LandscapeProblem(
        problem_id="truss72_discrete",
        label="72-Bar Truss (Discrete Sections)",
        lo=lo,
        hi=hi,
        evaluate=_make_72_eval(truss, snap=True),
        calibrate=None,
    )


def make_200bar_continuous() -> LandscapeProblem:
    base = ROOT / "PSO200BarTruss"
    truss = _load_module("truss200_cont", base / "truss200.py")

    state = {"avg_m": None, "avg_g": None}

    def calibrate(n_ref: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x_ref = rng.uniform(truss.A_MIN, truss.A_MAX, size=(n_ref, truss.N_GROUPS))
        m_list = np.zeros(n_ref)
        g_mat = np.zeros((n_ref, len(truss.FREE_DOFS)))
        for i in range(n_ref):
            res = truss.evaluate(x_ref[i])
            g = np.asarray(res["disp_violation"], dtype=float)
            m_list[i] = float(res["mass"])
            g_mat[i] = g
        state["avg_m"] = float(np.mean(m_list))
        state["avg_g"] = np.mean(g_mat, axis=0)

    def evaluate(x: np.ndarray):
        a = np.clip(np.asarray(x, dtype=float), truss.A_MIN, truss.A_MAX)
        res = truss.evaluate(a)
        g = np.asarray(res["disp_violation"], dtype=float)
        mass = float(res["mass"])
        if np.all(g <= 1e-12):
            objective = mass
        else:
            avg_g = state["avg_g"]
            avg_m = state["avg_m"]
            denom = float(np.sum(avg_g**2))
            if denom <= 1e-16:
                objective = mass + 1e5 * float(np.sum(g))
            else:
                penalty = abs(avg_m) * (avg_g / denom)
                objective = float(mass + np.dot(penalty, g))
        cv = float(np.sum(np.maximum(g, 0.0)))
        return objective, mass, cv

    lo, hi = truss.grouped_design_bounds()
    return LandscapeProblem(
        problem_id="truss200_continuous",
        label="200-Bar Planar Truss (Continuous)",
        lo=lo,
        hi=hi,
        evaluate=evaluate,
        calibrate=calibrate,
    )


def get_all_truss_problems() -> list[LandscapeProblem]:
    return [
        make_10bar_continuous(),
        make_10bar_discrete(),
        make_72bar_continuous_v2(),
        make_72bar_discrete(),
        make_200bar_continuous(),
    ]
