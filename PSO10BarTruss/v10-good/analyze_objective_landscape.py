"""
Objective landscape analysis for the 10-bar truss sizing problem.

Computes:
- Autocorrelation length
- Information content
- Local Optima Network (LON) structure
- Basins of attraction mapping (2D start-slice projection)
- Smoothness and narrow-basin diagnostics

Outputs:
- landscape_report.txt
- landscape_metrics.json
- landscape_autocorr.png
- landscape_information_content.png
- landscape_basins_map.png
- landscape_lon_basins.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from constraints import constraint_vector
from objectives import penalized_objective
from truss_model import Amin, Amax, mass_from_A, solve_displacements


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + u) / n_samples
    samples = np.zeros_like(grid)
    for dim in range(n_dim):
        samples[:, dim] = rng.permutation(grid[:, dim])
    return samples


def _clip_reflect(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = x.copy()
    below = y < lo
    above = y > hi
    y[below] = lo[below] + (lo[below] - y[below])
    y[above] = hi[above] - (y[above] - hi[above])
    return np.clip(y, lo, hi)


@dataclass
class LandscapeObjective:
    lo: np.ndarray
    hi: np.ndarray
    avg_m: float
    avg_g: np.ndarray

    @staticmethod
    def fit_reference(seed: int = 2026, n_ref: int = 800) -> "LandscapeObjective":
        rng = np.random.default_rng(seed)
        lo = np.full(10, Amin, dtype=float)
        hi = np.full(10, Amax, dtype=float)
        s01 = _lhs(n_ref, 10, rng)
        x_ref = lo + s01 * (hi - lo)

        m_list = np.zeros(n_ref)
        g_list = []
        for i in range(n_ref):
            u = solve_displacements(x_ref[i])
            g = constraint_vector(u)
            m_list[i] = mass_from_A(x_ref[i])
            g_list.append(g)
        g_mat = np.vstack(g_list)
        return LandscapeObjective(lo=lo, hi=hi, avg_m=float(np.mean(m_list)), avg_g=np.mean(g_mat, axis=0))

    def evaluate(self, x: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        u = solve_displacements(x)
        g = constraint_vector(u)
        m = float(mass_from_A(x))
        j = float(penalized_objective(m, g, self.avg_m, self.avg_g))
        cv = float(np.sum(np.maximum(g, 0.0)))
        return j, m, cv, g

    def evaluate_many(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = x.shape[0]
        j = np.zeros(n)
        m = np.zeros(n)
        cv = np.zeros(n)
        for i in range(n):
            j[i], m[i], cv[i], _ = self.evaluate(x[i])
        return j, m, cv


def random_walk_series(
    obj: LandscapeObjective,
    rng: np.random.Generator,
    n_steps: int,
    step_frac: float,
) -> np.ndarray:
    x = rng.uniform(obj.lo, obj.hi)
    span = obj.hi - obj.lo
    vals = np.zeros(n_steps)
    vals[0] = obj.evaluate(x)[0]
    sigma = step_frac * span
    for i in range(1, n_steps):
        dx = rng.normal(0.0, sigma)
        x = _clip_reflect(x + dx, obj.lo, obj.hi)
        vals[i] = obj.evaluate(x)[0]
    return vals


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    y = x - np.mean(x)
    var = np.var(y)
    if var < 1e-14:
        return np.ones(max_lag + 1)
    ac = np.zeros(max_lag + 1)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = np.mean(y[:-lag] * y[lag:]) / var
    return ac


def autocorrelation_length(ac: np.ndarray) -> float:
    r1 = float(np.clip(abs(ac[1]), 1e-8, 0.999999))
    return float(-1.0 / np.log(r1))


def information_content(series: np.ndarray, eps_values: np.ndarray) -> Dict[str, np.ndarray]:
    d = np.diff(series)
    s = np.std(d)
    if s < 1e-14:
        s = 1.0
    dn = d / s

    h_vals = np.zeros_like(eps_values)
    m_vals = np.zeros_like(eps_values)

    for k, eps in enumerate(eps_values):
        sym = np.zeros_like(dn, dtype=int)
        sym[dn > eps] = 1
        sym[dn < -eps] = -1

        pairs = np.stack([sym[:-1], sym[1:]], axis=1)
        bins = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        counts = np.array([np.sum((pairs[:, 0] == a) & (pairs[:, 1] == b)) for a, b in bins], dtype=float)
        p = counts / max(np.sum(counts), 1.0)
        nz = p > 0
        h = -np.sum(p[nz] * np.log2(p[nz]))
        h_vals[k] = h / np.log2(9.0)

        non_zero = sym[sym != 0]
        if len(non_zero) < 2:
            m_vals[k] = 0.0
        else:
            changes = np.sum(non_zero[:-1] != non_zero[1:])
            m_vals[k] = changes / (len(non_zero) - 1)

    return {"eps": eps_values, "H": h_vals, "M": m_vals}


def local_descent(
    obj: LandscapeObjective,
    x0: np.ndarray,
    max_iters: int = 40,
    init_step_frac: float = 0.12,
    min_step_frac: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    x = x0.copy()
    f, _, _, _ = obj.evaluate(x)
    span = obj.hi - obj.lo
    step = init_step_frac * span
    min_step = min_step_frac * span

    for _ in range(max_iters):
        improved = False
        best_x = x
        best_f = f
        for d in range(len(x)):
            for sgn in (-1.0, 1.0):
                cand = x.copy()
                cand[d] = np.clip(cand[d] + sgn * step[d], obj.lo[d], obj.hi[d])
                f_c, _, _, _ = obj.evaluate(cand)
                if f_c < best_f - 1e-10:
                    best_f = f_c
                    best_x = cand
                    improved = True
        if improved:
            x = best_x
            f = best_f
        else:
            step *= 0.5
            if np.all(step <= min_step):
                break
    return x, f


def _normalized_distance(x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    z = (x - y) / (hi - lo)
    return float(np.linalg.norm(z))


def cluster_minima(
    minima: List[np.ndarray],
    values: List[float],
    lo: np.ndarray,
    hi: np.ndarray,
    dist_tol: float = 0.05,
    f_tol: float = 1e-3,
) -> Tuple[np.ndarray, List[float], np.ndarray]:
    centers: List[np.ndarray] = []
    center_vals: List[float] = []
    labels = np.zeros(len(minima), dtype=int)

    for i, (x, f) in enumerate(zip(minima, values)):
        assigned = False
        for j, c in enumerate(centers):
            d = _normalized_distance(x, c, lo, hi)
            if d <= dist_tol and abs(f - center_vals[j]) <= f_tol:
                labels[i] = j
                assigned = True
                break
        if not assigned:
            labels[i] = len(centers)
            centers.append(x.copy())
            center_vals.append(float(f))

    return np.vstack(centers), center_vals, labels


def lon_structure(
    obj: LandscapeObjective,
    rng: np.random.Generator,
    n_starts: int = 36,
    n_perturb: int = 5,
) -> Dict:
    s01 = _lhs(n_starts, 10, rng)
    starts = obj.lo + s01 * (obj.hi - obj.lo)

    minima = []
    vals = []
    for i in range(n_starts):
        x_min, f_min = local_descent(obj, starts[i], max_iters=35)
        minima.append(x_min)
        vals.append(f_min)

    centers, center_vals, labels = cluster_minima(minima, vals, obj.lo, obj.hi)
    n_nodes = centers.shape[0]

    basin_sizes = np.array([np.sum(labels == i) for i in range(n_nodes)], dtype=int)
    probs = basin_sizes / np.sum(basin_sizes)
    basin_entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

    edges = np.zeros((n_nodes, n_nodes), dtype=int)
    span = obj.hi - obj.lo
    for i in range(n_nodes):
        for _ in range(n_perturb):
            x0 = centers[i] + rng.normal(0.0, 0.06 * span)
            x0 = np.clip(x0, obj.lo, obj.hi)
            x1, f1 = local_descent(obj, x0, max_iters=28)
            d_best = [
                _normalized_distance(x1, centers[j], obj.lo, obj.hi) + 5.0 * abs(f1 - center_vals[j])
                for j in range(n_nodes)
            ]
            j = int(np.argmin(d_best))
            edges[i, j] += 1

    edge_presence = (edges > 0).astype(int)
    np.fill_diagonal(edge_presence, 0)
    possible = max(n_nodes * (n_nodes - 1), 1)
    edge_density = float(np.sum(edge_presence) / possible)

    return {
        "centers": centers,
        "center_vals": np.array(center_vals),
        "labels": labels,
        "basin_sizes": basin_sizes,
        "edges": edges,
        "n_nodes": int(n_nodes),
        "edge_density": edge_density,
        "basin_entropy": basin_entropy,
        "starts": starts,
        "start_labels": labels,
    }


def sensitivity_dims(obj: LandscapeObjective, x_ref: np.ndarray) -> Tuple[int, int, np.ndarray]:
    span = obj.hi - obj.lo
    grad_mag = np.zeros(10)
    f0, _, _, _ = obj.evaluate(x_ref)
    for d in range(10):
        h = 0.01 * span[d]
        xp = x_ref.copy()
        xm = x_ref.copy()
        xp[d] = np.clip(xp[d] + h, obj.lo[d], obj.hi[d])
        xm[d] = np.clip(xm[d] - h, obj.lo[d], obj.hi[d])
        fp, _, _, _ = obj.evaluate(xp)
        fm, _, _, _ = obj.evaluate(xm)
        grad_mag[d] = abs(fp - fm) / max(2 * h, 1e-12)
    idx = np.argsort(-grad_mag)
    return int(idx[0]), int(idx[1]), grad_mag


def basin_map_2d(
    obj: LandscapeObjective,
    lon: Dict,
    x_anchor: np.ndarray,
    dim_i: int,
    dim_j: int,
    n_grid: int = 26,
) -> Dict:
    ai = np.linspace(obj.lo[dim_i], obj.hi[dim_i], n_grid)
    aj = np.linspace(obj.lo[dim_j], obj.hi[dim_j], n_grid)
    ids = np.zeros((n_grid, n_grid), dtype=int)
    fstart = np.zeros((n_grid, n_grid))

    centers = lon["centers"]
    cvals = lon["center_vals"]

    for r, vi in enumerate(ai):
        for c, vj in enumerate(aj):
            x0 = x_anchor.copy()
            x0[dim_i] = vi
            x0[dim_j] = vj
            f0, _, _, _ = obj.evaluate(x0)
            fstart[r, c] = f0
            xm, fm = local_descent(obj, x0, max_iters=24)
            score = [
                _normalized_distance(xm, centers[k], obj.lo, obj.hi) + 5.0 * abs(fm - cvals[k])
                for k in range(len(cvals))
            ]
            ids[r, c] = int(np.argmin(score))

    return {"grid_i": ai, "grid_j": aj, "ids": ids, "fstart": fstart}


def smoothness_metrics(
    obj: LandscapeObjective,
    rng: np.random.Generator,
    n_pairs: int = 300,
) -> Dict[str, float]:
    span = obj.hi - obj.lo
    slopes = np.zeros(n_pairs)
    for k in range(n_pairs):
        x = rng.uniform(obj.lo, obj.hi)
        y = _clip_reflect(x + rng.normal(0.0, 0.03 * span), obj.lo, obj.hi)
        fx, _, _, _ = obj.evaluate(x)
        fy, _, _, _ = obj.evaluate(y)
        d = _normalized_distance(x, y, obj.lo, obj.hi)
        slopes[k] = abs(fx - fy) / max(d, 1e-10)

    return {
        "slope_median": float(np.median(slopes)),
        "slope_q90": float(np.quantile(slopes, 0.90)),
        "slope_std": float(np.std(slopes)),
    }


def narrow_basin_metrics(
    obj: LandscapeObjective,
    rng: np.random.Generator,
    x_best: np.ndarray,
    f_best: float,
    n_dirs: int = 32,
    rise_abs: float = 100.0,
) -> Dict[str, float]:
    span = obj.hi - obj.lo
    widths = []
    for _ in range(n_dirs):
        d = rng.normal(0.0, 1.0, size=10)
        d = d / max(np.linalg.norm(d), 1e-12)
        hit = 0.5
        for alpha in np.linspace(0.005, 0.5, 40):
            cand = _clip_reflect(x_best + alpha * span * d, obj.lo, obj.hi)
            f_c, _, _, _ = obj.evaluate(cand)
            if f_c - f_best >= rise_abs:
                hit = alpha
                break
        widths.append(hit)

    widths = np.array(widths)
    return {
        "basin_width_mean_norm": float(np.mean(widths)),
        "basin_width_median_norm": float(np.median(widths)),
        "basin_width_q10_norm": float(np.quantile(widths, 0.10)),
        "rise_abs": float(rise_abs),
    }


def classify_landscape(
    ac_len: float,
    info_h: float,
    lon_nodes: int,
    lon_entropy: float,
    lon_density: float,
    smooth: Dict[str, float],
    narrow: Dict[str, float],
) -> Dict[str, str]:
    multimodal_score = 0
    if lon_nodes >= 4:
        multimodal_score += 1
    if lon_entropy >= 1.0:
        multimodal_score += 1
    if info_h >= 0.55:
        multimodal_score += 1
    if lon_density >= 0.25:
        multimodal_score += 1

    smooth_score = 0
    if ac_len >= 6.0:
        smooth_score += 1
    if info_h <= 0.45:
        smooth_score += 1
    if smooth["slope_q90"] <= 6.0 * max(smooth["slope_median"], 1e-9):
        smooth_score += 1

    narrow_score = 0
    if narrow["basin_width_median_norm"] <= 0.10:
        narrow_score += 1
    if narrow["basin_width_q10_norm"] <= 0.04:
        narrow_score += 1
    if smooth["slope_q90"] >= 3.0 * max(smooth["slope_median"], 1e-9):
        narrow_score += 1

    labels = []
    if multimodal_score >= 2:
        labels.append("multimodal")
    if smooth_score >= 2:
        labels.append("smooth-macro")
    if narrow_score >= 2:
        labels.append("narrow-basin")
    if not labels:
        labels.append("mixed/uncertain")

    summary = (
        f"Landscape class: {', '.join(labels)} | "
        f"multimodal_score={multimodal_score}, smooth_score={smooth_score}, narrow_score={narrow_score}"
    )

    return {
        "labels": ", ".join(labels),
        "summary": summary,
        "multimodal_score": str(multimodal_score),
        "smooth_score": str(smooth_score),
        "narrow_score": str(narrow_score),
    }


def recommend_pso_coefficients(
    classes: Dict[str, str],
    ac_len: float,
    info_h: float,
    lon_nodes: int,
    lon_density: float,
    narrow: Dict[str, float],
) -> Dict[str, object]:
    labels = classes["labels"]
    has_multimodal = "multimodal" in labels
    has_narrow = "narrow-basin" in labels
    has_smooth = "smooth-macro" in labels

    # Start from a conservative, commonly stable baseline.
    w = 0.68
    c1 = 1.35
    c2 = 1.55

    rationale = []

    if has_multimodal:
        c2 += 0.15
        c1 -= 0.10
        rationale.append("Multimodality detected (high info content / many attractors): increase social pull to help swarm consensus.")

    if has_narrow:
        w -= 0.08
        c1 += 0.10
        rationale.append("Narrow basin detected (small basin width): reduce inertia and keep enough cognitive pull for local refinement.")

    if has_smooth and not has_narrow:
        w += 0.05
        rationale.append("Smooth macro-landscape: slightly higher inertia supports broader traversal.")

    if lon_nodes >= 25 and lon_density < 0.15:
        c2 += 0.10
        rationale.append("Many weakly connected local optima (sparse LON): favor stronger social attraction to avoid fragmented sub-swarms.")

    if ac_len < 5.0:
        w -= 0.03
        rationale.append("Short autocorrelation length: lower inertia to avoid overshooting in rapidly changing terrain.")

    if info_h > 0.70:
        c2 += 0.05
        rationale.append("High information content: emphasize exploitation pressure after discovery.")

    if narrow["basin_width_median_norm"] < 0.04:
        w -= 0.03
        c1 += 0.05
        rationale.append("Very narrow basin estimate: further damp momentum and increase personal-best guidance.")

    # Practical bounds and stability-oriented projection.
    w = float(np.clip(w, 0.50, 0.78))
    c1 = float(np.clip(c1, 0.90, 2.20))
    c2 = float(np.clip(c2, 1.10, 2.40))
    csum = c1 + c2
    if csum < 2.0:
        scale = 2.0 / csum
        c1 *= scale
        c2 *= scale
    elif csum > 3.0:
        scale = 3.0 / csum
        c1 *= scale
        c2 *= scale

    # Optional robust alternative using constriction-style coefficients.
    robust_alt = {
        "w": 0.7298,
        "c1": 2.05,
        "c2": 2.05,
        "note": "Constriction-style robust setting (strong exploration/exploitation), useful when coefficient self-adaptation is disabled.",
    }

    # Suggested schedule for this landscape type.
    schedule = {
        "phase_1_explore": {"w": min(0.75, w + 0.05), "c1": max(0.95, c1 - 0.10), "c2": min(2.30, c2 + 0.10)},
        "phase_2_refine": {"w": max(0.52, w - 0.08), "c1": min(2.00, c1 + 0.10), "c2": max(1.20, c2 - 0.10)},
        "switch_fraction_of_iters": 0.60,
    }

    return {
        "recommended": {"w": round(w, 3), "c1": round(c1, 3), "c2": round(c2, 3)},
        "recommended_sum_c1_c2": round(float(c1 + c2), 3),
        "schedule": schedule,
        "robust_alternative": robust_alt,
        "rationale": rationale,
    }


def make_plots(
    out_dir: Path,
    ac: np.ndarray,
    info: Dict[str, np.ndarray],
    lon: Dict,
    basins: Dict,
    dim_i: int,
    dim_j: int,
) -> None:
    lags = np.arange(len(ac))
    plt.figure(figsize=(7, 4.5))
    plt.plot(lags, ac, linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Objective Random-Walk Autocorrelation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "landscape_autocorr.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(info["eps"], info["H"], label="Information content H(eps)", linewidth=2)
    plt.plot(info["eps"], info["M"], label="Partial info M(eps)", linewidth=2)
    plt.xlabel("epsilon")
    plt.ylabel("Normalized value")
    plt.title("Information Content Metrics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "landscape_information_content.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    ids = basins["ids"]
    plt.imshow(ids, origin="lower", aspect="auto", interpolation="nearest", cmap="tab20")
    plt.colorbar(label="Attractor ID")
    plt.xlabel(f"A[{dim_j + 1}] grid index")
    plt.ylabel(f"A[{dim_i + 1}] grid index")
    plt.title("Basins of Attraction Map (2D start slice)")
    plt.tight_layout()
    plt.savefig(out_dir / "landscape_basins_map.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    basin_sizes = lon["basin_sizes"]
    vals = lon["center_vals"]
    idx = np.argsort(vals)
    plt.bar(np.arange(len(idx)), basin_sizes[idx])
    plt.xlabel("Attractor rank (best to worst objective)")
    plt.ylabel("Basin size (count of starts)")
    plt.title("LON Node Basin Sizes")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "landscape_lon_basins.png", dpi=160)
    plt.close()


def build_report(
    metrics: Dict,
    out_path: Path,
) -> None:
    lines = []
    lines.append("10-Bar Truss Objective Landscape Report")
    lines.append("=" * 44)
    lines.append("")
    lines.append("Primary multimodality diagnostics")
    lines.append("-" * 34)
    lines.append(f"Autocorrelation length: {metrics['autocorrelation_length']:.3f}")
    lines.append(
        f"Information content (H at eps=0.05): {metrics['information_content_H_eps005']:.3f}"
    )
    lines.append(
        f"Partial information (M at eps=0.05): {metrics['information_content_M_eps005']:.3f}"
    )
    lines.append(f"LON node count (local attractors): {metrics['lon_nodes']}")
    lines.append(f"LON edge density: {metrics['lon_edge_density']:.3f}")
    lines.append(f"LON basin entropy: {metrics['lon_basin_entropy']:.3f}")
    lines.append("")
    lines.append("Smoothness diagnostics")
    lines.append("-" * 21)
    lines.append(f"Local slope median: {metrics['slope_median']:.3f}")
    lines.append(f"Local slope q90: {metrics['slope_q90']:.3f}")
    lines.append(f"Local slope std: {metrics['slope_std']:.3f}")
    lines.append("")
    lines.append("Narrow-basin diagnostics")
    lines.append("-" * 23)
    lines.append(
        f"Basin width median (normalized, rise={metrics['rise_abs']:.1f}): {metrics['basin_width_median_norm']:.4f}"
    )
    lines.append(f"Basin width q10 (normalized): {metrics['basin_width_q10_norm']:.4f}")
    lines.append("")
    lines.append("Classification")
    lines.append("-" * 14)
    lines.append(metrics["classification_summary"])
    lines.append("")
    lines.append("PSO Coefficient Recommendation")
    lines.append("-" * 30)
    rec = metrics["pso_recommendation"]
    lines.append(
        "Recommended fixed coefficients: "
        f"w={rec['recommended']['w']:.3f}, c1={rec['recommended']['c1']:.3f}, c2={rec['recommended']['c2']:.3f} "
        f"(c1+c2={rec['recommended_sum_c1_c2']:.3f})"
    )
    lines.append("Suggested 2-phase schedule:")
    lines.append(
        f"  Phase 1 (0-{int(100*rec['schedule']['switch_fraction_of_iters'])}% iters): "
        f"w={rec['schedule']['phase_1_explore']['w']:.3f}, "
        f"c1={rec['schedule']['phase_1_explore']['c1']:.3f}, "
        f"c2={rec['schedule']['phase_1_explore']['c2']:.3f}"
    )
    lines.append(
        f"  Phase 2 ({int(100*rec['schedule']['switch_fraction_of_iters'])}-100% iters): "
        f"w={rec['schedule']['phase_2_refine']['w']:.3f}, "
        f"c1={rec['schedule']['phase_2_refine']['c1']:.3f}, "
        f"c2={rec['schedule']['phase_2_refine']['c2']:.3f}"
    )
    lines.append(
        "Robust alternative: "
        f"w={rec['robust_alternative']['w']:.4f}, "
        f"c1={rec['robust_alternative']['c1']:.2f}, "
        f"c2={rec['robust_alternative']['c2']:.2f}"
    )
    lines.append("Reasoning:")
    for reason in rec["rationale"]:
        lines.append(f"  - {reason}")
    lines.append("")
    lines.append("Generated figures")
    lines.append("-" * 17)
    lines.append("- landscape_autocorr.png")
    lines.append("- landscape_information_content.png")
    lines.append("- landscape_basins_map.png")
    lines.append("- landscape_lon_basins.png")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 10-bar truss objective landscape")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-ref", type=int, default=800)
    parser.add_argument("--walk-steps", type=int, default=420)
    parser.add_argument("--walk-step-frac", type=float, default=0.03)
    parser.add_argument("--lon-starts", type=int, default=36)
    parser.add_argument("--lon-perturb", type=int, default=5)
    parser.add_argument("--basin-grid", type=int, default=26)
    parser.add_argument("--out-dir", type=str, default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    obj = LandscapeObjective.fit_reference(seed=args.seed, n_ref=args.n_ref)

    walk = random_walk_series(obj, rng, n_steps=args.walk_steps, step_frac=args.walk_step_frac)
    max_lag = min(60, len(walk) // 4)
    ac = autocorrelation(walk, max_lag=max_lag)
    ac_len = autocorrelation_length(ac)

    eps_values = np.linspace(0.01, 0.20, 20)
    info = information_content(walk, eps_values)
    eps_idx = int(np.argmin(np.abs(info["eps"] - 0.05)))
    h05 = float(info["H"][eps_idx])
    m05 = float(info["M"][eps_idx])

    lon = lon_structure(obj, rng, n_starts=args.lon_starts, n_perturb=args.lon_perturb)
    best_idx = int(np.argmin(lon["center_vals"]))
    x_best = lon["centers"][best_idx].copy()
    f_best = float(lon["center_vals"][best_idx])

    dim_i, dim_j, grad_mag = sensitivity_dims(obj, x_best)
    basins = basin_map_2d(
        obj,
        lon,
        x_anchor=x_best,
        dim_i=dim_i,
        dim_j=dim_j,
        n_grid=args.basin_grid,
    )

    smooth = smoothness_metrics(obj, rng)
    narrow = narrow_basin_metrics(obj, rng, x_best=x_best, f_best=f_best, n_dirs=32, rise_abs=100.0)

    classes = classify_landscape(
        ac_len=ac_len,
        info_h=h05,
        lon_nodes=lon["n_nodes"],
        lon_entropy=lon["basin_entropy"],
        lon_density=lon["edge_density"],
        smooth=smooth,
        narrow=narrow,
    )

    recommendation = recommend_pso_coefficients(
        classes=classes,
        ac_len=ac_len,
        info_h=h05,
        lon_nodes=lon["n_nodes"],
        lon_density=lon["edge_density"],
        narrow=narrow,
    )

    make_plots(out_dir, ac, info, lon, basins, dim_i=dim_i, dim_j=dim_j)

    metrics = {
        "seed": args.seed,
        "n_ref": args.n_ref,
        "walk_steps": args.walk_steps,
        "walk_step_frac": args.walk_step_frac,
        "autocorrelation_length": ac_len,
        "information_content_H_eps005": h05,
        "information_content_M_eps005": m05,
        "lon_nodes": int(lon["n_nodes"]),
        "lon_edge_density": float(lon["edge_density"]),
        "lon_basin_entropy": float(lon["basin_entropy"]),
        "best_local_objective": f_best,
        "best_local_design": x_best.tolist(),
        "top_sensitivity_dims": [int(dim_i), int(dim_j)],
        "top_sensitivity_magnitude": [float(grad_mag[dim_i]), float(grad_mag[dim_j])],
        "slope_median": smooth["slope_median"],
        "slope_q90": smooth["slope_q90"],
        "slope_std": smooth["slope_std"],
        "basin_width_mean_norm": narrow["basin_width_mean_norm"],
        "basin_width_median_norm": narrow["basin_width_median_norm"],
        "basin_width_q10_norm": narrow["basin_width_q10_norm"],
        "rise_abs": narrow["rise_abs"],
        "classification_labels": classes["labels"],
        "classification_summary": classes["summary"],
        "classification_multimodal_score": int(classes["multimodal_score"]),
        "classification_smooth_score": int(classes["smooth_score"]),
        "classification_narrow_score": int(classes["narrow_score"]),
        "pso_recommendation": recommendation,
    }

    (out_dir / "landscape_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    build_report(metrics, out_dir / "landscape_report.txt")

    print("Landscape analysis complete")
    print(f"Classification: {classes['labels']}")
    print(f"Autocorrelation length: {ac_len:.3f}")
    print(f"LON nodes: {lon['n_nodes']}, edge density: {lon['edge_density']:.3f}")
    print(f"Information content H(eps=0.05): {h05:.3f}")
    print(f"Basin width median (norm): {narrow['basin_width_median_norm']:.4f}")
    print(
        "Recommended PSO coeffs: "
        f"w={recommendation['recommended']['w']:.3f}, "
        f"c1={recommendation['recommended']['c1']:.3f}, "
        f"c2={recommendation['recommended']['c2']:.3f}"
    )
    print(f"Sensitivity slice dims: A[{dim_i + 1}], A[{dim_j + 1}]")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
