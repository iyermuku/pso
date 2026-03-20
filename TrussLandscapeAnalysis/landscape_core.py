from __future__ import annotations

import concurrent.futures as cf
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LandscapeProblem:
    problem_id: str
    label: str
    lo: np.ndarray
    hi: np.ndarray
    evaluate: Callable[[np.ndarray], Tuple[float, float, float]]
    calibrate: Callable[[int, int], None] | None = None

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n uniform samples within the problem bounds."""
        return rng.uniform(self.lo, self.hi, size=(n, len(self.lo)))


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin hypercube sample matrix in [0, 1]^n_dim."""
    u = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + u) / n_samples
    samples = np.zeros_like(grid)
    for dim in range(n_dim):
        samples[:, dim] = rng.permutation(grid[:, dim])
    return samples


def _clip_reflect(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Reflect values crossing bounds, then clip to enforce feasibility."""
    y = x.copy()
    below = y < lo
    above = y > hi
    y[below] = lo[below] + (lo[below] - y[below])
    y[above] = hi[above] - (y[above] - hi[above])
    return np.clip(y, lo, hi)


def evaluate_many(problem: LandscapeProblem, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate objective, mass, and constraint violation for each design in x."""
    n = x.shape[0]
    j = np.zeros(n)
    m = np.zeros(n)
    cv = np.zeros(n)
    for i in range(n):
        j[i], m[i], cv[i] = problem.evaluate(x[i])
    return j, m, cv


def random_walk_series(
    problem: LandscapeProblem,
    rng: np.random.Generator,
    n_steps: int,
    step_frac: float,
) -> np.ndarray:
    """Sample objective values along a bounded Gaussian random walk."""
    x = rng.uniform(problem.lo, problem.hi)
    span = problem.hi - problem.lo
    vals = np.zeros(n_steps)
    vals[0] = problem.evaluate(x)[0]
    sigma = step_frac * span
    for i in range(1, n_steps):
        dx = rng.normal(0.0, sigma)
        x = _clip_reflect(x + dx, problem.lo, problem.hi)
        vals[i] = problem.evaluate(x)[0]
    return vals


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized autocorrelation up to max_lag for a 1D series."""
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
    """Estimate correlation length from lag-1 autocorrelation magnitude."""
    r1 = float(np.clip(abs(ac[1]), 1e-8, 0.999999))
    return float(-1.0 / np.log(r1))


def information_content(series: np.ndarray, eps_values: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute information-content descriptors H(eps) and M(eps) for a series."""
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

        # Encode transition pairs (a,b) in {-1,0,1}^2 to bins 0..8 for fast counting.
        pair_codes = (sym[:-1] + 1) * 3 + (sym[1:] + 1)
        counts = np.bincount(pair_codes, minlength=9).astype(float)
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
    problem: LandscapeProblem,
    x0: np.ndarray,
    max_iters: int = 40,
    init_step_frac: float = 0.12,
    min_step_frac: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """Run coordinate-wise local descent with adaptive step halving."""
    x = x0.copy()
    f, _, _ = problem.evaluate(x)
    span = problem.hi - problem.lo
    step = init_step_frac * span
    min_step = min_step_frac * span

    for _ in range(max_iters):
        improved = False
        best_x = x
        best_f = f
        for d in range(len(x)):
            for sgn in (-1.0, 1.0):
                cand = x.copy()
                cand[d] = np.clip(cand[d] + sgn * step[d], problem.lo[d], problem.hi[d])
                f_c, _, _ = problem.evaluate(cand)
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
    """Return Euclidean distance after scaling each dimension to [0, 1] span."""
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
    """Cluster local minima into attractors using distance and objective tolerances."""
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
    problem: LandscapeProblem,
    rng: np.random.Generator,
    n_starts: int = 36,
    n_perturb: int = 5,
    n_threads: int = 1,
) -> Dict:
    """Build a local-optima-network approximation and basin statistics."""
    s01 = _lhs(n_starts, len(problem.lo), rng)
    starts = problem.lo + s01 * (problem.hi - problem.lo)

    minima: list[np.ndarray] = [None] * n_starts  # type: ignore[assignment]
    vals: list[float] = [0.0] * n_starts

    def _descent_start(i: int) -> tuple[int, np.ndarray, float]:
        x_min, f_min = local_descent(problem, starts[i], max_iters=35)
        return i, x_min, f_min

    if n_threads > 1 and n_starts > 1:
        with cf.ThreadPoolExecutor(max_workers=n_threads) as pool:
            for i, x_min, f_min in pool.map(_descent_start, range(n_starts)):
                minima[i] = x_min
                vals[i] = f_min
    else:
        for i in range(n_starts):
            _, x_min, f_min = _descent_start(i)
            minima[i] = x_min
            vals[i] = f_min

    centers, center_vals, labels = cluster_minima(minima, vals, problem.lo, problem.hi)
    n_nodes = centers.shape[0]

    basin_sizes = np.array([np.sum(labels == i) for i in range(n_nodes)], dtype=int)
    probs = basin_sizes / np.sum(basin_sizes)
    basin_entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

    edges = np.zeros((n_nodes, n_nodes), dtype=int)
    span = problem.hi - problem.lo

    inv_span = 1.0 / (problem.hi - problem.lo)

    x0_items: list[tuple[int, np.ndarray]] = []
    for i in range(n_nodes):
        for _ in range(n_perturb):
            x0 = centers[i] + rng.normal(0.0, 0.06 * span)
            x0 = np.clip(x0, problem.lo, problem.hi)
            x0_items.append((i, x0))

    center_vals_arr = np.asarray(center_vals, dtype=float)

    def _descent_edge(item: tuple[int, np.ndarray]) -> tuple[int, int]:
        i, x0 = item
        x1, f1 = local_descent(problem, x0, max_iters=28)
        z = (centers - x1) * inv_span
        d_best = np.linalg.norm(z, axis=1) + 5.0 * np.abs(center_vals_arr - f1)
        j = int(np.argmin(d_best))
        return i, j

    if n_threads > 1 and len(x0_items) > 1:
        with cf.ThreadPoolExecutor(max_workers=n_threads) as pool:
            for i, j in pool.map(_descent_edge, x0_items):
                edges[i, j] += 1
    else:
        for item in x0_items:
            i, j = _descent_edge(item)
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
    }


def sensitivity_dims(problem: LandscapeProblem, x_ref: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """Estimate per-dimension sensitivity via central finite differences."""
    span = problem.hi - problem.lo
    grad_mag = np.zeros(len(x_ref))
    for d in range(len(x_ref)):
        h = 0.01 * span[d]
        xp = x_ref.copy()
        xm = x_ref.copy()
        xp[d] = np.clip(xp[d] + h, problem.lo[d], problem.hi[d])
        xm[d] = np.clip(xm[d] - h, problem.lo[d], problem.hi[d])
        fp, _, _ = problem.evaluate(xp)
        fm, _, _ = problem.evaluate(xm)
        grad_mag[d] = abs(fp - fm) / max(2 * h, 1e-12)
    idx = np.argsort(-grad_mag)
    return int(idx[0]), int(idx[1]), grad_mag


def basin_map_2d(
    problem: LandscapeProblem,
    lon: Dict,
    x_anchor: np.ndarray,
    dim_i: int,
    dim_j: int,
    n_grid: int = 26,
    n_threads: int = 1,
) -> Dict:
    """Map attractor IDs on a 2D slice spanned by two sensitive dimensions."""
    ai = np.linspace(problem.lo[dim_i], problem.hi[dim_i], n_grid)
    aj = np.linspace(problem.lo[dim_j], problem.hi[dim_j], n_grid)
    ids = np.zeros((n_grid, n_grid), dtype=int)

    centers = lon["centers"]
    cvals = np.asarray(lon["center_vals"], dtype=float)
    inv_span = 1.0 / (problem.hi - problem.lo)

    rc_pairs = [(r, c) for r in range(n_grid) for c in range(n_grid)]

    def _basin_cell(rc: tuple[int, int]) -> tuple[int, int, int]:
        r, c = rc
        vi = ai[r]
        vj = aj[c]
        x0 = x_anchor.copy()
        x0[dim_i] = vi
        x0[dim_j] = vj
        xm, fm = local_descent(problem, x0, max_iters=24)
        z = (centers - xm) * inv_span
        score = np.linalg.norm(z, axis=1) + 5.0 * np.abs(cvals - fm)
        return r, c, int(np.argmin(score))

    if n_threads > 1 and len(rc_pairs) > 1:
        with cf.ThreadPoolExecutor(max_workers=n_threads) as pool:
            for r, c, val in pool.map(_basin_cell, rc_pairs):
                ids[r, c] = val
    else:
        for rc in rc_pairs:
            r, c, val = _basin_cell(rc)
            ids[r, c] = val

    return {"grid_i": ai, "grid_j": aj, "ids": ids}


def smoothness_metrics(problem: LandscapeProblem, rng: np.random.Generator, n_pairs: int = 300) -> Dict[str, float]:
    """Estimate landscape smoothness from local slope distribution statistics."""
    span = problem.hi - problem.lo
    slopes = np.zeros(n_pairs)
    for k in range(n_pairs):
        x = rng.uniform(problem.lo, problem.hi)
        y = _clip_reflect(x + rng.normal(0.0, 0.03 * span), problem.lo, problem.hi)
        fx, _, _ = problem.evaluate(x)
        fy, _, _ = problem.evaluate(y)
        d = _normalized_distance(x, y, problem.lo, problem.hi)
        slopes[k] = abs(fx - fy) / max(d, 1e-10)

    return {
        "slope_median": float(np.median(slopes)),
        "slope_q90": float(np.quantile(slopes, 0.90)),
        "slope_std": float(np.std(slopes)),
    }


def narrow_basin_metrics(
    problem: LandscapeProblem,
    rng: np.random.Generator,
    x_best: np.ndarray,
    f_best: float,
    n_dirs: int = 32,
    rise_abs: float = 100.0,
) -> Dict[str, float]:
    """Estimate basin narrowness by radial distance to a target objective rise."""
    span = problem.hi - problem.lo
    widths = []
    alpha_grid = np.linspace(0.005, 0.5, 40)
    for _ in range(n_dirs):
        d = rng.normal(0.0, 1.0, size=len(span))
        d = d / max(np.linalg.norm(d), 1e-12)
        hit = 0.5
        for alpha in alpha_grid:
            cand = _clip_reflect(x_best + alpha * span * d, problem.lo, problem.hi)
            f_c, _, _ = problem.evaluate(cand)
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
) -> Dict[str, object]:
    """Assign qualitative landscape classes using heuristic metric thresholds."""
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

    labels: List[str] = []
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
        "labels": labels,
        "summary": summary,
        "multimodal_score": multimodal_score,
        "smooth_score": smooth_score,
        "narrow_score": narrow_score,
    }


def recommend_pso_coefficients(
    classes: Dict[str, object],
    ac_len: float,
    info_h: float,
    lon_nodes: int,
    lon_density: float,
    narrow: Dict[str, float],
    n_dim: int = 10,
) -> Dict[str, object]:
    """Produce PSO coefficient, swarm-size, and iteration recommendations."""
    labels = classes["labels"]
    has_multimodal = "multimodal" in labels
    has_narrow = "narrow-basin" in labels
    has_smooth = "smooth-macro" in labels

    multimodal_score = int(classes["multimodal_score"])
    narrow_score = int(classes["narrow_score"])

    w = 0.68
    c1 = 1.35
    c2 = 1.55
    rationale = []

    if has_multimodal:
        c2 += 0.15
        c1 -= 0.10
        rationale.append("Multimodality: raise social pull for swarm consensus across local basins.")

    if has_narrow:
        w -= 0.08
        c1 += 0.10
        rationale.append("Narrow basin: reduce inertia and retain cognitive pull for local refinement.")
    if has_smooth and not has_narrow:
        w += 0.05
        rationale.append("Smooth macro-landscape: slightly higher inertia supports broader traversal.")
    if lon_nodes >= 25 and lon_density < 0.15:
        c2 += 0.10
        rationale.append("Many weakly connected attractors: increase social attraction to reduce swarm fragmentation.")
    if ac_len < 5.0:
        w -= 0.03
        rationale.append("Short autocorrelation length: lower inertia to avoid overshoot.")
    if info_h > 0.70:
        c2 += 0.05
        rationale.append("High information content: increase exploitation pressure after discovery.")
    if narrow["basin_width_median_norm"] < 0.04:
        w -= 0.03
        c1 += 0.05
        rationale.append("Very narrow basin estimate: damp momentum and increase pbest guidance.")

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

    # ---- Swarm size recommendation ----------------------------------------
    # Base: 10 × dimension; scale up for multimodal/narrow landscapes.
    swarm_base = 10 * n_dim
    swarm_size = swarm_base + 5 * multimodal_score + 5 * narrow_score
    if lon_nodes >= 20:
        swarm_size += 10  # extra diversity needed for many local optima
    swarm_size = int(np.clip(swarm_size, 30, 150))
    rationale.append(
        f"Swarm size {swarm_size}: 10×dim={swarm_base} base"
        + (f" +{5 * multimodal_score} multimodal" if multimodal_score else "")
        + (f" +{5 * narrow_score} narrow" if narrow_score else "")
        + (f" +10 LON" if lon_nodes >= 20 else "")
        + ", clipped to [30, 150]."
    )

    # ---- Iteration count recommendation -----------------------------------
    # Base 200; increase for highly multimodal or narrow landscapes.
    iters_base = 200
    n_iters = iters_base + 20 * multimodal_score + 30 * narrow_score
    if lon_nodes >= 30:
        n_iters += 50  # many attractors need longer search
    if ac_len < 3.0:
        n_iters += 50  # very rugged: more iterations to escape local traps
    n_iters = int(np.clip(n_iters, 150, 600))
    rationale.append(
        f"Iterations {n_iters}: {iters_base} base"
        + (f" +{20 * multimodal_score} multimodal" if multimodal_score else "")
        + (f" +{30 * narrow_score} narrow" if narrow_score else "")
        + (f" +50 LON" if lon_nodes >= 30 else "")
        + (f" +50 ruggedness" if ac_len < 3.0 else "")
        + ", clipped to [150, 600]."
    )

    return {
        "recommended": {"w": round(w, 3), "c1": round(c1, 3), "c2": round(c2, 3)},
        "recommended_sum_c1_c2": round(float(c1 + c2), 3),
        "recommended_swarm_size": swarm_size,
        "recommended_iters": n_iters,
        "schedule": {
            "phase_1_explore": {"w": min(0.75, w + 0.05), "c1": max(0.95, c1 - 0.10), "c2": min(2.30, c2 + 0.10)},
            "phase_2_refine": {"w": max(0.52, w - 0.08), "c1": min(2.00, c1 + 0.10), "c2": max(1.20, c2 - 0.10)},
            "switch_fraction_of_iters": 0.60,
        },
        "rationale": rationale,
    }


def _plot_problem_outputs(
    out_dir: Path,
    prefix: str,
    ac: np.ndarray,
    info: Dict[str, np.ndarray],
    basin_ids: np.ndarray,
    lon_basin_sizes: np.ndarray,
    lon_center_vals: np.ndarray,
) -> None:
    """Generate and save diagnostic plots for autocorrelation, info content, and basins."""
    lags = np.arange(len(ac))
    plt.figure(figsize=(7, 4.5))
    plt.plot(lags, ac, linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(f"{prefix}: Random-Walk Autocorrelation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_autocorr.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(info["eps"], info["H"], label="H(eps)", linewidth=2)
    plt.plot(info["eps"], info["M"], label="M(eps)", linewidth=2)
    plt.xlabel("epsilon")
    plt.ylabel("Normalized value")
    plt.title(f"{prefix}: Information Content")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_information_content.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.imshow(basin_ids, origin="lower", aspect="auto", interpolation="nearest", cmap="tab20")
    plt.colorbar(label="Attractor ID")
    plt.title(f"{prefix}: Basins of Attraction (2D slice)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_basins_map.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    idx = np.argsort(lon_center_vals)
    plt.bar(np.arange(len(idx)), lon_basin_sizes[idx])
    plt.xlabel("Attractor rank (best to worst)")
    plt.ylabel("Basin size")
    plt.title(f"{prefix}: LON Basin Sizes")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_lon_basins.png", dpi=160)
    plt.close()


def analyze_problem(
    problem: LandscapeProblem,
    out_dir: Path,
    seed: int = 2026,
    n_ref: int = 600,
    walk_steps: int = 360,
    walk_step_frac: float = 0.03,
    lon_starts: int = 30,
    lon_perturb: int = 4,
    basin_grid: int = 24,
    n_threads: int = 1,
) -> Dict[str, object]:
    """Run full landscape analysis, save outputs, and return computed metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if problem.calibrate is not None:
        problem.calibrate(n_ref, seed)

    cache: Dict[bytes, Tuple[float, float, float]] = {}
    cache_hits = 0
    cache_misses = 0
    base_evaluate = problem.evaluate

    def _evaluate_cached(x: np.ndarray) -> Tuple[float, float, float]:
        # Byte-keyed memoization preserves exact values while avoiding repeated FE evaluations.
        nonlocal cache_hits, cache_misses
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=float))
        key = x_arr.tobytes()
        val = cache.get(key)
        if val is None:
            cache_misses += 1
            val = base_evaluate(x_arr)
            cache[key] = val
        else:
            cache_hits += 1
        return val

    problem = replace(problem, evaluate=_evaluate_cached)

    rng = np.random.default_rng(seed)

    walk = random_walk_series(problem, rng, n_steps=walk_steps, step_frac=walk_step_frac)
    max_lag = min(60, len(walk) // 4)
    ac = autocorrelation(walk, max_lag=max_lag)
    ac_len = autocorrelation_length(ac)

    eps_values = np.linspace(0.01, 0.20, 20)
    info = information_content(walk, eps_values)
    eps_idx = int(np.argmin(np.abs(info["eps"] - 0.05)))
    h05 = float(info["H"][eps_idx])
    m05 = float(info["M"][eps_idx])

    lon = lon_structure(problem, rng, n_starts=lon_starts, n_perturb=lon_perturb, n_threads=n_threads)
    best_idx = int(np.argmin(lon["center_vals"]))
    x_best = lon["centers"][best_idx].copy()
    f_best = float(lon["center_vals"][best_idx])

    dim_i, dim_j, grad_mag = sensitivity_dims(problem, x_best)
    basin_map = basin_map_2d(
        problem,
        lon,
        x_anchor=x_best,
        dim_i=dim_i,
        dim_j=dim_j,
        n_grid=basin_grid,
        n_threads=n_threads,
    )

    smooth = smoothness_metrics(problem, rng)
    narrow = narrow_basin_metrics(problem, rng, x_best=x_best, f_best=f_best, n_dirs=28, rise_abs=100.0)

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
        n_dim=problem.lo.size,
    )

    center_vals = np.asarray(lon["center_vals"], dtype=float)
    center_order = np.argsort(center_vals)
    detected_optima = [
        {
            "rank": int(rank + 1),
            "objective": float(center_vals[idx]),
            "design_variables": np.asarray(lon["centers"][idx], dtype=float).tolist(),
            "basin_size": int(lon["basin_sizes"][idx]),
        }
        for rank, idx in enumerate(center_order)
    ]

    metrics: Dict[str, object] = {
        "problem_id": problem.problem_id,
        "label": problem.label,
        "seed": seed,
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
        "classification_labels": ", ".join(classes["labels"]),
        "classification_summary": classes["summary"],
        "classification_multimodal_score": int(classes["multimodal_score"]),
        "classification_smooth_score": int(classes["smooth_score"]),
        "classification_narrow_score": int(classes["narrow_score"]),
        "detected_optima": detected_optima,
        "pso_recommendation": recommendation,
        "analysis_threads": int(n_threads),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "cache_total_queries": int(cache_hits + cache_misses),
        "cache_unique_evals": int(len(cache)),
        "cache_hit_rate": float(cache_hits / max(cache_hits + cache_misses, 1)),
    }

    prefix = problem.problem_id
    _plot_problem_outputs(
        out_dir=out_dir,
        prefix=prefix,
        ac=ac,
        info=info,
        basin_ids=basin_map["ids"],
        lon_basin_sizes=lon["basin_sizes"],
        lon_center_vals=lon["center_vals"],
    )

    (out_dir / f"{prefix}_landscape_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_lines = [
        f"{problem.label} Landscape Report",
        "=" * (len(problem.label) + 17),
        "",
        f"Classification: {metrics['classification_labels']}",
        f"Autocorrelation length: {metrics['autocorrelation_length']:.3f}",
        f"Information content H(eps=0.05): {metrics['information_content_H_eps005']:.3f}",
        f"LON nodes: {metrics['lon_nodes']}",
        f"LON edge density: {metrics['lon_edge_density']:.3f}",
        f"Basin width median (norm): {metrics['basin_width_median_norm']:.4f}",
        "",
        "Recommended PSO coefficients:",
        (
            f"w={recommendation['recommended']['w']:.3f}, "
            f"c1={recommendation['recommended']['c1']:.3f}, "
            f"c2={recommendation['recommended']['c2']:.3f}"
        ),
        f"Recommended swarm size: {recommendation['recommended_swarm_size']}",
        f"Recommended iterations: {recommendation['recommended_iters']}",
        f"Detected local optima (LON centers): {len(detected_optima)}",
        "",
        "Reasoning:",
    ]
    report_lines.extend([f"- {r}" for r in recommendation["rationale"]])
    report_lines.append("")
    report_lines.append("Generated plots:")
    report_lines.append(f"- {prefix}_autocorr.png")
    report_lines.append(f"- {prefix}_information_content.png")
    report_lines.append(f"- {prefix}_basins_map.png")
    report_lines.append(f"- {prefix}_lon_basins.png")

    (out_dir / f"{prefix}_landscape_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    return metrics
