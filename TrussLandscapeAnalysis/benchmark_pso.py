from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + u) / n_samples
    samples = np.zeros_like(grid)
    for dim in range(n_dim):
        samples[:, dim] = rng.permutation(grid[:, dim])
    return samples


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_value: float
    history: dict[str, np.ndarray] | None = None


def pso_minimize(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    swarm_size: int,
    iters: int,
    inertia: float,
    c1: float,
    c2: float,
    seed: int | None = None,
    track_history: bool = False,
) -> PSOResult:
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo
    n_dim = len(bounds)

    x = lo + _lhs(swarm_size, n_dim, rng) * span
    v = rng.uniform(-0.1, 0.1, size=(swarm_size, n_dim)) * span

    pbest_x = x.copy()
    pbest_val = np.array([func(row) for row in x], dtype=float)
    best_idx = int(np.argmin(pbest_val))
    gbest_x = pbest_x[best_idx].copy()
    gbest_val = float(pbest_val[best_idx])

    if track_history:
        x_hist = [x.copy()]
        g_hist = [gbest_val]
        gbest_hist = [gbest_x.copy()]

    for _ in range(iters):
        r1 = rng.random((swarm_size, n_dim))
        r2 = rng.random((swarm_size, n_dim))
        v = inertia * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        x = x + v

        for dim in range(n_dim):
            mask_lo = x[:, dim] < lo[dim]
            if np.any(mask_lo):
                x[mask_lo, dim] = lo[dim] + (lo[dim] - x[mask_lo, dim])
                v[mask_lo, dim] *= -1
            mask_hi = x[:, dim] > hi[dim]
            if np.any(mask_hi):
                x[mask_hi, dim] = hi[dim] - (x[mask_hi, dim] - hi[dim])
                v[mask_hi, dim] *= -1
            x[:, dim] = np.clip(x[:, dim], lo[dim], hi[dim])

        values = np.array([func(row) for row in x], dtype=float)
        improved = values < pbest_val
        if np.any(improved):
            pbest_val[improved] = values[improved]
            pbest_x[improved] = x[improved]

        best_idx = int(np.argmin(pbest_val))
        if float(pbest_val[best_idx]) < gbest_val:
            gbest_val = float(pbest_val[best_idx])
            gbest_x = pbest_x[best_idx].copy()

        if track_history:
            x_hist.append(x.copy())
            g_hist.append(gbest_val)
            gbest_hist.append(gbest_x.copy())

    history = None
    if track_history:
        history = {
            "X_history": np.asarray(x_hist),
            "gbest_history": np.asarray(g_hist),
            "gbest_X_history": np.asarray(gbest_hist),
        }

    return PSOResult(best_position=gbest_x, best_value=gbest_val, history=history)
