"""
Particle Swarm Optimization for the Michaelewicz Function.

The Michaelewicz function is a highly multimodal benchmark for optimization:

    f(x) = - \sum_{i=1}^d \sin(x_i) * \sin^{2m}(i * x_i^2 / \pi),
    with 0 <= x_i <= pi and m typically chosen as 10.

The global minimum lies inside the search space with value approximately
-1.8013 (for 2-D with m=10).

This module provides a generic PSO minimizer that can handle arbitrary
dimension, though we demonstrate and solve for 2 dimensions as requested.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Sequence


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator):
    """Latin Hypercube Sampling in [0,1]^n_dim; shuffle per dimension."""
    U = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + U) / n_samples
    samples = np.zeros_like(grid)
    for j in range(n_dim):
        samples[:, j] = rng.permutation(grid[:, j])
    return samples


def michaelewicz(x: np.ndarray, m: float = 10.0) -> float:
    """Evaluate the Michaelewicz function.

    Parameters
    ----------
    x : np.ndarray
        1-D array of length d representing a point in the search space.
    m : float
        Shape parameter (typically 10).

    Returns
    -------
    float
        The value of the objective function (to be minimized).
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    s = 0.0
    for i in range(d):
        s += np.sin(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi) ** (2 * m))
    return -s  # negative because we want to minimize


def pso_minimize(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    n_dim: int,
    swarm_size: int = 30,
    iters: int = 100,
    inertia: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int | None = None,
    track_history: bool = False,
) -> Tuple[np.ndarray, float, dict | None]:
    """Run a simple PSO to minimize ``func`` over a hyperrectangle.

    Parameters
    ----------
    func
        Objective function, takes a 1-D array of length ``n_dim`` and returns
        a scalar. PSO will attempt to minimize this value.
    bounds
        Sequence of (lower, upper) pairs for each dimension.  ``len(bounds)``
        must equal ``n_dim``.
    n_dim
        Dimensionality of the problem.
    swarm_size
        Number of particles in the swarm.
    iters
        Number of iterations to perform.
    inertia
        Inertia weight for the velocity update.
    c1
        Cognitive acceleration coefficient.
    c2
        Social acceleration coefficient.
    seed
        Optional random seed for reproducibility.
    track_history
        If True, record particle positions each iteration and best-value
        evolution. The returned history dict will have keys
        ``'X_history'``, ``'gbest_history'``, and ``'gbest_X_history'``.

    Returns
    -------
    Tuple[np.ndarray, float, dict | None]
        ``(best_position, best_value, history)``; ``history`` is ``None``
        unless ``track_history`` is True.
    """
    rng = np.random.default_rng(seed)

    # convert bounds to arrays
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo

    # initialize particle positions and velocities using LHS
    S01 = _lhs(swarm_size, n_dim, rng)
    X = lo + S01 * span
    V = rng.uniform(-span, span, size=(swarm_size, n_dim))

    # personal bests
    pbest_X = X.copy()
    pbest_val = np.array([func(x) for x in X])

    # global best
    idx = int(np.argmin(pbest_val))
    gbest_X = pbest_X[idx].copy()
    gbest_val = float(pbest_val[idx])

    # prepare history containers if requested
    history: dict | None = None
    if track_history:
        # record particles and global best (position+value)
        history = {
            "X_history": np.zeros((iters + 1, swarm_size, n_dim)),
            "gbest_history": np.zeros(iters + 1),
            "gbest_X_history": np.zeros((iters + 1, n_dim)),
        }
        history["X_history"][0, :, :] = X.copy()
        history["gbest_history"][0] = gbest_val
        history["gbest_X_history"][0, :] = gbest_X.copy()

    for k in range(1, iters + 1):
        # velocity update
        r1 = rng.random((swarm_size, n_dim))
        r2 = rng.random((swarm_size, n_dim))
        V = (
            inertia * V
            + c1 * r1 * (pbest_X - X)
            + c2 * r2 * (gbest_X - X)
        )
        # optionally clamp velocity to fraction of span
        V = np.clip(V, -span, span)

        # position update with simple boundary handling (reflect)
        X = X + V
        below = X < lo
        above = X > hi
        X = np.where(below, lo + (lo - X), X)
        X = np.where(above, hi - (X - hi), X)
        V[below | above] *= -0.5

        # evaluate
        vals = np.array([func(x) for x in X])

        # update personal bests (for minimization, smaller is better)
        better = vals < pbest_val
        pbest_X[better] = X[better]
        pbest_val[better] = vals[better]

        # update global best
        i_min = int(np.argmin(pbest_val))
        if pbest_val[i_min] < gbest_val:
            gbest_val = float(pbest_val[i_min])
            gbest_X = pbest_X[i_min].copy()

        if track_history:
            history["X_history"][k, :, :] = X.copy()
            history["gbest_history"][k] = gbest_val
            history["gbest_X_history"][k, :] = gbest_X.copy()

    return gbest_X, gbest_val, history


if __name__ == "__main__":
    # demonstration: optimize in 2 dimensions
    n = 2
    m = 10
    bounds = [(0.0, np.pi)] * n
    best_pos, best_val, _ = pso_minimize(
        lambda x: michaelewicz(x, m=m),
        bounds,
        n_dim=n,
        swarm_size=100,
        iters=500,
        seed=42,
    )
    print(f"2D Michaelewicz (m={m}) PSO result:")
    print(f"  best position: {best_pos}")
    print(f"  best value: {best_val}")
