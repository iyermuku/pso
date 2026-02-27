"""
Simple Particle Swarm Optimization for the multimodal test function
from Xin-She Yang, *Engineering Optimization*, Chapter 15.

Equation 15.12 (maximization version):
    f(x) = (\sum_{i=1}^d x_i) * exp\left[-\sum_{i=1}^d \sin(x_i^2)\right],
    with -2\pi \le x_i \le 2\pi.

The global **minimum** of this function is 0 at x=0 (all dimensions), but
this script uses PSO to **maximize** the same expression.  The code is
written for arbitrary dimension `n_dim` but the `main` block demonstrates
it in 2 dimensions as requested by the user.

This module lives inside the `MultimodalFunctionTest` folder and is
independent of the truss-pso code already present elsewhere in the
repository.
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


def multimodal_15_12(x: np.ndarray) -> float:
    """Evaluate equation 15.12 of Yang (multimodal test function).

    Parameters
    ----------
    x : np.ndarray
        1-D array of length d representing a point in the search space.

    Returns
    -------
    float
        The value of the objective function (to be maximized).
    """
    x = np.asarray(x, dtype=float)
    # use sum of absolute values rather than plain sum
    s1 = np.sum(np.abs(x))
    s2 = np.sum(np.sin(x**2))
    return s1 * np.exp(-s2)


def pso_maximize(
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
    """Run a simple PSO to maximize ``func`` over a hyperrectangle.

    Parameters
    ----------
    func
        Objective function, takes a 1-D array of length ``n_dim`` and returns
        a scalar. PSO will attempt to maximize this value.
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
        evolution.  The returned history dict will have keys
        ``'X_history'`` and ``'gbest_history'``.

    Returns
    -------
    Tuple[np.ndarray, float, dict | None]
        ``(best_position, best_value, history)``; ``history`` is ``None``
        unless ``track_history`` is True.
    """
    """Run a simple PSO to maximize ``func`` over a hyperrectangle.

    Parameters
    ----------
    func
        Objective function, takes a 1-D array of length ``n_dim`` and returns
        a scalar. PSO will attempt to maximize this value.
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

    Returns
    -------
    Tuple[np.ndarray, float]
        ``(best_position, best_value)`` found by the swarm.
    """
    rng = np.random.default_rng(seed)

    # convert bounds to arrays
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo

    # initialize particle positions using Latin Hypercube Sampling
    S01 = _lhs(swarm_size, n_dim, rng)
    X = lo + S01 * span
    V = rng.uniform(-span, span, size=(swarm_size, n_dim))

    # personal bests
    pbest_X = X.copy()
    pbest_val = np.array([func(x) for x in X])

    # global best
    idx = int(np.argmax(pbest_val))
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

        # update personal bests
        better = vals > pbest_val
        pbest_X[better] = X[better]
        pbest_val[better] = vals[better]

        # update global best
        i_max = int(np.argmax(pbest_val))
        if pbest_val[i_max] > gbest_val:
            gbest_val = float(pbest_val[i_max])
            gbest_X = pbest_X[i_max].copy()

        if track_history:
            history["X_history"][k, :, :] = X.copy()
            history["gbest_history"][k] = gbest_val
            history["gbest_X_history"][k, :] = gbest_X.copy()

    return gbest_X, gbest_val, history


if __name__ == "__main__":
    # demonstration: optimize in 2 dimensions
    n = 2
    bounds = [(-2 * np.pi, 2 * np.pi)] * n
    best_pos, best_val = pso_maximize(
        multimodal_15_12, bounds, n_dim=n, swarm_size=100, iters=500, seed=42
    )
    print("2D PSO result for equation 15.12")
    print("best position:", best_pos)
    print("best value:", best_val)
