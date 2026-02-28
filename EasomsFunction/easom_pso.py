"""
PSO implementation for Easom's function minimization
Includes scaling to handle different axis ranges
"""

import numpy as np
from typing import Callable, Tuple, Sequence, Optional


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator):
    """Latin Hypercube Sampling in [0,1]^n_dim; shuffle per dimension."""
    U = rng.random((n_samples, n_dim))
    grid = (np.arange(n_samples)[:, None] + U) / n_samples
    samples = np.zeros_like(grid)
    for j in range(n_dim):
        samples[:, j] = rng.permutation(grid[:, j])
    return samples


def easom(x, y):
    """
    Easom's function.
    
    f(x,y) = -cos(x) * cos(y/100) * exp(-(x-π)² - (y/(100π) - 1)²)
    
    Global minimum at (π, 100π) ≈ (3.14159, 314.159) with f ≈ -0.9995
    """
    return -np.cos(x) * np.cos(y / 100) * np.exp(-(x - np.pi)**2 - (y / (100 * np.pi) - 1)**2)


def easom_vec(X):
    """Vectorized Easom function for array of positions."""
    if X.ndim == 1:
        return easom(X[0], X[1])
    else:
        return np.array([easom(x[0], x[1]) for x in X])


def pso_minimize(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    n_dim: int,
    swarm_size: int = 50,
    iters: int = 300,
    inertia: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: Optional[int] = None,
    track_history: bool = False,
) -> Tuple[np.ndarray, float, Optional[dict]]:
    """
    Run PSO to minimize func over a hyperrectangle with scaling.
    
    Parameters
    ----------
    func : Callable
        Objective function to minimize (takes 1-D array, returns scalar)
    bounds : Sequence[Tuple[float, float]]
        Bounds for each dimension [(lo, hi), ...]
    n_dim : int
        Number of dimensions
    swarm_size : int
        Number of particles
    iters : int
        Number of iterations
    inertia : float
        Inertia weight
    c1 : float
        Cognitive coefficient
    c2 : float
        Social coefficient
    seed : int, optional
        Random seed
    track_history : bool
        If True, track particle positions and gbest history
    
    Returns
    -------
    best_pos : np.ndarray
        Best position found
    best_val : float
        Best function value found
    history : dict or None
        If track_history=True, contains 'X_history', 'gbest_history', 'gbest_X_history'
    """
    rng = np.random.default_rng(seed)
    
    # Convert bounds to arrays
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo
    
    # Initialize with Latin Hypercube Sampling
    S01 = _lhs(swarm_size, n_dim, rng)
    X = lo + S01 * span
    
    # Initialize velocities in scaled space (fraction of domain span)
    V = rng.uniform(-0.1, 0.1, size=(swarm_size, n_dim)) * span
    
    # Personal bests
    pbest_X = X.copy()
    pbest_val = np.array([func(x) for x in X])
    
    # Global best
    idx = int(np.argmin(pbest_val))
    gbest_X = pbest_X[idx].copy()
    gbest_val = pbest_val[idx]
    
    # History tracking
    if track_history:
        X_history = [X.copy()]
        gbest_history = [gbest_val]
        gbest_X_history = [gbest_X.copy()]
    
    # Main PSO loop
    for iteration in range(iters):
        # Random coefficients
        r1 = rng.random((swarm_size, n_dim))
        r2 = rng.random((swarm_size, n_dim))
        
        # Update velocities
        V = (inertia * V + 
             c1 * r1 * (pbest_X - X) + 
             c2 * r2 * (gbest_X - X))
        
        # Update positions
        X = X + V
        
        # Enforce bounds with reflection
        for d in range(n_dim):
            # Reflect particles that go out of bounds
            mask_lo = X[:, d] < lo[d]
            X[mask_lo, d] = lo[d] + (lo[d] - X[mask_lo, d])
            V[mask_lo, d] *= -1
            
            mask_hi = X[:, d] > hi[d]
            X[mask_hi, d] = hi[d] - (X[mask_hi, d] - hi[d])
            V[mask_hi, d] *= -1
            
            # Final clip to ensure bounds
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])
        
        # Evaluate new positions
        vals = np.array([func(x) for x in X])
        
        # Update personal bests
        improved = vals < pbest_val
        pbest_val[improved] = vals[improved]
        pbest_X[improved] = X[improved]
        
        # Update global best
        min_idx = np.argmin(pbest_val)
        if pbest_val[min_idx] < gbest_val:
            gbest_val = pbest_val[min_idx]
            gbest_X = pbest_X[min_idx].copy()
        
        # Track history
        if track_history:
            X_history.append(X.copy())
            gbest_history.append(gbest_val)
            gbest_X_history.append(gbest_X.copy())
    
    history = None
    if track_history:
        history = {
            'X_history': np.array(X_history),
            'gbest_history': np.array(gbest_history),
            'gbest_X_history': np.array(gbest_X_history),
        }
    
    return gbest_X, gbest_val, history


if __name__ == "__main__":
    # Quick test
    bounds = [(0, 2*np.pi), (0, 200*np.pi)]
    best_pos, best_val, _ = pso_minimize(easom_vec, bounds, n_dim=2, swarm_size=50, iters=300, seed=42)
    print(f"Best position: {best_pos}")
    print(f"Best value: {best_val}")
    print(f"Expected: x ≈ {np.pi:.6f}, y ≈ {100*np.pi:.6f}, f ≈ -0.9995")
