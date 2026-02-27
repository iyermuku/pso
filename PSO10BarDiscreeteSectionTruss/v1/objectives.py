"""
Objective utilities for the PSO.
- penalized_objective implements a parameter-less adaptive penalty.
"""
import numpy as np


def penalized_objective(m: float, g: np.ndarray, avg_m: float, avg_g: np.ndarray) -> float:
    """Parameter-less adaptive penalty: f'(x)=f(x) if feasible;
    else f'(x)=f(x)+sum k_i g_i(x), where k_i = |\bar{f}| * \bar{g}_i / sum(\bar{g}^2).
    """
    if np.all(g <= 1e-12):
        return m
    denom = np.sum(avg_g**2)
    if denom <= 1e-16:
        return m + 1e3 * float(np.sum(g))
    k = abs(avg_m) * (avg_g / denom)
    return float(m + np.dot(k, g))
