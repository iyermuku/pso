"""
Constraint handling utilities.
- project_params: enforce PSO parameter feasibility.
- constraint_vector: displacement constraint violations.
"""
import numpy as np
from truss_model import U_ALLOW, S_ALLOW, member_stresses


def project_params(w: float, c1: float, c2: float, eps: float = 1e-6):
    c1 = max(0.0, float(c1))
    c2 = max(0.0, float(c2))
    csum = c1 + c2
    """
    first try the basic constraint 0 < c1+c2 < 4
    if (csum < eps) or (csum > 4.0 - eps) :
        c1 = 2.0
        c2 = 2.0
        csum = 4.0
    """
    """Project (w,c1,c2) onto the feasible region:
    2 <= c1+c2 <= 3 and (c1+c2)/2 - 1 < w < 1. Also enforce c1,c2 >= 0.
    We scale (c1,c2) to meet sum bounds while preserving their ratio.
    """
    if csum < 2.0:
        scale = 2.0 / max(csum, eps)
        c1 *= scale
        c2 *= scale
        csum = c1 + c2
    elif csum > 3.0:
        scale = 3.0 / csum
        c1 *= scale
        c2 *= scale
        csum = c1 + c2
    """
     R.E.Perez,K.Behdinan/ComputersandStructures85(2007) pg 1584
     keep c1c2 between 1.4 and 2
    if csum < 2.8 + eps:
        scale = (2.8 + eps) / max(csum, eps)
        c1 *= scale
        c2 *= scale
        csum = c1 + c2
    elif csum > 4.0 - eps:
        scale = (4.0 - eps) / csum
        c1 *= scale
        c2 *= scale
        csum = c1 + c2
    """
    # Now bound w
    w_lb = csum / 2.0 - 1.0 + eps
    w_ub = 1.0 - eps
    w = min(max(float(w), w_lb), w_ub)
    return w, c1, c2


def constraint_vector(U: np.ndarray, u_allow: float = U_ALLOW) -> np.ndarray:
    """
    Return concatenated constraint violations:
    - displacement violations per DOF: max(0, |U| - U_ALLOW)
    - stress violations per member:    max(0, |sigma| - S_ALLOW)
    """
    disp_viol = np.maximum(0.0, np.abs(U) - u_allow)
    sigma = member_stresses(U)  # ksi
    stress_viol = np.maximum(0.0, np.abs(sigma) - S_ALLOW)
    return np.concatenate([disp_viol, stress_viol])
