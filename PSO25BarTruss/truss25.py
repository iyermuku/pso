"""
25-bar space truss (3D) — classic Schmit & Farshi (1974) / Kamat & Hayduk (1985) benchmark.
geometry and loading as per Perez Behdinan 2006

Geometry & Groups
-----------------
10 nodes (1-indexed):
  Nodes 1–6 : free  (3 DOF each → 18 free DOFs)
  Nodes 7–10: pin supports (fully fixed, all 3 DOFs = 0)

25 members in 8 symmetry groups (double symmetry about x-z and y-z planes):
  A1 : member 1
  A2 : members  2– 5
  A3 : members  6– 9
  A4 : members 10–11
  A5 : members 12–13
  A6 : members 14–17
  A7 : members 18–21
  A8 : members 22–25


Constraints
-----------
  Displacement: |u_i| ≤ 0.35 in   for all free DOFs under every load case
  Stress:       |σ_e| ≤ 40.0 ksi  for all members under every load case

Design Variables
----------------
  8 cross-sectional areas (in²):  A_min = 0.01,  A_max = 3.4

Material
--------
  E   = 10,000 ksi    (elastic modulus)
  ρ   = 0.1   lb/in³  (mass density)

References
----------
  Schmit, L.A., Farshi, B. (1974). Some approximation concepts for structural synthesis.
    AIAA Journal, 12(5), 692–699.
  Kamat, M.P., Hayduk, R.J. (1985). Recent advances in structural optimization.
    Journal of Aircraft, 22(12), 1065–1071.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Material & design constants
# ---------------------------------------------------------------------------
E: float       = 10_000.0  # ksi  (elastic modulus)
RHO: float     = 0.1       # lb/in³ (mass density)
U_ALLOW: float = 0.35      # in  (±0.35 in each global direction for all free DOFs)
S_ALLOW: float = 40.0      # ksi (allowable stress, tension and compression)
A_MIN: float   = 0.01      # in²
A_MAX: float   = 3.4       # in²

N_GROUPS: int = 8
N_NODES: int  = 10
N_ELEMS: int  = 25
N_DOF: int    = 30          # 3 DOF per node × 10 nodes

# As per Perez Behdinan 2006.
# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# Node coordinates in inches: row i → (x, y, z) for node (i+1)
NODES: np.ndarray = np.array(
    [
        [ -37.5,   0.0,  200.0],   # Node 1  (free)
        [ 37.5,   0.0,  200.0],   # Node 2  (free)
        [-37.5,   37.5,  100.0],   # Node 3  (free)
        [ 37.5,   37.5,  100.0],   # Node 4  (free)
        [ 37.5,  -37.5,  100.0],   # Node 5  (free)
        [-37.5,  -37.5,  100.0],   # Node 6  (free)
        [-100.0,  100.0,   0.0],   # Node 7  (pinned)
        [ 100.0,  100.0,   0.0],   # Node 8  (pinned)
        [ 100.0, -100.0,   0.0],   # Node 9  (pinned)
        [-100.0, -100.0,   0.0],   # Node 10 (pinned)
    ],
    dtype=float,
)

FIXED_NODES: List[int] = [7, 8, 9, 10]

# Element connectivity: 25 rows of (node_i, node_j), 1-indexed
ELEMENTS: np.ndarray = np.array(
    [
        [ 1,  2],   #  1 — A1
        [ 1,  4],   #  2 — A2  (direct: same y-half)
        [ 2,  3],   #  3 — A2
        [ 1,  5],   #  4 — A2
        [ 2,  6],   #  5 — A2
        [ 2,  4],   #  6 — A3  (cross: opposite y-half)
        [ 2,  5],   #  7 — A3
        [ 1,  3],   #  8 — A3
        [ 1,  6],   #  9 — A3
        [ 3,  6],   # 10 — A4  (upper-ring, parallel to x-axis)
        [ 4,  5],   # 11 — A4
        [ 3,  4],   # 12 — A5  (upper-ring, parallel to y-axis)
        [ 5,  6],   # 13 — A5
        [ 3,  10],   # 14 — A6  (direct diagonal: same quadrant)
        [ 6,  7],   # 15 — A6
        [ 4,  9],   # 16 — A6
        [ 5, 8],   # 17 — A6
        [ 4,  7],   # 18 — A7  (cross-x diagonal: opposite x, same y-half)
        [ 3,  8],   # 19 — A7
        [ 5, 10],   # 20 — A7
        [ 6,  9],   # 21 — A7
        [ 6,  10],   # 22 — A8  
        [ 3, 7],   # 23 — A8
        [ 4,  8],   # 24 — A8
        [ 5,  9],   # 25 — A8
    ],
    dtype=int,
)

# 8 element groups (1-indexed members)
GROUPS: Dict[int, List[int]] = {
    1: [1],
    2: [2, 3, 4, 5],
    3: [6, 7, 8, 9],
    4: [10, 11],
    5: [12, 13],
    6: [14, 15, 16, 17],
    7: [18, 19, 20, 21],
    8: [22, 23, 24, 25],
}

# ---------------------------------------------------------------------------
# Load cases (forces in kips; positive = positive global axis)
# ---------------------------------------------------------------------------

_LOADS_CASE1: Dict[int, Tuple[float, float, float]] = {
    1: (1.0,  -10.0, -10.0),
    2: (0.0,  -10.0, -10.0),
    3: (0.5,   0.0,  0.0),
    6: (0.5,   0.0,  0.0),
}


def _build_load_vector(loads: Dict[int, Tuple[float, float, float]]) -> np.ndarray:
    F = np.zeros(N_DOF, dtype=float)
    for node_id, (px, py, pz) in loads.items():
        base = 3 * (node_id - 1)
        F[base]     += px
        F[base + 1] += py
        F[base + 2] += pz
    return F


LOAD_VECTORS: List[np.ndarray] = [
    _build_load_vector(_LOADS_CASE1)
]

# ---------------------------------------------------------------------------
# DOF bookkeeping
# ---------------------------------------------------------------------------

def _build_dof_sets() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    fixed: List[int] = []
    for n in FIXED_NODES:
        base = 3 * (n - 1)
        fixed.extend([base, base + 1, base + 2])
    fixed_set = set(fixed)
    free = tuple(d for d in range(N_DOF) if d not in fixed_set)
    return tuple(fixed), free


FIXED_DOFS: Tuple[int, ...]
FREE_DOFS: Tuple[int, ...]
FIXED_DOFS, FREE_DOFS = _build_dof_sets()

FREE_NODES: Tuple[int, ...] = tuple(
    n for n in range(1, N_NODES + 1) if n not in FIXED_NODES
)

# ---------------------------------------------------------------------------
# Pre-computed element data
# ---------------------------------------------------------------------------

def _element_direction(node_i: int, node_j: int) -> Tuple[float, float, float, float]:
    """Return (length, cx, cy, cz) for element from node_i to node_j (1-indexed)."""
    xi, yi, zi = NODES[node_i - 1]
    xj, yj, zj = NODES[node_j - 1]
    dx, dy, dz = xj - xi, yj - yi, zj - zi
    length = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if length < 1e-12:
        raise ValueError(f"Zero-length element between nodes {node_i} and {node_j}")
    return length, dx / length, dy / length, dz / length


ELEMENT_LENGTHS: np.ndarray = np.array(
    [_element_direction(i, j)[0] for i, j in ELEMENTS],
    dtype=float,
)

# ---------------------------------------------------------------------------
# FEA functions
# ---------------------------------------------------------------------------

def areas_from_groups(group_areas: Sequence[float]) -> np.ndarray:
    """Map 8 group design variables to per-element areas (length 25)."""
    ga = np.asarray(group_areas, dtype=float).ravel()
    if ga.size != N_GROUPS:
        raise ValueError(f"Expected {N_GROUPS} group areas, got {ga.size}")
    areas = np.empty(N_ELEMS, dtype=float)
    for gid, members in GROUPS.items():
        for m in members:
            areas[m - 1] = ga[gid - 1]
    return areas


def mass_from_groups(group_areas: Sequence[float]) -> float:
    """Total structural mass in lbs: ρ · Σ(Aᵢ · Lᵢ)."""
    a = areas_from_groups(group_areas)
    return float(RHO * np.dot(a, ELEMENT_LENGTHS))


def assemble_global_stiffness(group_areas: Sequence[float]) -> np.ndarray:
    """Assemble N_DOF × N_DOF global stiffness matrix (direct stiffness method).

    For each 3-D bar element, the 6×6 contribution in global coordinates is:
        k_elem = (E · A / L) · aᵀa,   a = [-cx, -cy, -cz, cx, cy, cz]
    """
    elem_areas = areas_from_groups(group_areas)
    K = np.zeros((N_DOF, N_DOF), dtype=float)
    for idx, (ni, nj) in enumerate(ELEMENTS):
        L, cx, cy, cz = _element_direction(ni, nj)
        a = np.array([-cx, -cy, -cz, cx, cy, cz], dtype=float)
        k_elem = (E * elem_areas[idx] / L) * np.outer(a, a)
        dofs = [
            3 * (ni - 1), 3 * (ni - 1) + 1, 3 * (ni - 1) + 2,
            3 * (nj - 1), 3 * (nj - 1) + 1, 3 * (nj - 1) + 2,
        ]
        K[np.ix_(dofs, dofs)] += k_elem
    return K


def solve_displacements(group_areas: Sequence[float]) -> List[np.ndarray]:
    """Solve for nodal displacements under each load case.

    Returns a list of full displacement vectors (length N_DOF = 30) for each
    load case.  The 12 fixed DOFs remain zero; the 18 free DOFs are computed.
    """
    areas = np.clip(np.asarray(group_areas, dtype=float), A_MIN, A_MAX)
    K = assemble_global_stiffness(areas)
    free_idx = np.asarray(FREE_DOFS, dtype=int)
    K_ff = K[np.ix_(free_idx, free_idx)]

    all_U: List[np.ndarray] = []
    for F in LOAD_VECTORS:
        F_f = F[free_idx]
        U_f = np.linalg.solve(K_ff, F_f)
        U = np.zeros(N_DOF, dtype=float)
        U[free_idx] = U_f
        all_U.append(U)
    return all_U


def member_stresses(U: np.ndarray) -> np.ndarray:
    """Compute axial stress (ksi) in each of the 25 members.

    σ = (E / L) · [-cx, -cy, -cz, cx, cy, cz] · u_element

    Positive = tension; negative = compression.
    Area does not appear in the stress formula for a truss element.
    """
    U = np.asarray(U, dtype=float)
    stresses = np.empty(N_ELEMS, dtype=float)
    for idx, (ni, nj) in enumerate(ELEMENTS):
        L, cx, cy, cz = _element_direction(ni, nj)
        dofs = [
            3 * (ni - 1), 3 * (ni - 1) + 1, 3 * (ni - 1) + 2,
            3 * (nj - 1), 3 * (nj - 1) + 1, 3 * (nj - 1) + 2,
        ]
        a = np.array([-cx, -cy, -cz, cx, cy, cz], dtype=float)
        stresses[idx] = (E / L) * float(np.dot(a, U[dofs]))
    return stresses


def displacement_violation_vector(all_U: List[np.ndarray]) -> np.ndarray:
    """Per-free-DOF violation: max(0, max_over_loadcases(|u|) − U_ALLOW).

    Returns array of length len(FREE_DOFS) = 18.
    """
    free_idx = np.asarray(FREE_DOFS, dtype=int)
    worst = np.zeros(len(FREE_DOFS), dtype=float)
    for U in all_U:
        worst = np.maximum(worst, np.abs(U[free_idx]))
    return np.maximum(0.0, worst - U_ALLOW)


def stress_violation_vector(all_stresses: List[np.ndarray]) -> np.ndarray:
    """Per-element violation: max(0, max_over_loadcases(|σ|) − S_ALLOW).

    Returns array of length N_ELEMS = 25.
    """
    worst = np.zeros(N_ELEMS, dtype=float)
    for sigma in all_stresses:
        worst = np.maximum(worst, np.abs(np.asarray(sigma, dtype=float)))
    return np.maximum(0.0, worst - S_ALLOW)


def evaluate(group_areas: Sequence[float]) -> Dict[str, object]:
    """Full evaluation of the 25-bar space truss.

    Parameters
    ----------
    group_areas : array-like of shape (8,)
        Cross-sectional areas for the 8 design variable groups.

    Returns
    -------
    dict with keys:
        mass             float    structural mass (lb)
        U                list     displacement vectors per load case  [len-2 list of ndarray(30,)]
        stresses         list     axial stress vectors per load case  [len-2 list of ndarray(25,)]
        max_disp         float    max |displacement| over all free DOFs and load cases
        max_stress       float    max |stress| over all elements and load cases
        disp_violation   ndarray  per-free-DOF violation vector  (len=18)
        stress_violation ndarray  per-element violation vector   (len=25)
    """
    areas = np.clip(np.asarray(group_areas, dtype=float), A_MIN, A_MAX)
    all_U = solve_displacements(areas)
    all_stresses = [member_stresses(U) for U in all_U]

    dv = displacement_violation_vector(all_U)
    sv = stress_violation_vector(all_stresses)

    free_idx = np.asarray(FREE_DOFS, dtype=int)
    max_disp   = float(max(np.max(np.abs(U[free_idx])) for U in all_U))
    max_stress = float(max(np.max(np.abs(s))             for s in all_stresses))

    return {
        "mass":             mass_from_groups(areas),
        "U":                all_U,
        "stresses":         all_stresses,
        "max_disp":         max_disp,
        "max_stress":       max_stress,
        "disp_violation":   dv,
        "stress_violation": sv,
    }


def grouped_design_bounds() -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) arrays of shape (N_GROUPS,) for the design variable bounds."""
    lo = np.full(N_GROUPS, A_MIN, dtype=float)
    hi = np.full(N_GROUPS, A_MAX, dtype=float)
    return lo, hi


def node_coordinates() -> np.ndarray:
    return NODES.copy()


def element_connectivity() -> np.ndarray:
    return ELEMENTS.copy()


def applied_loads() -> List[np.ndarray]:
    return [F.copy() for F in LOAD_VECTORS]
