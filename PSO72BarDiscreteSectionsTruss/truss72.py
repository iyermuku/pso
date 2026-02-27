"""
72-bar truss model (3D) with 16 design variables grouped as in Table 6, and
load cases as in Table 7 (Perez & Behdinan / Sedaghati benchmark).

This module assembles a 3D truss FE model and provides:
- areas_from_groups(A_groups16): map 16 design variables to 72 member areas
- mass_from_A(A_groups16): objective mass (rho * sum(A_i * L_i))
- solve_displacements(A_groups16): nodal displacements for each load case
- member_stresses(U): axial stress for each bar under a given displacement vector

Geometry & numbering:
---------------------
To keep this module self-contained, we include a *synthetic but consistent* 72-bar
space-truss geometry that produces a connected model; the element numbers 1..72
exist and can be grouped per Table 6. For production runs, you should replace the
synthetic geometry with the canonical dataset (node coordinates and element
connectivity) from the literature (e.g., Sedaghati 2005; CoFE example files). See:
- 72-bar truss design groups and load cases (CoFE page):
  https://vtpasquale.github.io/NASTRAN_CoFE/2._Examples/b._Optimization/4._72-Bar_Truss_Optimization/  
  (Young's modulus 10^7 psi, density 0.1 lbm/in^3, ±25 ksi stress, ±0.25 in disp, load cases)  
  LOAD CASE 1: node 1, Fx=5000 lbf, Fy=5000 lbf, Fz=-5000 lbf;  
  LOAD CASE 2: nodes 1..4, Fz=-5000 lbf each.  
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

# -------------------------------------------------------------
# Material & constraints (per benchmark references)
# -------------------------------------------------------------
E = 1.0e7        # psi (10^7 psi)  \u2014 Young's modulus  (CoFE)  
RHO = 0.1        # lbm/in^3         \u2014 specific mass    (CoFE)  
U_ALLOW = 0.25   # in               \u2014 disp limit nodes 1-4 in X,Y (CoFE)  
S_ALLOW = 25.0   # ksi               \u2014 stress limit ±25 ksi (CoFE)

# Area bounds commonly used in the benchmark
A_MIN = 0.10     # in^2
A_MAX = 10.0     # in^2 (set generous upper bound)
# Available cross-sectional areas (in^2)
# Wu&Chow1995, Table 9 (pp 986)
available_A = np.array([
    0.111, 0.141, 0.196, 0.250, 0.307, 0.391, 0.442, 0.563, 0.602, 0.766, 
    0.785, 0.994, 1.000, 1.228, 1.266, 1.457, 1.563, 1.620, 1.800, 1.990, 
    2.130, 2.380, 2.620, 2.630, 2.880, 2.930, 3.090, 3.130, 3.380, 3.470, 
    3.550, 3.630, 3.840, 3.870, 3.880, 4.180, 4.220, 4.490, 4.590, 4.800, 
    4.970, 5.120, 5.740, 7.220, 7.970, 8.530, 9.300, 10.850, 11.500, 13.500, 
    13.900, 14.200, 15.500, 16.000, 16.900, 18.800, 19.900, 22.000, 22.900, 24.500, 
    26.500, 28.000, 30.000, 33.500
])

# -------------------------------------------------------------
# 72-bar element grouping \u2014 Table 6
# -------------------------------------------------------------
GROUPS: Dict[int, List[int]] = {
    1:  list(range(1, 4+1)),
    2:  list(range(5, 12+1)),
    3:  list(range(13, 16+1)),
    4:  [17, 18],
    5:  list(range(19, 22+1)),
    6:  list(range(23, 30+1)),
    7:  [31, 32, 33, 34],
    8:  [35, 36],
    9:  [37, 38, 39, 40],
    10: list(range(41, 48+1)),
    11: [49, 50, 51, 52],
    12: [53, 54],
    13: [55, 56, 57, 58],
    14: list(range(59, 66+1)),
    15: [67, 68, 69, 70],
    16: [71, 72],
}

# -------------------------------------------------------------
# Synthetic geometry for testing (connected 4-level square tower)
# -------------------------------------------------------------
# We create 5 levels in z (0, L, 2L, 3L,4L) and 4 nodes per level (square of side s).
# Then we add verticals, perimeters, intra-level diagonals, and inter-level X-bracing
# to reach 72 bars. Element numbering starts at 1, consistent with GROUPS.

@dataclass
class Geometry:
    nodes: np.ndarray     # shape (N,3) coordinates in inches
    elems: np.ndarray     # shape (M,2) connectivity (1-based node IDs)
    fixed_nodes: List[int]# nodes fully constrained (X,Y,Z fixed)


def synthetic_72bar_geometry(s: float = 60.0, L: float = 60.0) -> Geometry:
    # Nodes: 5 levels, 4 per level
    coords = []
    for k in range(5):
        z = (4-k) * L
        coords.extend([
            [0.0,   0.0,   z],
            [2*s,     0.0,   z],
            [2*s,     2*s,     z],
            [0.0,   2*s,     z],
        ])
    nodes = np.array(coords, dtype=float)  # IDs 1..20

    # Element list (pairs of node IDs, 1-based)
    elems: List[Tuple[int,int]] = []

    # Helper to add unique element
    def add(i,j):
        if i==j: return
        #if i>j: i,j = j,i
        elems.append((i,j))

    add( 1, 5);add( 2, 6);add( 3, 7);add( 4, 8);
    add( 5, 2);add( 1, 6);add( 6, 3);add( 2, 7);add( 7, 4);add( 3, 8);add( 8, 1);add( 4, 5);
    add( 1, 2);add( 2, 3);add( 3, 4);add( 4, 1);
    add( 1, 3);add( 2, 4);
    add( 5, 9);add( 6,10);add( 7,11);add( 8,12);
    add( 9, 6);add( 5,10);add(10, 7);add( 6,11);add(11, 8);add( 7,12);add(12, 5);add( 8, 9);
    add( 5, 6);add( 6, 7);add( 7, 8);add( 8, 5);
    add( 5, 7);add( 6, 8);
    add( 9,13);add(10,14);add(11,15);add(12,16);
    add(13,10);add( 9,14);add(14,11);add(10,15);add(15,12);add(11,16);add(16, 9);add(12,13);
    add( 9,10);add(10,11);add(11,12);add(12, 9);
    add( 9,11);add(10,12);
    add(13,17);add(14,18);add(15,19);add(16,20);
    add(17,14);add(13,18);add(18,15);add(14,19);add(19,16);add(15,20);add(20,13);add(16,17);
    add(13,14);add(14,15);add(15,16);add(16,13);
    add(13,15);add(14,16);

    # # Vertical columns between levels
    # for level in range(4):
        # base = level*4
        # nextb = (level+1)*4
        # for p in range(4):
            # add(base+p+1, nextb+p+1)
        # #add the diagonals
        # for p in range(4):
            # add(nextb+p+1, base+p+2)
            # add(nextb+p+2, base+p+1)
        # for p in range(4):
            # add(nextb+p+1, base+p+1)

    # # Perimeter edges at each level
    # for level in range(4):
        # base = level*4
        # add(base+1, base+2)
        # add(base+2, base+3)
        # add(base+3, base+4)
        # add(base+4, base+1)
        # # add diagonals across square
        # add(base+1, base+3)
        # add(base+2, base+4)

    # # Inter-level X bracing on each side (between level l and l+1)
    # # Sides: (1-2), (2-3), (3-4), (4-1) across levels
    # for level in range(4):
        # b = level*4; n = (level+1)*4
        # add(b+1, n+2); add(b+2, n+1)  # side 1-2
        # add(b+2, n+3); add(b+3, n+2)  # side 2-3
        # add(b+3, n+4); add(b+4, n+3)  # side 3-4
        # add(b+4, n+1); add(b+1, n+4)  # side 4-1

    # # Add some cross-level diagonals (two-level skip) to reach 72
    # add(1, 9); add(2, 10); add(3, 11); add(4, 12)
    # add(5, 13); add(6, 14); add(7, 15); add(8, 16)

    # # If still fewer than 72, add cross-braces between non-adjacent corners across levels
    # while len(elems) < 72:
        # # deterministic extra chords
        # for (i,j) in [(1,6),(2,5),(3,8),(4,7),(9,14),(10,13),(11,16),(12,15)]:
            # if len(elems) >= 72: break
            # add(i,j)

    elems_arr = np.array(elems[:72], dtype=int)

    # Supports: fix bottom level nodes (IDs 17..20) completely
    fixed_nodes = [17,18,19,20]
    return Geometry(nodes=nodes, elems=elems_arr, fixed_nodes=fixed_nodes)

GEOM = synthetic_72bar_geometry()
#print(GEOM.nodes)
#print(GEOM.elems)
# -------------------------------------------------------------
# Load cases (Table 7 / CoFE): LC1 and LC2
# -------------------------------------------------------------
# LC1: Node 1, Fx=+5000, Fy=+5000, Fz=-5000
# LC2: Nodes 1..4, Fz=-5000 each

@dataclass
class LoadCase:
    forces: Dict[int, np.ndarray]  # node_id -> [Fx,Fy,Fz] (lbf)

LOAD_CASES: List[LoadCase] = [
    LoadCase(forces={1: np.array([+5000.0, +5000.0, -5000.0], dtype=float)}),
    LoadCase(forces={1: np.array([0.0, 0.0, -5000.0]),
                     2: np.array([0.0, 0.0, -5000.0]),
                     3: np.array([0.0, 0.0, -5000.0]),
                     4: np.array([0.0, 0.0, -5000.0])}),
]

# -------------------------------------------------------------
# FE utilities (3D truss)
# -------------------------------------------------------------

def _elem_length_dir(i: int, j: int, nodes: np.ndarray) -> Tuple[float, np.ndarray]:
    xi = nodes[i-1]; xj = nodes[j-1]
    d = xj - xi
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        raise ValueError(f"Zero-length element between nodes {i}-{j}")
    t = d / L  # direction cosines (3,)
    return L, t


def areas_from_groups(A_groups16: np.ndarray) -> np.ndarray:
    """Map 16 design variables to 72 member areas (in^2)."""
    A_groups16 = np.asarray(A_groups16, dtype=float)
    if A_groups16.shape != (16,):
        raise ValueError("A_groups16 must have shape (16,)")
    A_members = np.zeros(72, dtype=float)
    for g in range(1,17):
        for eidx in GROUPS[g]:
            A_members[eidx-1] = A_groups16[g-1]
    return A_members


def mass_from_A(A_groups16: np.ndarray) -> float:
    """Total mass (lbm) = sum(rho * A * L) over 72 members."""
    A_members = areas_from_groups(A_groups16)
    m = 0.0
    for k,(i,j) in enumerate(GEOM.elems, start=1):
        L,_ = _elem_length_dir(i,j,GEOM.nodes)
        A = A_members[k-1]
        m += RHO * A * L
    return float(m)


def _assemble_global_K(A_members: np.ndarray) -> np.ndarray:
    """Assemble global stiffness matrix K (size 3N x 3N)."""
    nnode = GEOM.nodes.shape[0]
    K = np.zeros((3*nnode, 3*nnode), dtype=float)
    for k,(i,j) in enumerate(GEOM.elems, start=1):
        L,t = _elem_length_dir(i,j,GEOM.nodes)
        A = A_members[k-1]
        k_axial = (E*A)/L
        # 3x3 outer product
        tt = np.outer(t,t)
        Kii = k_axial * tt
        Kjj = k_axial * tt
        Kij = -k_axial * tt
        Kji = -k_axial * tt
        # global DOF indices
        ii = slice(3*(i-1), 3*(i-1)+3)
        jj = slice(3*(j-1), 3*(j-1)+3)
        K[ii,ii] += Kii
        K[jj,jj] += Kjj
        K[ii,jj] += Kij
        K[jj,ii] += Kji
    return K


def _apply_bc(K: np.ndarray, F: np.ndarray, fixed_nodes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply essential BCs by zeroing rows/cols for fixed DOFs and setting RHS."""
    n = K.shape[0]
    fixed_dofs = []
    for nid in fixed_nodes:
        fixed_dofs.extend([3*(nid-1)+0, 3*(nid-1)+1, 3*(nid-1)+2])
    fixed_dofs = sorted(set(fixed_dofs))
    free = np.array([d for d in range(n) if d not in fixed_dofs], dtype=int)
    # Partition
    Kff = K[np.ix_(free,free)]
    Ff = F[free]
    return Kff, Ff, free


def solve_displacements(A_groups16: np.ndarray) -> List[np.ndarray]:
    """Solve U for each load case; returns list of U (size 3N)."""
    A_members = areas_from_groups(A_groups16)
    K = _assemble_global_K(A_members)
    nnode = GEOM.nodes.shape[0]
    U_all = []
    for lc in LOAD_CASES:
        # Build global force vector
        F = np.zeros(3*nnode, dtype=float)
        for nid, vec in lc.forces.items():
            F[3*(nid-1):3*(nid-1)+3] += vec
        # Apply BCs
        Kff, Ff, free = _apply_bc(K, F, GEOM.fixed_nodes)
        # Solve
        Uf = np.linalg.solve(Kff, Ff)
        U = np.zeros(3*nnode, dtype=float)
        U[free] = Uf
        U_all.append(U)
    return U_all


def member_stresses(U: np.ndarray, A_members: np.ndarray) -> np.ndarray:
    """Axial stress (ksi) per member for displacement U (in), using sigma = (E/L) * t^T*(uj-ui)."""
    sig = np.zeros(72, dtype=float)
    for k,(i,j) in enumerate(GEOM.elems, start=1):
        L,t = _elem_length_dir(i,j,GEOM.nodes)
        ui = U[3*(i-1):3*(i-1)+3]
        uj = U[3*(j-1):3*(j-1)+3]
        axial_strain = float(np.dot(t, (uj - ui))) / L
        # For truss under small strain, axial stress = E * axial_strain (psi). Convert to ksi
        sigma_psi = E * axial_strain
        sig[k-1] = sigma_psi / 1000.0
    return sig

# Convenience API: evaluate displacements, stresses, mass for all load cases

def evaluate(A_groups16: np.ndarray) -> Dict[str, object]:
    A_members = areas_from_groups(A_groups16)
    mass = mass_from_A(A_groups16)
    U_list = solve_displacements(A_groups16)
    stresses = [member_stresses(U, A_members) for U in U_list]
    return {"mass": mass, "U": U_list, "stresses": stresses}

# Script entry (quick demo)
if __name__ == '__main__':
    # Example areas (clipped to [A_MIN, A_MAX])
    A = np.array([0.1565,0.5456,0.4104,0.5697,0.5237,0.5171,0.1,0.1,1.268,0.5117,0.1,0.1,1.886,0.5123,0.1,0.1])
    A = np.clip(A, A_MIN, A_MAX)
    res = evaluate(A)
    print(f"Mass (lbm) = {res['mass']:.4f}")
    for i,U in enumerate(res['U'], start=1):
        maxU = float(np.max(np.abs(U)))
        print(f"Load case {i}: max |U| = {maxU:.6f} in")
        sig = res['stresses'][i-1]
        print(f"            max |sigma| = {np.max(np.abs(sig)):.6f} ksi")
