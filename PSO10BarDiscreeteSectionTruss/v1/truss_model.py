"""
Finite element model and problem data for the 10-member truss sizing problem.
Contains geometry, materials, loading, DOF helpers, and FE assembly/solve.
"""
import numpy as np

# -----------------------------
# Problem data (units in inches, kips, lbm)
# -----------------------------
L = 360.0  # in
E = 1.0e4  # ksi (kips/in^2)
P = 100.0  # kip downward at nodes 4 and 2
rho = 0.1  # lbm/in^3
Amin, Amax = 0.1, 35.0  # in^2
U_ALLOW = 2.0
S_ALLOW = 25.0  # ksi (25,000 psi) allowable axial stress in tension/compression
  # in (ux, uy <= 2)

# Available cross-sectional areas (in^2)
#41 field discrete data from Camp1998
available_A = np.array([
    1.62, 1.8, 1.99, 2.13, 2.38, 2.62, 2.88, 2.93, 3.09, 3.13,
    3.38, 3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22, 4.49,
    4.59, 4.80, 4.97, 5.12, 5.74, 7.22, 7.97, 11.5, 13.50, 13.90,
    14.2, 15.5, 16.0, 16.9, 18.8, 19.9, 22.0, 22.9, 26.5, 30.0, 33.5
])

# -----------------------------
# Geometry
# -----------------------------
nodes = {
    1: np.array([2*L, L]),
    2: np.array([2*L, 0.0]),
    3: np.array([L, L]),
    4: np.array([L, 0.0]),
    5: np.array([0.0, L]),
    6: np.array([0.0, 0.0]),
}

members = {
    1: (5, 3),
    2: (3, 1),
    3: (6, 4),
    4: (4, 2),
    5: (3, 4),
    6: (1, 2),
    7: (5, 4),
    8: (6, 3),
    9: (3, 2),
    10: (4, 1),
}

ndof = 2 * len(nodes)

# -----------------------------
# DOF helpers and member precomputations
# -----------------------------
def dof_index(node_id: int):
    b = 2 * (node_id - 1)
    return b, b + 1

member_lengths = {}
member_cs = {}
member_dof_idx = {}

for m_id, (i, j) in members.items():
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    dx, dy = xj - xi, yj - yi
    Lm = float(np.hypot(dx, dy))
    c, s = dx / Lm, dy / Lm
    member_lengths[m_id] = Lm
    member_cs[m_id] = (c, s)
    iux, iuy = dof_index(i)
    jux, juy = dof_index(j)
    member_dof_idx[m_id] = [iux, iuy, jux, juy]

F_base = np.zeros(ndof)
_, uy4 = dof_index(4)
_, uy2 = dof_index(2)
F_base[uy4] -= P
F_base[uy2] -= P

fixed_dofs = [*dof_index(5), *dof_index(6)]
free_dofs = [d for d in range(ndof) if d not in fixed_dofs]

# -----------------------------
# FE assembly and solve
# -----------------------------
def assemble_K(A: np.ndarray) -> np.ndarray:
    K = np.zeros((ndof, ndof))
    for m_id, (i, j) in members.items():
        Lm = member_lengths[m_id]
        c, s = member_cs[m_id]
        k = (A[m_id - 1] * E / Lm) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s],
        ])
        idx = member_dof_idx[m_id]
        for a in range(4):
            for b in range(4):
                K[idx[a], idx[b]] += k[a, b]
    return K


def solve_displacements(A: np.ndarray) -> np.ndarray:
    K = assemble_K(A)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Ff = F_base[free_dofs]
    Uf = np.linalg.solve(Kff, Ff)
    U = np.zeros(ndof)
    U[free_dofs] = Uf
    return U



# -----------------------------------------------------
# Member axial stresses from nodal displacements
# -----------------------------------------------------
def member_stresses(U: np.ndarray) -> np.ndarray:
    """
    Compute axial stress (ksi) in each member given global displacements U.
    sigma_m = (E / L_m) * (-c*u_ix - s*u_iy + c*u_jx + s*u_jy)
    Returns array of shape (10,) in ksi.
    """
    sig = np.zeros(10)
    for m_id, (i, j) in members.items():
        Lm = member_lengths[m_id]
        c, s = member_cs[m_id]
        iux, iuy = member_dof_idx[m_id][0], member_dof_idx[m_id][1]
        jux, juy = member_dof_idx[m_id][2], member_dof_idx[m_id][3]
        delta = (-c * U[iux]) + (-s * U[iuy]) + (c * U[jux]) + (s * U[juy])
        sig[m_id - 1] = (E / Lm) * delta  # ksi
    return sig

# Convenience: solve stresses directly from areas
def solve_stresses(A: np.ndarray) -> np.ndarray:
    U = solve_displacements(A)
    return member_stresses(U)

def mass_from_A(A: np.ndarray) -> float:
    lengths = np.array([member_lengths[m] for m in range(1, 11)])
    return rho * np.dot(A, lengths)
