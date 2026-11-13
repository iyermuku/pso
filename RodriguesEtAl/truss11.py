"""11-bar planar truss benchmark from Rodrigues da Silva et al. (2024).

Paper context:
- 11-bar planar truss with 3 shape variables (y4, y5, y6) and 11 area variables
- Nodes A-F arranged on three columns and two rows
- Supports at A and D
- Downward 100 kN loads at B and C

Units used here:
- length: cm
- force: kN
- stress / modulus: kN/cm^2
- density: kg/cm^3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


E: float = 20_000.0  # 200 GPa = 20,000 kN/cm^2
RHO: float = 0.00785  # 7850 kg/m^3 = 0.00785 kg/cm^3
S_ALLOW: float = 15.0  # 60 MPa = 6.0 kN/cm^2
A_MIN: float = 0.10
A_MAX: float = 30.0
LOAD_MAG: float = 100.0

X_SPAN: float = 200.0
Y_BOUNDS: Tuple[float, float] = (5.0, 300.0)


NODE_NAMES: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
NODE_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Member numbering follows the figure in the paper.
ELEMENTS: np.ndarray = np.array(
    [
        [1, 4],
        [2, 5],
        [3, 6],
        [1, 2],
        [2, 3],
        [4, 5],
        [5, 6],
        [4, 2],
        [1, 5],
        [5, 3],
        [2, 6],
    ],
    dtype=int,
)

N_NODES: int = 6
N_ELEMS: int = 11
N_DOF: int = 12


@dataclass(frozen=True)
class Geometry:
    nodes: np.ndarray
    elements: np.ndarray
    supports: Tuple[int, ...]


def build_geometry(y4: float = 200.0, y5: float = 200.0, y6: float = 200.0) -> Geometry:
    """Build the 11-bar truss geometry for a given shape configuration."""
    nodes = np.array(
        [
            [0.0, 0.0],
            [X_SPAN, 0.0],
            [2.0 * X_SPAN, 0.0],
            [0.0, float(y4)],
            [X_SPAN, float(y5)],
            [2.0 * X_SPAN, float(y6)],
        ],
        dtype=float,
    )
    return Geometry(nodes=nodes, elements=ELEMENTS.copy(), supports=(1, 4))


GEOMETRY = build_geometry()


def node_coordinates(y4: float = 200.0, y5: float = 200.0, y6: float = 200.0) -> np.ndarray:
    """Return node coordinates for the current shape design variables."""
    return build_geometry(y4=y4, y5=y5, y6=y6).nodes


def element_connectivity() -> np.ndarray:
    return ELEMENTS.copy()


def dof_index(node_id: int) -> Tuple[int, int]:
    base = 2 * (node_id - 1)
    return base, base + 1


def support_dofs() -> Tuple[int, ...]:
    dofs: List[int] = []
    for node_id in GEOMETRY.supports:
        dofs.extend(dof_index(node_id))
    return tuple(dofs)


FIXED_DOFS: Tuple[int, ...] = support_dofs()
FREE_DOFS: Tuple[int, ...] = tuple(d for d in range(N_DOF) if d not in FIXED_DOFS)


def applied_loads() -> np.ndarray:
    """Global nodal load vector for the benchmark loading in the paper."""
    loads = np.zeros(N_DOF, dtype=float)
    _, uy_b = dof_index(2)
    _, uy_c = dof_index(3)
    loads[uy_b] -= LOAD_MAG
    loads[uy_c] -= LOAD_MAG
    return loads


LOAD_VECTOR = applied_loads()


def _element_geometry(node_i: int, node_j: int, nodes: np.ndarray) -> Tuple[float, float, float]:
    xi, yi = nodes[node_i - 1]
    xj, yj = nodes[node_j - 1]
    dx = xj - xi
    dy = yj - yi
    length = float(np.hypot(dx, dy))
    if length <= 1e-12:
        raise ValueError(f"Zero-length element between nodes {node_i} and {node_j}")
    return length, dx / length, dy / length


def _resolve_shape(shape_y: Sequence[float] | None = None) -> Tuple[float, float, float]:
    if shape_y is None:
        return 200.0, 200.0, 200.0
    values = np.asarray(shape_y, dtype=float).reshape(-1)
    if values.size != 3:
        raise ValueError(f"Expected 3 shape variables (y4, y5, y6), got {values.size}")
    return float(values[0]), float(values[1]), float(values[2])


def areas_from_members(member_areas: Sequence[float]) -> np.ndarray:
    """Validate and return the 11 per-member areas (cm^2)."""
    areas = np.asarray(member_areas, dtype=float).reshape(-1)
    if areas.size != N_ELEMS:
        raise ValueError(f"Expected {N_ELEMS} member areas, got {areas.size}")
    return np.clip(areas, A_MIN, A_MAX)


def mass_from_areas(member_areas: Sequence[float], shape_y: Sequence[float] | None = None) -> float:
    """Structural mass in kg."""
    areas = areas_from_members(member_areas)
    nodes = node_coordinates(*_resolve_shape(shape_y))
    total = 0.0
    for area, (node_i, node_j) in zip(areas, ELEMENTS):
        length, _, _ = _element_geometry(node_i, node_j, nodes)
        total += RHO * area * length
    return float(total)


def assemble_global_stiffness(member_areas: Sequence[float], shape_y: Sequence[float] | None = None) -> np.ndarray:
    """Assemble the 12x12 global stiffness matrix."""
    areas = areas_from_members(member_areas)
    nodes = node_coordinates(*_resolve_shape(shape_y))
    k_global = np.zeros((N_DOF, N_DOF), dtype=float)

    for area, (node_i, node_j) in zip(areas, ELEMENTS):
        length, c, s = _element_geometry(node_i, node_j, nodes)
        k = (E * area / length) * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ],
            dtype=float,
        )
        dofs = [*dof_index(node_i), *dof_index(node_j)]
        k_global[np.ix_(dofs, dofs)] += k

    return k_global


def assemble_equilibrium_matrix(shape_y: Sequence[float] | None = None) -> np.ndarray:
    """Assemble the equilibrium matrix M such that M @ N = P."""
    nodes = node_coordinates(*_resolve_shape(shape_y))
    m = np.zeros((N_DOF, N_ELEMS), dtype=float)

    for element_index, (node_i, node_j) in enumerate(ELEMENTS):
        length, c, s = _element_geometry(node_i, node_j, nodes)
        dofs = [*dof_index(node_i), *dof_index(node_j)]
        m[np.asarray(dofs, dtype=int), element_index] = [-c, -s, c, s]

    return m


def equilibrium_matrix_rank(shape_y: Sequence[float] | None = None) -> int:
    """Return the numerical rank of the truss equilibrium matrix."""
    return int(np.linalg.matrix_rank(assemble_equilibrium_matrix(shape_y=shape_y)))


def equilibrium_svd(shape_y: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the SVD factors U, singular values, and V^T for the equilibrium matrix."""
    m = assemble_equilibrium_matrix(shape_y=shape_y)
    u, s, vh = np.linalg.svd(m, full_matrices=True)
    return u, s, vh


def equilibrium_uvw(shape_y: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the SVD factors in the textbook form M = U V W^T.

    U is n_r x n_r orthogonal, W is n_e x n_e orthogonal, and V is the diagonal
    singular-value matrix with shape n_r x n_e.
    """
    m = assemble_equilibrium_matrix(shape_y=shape_y)
    u, s, vh = np.linalg.svd(m, full_matrices=True)
    w = vh.conj().T
    v = np.zeros((m.shape[0], m.shape[1]), dtype=float)
    r = min(m.shape[0], m.shape[1])
    v[:r, :r] = np.diag(s[:r])
    return u, v, w


def equilibrium_null_space_basis(shape_y: Sequence[float] | None = None) -> np.ndarray:
    """Return the null-space basis matrix V from the SVD of the equilibrium matrix."""
    m = assemble_equilibrium_matrix(shape_y=shape_y)
    _, s, vh = np.linalg.svd(m, full_matrices=True)
    rank = np.linalg.matrix_rank(m)
    nullity = m.shape[1] - rank
    if nullity <= 0:
        return np.zeros((m.shape[1], 0), dtype=float)
    v = vh.conj().T[:, rank:]
    return v


def solve_displacements(member_areas: Sequence[float], shape_y: Sequence[float] | None = None) -> np.ndarray:
    """Solve the displacements for the benchmark load case."""
    k_global = assemble_global_stiffness(member_areas, shape_y=shape_y)
    k_ff = k_global[np.ix_(FREE_DOFS, FREE_DOFS)]
    f_f = LOAD_VECTOR[np.asarray(FREE_DOFS, dtype=int)]
    u_f = np.linalg.solve(k_ff, f_f)
    u = np.zeros(N_DOF, dtype=float)
    u[np.asarray(FREE_DOFS, dtype=int)] = u_f
    return u


def member_stresses(displacements: Sequence[float], shape_y: Sequence[float] | None = None) -> np.ndarray:
    """Axial stress in each member, in kN/cm^2."""
    nodes = node_coordinates(*_resolve_shape(shape_y))
    u = np.asarray(displacements, dtype=float).reshape(-1)
    stresses = np.zeros(N_ELEMS, dtype=float)

    for idx, (node_i, node_j) in enumerate(ELEMENTS):
        length, c, s = _element_geometry(node_i, node_j, nodes)
        dofs = [*dof_index(node_i), *dof_index(node_j)]
        axial_extension = float(np.dot(np.array([-c, -s, c, s], dtype=float), u[dofs]))
        stresses[idx] = E * axial_extension / length

    return stresses


def evaluate(member_areas: Sequence[float], shape_y: Sequence[float] | None = None) -> dict[str, object]:
    """Convenience evaluation returning mass, displacements, and stresses."""
    areas = areas_from_members(member_areas)
    y4, y5, y6 = _resolve_shape(shape_y)
    displacements = solve_displacements(areas, shape_y=(y4, y5, y6))
    stresses = member_stresses(displacements, shape_y=(y4, y5, y6))
    free_disp = np.abs(displacements[np.asarray(FREE_DOFS, dtype=int)])
    return {
        "areas": areas.copy(),
        "shape_y": np.array([y4, y5, y6], dtype=float),
        "mass": mass_from_areas(areas, shape_y=(y4, y5, y6)),
        "U": displacements,
        "stresses": stresses,
        "max_disp": float(np.max(free_disp)),
        "max_stress": float(np.max(np.abs(stresses))),
    }


def shape_variable_bounds() -> Tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, Y_BOUNDS[0], dtype=float)
    hi = np.full(3, Y_BOUNDS[1], dtype=float)
    return lo, hi


def area_bounds() -> Tuple[np.ndarray, np.ndarray]:
    lo = np.full(N_ELEMS, A_MIN, dtype=float)
    hi = np.full(N_ELEMS, A_MAX, dtype=float)
    return lo, hi


if __name__ == "__main__":
    demo_areas = np.array([0.12, 0.98, 8.35, 9.69, 0.18, 8.53, 7.73, 6.99, 1.88, 0.19, 5.71], dtype=float)
    result = evaluate(demo_areas, shape_y=(296.54, 195.37, 6.59))
    print(f"Mass (kg): {result['mass']:.4f}")
    print(f"Max |disp| (cm): {result['max_disp']:.6f}")
    print(f"Max |stress| (kN/cm^2): {result['max_stress']:.6f}")
    print(f"Equilibrium matrix rank: {equilibrium_matrix_rank()}")
    U, V, W = equilibrium_uvw()
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    print(f"Singular values: {np.diag(V)[:9]}")
    Vnull = equilibrium_null_space_basis()
    print(f"Nullity: {Vnull.shape[1]}")
    print(f"Null-space basis matrix:\n{Vnull}")