from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


E = 30000.0  # ksi = kip / in^2
RHO = 0.283  # lb / in^3
U_ALLOW = 4.0  # in
A_MIN = 0.10  # in^2
A_MAX = 2.00  # in^2

PANEL_WIDTH = 240.0
PANEL_HEIGHT = 144.0
SUPPORT_DROP = 360.0


GROUPS: Dict[int, List[int]] = {
    1: list(range(1, 5)),
    2: [5, 8, 11, 14, 17],
    3: list(range(19, 25)),
    4: [18, 25, 56, 63, 94, 101, 132, 139, 170, 177],
    5: [26, 29, 32, 35, 38],
    6: [6, 7, 9, 10, 12, 13, 15, 16, 27, 28, 30, 31, 33, 34, 36, 37],
    7: list(range(39, 43)),
    8: [43, 46, 49, 52, 55],
    9: list(range(57, 63)),
    10: [64, 67, 70, 73, 76],
    11: [44, 45, 47, 48, 50, 51, 53, 54, 65, 66, 68, 69, 71, 72, 74, 75],
    12: list(range(77, 81)),
    13: [81, 84, 87, 90, 93],
    14: list(range(95, 101)),
    15: [102, 105, 108, 111, 114],
    16: [82, 83, 85, 86, 88, 89, 91, 92, 103, 104, 106, 107, 109, 110, 112, 113],
    17: list(range(115, 119)),
    18: [119, 122, 125, 128, 131],
    19: list(range(133, 139)),
    20: [140, 143, 146, 149, 152],
    21: [120, 121, 123, 124, 126, 127, 129, 130, 141, 142, 144, 145, 147, 148, 150, 151],
    22: list(range(153, 157)),
    23: [157, 160, 163, 166, 169],
    24: list(range(171, 177)),
    25: [178, 181, 184, 187, 190],
    26: [158, 159, 161, 162, 164, 165, 167, 168, 179, 180, 182, 183, 185, 186, 188, 189],
    27: list(range(191, 195)),
    28: [195, 197, 198, 200],
    29: [196, 199],
}


PX_LOAD_NODES = [1, 6, 15, 20, 29, 34, 43, 48, 57, 62]
PY_LOAD_NODES = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28,
    29, 30, 31, 32, 33, 34, 36, 38, 40, 42, 43, 44, 45, 46, 47, 48, 50, 52, 54,
    56, 57, 58, 59, 60, 61, 62, 64, 66, 68, 70, 71, 72, 73, 74, 75,
]


FULL_ROWS: List[List[int]] = [
    [1, 2, 3, 4, 5],
    [15, 16, 17, 18, 19],
    [29, 30, 31, 32, 33],
    [43, 44, 45, 46, 47],
    [57, 58, 59, 60, 61],
    [71, 72, 73, 74, 75],
]

MIXED_ROWS: List[List[int]] = [
    [6, 7, 8, 9, 10, 11, 12, 13, 14],
    [20, 21, 22, 23, 24, 25, 26, 27, 28],
    [34, 35, 36, 37, 38, 39, 40, 41, 42],
    [48, 49, 50, 51, 52, 53, 54, 55, 56],
    [62, 63, 64, 65, 66, 67, 68, 69, 70],
]


@dataclass(frozen=True)
class Geometry:
    nodes: np.ndarray
    elements: np.ndarray
    supports: Tuple[int, ...]


def _build_nodes() -> np.ndarray:
    coords: Dict[int, Tuple[float, float]] = {}

    full_x = [0.0, 240.0, 480.0, 720.0, 960.0]
    mixed_x = [0.0, 120.0, 240.0, 360.0, 480.0, 600.0, 720.0, 840.0, 960.0]
    y_levels = [1440.0, 1296.0, 1152.0, 1008.0, 864.0, 720.0, 576.0, 432.0, 288.0, 144.0, 0.0]

    for row_idx, nodes in enumerate(FULL_ROWS):
        y = y_levels[row_idx * 2]
        for node_id, x in zip(nodes, full_x):
            coords[node_id] = (x, y)

    for row_idx, nodes in enumerate(MIXED_ROWS):
        y = y_levels[row_idx * 2 + 1]
        for node_id, x in zip(nodes, mixed_x):
            coords[node_id] = (x, y)

    coords[76] = (240.0, -SUPPORT_DROP)
    coords[77] = (720.0, -SUPPORT_DROP)

    return np.array([coords[idx] for idx in range(1, 78)], dtype=float)


def _append_full_to_mixed(elements: List[Tuple[int, int]], full_row: Sequence[int], mixed_row: Sequence[int]) -> None:
    for col in range(4):
        elements.append((full_row[col], mixed_row[2 * col]))
        elements.append((full_row[col], mixed_row[2 * col + 1]))
        elements.append((full_row[col + 1], mixed_row[2 * col + 1]))
    elements.append((full_row[4], mixed_row[8]))


def _append_mixed_to_full(elements: List[Tuple[int, int]], mixed_row: Sequence[int], full_row: Sequence[int]) -> None:
    for col in range(4):
        elements.append((mixed_row[2 * col], full_row[col]))
        elements.append((mixed_row[2 * col + 1], full_row[col]))
        elements.append((mixed_row[2 * col + 1], full_row[col + 1]))
    elements.append((mixed_row[8], full_row[4]))


def _append_horizontal_chain(elements: List[Tuple[int, int]], row_nodes: Sequence[int]) -> None:
    for left, right in zip(row_nodes[:-1], row_nodes[1:]):
        elements.append((left, right))


def _build_elements() -> np.ndarray:
    elements: List[Tuple[int, int]] = []

    _append_horizontal_chain(elements, FULL_ROWS[0])
    _append_full_to_mixed(elements, FULL_ROWS[0], MIXED_ROWS[0])
    _append_horizontal_chain(elements, MIXED_ROWS[0])
    _append_mixed_to_full(elements, MIXED_ROWS[0], FULL_ROWS[1])

    for block in range(1, 5):
        _append_horizontal_chain(elements, FULL_ROWS[block])
        _append_full_to_mixed(elements, FULL_ROWS[block], MIXED_ROWS[block])
        _append_horizontal_chain(elements, MIXED_ROWS[block])
        _append_mixed_to_full(elements, MIXED_ROWS[block], FULL_ROWS[block + 1])

    _append_horizontal_chain(elements, FULL_ROWS[-1])
    elements.extend([
        (71, 76),
        (72, 76),
        (73, 76),
        (73, 77),
        (74, 77),
        (75, 77),
    ])

    if len(elements) != 200:
        raise RuntimeError(f"Expected 200 elements, found {len(elements)}")
    return np.asarray(elements, dtype=int)


GEOMETRY = Geometry(nodes=_build_nodes(), elements=_build_elements(), supports=(76, 77))

N_NODES = GEOMETRY.nodes.shape[0]
N_ELEMS = GEOMETRY.elements.shape[0]
N_GROUPS = len(GROUPS)
N_DOF = 2 * N_NODES


def _support_dofs() -> List[int]:
    dofs: List[int] = []
    for node_id in GEOMETRY.supports:
        base = 2 * (node_id - 1)
        dofs.extend([base, base + 1])
    return dofs


FIXED_DOFS = tuple(_support_dofs())
FREE_DOFS = tuple(dof for dof in range(N_DOF) if dof not in FIXED_DOFS)
FREE_NODES = tuple(node_id for node_id in range(1, N_NODES + 1) if node_id not in GEOMETRY.supports)


def build_load_vector() -> np.ndarray:
    load = np.zeros(N_DOF, dtype=float)
    for node_id in PX_LOAD_NODES:
        load[2 * (node_id - 1)] += 1.0
    for node_id in PY_LOAD_NODES:
        load[2 * (node_id - 1) + 1] -= 10.0
    return load


LOAD_VECTOR = build_load_vector()


def _element_geometry(node_i: int, node_j: int) -> Tuple[float, float, float]:
    xi, yi = GEOMETRY.nodes[node_i - 1]
    xj, yj = GEOMETRY.nodes[node_j - 1]
    dx = xj - xi
    dy = yj - yi
    length = float(np.hypot(dx, dy))
    if length <= 1e-12:
        raise ValueError(f"Zero-length element between nodes {node_i} and {node_j}")
    c = dx / length
    s = dy / length
    return length, c, s


ELEMENT_LENGTHS = np.array([_element_geometry(i, j)[0] for i, j in GEOMETRY.elements], dtype=float)


def areas_from_groups(group_areas: Sequence[float]) -> np.ndarray:
    group_areas = np.asarray(group_areas, dtype=float).reshape(-1)
    if group_areas.size != N_GROUPS:
        raise ValueError(f"Expected {N_GROUPS} group areas, got {group_areas.size}")
    areas = np.zeros(N_ELEMS, dtype=float)
    for group_id, members in GROUPS.items():
        areas[np.asarray(members, dtype=int) - 1] = group_areas[group_id - 1]
    return areas


def mass_from_groups(group_areas: Sequence[float]) -> float:
    elem_areas = areas_from_groups(group_areas)
    return float(np.sum(RHO * elem_areas * ELEMENT_LENGTHS))


def assemble_global_stiffness(group_areas: Sequence[float]) -> np.ndarray:
    elem_areas = areas_from_groups(group_areas)
    k_global = np.zeros((N_DOF, N_DOF), dtype=float)

    for elem_idx, (node_i, node_j) in enumerate(GEOMETRY.elements):
        length, c, s = _element_geometry(node_i, node_j)
        k = (E * elem_areas[elem_idx] / length) * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ],
            dtype=float,
        )
        dofs = [2 * (node_i - 1), 2 * (node_i - 1) + 1, 2 * (node_j - 1), 2 * (node_j - 1) + 1]
        k_global[np.ix_(dofs, dofs)] += k

    return k_global


def solve_displacements(group_areas: Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(group_areas, dtype=float), A_MIN, A_MAX)
    k_global = assemble_global_stiffness(clipped)
    k_ff = k_global[np.ix_(FREE_DOFS, FREE_DOFS)]
    f_f = LOAD_VECTOR[np.asarray(FREE_DOFS, dtype=int)]
    u_f = np.linalg.solve(k_ff, f_f)
    u = np.zeros(N_DOF, dtype=float)
    u[np.asarray(FREE_DOFS, dtype=int)] = u_f
    return u


def element_stresses(displacements: Sequence[float], group_areas: Sequence[float]) -> np.ndarray:
    u = np.asarray(displacements, dtype=float)
    stresses = np.zeros(N_ELEMS, dtype=float)
    for idx, (node_i, node_j) in enumerate(GEOMETRY.elements):
        length, c, s = _element_geometry(node_i, node_j)
        dofs = [2 * (node_i - 1), 2 * (node_i - 1) + 1, 2 * (node_j - 1), 2 * (node_j - 1) + 1]
        axial_extension = np.dot(np.array([-c, -s, c, s], dtype=float), u[dofs])
        stresses[idx] = E * axial_extension / length
    return stresses


def displacement_violation_vector(displacements: Sequence[float]) -> np.ndarray:
    u = np.asarray(displacements, dtype=float)
    free_disp = np.abs(u[np.asarray(FREE_DOFS, dtype=int)])
    return np.maximum(0.0, free_disp - U_ALLOW)


def evaluate(group_areas: Sequence[float]) -> Dict[str, object]:
    areas = np.clip(np.asarray(group_areas, dtype=float), A_MIN, A_MAX)
    u = solve_displacements(areas)
    stress = element_stresses(u, areas)
    return {
        "areas": areas.copy(),
        "mass": mass_from_groups(areas),
        "U": u,
        "stresses": stress,
        "max_disp": float(np.max(np.abs(u[np.asarray(FREE_DOFS, dtype=int)]))),
        "max_stress": float(np.max(np.abs(stress))),
        "disp_violation": displacement_violation_vector(u),
    }


def grouped_design_bounds() -> Tuple[np.ndarray, np.ndarray]:
    lo = np.full(N_GROUPS, A_MIN, dtype=float)
    hi = np.full(N_GROUPS, A_MAX, dtype=float)
    return lo, hi


def group_members(group_id: int) -> List[int]:
    return GROUPS[group_id]


def node_coordinates() -> np.ndarray:
    return GEOMETRY.nodes.copy()


def element_connectivity() -> np.ndarray:
    return GEOMETRY.elements.copy()


def applied_loads() -> np.ndarray:
    return LOAD_VECTOR.copy()
