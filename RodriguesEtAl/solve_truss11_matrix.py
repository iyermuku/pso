"""Solve the 12-by-15 equilibrium matrix for the 11-bar truss.

The equilibrium equations alone do not uniquely determine all 15 unknowns.
This script returns the minimum-norm solution and the null-space basis, so
all equilibrium solutions can be written as x = x_min + Z @ q.
"""

from __future__ import annotations

import numpy as np

import truss11 as T


UNKNOWN_NAMES = (
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "Ax",
    "Ay",
    "Dx",
    "Dy",
)

DEFAULT_MEMBER_AREAS = np.array(
    [0.12, 0.98, 8.35, 9.69, 0.18, 8.53, 7.73, 6.99, 1.88, 0.19, 5.71],
    dtype=float,
)


def equilibrium_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Return the attached coefficient matrix A and load vector B."""
    c = 1.0 / np.sqrt(2.0)
    matrix = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, c, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, -c, 0, 0, c, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, c, 0, 0, c, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0, -c, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, c, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, c, 0, 0, 0, 0, 0, 1, 0],
            [-1, 0, 0, 0, 0, 0, 0, -c, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, -1, 1, 0, -c, c, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, -c, -c, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -c, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -c, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    load_vector = np.array(
        [0, 0, 0, 100, 0, 100, 0, 0, 0, 0, 0, 0],
        dtype=float,
    )
    return matrix, load_vector


def solve_equilibrium(
    member_areas: np.ndarray | None = None,
    shape_y: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Solve all member forces and reactions using truss compatibility."""
    matrix, load_vector = equilibrium_matrix()
    areas = DEFAULT_MEMBER_AREAS if member_areas is None else T.areas_from_members(member_areas)
    displacements = T.solve_displacements(areas, shape_y=shape_y)
    stresses = T.member_stresses(displacements, shape_y=shape_y)
    member_forces = areas * stresses

    stiffness = T.assemble_global_stiffness(areas, shape_y=shape_y)
    reactions = stiffness @ displacements - T.LOAD_VECTOR
    solution = np.concatenate((member_forces, reactions[[0, 1, 6, 7]]))
    rank = int(np.linalg.matrix_rank(matrix))
    _, _, vh = np.linalg.svd(matrix, full_matrices=True)
    null_basis = vh[rank:].T
    residual_norm = float(np.linalg.norm(matrix @ solution - load_vector))

    return solution, null_basis, int(rank), residual_norm


def main() -> None:
    solution, null_basis, rank, residual_norm = solve_equilibrium()

    print("Unknown order:")
    print("  " + ", ".join(UNKNOWN_NAMES))
    print("\nCompatible truss force solution:")
    for name, value in zip(UNKNOWN_NAMES, solution):
        print(f"  {name:>3} = {value: .8f}")

    print(f"\nMatrix rank: {rank}")
    print(f"Equilibrium-matrix nullity: {null_basis.shape[1]}")
    print("Compatibility selects one solution from these equilibrium force modes.")
    print(f"Residual norm: {residual_norm:.3e}")


if __name__ == "__main__":
    main()
