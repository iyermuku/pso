"""Check feasibility of a 25-bar truss design vector.

Accepts 8 comma-separated design variables and reports:
- feasible: yes/no
- maximum displacement
- maximum stress
- if infeasible: failed DOFs and failed bars

Usage:
  python PSO25BarTruss/check_truss25_design.py --x "0.1,0.5,3.4,0.1,2.0,1.0,0.34,3.4"
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np

import truss25 as truss

TOL = 1e-9
AXIS = ("x", "y", "z")


def parse_design(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != truss.N_GROUPS:
        raise ValueError(
            f"Expected {truss.N_GROUPS} comma-separated values, got {len(parts)}"
        )
    try:
        x = np.array([float(v) for v in parts], dtype=float)
    except ValueError as exc:
        raise ValueError("All design variables must be numeric.") from exc
    return x


def dof_to_node_axis(global_dof: int) -> tuple[int, str]:
    node = global_dof // 3 + 1
    axis = AXIS[global_dof % 3]
    return node, axis


def worst_displacement_per_free_dof(all_u: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (worst_abs_disp, loadcase_idx) per free DOF."""
    free_idx = np.asarray(truss.FREE_DOFS, dtype=int)
    per_case = np.array([np.abs(u[free_idx]) for u in all_u])
    loadcase_idx = np.argmax(per_case, axis=0)
    worst = np.max(per_case, axis=0)
    return worst, loadcase_idx


def worst_stress_per_bar(all_sigma: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (worst_abs_stress, loadcase_idx) per bar."""
    per_case = np.array([np.abs(s) for s in all_sigma])
    loadcase_idx = np.argmax(per_case, axis=0)
    worst = np.max(per_case, axis=0)
    return worst, loadcase_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 25-bar truss design feasibility")
    parser.add_argument(
        "--x",
        type=str,
        required=False,
        help="8 comma-separated design variables",
    )
    args = parser.parse_args()

    x_text = args.x
    if not x_text:
        x_text = input("Enter 8 design variables (comma-separated): ").strip()

    try:
        x = parse_design(x_text)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2

    x_eval = np.clip(x, truss.A_MIN, truss.A_MAX)
    if not np.allclose(x, x_eval):
        print("Note: values were clipped to bounds before evaluation.")

    res = truss.evaluate(x_eval)
    mass = float(res["mass"])
    max_disp = float(res["max_disp"])
    max_stress = float(res["max_stress"])
    dv = np.asarray(res["disp_violation"], dtype=float)
    sv = np.asarray(res["stress_violation"], dtype=float)

    feasible = bool(np.all(dv <= TOL) and np.all(sv <= TOL))

    print(f"feasible: {'yes' if feasible else 'no'}")
    print(f"mass: {mass:.6f} lb")
    print(f"maximum displacement: {max_disp:.6f} in")
    print(f"maximum stress: {max_stress:.6f} ksi")

    if feasible:
        return 0

    all_u = res["U"]
    all_sigma = res["stresses"]

    worst_u, lc_u = worst_displacement_per_free_dof(all_u)
    worst_s, lc_s = worst_stress_per_bar(all_sigma)

    failed_dof_idx = np.where(dv > TOL)[0]
    failed_bar_idx = np.where(sv > TOL)[0]

    print("\nfailed displacement DOFs:")
    if failed_dof_idx.size == 0:
        print("  none")
    else:
        for local_idx in failed_dof_idx:
            global_dof = truss.FREE_DOFS[int(local_idx)]
            node, axis = dof_to_node_axis(global_dof)
            print(
                f"  node {node} {axis}: |u|={worst_u[local_idx]:.6f} in "
                f"(limit {truss.U_ALLOW:.6f}, violation {dv[local_idx]:.6f}) "
                f"[load case {int(lc_u[local_idx]) + 1}]"
            )

    print("\nfailed stress bars:")
    if failed_bar_idx.size == 0:
        print("  none")
    else:
        for bar_idx in failed_bar_idx:
            bar_num = int(bar_idx) + 1
            n1, n2 = truss.ELEMENTS[bar_idx]
            print(
                f"  bar {bar_num} ({n1}-{n2}): |sigma|={worst_s[bar_idx]:.6f} ksi "
                f"(limit {truss.S_ALLOW:.6f}, violation {sv[bar_idx]:.6f}) "
                f"[load case {int(lc_s[bar_idx]) + 1}]"
            )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
