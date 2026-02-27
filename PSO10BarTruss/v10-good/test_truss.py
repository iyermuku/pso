#!/usr/bin/env python3
"""
Test script for the 10-member truss.

Usage examples:
  python test_truss.py 1 1 1 1 1 1 1 1 1 1
  python test_truss.py 0.5 0.8 1.2 0.9 0.7 1.1 0.6 0.6 0.5 0.5 --csv results.csv

Inputs:
  Areas (in^2) for members 1..10 in the order defined in truss_model.py.
Outputs:
  Member stresses (ksi), nodal displacements (in), and total mass (lbm).
"""
import argparse
import numpy as np
from truss_model import (
    solve_displacements,
    solve_stresses,
    mass_from_A,
    nodes,
    dof_index,
    members,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute axial stresses (ksi), nodal displacements (in), and mass (lbm) "
            "for the 10-member truss given 10 areas (in^2)."
        )
    )
    parser.add_argument(
        "areas",
        nargs=10,
        type=float,
        metavar="A",
        help=(
            "Cross-sectional areas for members 1..10 (in^2), in the order defined "
            "by truss_model.members."
        ),
    )
    parser.add_argument(
        "--csv",
        metavar="OUT.csv",
        help="Optional path to save results as CSV",
    )
    args = parser.parse_args()

    A = np.array(args.areas, dtype=float)

    # Solve: global displacements, member stresses, total mass
    U = solve_displacements(A)
    stresses = solve_stresses(A)
    mass = mass_from_A(A)

    # Pretty print results
    print("INPUT")
    print(f"Areas (in^2) [m01..m10]: {A.tolist()}")

    print("\nRESULTS")
    print("Member axial stresses (ksi):")
    for m in range(1, 11):
        i, j = members[m]
        print(f"  m{m:02d} ({i}–{j}): {stresses[m-1]: .6f}")

    print("\nNodal displacements (in):")
    for nid in sorted(nodes.keys()):
        ux_idx, uy_idx = dof_index(nid)
        print(f"  node {nid}: ux = {U[ux_idx]: .6f}, uy = {U[uy_idx]: .6f}")

    print(f"\nTotal structural mass (lbm): {mass:.6f}")

    # Optional CSV output
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["member", "i", "j", "stress_ksi"])
            for m in range(1, 11):
                i, j = members[m]
                w.writerow([m, i, j, stresses[m - 1]])

            w.writerow([])
            w.writerow(["node", "ux_in", "uy_in"])
            for nid in sorted(nodes.keys()):
                ux_idx, uy_idx = dof_index(nid)
                w.writerow([nid, U[ux_idx], U[uy_idx]])

            w.writerow([])
            w.writerow(["mass_lbm", mass])
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
