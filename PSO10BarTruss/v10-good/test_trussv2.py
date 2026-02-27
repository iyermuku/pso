#!/usr/bin/env python3
"""
Extended test script for the 10‑member truss.

New features:
  • Accept an input CSV of multiple experiments (columns of areas) and write a single output CSV
    containing, for each experiment, per‑member stresses, end‑node displacements, mass, and
    a short status message about constraint satisfaction (stress/displacement).
  • Compare reported (provided) weight from the input file against the computed mass and flag
    a mismatch when the absolute difference exceeds a tolerance (default 1 lbm).

Input CSV format expected (like the file derived from the two tables in the paper image):
  - First column: "Truss area" (rows "01".."10" hold the cross‑sectional areas in in^2).
  - Last row: "Weight" with values per experiment (optional; can be absent).
  - All other columns are experiment names (e.g., "PSO best", "Haftka and Grdal", etc.).
  - Any non‑numeric columns besides the first (e.g., accidental "Table") will be ignored.

Examples:
  # single run (unchanged behavior)
  python test_truss.py 33.5 0.1 22.766 14.417 0.1 0.1 7.534 20.467 20.392 0.1 --csv out_single.csv

  # batch run from CSV
  python test_truss.py --in-csv 10bartruss_results.csv --out-csv results_out.csv

Outputs:
  • A CSV with columns:
      experiment, member, i_node, j_node, stress_ksi,
      ui_x_in, ui_y_in, uj_x_in, uj_y_in,
      mass_lbm, status,
      provided_weight_lbm, computed_weight_lbm, weight_diff_lbm, weight_mismatch_flag

Units and constraints:
  • Stresses in ksi; displacements in inches; mass in lbm (consistent with canonical 10‑bar truss).
  • Stress limit: ±25 ksi (configurable via --stress-limit).
  • Displacement limit: ±2 in (configurable via --disp-limit).

"""
import argparse
import csv
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

# --- truss model API (must be provided by your project) ---
from truss_model import (
    solve_displacements,  # U = solve_displacements(A)
    solve_stresses,       # stresses = solve_stresses(A)
    mass_from_A,          # mass_lbm = mass_from_A(A)
    nodes,                # dict of node_id -> (x,y) or similar (used for iteration)
    dof_index,            # dof_index(node_id) -> (ux_idx, uy_idx)
    members,              # dict: member_id (1..10) -> (i_node, j_node)
)

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def read_experiments_from_csv(path: str) -> Tuple[List[str], Dict[str, List[float]], Dict[str, Optional[float]]]:
    """Read the batch input CSV and return experiment names, areas per experiment,
    and provided weights per experiment (if present)."""
    import pandas as pd
    df = pd.read_csv(path)

    if df.columns[0].strip().lower() not in {"truss area", "truss_area", "area"}:
        raise ValueError(
            "Expected first column to be 'Truss area' with rows '01'..'10' and a final 'Weight' row."
        )

    # Normalize the first column values as strings like '01'..'10' or 'Weight'
    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str).str.strip()

    # Candidate experiment columns: all except first; drop clearly non-numeric columns (e.g., 'Table')
    exp_cols = []
    for c in df.columns[1:]:
        # If column has at least one numeric in the area rows, keep it
        area_rows = df[df[first_col].str.match(r"^(0?[1-9]|10)$")]
        try:
            _ = pd.to_numeric(area_rows[c], errors='coerce')
            if _.notna().any():
                exp_cols.append(c)
        except Exception:
            # Not numeric; skip
            pass

    if not exp_cols:
        raise ValueError("No experiment columns with numeric areas found in input CSV.")

    # Build areas per experiment
    areas_by_exp: Dict[str, List[float]] = {}
    provided_weight_by_exp: Dict[str, Optional[float]] = {}

    # Ensure area rows are ordered 01..10
    area_df = df[df[first_col].str.match(r"^(0?[1-9]|10)$")].copy()
    area_df[first_col] = area_df[first_col].str.zfill(2)
    area_df = area_df.sort_values(first_col)

    # Optional weight row
    weight_row = df[df[first_col].str.lower().eq("weight")]

    for exp in exp_cols:
        vals = pd.to_numeric(area_df[exp], errors='coerce').tolist()
        if len(vals) != 10 or any(math.isnan(v) for v in vals):
            raise ValueError(f"Experiment '{exp}' must have 10 numeric area values (rows 01..10).")
        areas_by_exp[exp] = vals
        if not weight_row.empty:
            w = pd.to_numeric(weight_row[exp], errors='coerce')
            provided_weight_by_exp[exp] = float(w.iloc[0]) if w.notna().any() else None
        else:
            provided_weight_by_exp[exp] = None

    return exp_cols, areas_by_exp, provided_weight_by_exp


def analyze_experiment(exp_name: str, A: List[float], stress_limit: float, disp_limit: float,
                       provided_weight: Optional[float], weight_tol: float) -> Tuple[np.ndarray, np.ndarray, float, str, float, bool]:
    """Run the truss analysis and assemble status information.
    Returns (stresses, U, mass, status, weight_diff, mismatch_flag).
    """
    A_vec = np.array(A, dtype=float)
    U = solve_displacements(A_vec)
    stresses = solve_stresses(A_vec)
    mass = mass_from_A(A_vec)

    # Constraints
    stress_viol = bool(np.any(np.abs(stresses) > stress_limit))
    disp_viol = bool(np.any(np.abs(U) > disp_limit))
    if stress_viol and disp_viol:
        status = "both violated"
    elif stress_viol:
        status = "stress violated"
    elif disp_viol:
        status = "displacement violated"
    else:
        status = "all constraints satisfied"

    # Weight comparison
    if provided_weight is not None:
        weight_diff = abs(float(provided_weight) - float(mass))
        mismatch_flag = bool(weight_diff > weight_tol)
    else:
        weight_diff = float('nan')
        mismatch_flag = False

    return stresses, U, mass, status, weight_diff, mismatch_flag


def write_output_csv(out_path: str, results_rows: List[Dict]):
    fields = [
        "experiment", "member", "i_node", "j_node", "stress_ksi",
        "ui_x_in", "ui_y_in", "uj_x_in", "uj_y_in",
        "mass_lbm", "status",
        "provided_weight_lbm", "computed_weight_lbm", "weight_diff_lbm", "weight_mismatch_flag",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results_rows:
            w.writerow(row)


# -----------------------------------------------------------
# Main CLI
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute axial stresses (ksi), nodal displacements (in), and mass (lbm) for the 10-member "
            "truss. Accepts either 10 areas via CLI or a CSV with multiple experiments."
        )
    )
    parser.add_argument("areas", nargs='*', type=float, metavar="A",
                        help="Cross-sectional areas for members 1..10 (in^2). If omitted, use --in-csv.")
    parser.add_argument("--csv", metavar="OUT.csv", help="Optional path to save results as CSV (single run).")

    # Batch processing options
    parser.add_argument("--in-csv", metavar="IN.csv", help="Path to input CSV containing experiments.")
    parser.add_argument("--out-csv", metavar="OUT.csv", help="Path to output CSV for batch results.")

    # Constraint and tolerance settings
    parser.add_argument("--stress-limit", type=float, default=25.0, help="Allowable |stress| in ksi (default 25).")
    parser.add_argument("--disp-limit", type=float, default=2.0, help="Allowable |displacement| in inches (default 2).")
    parser.add_argument("--weight-tolerance", type=float, default=1.0,
                        help="Tolerance in lbm to flag provided vs computed weight mismatch (default 1 lbm).")

    args = parser.parse_args()

    # Mode selection
    if args.in_csv:
        if not args.out_csv:
            raise SystemExit("--out-csv is required when using --in-csv")
        exp_names, areas_by_exp, provided_weights = read_experiments_from_csv(args.in_csv)

        results: List[Dict] = []
        for exp in exp_names:
            stresses, U, mass, status, wdiff, mismatch = analyze_experiment(
                exp, areas_by_exp[exp], args.stress_limit, args.disp_limit, provided_weights.get(exp), args.weight_tolerance
            )
            # Per-member rows
            for m in range(1, 11):
                i_node, j_node = members[m]
                ux_i, uy_i = dof_index(i_node)
                ux_j, uy_j = dof_index(j_node)
                row = {
                    "experiment": exp,
                    "member": m,
                    "i_node": i_node,
                    "j_node": j_node,
                    "stress_ksi": float(stresses[m-1]),
                    "ui_x_in": float(U[ux_i]),
                    "ui_y_in": float(U[uy_i]),
                    "uj_x_in": float(U[ux_j]),
                    "uj_y_in": float(U[uy_j]),
                    "mass_lbm": float(mass),
                    "status": status,
                    "provided_weight_lbm": float(provided_weights.get(exp)) if provided_weights.get(exp) is not None else float('nan'),
                    "computed_weight_lbm": float(mass),
                    "weight_diff_lbm": float(wdiff),
                    "weight_mismatch_flag": bool(mismatch),
                }
                results.append(row)
        write_output_csv(args.out_csv, results)
        print(f"Saved batch results to: {args.out_csv}")
        return

    # Single-run mode (backwards compatible)
    if len(args.areas) != 10:
        raise SystemExit("Provide exactly 10 areas or use --in-csv for batch mode.")
    A_vec = np.array(args.areas, dtype=float)
    U = solve_displacements(A_vec)
    stresses = solve_stresses(A_vec)
    mass = mass_from_A(A_vec)

    # Pretty print (unchanged)
    print("INPUT")
    print(f"Areas (in^2) [m01..m10]: {A_vec.tolist()}")
    print("\nRESULTS")
    print("Member axial stresses (ksi):")
    for m in range(1, 11):
        i, j = members[m]
        print(f" m{m:02d} ({i}-{j}): {stresses[m-1]: .6f}")
    print("\nNodal displacements (in):")
    for nid in sorted(nodes.keys()):
        ux_idx, uy_idx = dof_index(nid)
        print(f" node {nid}: ux = {U[ux_idx]: .6f}, uy = {U[uy_idx]: .6f}")
    print(f"\nTotal structural mass (lbm): {mass:.6f}")

    if args.csv:
        # Compose single-result CSV
        fields = [
            "experiment", "member", "i_node", "j_node", "stress_ksi",
            "ui_x_in", "ui_y_in", "uj_x_in", "uj_y_in",
            "mass_lbm",
        ]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for m in range(1, 11):
                i_node, j_node = members[m]
                ux_i, uy_i = dof_index(i_node)
                ux_j, uy_j = dof_index(j_node)
                w.writerow({
                    "experiment": "manual",
                    "member": m,
                    "i_node": i_node,
                    "j_node": j_node,
                    "stress_ksi": float(stresses[m-1]),
                    "ui_x_in": float(U[ux_i]),
                    "ui_y_in": float(U[uy_i]),
                    "uj_x_in": float(U[ux_j]),
                    "uj_y_in": float(U[uy_j]),
                    "mass_lbm": float(mass),
                })
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
