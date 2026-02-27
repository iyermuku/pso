
#!/usr/bin/env python3
"""
Test constraints for 72-bar truss benchmark columns in a CSV (Table 8).
Input: table8_72bar_truss.csv (with rows A01..A16 and Weight; columns are benchmarks)
Output: console report per column:
 - Whether stress (S_ALLOW) or displacement (U_ALLOW) was violated (or both / satisfied)
 - If violation, by how much (absolute exceedance)
 - If weight mismatch vs table Weight, the signed difference (computed_mass - table_weight)

Updates:
- Constraints considered satisfied within 1% tolerance of allowable values.
  (i.e., violation only if max value > 1.01 * allowable)
- Weight mismatch reported only when its magnitude exceeds 1% of the table weight.
  Suppressed in the CSV unless over 1%.

Assumptions & references:
- 72-bar truss benchmark grouping and load cases from CoFE example:
  https://vtpasquale.github.io/NASTRAN_CoFE/2._Examples/b._Optimization/4._72-Bar_Truss_Optimization/
- This script uses the self-contained synthetic geometry in truss72.py for analysis.
  For benchmark-grade reproduction, swap in canonical node/connectivity data.
"""
import sys
import numpy as np
import pandas as pd
import truss72 as t72

CSV_DEFAULT = 'table8_72bar_truss.csv'

# --- New defaults ---
TOL_CONSTRAINT = 0.01  # 1% tolerance for U_ALLOW and S_ALLOW
TOL_WEIGHT = 0.01      # 1% tolerance for table Weight


def read_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect 'Area group' plus benchmark columns; ensure A01..A16 and Weight present
    required_rows = [f'A{str(i).zfill(2)}' for i in range(1, 17)] + ['Weight']
    if not set(required_rows).issubset(set(df['Area group'])):
        raise ValueError('CSV missing required rows A01..A16 and Weight')
    return df


def check_constraints(A16: np.ndarray, table_weight: float,
                      tol_constraint: float = TOL_CONSTRAINT,
                      tol_weight: float = TOL_WEIGHT):
    # clip areas to bounds
    A16 = np.clip(np.asarray(A16, dtype=float), t72.A_MIN, t72.A_MAX)
    res = t72.evaluate(A16)
    mass = float(res['mass'])

    # Displacement check: nodes 1..4, X/Y components only as per benchmark
    max_disp = 0.0
    for U in res['U']:
        for nid in [1, 2, 3, 4]:
            ux = abs(U[3 * (nid - 1) + 0])
            uy = abs(U[3 * (nid - 1) + 1])
            max_disp = max(max_disp, ux, uy)

    disp_allow_nominal = float(t72.U_ALLOW)
    disp_allow_eff = (1.0 + tol_constraint) * disp_allow_nominal  # effective allowable w/ 1% slack
    # violation only if above the effective allowable
    disp_violation = max(0.0, max_disp - disp_allow_eff)

    # Stress check: max sigma over members across load cases
    max_stress = 0.0
    A_members = t72.areas_from_groups(A16)
    for i, U in enumerate(res['U']):
        sig = t72.member_stresses(U, A_members)
        max_stress = max(max_stress, float(np.max(np.abs(sig))))

    stress_allow_nominal = float(t72.S_ALLOW)
    stress_allow_eff = (1.0 + tol_constraint) * stress_allow_nominal  # effective allowable w/ 1% slack
    # violation only if above the effective allowable
    stress_violation = max(0.0, max_stress - stress_allow_eff)

    # Weight mismatch (signed)
    weight_mismatch = mass - float(table_weight)
    # Only flag if magnitude exceeds 1% of table weight
    weight_thresh = abs(tol_weight * float(table_weight))
    weight_mismatch_over_1pct = abs(weight_mismatch) > weight_thresh

    # Status
    if disp_violation <= 1e-12 and stress_violation <= 1e-12:
        status = 'constraints satisfied'
    elif disp_violation > 1e-12 and stress_violation > 1e-12:
        status = 'both violated'
    elif disp_violation > 1e-12:
        status = 'displacement violated'
    else:
        status = 'stress violated'

    return {
        'mass': mass,
        'max_disp': max_disp,
        'disp_allow_nominal': disp_allow_nominal,
        'disp_allow_eff': disp_allow_eff,
        'disp_violation': disp_violation,
        'max_stress': max_stress,
        'stress_allow_nominal': stress_allow_nominal,
        'stress_allow_eff': stress_allow_eff,
        'stress_violation': stress_violation,
        'weight_mismatch': weight_mismatch,
        'weight_mismatch_over_1pct': weight_mismatch_over_1pct,
        'weight_thresh_abs': weight_thresh,
        'status': status,
    }


def main(csv_path: str):
    df = read_table(csv_path)
    area_rows = [f'A{str(i).zfill(2)}' for i in range(1, 17)]
    # Columns to evaluate: everything except 'Area group'
    columns = [c for c in df.columns if c.strip().lower() != 'area group']
    # Build row lookup
    df_idx = df.set_index('Area group')

    print(f"\n=== Constraint & weight checks for: {csv_path} ===")
    print(f"(Allowables from truss72.py) U_ALLOW = {t72.U_ALLOW} in, "
          f"S_ALLOW = {t72.S_ALLOW} ksi")
    print(f"Constraint tolerance: {int(TOL_CONSTRAINT*100)}% "
          f"(i.e., effective allowables = 1+{TOL_CONSTRAINT:.2%} of nominal)")
    print(f"Weight mismatch reporting threshold: {int(TOL_WEIGHT*100)}% of table weight\n")

    results = []

    for col in columns:
        try:
            A16 = df_idx.loc[area_rows, col].astype(float).to_numpy()
        except Exception:
            print(f"[WARN] Column '{col}' has non-numeric area entries; skipping.")
            continue

        table_weight = float(df_idx.loc['Weight', col])
        r = check_constraints(A16, table_weight)
        results.append((col, r))

        print(f"Column: {col}")
        print(f"  Status: {r['status']}")
        print(f"  Displacement: max U={r['max_disp']:.6f} in; "
              f"nominal allow={r['disp_allow_nominal']:.6f} in; "
              f"effective allow (1% tol)={r['disp_allow_eff']:.6f} in; "
              f"violation={r['disp_violation']:.6f} in")
        print(f"  Stress: max sigma={r['max_stress']:.6f} ksi; "
              f"nominal allow={r['stress_allow_nominal']:.6f} ksi; "
              f"effective allow (1% tol)={r['stress_allow_eff']:.6f} ksi; "
              f"violation={r['stress_violation']:.6f} ksi")

        if r['weight_mismatch_over_1pct']:
            print(f"  Weight: table={table_weight:.4f} lbm; "
                  f"computed={r['mass']:.4f} lbm; "
                  f"mismatch (computed - table) = {r['weight_mismatch']:.4f} lbm "
                  f"(> 1% threshold of {r['weight_thresh_abs']:.4f} lbm)")
        else:
            print(f"  Weight: table={table_weight:.4f} lbm; "
                  f"computed={r['mass']:.4f} lbm; mismatch within 1% "
                  f"(threshold {r['weight_thresh_abs']:.4f} lbm)")
        print()  # blank line between columns

    # Optional: save a CSV summary
    out_rows = []
    for col, r in results:
        # Only record weight mismatch when it exceeds 1%; otherwise leave blank
        out_rows.append({
            'column': col,
            'status': r['status'],
            'max_disp_in': r['max_disp'],
            'disp_allow_nominal_in': r['disp_allow_nominal'],
            'disp_allow_effective_in': r['disp_allow_eff'],
            'disp_violation_in': r['disp_violation'],
            'max_stress_ksi': r['max_stress'],
            'stress_allow_nominal_ksi': r['stress_allow_nominal'],
            'stress_allow_effective_ksi': r['stress_allow_eff'],
            'stress_violation_ksi': r['stress_violation'],
            'computed_mass_lbm': r['mass'],
            'weight_mismatch_lbm': (
                r['weight_mismatch'] if r['weight_mismatch_over_1pct'] else ''
            ),
            'weight_mismatch_over_1pct': r['weight_mismatch_over_1pct'],
        })
    out_df = pd.DataFrame(out_rows)
    out_path = 'constraint_check_summary.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Summary CSV written: {out_path}")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    main(csv_path)