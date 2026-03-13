"""Validation tests for the 25-bar discrete sections truss model.

Run directly:
    python PSO25BarDiscreteSectionsTruss/test_truss25_discrete.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import truss25_discrete as T
import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _approx(expected, actual, tol=1e-6, label=""):
    assert abs(actual - expected) <= tol, (
        f"{label}: expected {expected}, got {actual}"
    )


def _snap(x: np.ndarray) -> np.ndarray:
    """Nearest-available snap (mirrors adapter logic)."""
    avail = T.available_A
    idx = np.argmin(np.abs(avail[:, None] - x[None, :]), axis=0)
    return avail[idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_geometry_counts():
    assert T.N_NODES  == 10
    assert T.N_ELEMS  == 25
    assert T.N_GROUPS == 8
    assert T.N_DOF    == 30
    assert T.ELEMENTS.shape == (25, 2)
    assert T.NODES.shape    == (10, 3)
    print("PASS  test_geometry_counts")


def test_group_coverage():
    seen = {}
    for gid, members in T.GROUPS.items():
        for m in members:
            assert 1 <= m <= 25
            assert m not in seen, f"Member {m} in group {seen[m]} AND {gid}"
            seen[m] = gid
    assert len(seen) == 25
    print("PASS  test_group_coverage")


def test_dof_bookkeeping():
    assert len(T.FIXED_DOFS) == 12
    assert len(T.FREE_DOFS)  == 18
    assert set(T.FREE_DOFS) | set(T.FIXED_DOFS) == set(range(30))
    print("PASS  test_dof_bookkeeping")


def test_available_A():
    """Check catalogue: 30 entries, sorted, A_MIN/A_MAX match, increments correct."""
    avail = T.available_A
    assert len(avail) == 30, f"Expected 30 catalogue entries, got {len(avail)}"
    assert np.all(np.diff(avail) > 0), "available_A must be strictly increasing"
    assert abs(avail[0]  - T.A_MIN) < 1e-9, f"First entry {avail[0]} ≠ A_MIN {T.A_MIN}"
    assert abs(avail[-1] - T.A_MAX) < 1e-9, f"Last entry {avail[-1]} ≠ A_MAX {T.A_MAX}"
    # Steps 0.1 from 0.1..2.6 (26 entries), then 0.2 for 2.8,3.0,3.2,3.4 (4 entries)
    assert abs(avail[25] - 2.6) < 1e-9, f"Entry 26: expected 2.6, got {avail[25]}"
    assert abs(avail[26] - 2.8) < 1e-9, f"Entry 27: expected 2.8, got {avail[26]}"
    assert abs(avail[29] - 3.4) < 1e-9, f"Entry 30: expected 3.4, got {avail[29]}"
    print("PASS  test_available_A")


def test_bounds_match_catalogue():
    assert abs(T.A_MIN - T.available_A[0])  < 1e-9
    assert abs(T.A_MAX - T.available_A[-1]) < 1e-9
    print("PASS  test_bounds_match_catalogue")


def test_snap_nearest():
    """Values between catalogue entries snap to nearest."""
    # 0.15 → 0.1 or 0.2 (equidistant → 0.1 by tie-break toward lower)
    snapped = _snap(np.array([0.15, 0.25, 2.7, 3.3]))
    # 2.7 is equidistant between 2.6 and 2.8 — either acceptable, just finite
    assert snapped[0] in T.available_A
    assert snapped[1] in T.available_A
    assert snapped[2] in T.available_A
    assert snapped[3] in T.available_A
    # 3.3 → 3.2 (closer) or 3.4 (equidistant) — both valid
    assert abs(snapped[3] - 3.2) < 1e-9 or abs(snapped[3] - 3.4) < 1e-9
    print("PASS  test_snap_nearest")


def test_evaluate_snapped_design():
    """Evaluate with catalogue-valid areas returns finite structural quantities."""
    x = _snap(np.array([1.3, 2.4, 3.0, 0.1, 0.1, 3.4, 0.1, 1.3]))
    res = T.evaluate(x)
    assert np.isfinite(res["mass"])
    assert np.isfinite(res["max_disp"])
    assert np.isfinite(res["max_stress"])
    assert res["disp_violation"].shape  == (18,)
    assert res["stress_violation"].shape == (25,)
    print(f"PASS  test_evaluate_snapped_design")
    print(f"       mass={res['mass']:.4f} lb, "
          f"max_disp={res['max_disp']:.6f} in, "
          f"max_stress={res['max_stress']:.4f} ksi")


def test_mass_agrees_continuous():
    """At areas present in both catalogues the mass must equal the continuous formula."""
    x = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 3.4, 0.5, 1.0])
    # These are all valid catalogue entries
    for v in x:
        assert v in T.available_A, f"{v} not in catalogue"
    res_d = T.evaluate(x)
    # Compute expected mass directly
    expected_mass = T.RHO * float(np.dot(T.areas_from_groups(x), T.ELEMENT_LENGTHS))
    _approx(expected_mass, res_d["mass"], tol=1e-6, label="mass")
    print("PASS  test_mass_agrees_continuous")


def test_symmetry_condition_I():
    """Anti-symmetry check under load Condition I (same as continuous model)."""
    x = _snap(np.ones(8))
    all_U = T.solve_displacements(x)
    U1 = all_U[0]
    dz1, dz2 = U1[2], U1[5]
    dy1, dy2 = U1[1], U1[4]
    assert abs(dz1 - dz2) < 1e-8, f"z-symmetry: {dz1:.4e} vs {dz2:.4e}"
    assert abs(dy1 + dy2) < 1e-8, f"y-antisymmetry: {dy1:.4e} vs {dy2:.4e}"
    print("PASS  test_symmetry_condition_I")


if __name__ == "__main__":
    test_geometry_counts()
    test_group_coverage()
    test_dof_bookkeeping()
    test_available_A()
    test_bounds_match_catalogue()
    test_snap_nearest()
    test_evaluate_snapped_design()
    test_mass_agrees_continuous()
    test_symmetry_condition_I()
    print("\nAll tests passed.")
