"""Validation tests for the 25-bar space truss model (truss25.py).

Run directly:
    python PSO25BarTruss/test_truss25.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import truss25 as T
import numpy as np


def test_geometry_counts():
    assert T.N_NODES  == 10,  f"Expected 10 nodes, got {T.N_NODES}"
    assert T.N_ELEMS  == 25,  f"Expected 25 elements, got {T.N_ELEMS}"
    assert T.N_GROUPS == 8,   f"Expected 8 groups, got {T.N_GROUPS}"
    assert T.N_DOF    == 30,  f"Expected 30 DOFs, got {T.N_DOF}"
    assert T.ELEMENTS.shape == (25, 2), f"Elements shape wrong: {T.ELEMENTS.shape}"
    assert T.NODES.shape    == (10, 3), f"Nodes shape wrong: {T.NODES.shape}"
    print("PASS  test_geometry_counts")


def test_group_coverage():
    """Every one of the 25 members must appear in exactly one group."""
    seen = {}
    for gid, members in T.GROUPS.items():
        for m in members:
            assert 1 <= m <= 25, f"Member {m} out of range"
            assert m not in seen, f"Member {m} listed in group {seen[m]} AND {gid}"
            seen[m] = gid
    assert len(seen) == 25, f"Only {len(seen)} distinct members covered (expected 25)"
    # Verify counts per group
    expected = {1: 1, 2: 4, 3: 4, 4: 2, 5: 2, 6: 4, 7: 4, 8: 4}
    for gid, cnt in expected.items():
        actual = len(T.GROUPS[gid])
        assert actual == cnt, f"Group {gid}: expected {cnt} members, got {actual}"
    print("PASS  test_group_coverage")


def test_dof_bookkeeping():
    """Fixed nodes 7-10 give 12 fixed DOFs; nodes 1-6 give 18 free DOFs."""
    assert len(T.FIXED_DOFS) == 12, f"Expected 12 fixed DOFs, got {len(T.FIXED_DOFS)}"
    assert len(T.FREE_DOFS)  == 18, f"Expected 18 free DOFs, got {len(T.FREE_DOFS)}"
    assert set(T.FREE_DOFS) | set(T.FIXED_DOFS) == set(range(30))
    assert T.FREE_NODES == (1, 2, 3, 4, 5, 6)
    print("PASS  test_dof_bookkeeping")


def test_load_vectors():
    """Verify load vectors have correct non-zero entries."""
    F1, F2 = T.LOAD_VECTORS

    # Condition I: Node 1 → Py=+20, Pz=-5; Node 2 → Py=-20, Pz=-5
    assert F1[3 * 0 + 1] == pytest_approx(20.0),   "F1: Node1 Py"
    assert F1[3 * 0 + 2] == pytest_approx(-5.0),   "F1: Node1 Pz"
    assert F1[3 * 1 + 1] == pytest_approx(-20.0),  "F1: Node2 Py"
    assert F1[3 * 1 + 2] == pytest_approx(-5.0),   "F1: Node2 Pz"
    assert F1[3 * 0 + 0] == pytest_approx(0.0),    "F1: Node1 Px should be 0"

    # Condition II
    assert F2[3 * 0 + 0] == pytest_approx(1.0),   "F2: Node1 Px"
    assert F2[3 * 0 + 1] == pytest_approx(10.0),  "F2: Node1 Py"
    assert F2[3 * 2 + 0] == pytest_approx(0.5),   "F2: Node3 Px"
    assert F2[3 * 5 + 0] == pytest_approx(0.5),   "F2: Node6 Px"
    print("PASS  test_load_vectors")


def pytest_approx(val, rel=1e-9):
    """Thin float comparison helper (no pytest needed)."""
    class _Approx:
        def __init__(self, v):
            self.v = v
        def __eq__(self, other):
            if self.v == 0.0:
                return abs(other) < 1e-12
            return abs(other - self.v) / max(abs(self.v), 1e-12) < rel
        def __repr__(self):
            return f"~{self.v}"
    return _Approx(val)


def test_element_lengths():
    """Spot-check a few element lengths by hand."""
    # Member 1: (1,2) → (0,37.5,200)→(0,-37.5,200), length=75
    assert abs(T.ELEMENT_LENGTHS[0] - 75.0) < 1e-6, f"Member 1 length: {T.ELEMENT_LENGTHS[0]}"
    # Member 2: (1,3) → (0,37.5,200)→(-37.5,37.5,100), Δ=(-37.5,0,-100)
    # length = sqrt(37.5²+0²+100²) = sqrt(11406.25) ≈ 106.802
    expected_A2 = np.sqrt(37.5**2 + 100.0**2)
    assert abs(T.ELEMENT_LENGTHS[1] - expected_A2) < 1e-6, f"Member 2 length: {T.ELEMENT_LENGTHS[1]}"
    # Member 10: (3,4) → (-37.5,37.5,100)→(37.5,37.5,100), Δ=(75,0,0), length=75
    assert abs(T.ELEMENT_LENGTHS[9] - 75.0) < 1e-6, f"Member 10 length: {T.ELEMENT_LENGTHS[9]}"
    print("PASS  test_element_lengths")


def test_uniform_design_evaluates():
    """All-A_MIN design must solve without error; all-ones gives reasonable mass."""
    areas_min = np.full(8, T.A_MIN)
    res_min = T.evaluate(areas_min)
    assert np.isfinite(res_min["mass"]), "mass must be finite"
    assert np.isfinite(res_min["max_disp"]), "max_disp must be finite"
    assert np.isfinite(res_min["max_stress"]), "max_stress must be finite"
    assert res_min["disp_violation"].shape  == (18,), "disp violation shape"
    assert res_min["stress_violation"].shape == (25,), "stress violation shape"

    areas_one = np.ones(8)
    res_one = T.evaluate(areas_one)
    # Mass with uniform 1.0 in²: ρ × Σ Lᵢ
    expected_mass = T.RHO * float(np.sum(T.ELEMENT_LENGTHS))
    assert abs(res_one["mass"] - expected_mass) < 1e-3, (
        f"Mass mismatch: {res_one['mass']:.4f} vs expected {expected_mass:.4f}"
    )
    print(f"PASS  test_uniform_design_evaluates")
    print(f"       uniform 1.0 in²: mass={res_one['mass']:.4f} lb, "
          f"max_disp={res_one['max_disp']:.6f} in, "
          f"max_stress={res_one['max_stress']:.4f} ksi")


def test_groups_produce_correct_areas():
    """Check that areas_from_groups correctly expands groups."""
    areas = np.arange(1.0, 9.0)   # group values 1.0 … 8.0
    elem_areas = T.areas_from_groups(areas)
    assert elem_areas[0]  == 1.0, f"Element 1 (Group 1): {elem_areas[0]}"   # A1
    assert elem_areas[1]  == 2.0, f"Element 2 (Group 2): {elem_areas[1]}"   # A2
    assert elem_areas[9]  == 4.0, f"Element 10 (Group 4): {elem_areas[9]}"  # A4
    assert elem_areas[24] == 8.0, f"Element 25 (Group 8): {elem_areas[24]}" # A8
    print("PASS  test_groups_produce_correct_areas")


def test_symmetry_condition_I():
    """Under Condition I the structure has anti-symmetry about y=0.

    Nodes 1 & 2 are symmetric about y=0, as are force magnitudes (Py anti-symmetric).
    Consequently, the z-displacements of node 1 and node 2 should be equal,
    and the y-displacements should be equal in magnitude but opposite.
    """
    areas = np.full(8, 1.0)
    all_U = T.solve_displacements(areas)
    U1 = all_U[0]   # load case I

    # z-disp node 1 (DOF 2) vs z-disp node 2 (DOF 5)
    dz1 = U1[2]
    dz2 = U1[5]
    assert abs(dz1 - dz2) < 1e-8, f"Pz symmetry violated: dz1={dz1:.4e}, dz2={dz2:.4e}"

    # y-disp node 1 (DOF 1) vs y-disp node 2 (DOF 4) should be equal & opposite
    dy1 = U1[1]
    dy2 = U1[4]
    assert abs(dy1 + dy2) < 1e-8, f"Py anti-symmetry violated: dy1={dy1:.4e}, dy2={dy2:.4e}"
    print("PASS  test_symmetry_condition_I")


if __name__ == "__main__":
    test_geometry_counts()
    test_group_coverage()
    test_dof_bookkeeping()
    test_load_vectors()
    test_element_lengths()
    test_uniform_design_evaluates()
    test_groups_produce_correct_areas()
    test_symmetry_condition_I()
    print("\nAll tests passed.")
