from __future__ import annotations

import numpy as np

import truss200


def test_geometry_counts() -> None:
    assert truss200.N_NODES == 77
    assert truss200.N_ELEMS == 200
    assert truss200.N_GROUPS == 29


def test_group_members_cover_all_elements_once() -> None:
    members = []
    for group_id in range(1, truss200.N_GROUPS + 1):
        members.extend(truss200.group_members(group_id))
    assert sorted(members) == list(range(1, truss200.N_ELEMS + 1))


def test_supports_and_loads() -> None:
    loads = truss200.applied_loads()
    assert loads.shape == (2 * truss200.N_NODES,)
    assert np.isclose(loads[0], 1.0)
    assert np.isclose(loads[1], -10.0)
    assert np.isclose(loads[2 * (76 - 1)], 0.0)
    assert np.isclose(loads[2 * (77 - 1) + 1], 0.0)


def test_uniform_design_solves() -> None:
    areas = np.full(truss200.N_GROUPS, 1.5, dtype=float)
    result = truss200.evaluate(areas)
    assert result["mass"] > 0.0
    assert result["max_disp"] >= 0.0
    assert result["U"].shape == (2 * truss200.N_NODES,)
    assert np.all(np.isfinite(result["U"]))
