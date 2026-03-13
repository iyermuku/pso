# 25-Bar Space Truss — PSO Minimum-Weight Optimisation

## Problem Description

The 25-bar space truss is a classical benchmark in structural optimisation,
introduced by Schmit & Farshi (1974) and widely used to compare metaheuristic
methods.  It consists of a 3-D tower with **10 nodes**, **25 members**, and
**8 independent design variables** (cross-sectional areas) arising from
the double symmetry of the structure about the x-z and y-z planes.

---

## Geometry

| Level | Nodes | Coordinates (x, y, z) in inches |
|-------|-------|----------------------------------|
| Top   | 1, 2  | (0, ±37.5, 200) |
| Mid   | 3–6   | (±37.5, ±37.5, 100) |
| Base  | 7–10  | (±100, ±100, 0) — **pin supports** |

The base spans 200 in × 200 in; the tower rises 200 in (two 100-in tiers).

---

## Member Groups

| Group | Members | Description |
|-------|---------|-------------|
| A₁    | 1 | Top chord (nodes 1–2) |
| A₂    | 2, 3, 4, 5 | Direct diagonals, top → mid (same y-half) |
| A₃    | 6, 7, 8, 9 | Cross diagonals, top → mid (opposite y-half) |
| A₄    | 10, 11 | Upper-ring, parallel to x-axis |
| A₅    | 12, 13 | Upper-ring, parallel to y-axis |
| A₆    | 14–17 | Direct diagonals, mid → base (same quadrant) |
| A₇    | 18–21 | Cross-x diagonals, mid → base (opposite x, same y) |
| A₈    | 22–25 | Full cross diagonals, mid → base (longest members) |

---

## Load Cases

Two load conditions are applied simultaneously; both must be satisfied.

**Condition I** (kips):

| Node | Pₓ | Pᵧ | P_z |
|------|----|----|-----|
| 1    | 0  | +20| −5  |
| 2    | 0  | −20| −5  |

**Condition II** (kips):

| Node | Pₓ  | Pᵧ | P_z |
|------|-----|----|-----|
| 1    | 1.0 | +10| −5  |
| 2    | 0.0 | +10| −5  |
| 3    | 0.5 | 0  |  0  |
| 6    | 0.5 | 0  |  0  |

---

## Material & Constraint Parameters

| Parameter | Value |
|-----------|-------|
| Elastic modulus E | 10,000 ksi |
| Mass density ρ | 0.1 lb/in³ |
| Allowable displacement | ±0.35 in (all x, y, z directions, all free nodes) |
| Allowable stress | ±40 ksi (tension and compression) |
| Area bounds | A_min = 0.01 in², A_max = 3.4 in² |

---

## Objective

Minimise total structural mass:

    mass = ρ · Σᵢ Aᵢ · Lᵢ   (lb)

subject to displacement and stress constraints satisfied under **both** load cases.

---

## File Structure

```
PSO25BarTruss/
├── truss25.py          # 3-D FEA model: geometry, solver, evaluate()
├── test_truss25.py     # Validation tests (8 tests, run directly)
└── README.md           # This file
```

### `truss25.py` — key API

| Symbol / Function | Description |
|---|---|
| `E`, `RHO`, `U_ALLOW`, `S_ALLOW` | Material and constraint constants |
| `A_MIN`, `A_MAX` | Area bounds |
| `NODES` | 10 × 3 coordinate array |
| `ELEMENTS` | 25 × 2 connectivity (1-indexed) |
| `GROUPS` | Dict mapping group id → member list |
| `LOAD_VECTORS` | List of two force vectors (kips) |
| `FREE_DOFS`, `FIXED_DOFS` | DOF index tuples |
| `ELEMENT_LENGTHS` | Pre-computed element lengths |
| `areas_from_groups(x)` | Expand 8-vector → 25 element areas |
| `mass_from_groups(x)` | Compute mass (lb) |
| `solve_displacements(x)` | Return list of displacement vectors (one per load case) |
| `member_stresses(U)` | Axial stresses (ksi) for one displacement vector |
| `displacement_violation_vector(all_U)` | Max violation per free DOF over all load cases |
| `stress_violation_vector(all_stresses)` | Max violation per element over all load cases |
| `evaluate(x)` | Full evaluation dict (mass, U, stresses, max_disp, max_stress, violations) |
| `grouped_design_bounds()` | Return (lo, hi) arrays of shape (8,) |

---

## Baseline Results (Uniform 1.0 in² Design)

| Quantity | Value |
|----------|-------|
| Structural mass | 345.73 lb |
| Max displacement | 0.859 in  (exceeds 0.35 in limit — needs optimisation) |
| Max stress | 19.13 ksi (within ±40 ksi limit) |

---

## Running Tests

```bash
python PSO25BarTruss/test_truss25.py
```

All 8 tests cover:
1. Geometry counts (nodes, elements, groups, DOFs)
2. Group member coverage (all 25 members in exactly one group)
3. DOF bookkeeping (18 free, 12 fixed)
4. Load vector assembly
5. Element length spot-checks
6. Uniform design evaluation (finite mass / disp / stress)
7. `areas_from_groups` expansion
8. Anti-symmetry check under Condition I

---

## Running PSO Optimisation

```bash
# Landscape analysis first (determines recommended PSO coefficients)
python TrussLandscapeAnalysis/run_all_truss_landscapes.py --problems truss25_continuous

# PSO with fixed (landscape-recommended) coefficients
python "PSO FEA/run_pso_fea.py" --problem truss25_continuous --coeff-mode fixed

# PSO with two-phase exploration/refinement schedule
python "PSO FEA/run_pso_fea.py" --problem truss25_continuous --coeff-mode two-phase
```

---

## References

1. Schmit, L.A., Farshi, B. (1974). Some approximation concepts for structural synthesis.
   *AIAA Journal*, 12(5), 692–699.
2. Kamat, M.P., Hayduk, R.J. (1985). Recent advances in structural optimization.
   *Journal of Aircraft*, 22(12), 1065–1071.
3. Kennedy, J., Eberhart, R. (1995). Particle swarm optimization.
   *Proc. IEEE ICNN*, 1942–1948.
