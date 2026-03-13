# PSO200BarTruss

Continuous-size 200-bar planar truss benchmark based on the attached topology and grouping.

## Problem definition

- Material modulus: `E = 30,000 ksi`
- Material density: `rho = 0.283 lb/in^3`
- Area bounds: `0.1 <= A_g <= 2.0 in^2` for each of the 29 groups
- Objective: minimize total structural weight
- Constraint: maximum absolute nodal displacement over all free DOFs must not exceed `4 in`
- Loads:
  - `+1 kip` in `x` at joints `1, 6, 15, 20, 29, 34, 43, 48, 57, 62`
  - `-10 kip` in `y` at joints `1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 29, 30, 31, 32, 33, 34, 36, 38, 40, 42, 43, 44, 45, 46, 47, 48, 50, 52, 54, 56, 58, 59, 60, 61, 62, 64, 66, 68, 70, 71, 72, 73, 74, 75`

## Files

- `truss200.py`: 2D truss geometry, grouped areas, load vector, stiffness assembly, and evaluation helpers.
- `test_truss200.py`: basic geometry and solver validation.

## Topology notes

- Nodes `1..75` follow the figure numbering.
- Nodes `76` and `77` are the lower support nodes.
- The rectangular tower uses alternating 5-node and 9-node rows.
- Members `1..200` and groups `1..29` match the attached numbering and grouping table.

## Running the analyses

The shared analysis tooling now exposes this problem as `truss200_continuous`.

Landscape analysis:

```bash
python "TrussLandscapeAnalysis/run_all_truss_landscapes.py" --problems truss200_continuous
```

Fixed-coefficient PSO using the landscape recommendation:

```bash
python "PSO FEA/run_pso_fea.py" --problem truss200_continuous --coeff-mode fixed
```

Outputs are written to the shared results folders:

- `TrussLandscapeAnalysis/results/truss200_continuous/`
- `PSO FEA/results/truss200_continuous/`