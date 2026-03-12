# PSO FEA

Common PSO runner for all truss FEA problems in this repository.

## Included problems
- `truss10_continuous`
- `truss10_discrete`
- `truss72_continuous`
- `truss72_discrete`

## Design
- Reuses model / FEA logic from the existing truss folders.
- Uses landscape-analysis recommended coefficients from `TrussLandscapeAnalysis/results`.
- Uses Latin Hypercube Sampling (LHS) for initial swarm placement.
- Supports two coefficient modes:
	- `fixed`: fixed recommended `w`, `c1`, `c2`
	- `two-phase`: uses phase-1 and phase-2 schedules from landscape metrics
- Uses reflection at bounds and a constraint-violation reflection step during exploration.
- Uses Deb-style comparison for feasible/infeasible designs.

## Main files
- `problem_adapters.py` - extracts the truss-specific evaluation logic
- `common_pso.py` - shared PSO implementation
- `run_pso_fea.py` - single-problem CLI runner
- `run_batch_pso_fea.py` - multi-problem batch runner

## Usage

Run from the repository root:

```bash
python "PSO FEA/run_pso_fea.py" --problem truss10_continuous
python "PSO FEA/run_pso_fea.py" --problem truss10_discrete
python "PSO FEA/run_pso_fea.py" --problem truss72_continuous
python "PSO FEA/run_pso_fea.py" --problem truss72_discrete

# use landscape two-phase schedule
python "PSO FEA/run_pso_fea.py" --problem truss10_continuous --coeff-mode two-phase
```

Optional arguments:
- `--swarm-size`
- `--iters`
- `--seed`
- `--out-dir`
- `--coeff-mode` (`fixed` or `two-phase`)

## Batch usage

Run all problems:

```bash
python "PSO FEA/run_batch_pso_fea.py" --problems all --coeff-mode fixed
```

Run selected problems with two-phase scheduling:

```bash
python "PSO FEA/run_batch_pso_fea.py" --problems truss10_continuous,truss72_continuous --coeff-mode two-phase
```

Batch output files:
- `PSO FEA/results/batch_summary.json`
- `PSO FEA/results/batch_summary.md`

Batch summaries include:
- design variable vector
- best mass
- max displacement from best design
- max stress from best design
- final feasible fraction

## Outputs
For each problem run, results are written to:
- `PSO FEA/results/<problem_id>/`

Generated files include:
- `<problem_id>_run_summary.json`
- `<problem_id>_objective_convergence.png`
- `<problem_id>_mass_feasibility.png`
- `<problem_id>_constraint_violation.png`

`<problem_id>_run_summary.json` includes:
- `design_variables`
- `best_mass`
- `max_displacement`
- `max_stress`
- `final_feasible_fraction`
