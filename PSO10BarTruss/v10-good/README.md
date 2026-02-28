# 10-Bar Truss PSO Optimization

This directory contains Particle Swarm Optimization (PSO) implementations applied to the classic 10-bar truss design problem. The goal is to minimize structural mass while satisfying displacement and stress constraints.

## Problem Description

The 10-bar truss optimization problem involves:
- **Design variables**: 10 cross-sectional areas
- **Objective**: Minimize total structural mass
- **Constraints**: Maximum nodal displacement (≤2 inches) and member stress (tension/compression limits)
- **Solution approach**: PSO with Deb's feasibility rule and Latin Hypercube Sampling

## Dependencies

- Python 3.9 or higher
- NumPy
- Matplotlib
- Pandas
- SciPy

## Scripts

### Core Implementation

- **`pso.py`**: Main PSO implementation with two variants
  - **`pso_robust`**: Standard PSO with constriction factor and fixed parameters
  - **`pso_single_run`**: Adaptive PSO where each particle carries its own w, c1, c2 coefficients that evolve during optimization
  - Both use Latin Hypercube Sampling (LHS) for initialization
  - Deb's feasibility rule for constraint handling
  - Optional stall detection and restart mechanisms

- **`main.py`**: Execution script with command-line interface
  - Supports both robust and single-run modes
  - Generates comprehensive plots and animations
  - Outputs: best design, convergence plots, parameter evolution, constraint violation tracking

- **`truss_model.py`**: Finite element analysis for the 10-bar truss
  - Displacement calculations
  - Stress computations
  - Constraint evaluations

- **`objectives.py`**: Objective function and penalty formulation

- **`constraints.py`**: Constraint checking functions

### Analysis Scripts

- **`study_swarm_iters.py`**: Hyperparameter grid search
  - Tests multiple combinations of swarm size and iteration counts
  - Runs multiple trials per combination
  - Generates heatmaps showing success rates and best masses
  - Outputs: CSV results, summary statistics

- **`perturb_design_study.py`**: Design sensitivity analysis
  - Perturbs the best design found by PSO
  - Uses Latin Hypercube Sampling to generate design variations
  - Evaluates how constraint violations change with perturbations
  - Plots percentage deviation from constraints
  - Useful for understanding design robustness

### Test Scripts

- **`test_truss.py`**: Basic unit tests for truss model
- **`test_trussv2.py`**: Extended testing
- **`test_trussv3.py`**: Additional validation tests

## Usage

### Basic Single Run

Run PSO optimization with default parameters:

```bash
python main.py --mode single --num_runs 1 --iters 400 --swarm 40
```

### Multiple Runs for Statistics

Run multiple independent trials:

```bash
python main.py --mode single --num_runs 25 --iters 400 --swarm 40 --max_restarts 2 --stall_window 20
```

### Robust PSO Mode

Use standard PSO with fixed parameters:

```bash
python main.py --mode robust --num_runs 10 --iters 500 --swarm 50
```

### Parameter Study

Find optimal swarm size and iteration count:

```bash
python study_swarm_iters.py
```

### Design Perturbation Analysis

Analyze sensitivity of the best design:

```bash
python perturb_design_study.py
```

## Command-Line Arguments

| Argument         | Description                              | Default |
|------------------|------------------------------------------|---------|
| `--mode`         | PSO variant: 'robust' or 'single'       | 'single' |
| `--num_runs`     | Number of independent trials            | 1 |
| `--iters`        | Iterations per run                      | 400 |
| `--swarm`        | Swarm size (number of particles)        | 40 |
| `--max_restarts` | Maximum restarts on stall               | 2 |
| `--stall_window` | Iterations before restart trigger       | 20 |
| `--seed`         | Random seed for reproducibility         | None |

## Output Files

### Main Optimization

- **`pso_best_mass_vs_iteration.png`**: Mass convergence over iterations
- **`pso_best_max_disp_vs_iteration.png`**: Maximum displacement evolution
- **`pso_single_params_vs_iteration.png`**: Evolution of w, c1, c2 parameters (single-run mode)
- **CSV files**: Best designs and performance metrics

### Study Scripts

- **`study_swarm_iters.py`**: Heatmaps and CSV results
- **`perturb_design_study.py`**: Perturbation plots showing constraint violations

## Key Features

- **Latin Hypercube Sampling**: Space-filling initialization for better exploration
- **Deb's Feasibility Rule**: Lexicographic constraint handling (feasible solutions always preferred)
- **Adaptive Parameters**: Single-run mode evolves PSO coefficients during optimization
- **Stall Detection**: Automatic restart mechanism when progress stalls
- **Comprehensive Visualization**: Mass, displacement, constraint, and parameter evolution plots

## References

See [README_single_run.md](README_single_run.md) for detailed documentation of the adaptive single-run PSO algorithm.
