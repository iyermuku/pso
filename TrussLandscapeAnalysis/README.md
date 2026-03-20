# TrussLandscapeAnalysis

Objective-landscape analysis for truss optimization problems in this repository.
The code computes landscape diagnostics (ruggedness, multimodality, basin geometry),
classifies each problem, and then derives PSO settings for:

- fixed coefficients (`w, c1, c2`)
- two-phase coefficients (exploration phase + refinement phase)

This README is written for paper/research presentation and maps directly to the
implementation in `TrussLandscapeAnalysis/landscape_core.py`.

## Scope and workflow

1. Define each truss problem through `problem_adapters.py`.
2. Sample/evaluate objective values through random walks and local descents.
3. Compute diagnostic metrics: autocorrelation, information content, LON, basin widths.
4. Convert metrics into three score axes: multimodal, smooth, narrow.
5. Assign class labels from scores.
6. Deduce PSO hyperparameters (fixed and two-phase) from class + metric values.

## Definitions with literature context

### Multimodal
Multimodal means the objective landscape contains multiple local optima with distinct
basins of attraction. In this code, multimodality is inferred from Local Optima Network
(LON) structure and random-walk information characteristics.

LON perspective:

- nodes = local optima
- edges = transitions between attraction basins under perturbation + local search

Key references: Ochoa et al. (2008), Tomassini et al. (2008), Daolio et al. (2010).

### Narrow basin
Narrow means promising basins occupy a small normalized region around local optima,
so optimization requires precise local motion and low overshoot.

In this code, narrowness is estimated by directional basin width around the best
local optimum: how far one can move (normalized to variable span) before the objective
rises by a fixed absolute amount (`rise_abs`).

### Smooth ("smooth-macro")
Smooth here means the macro-scale trend is correlated over longer steps (funnel-like),
not necessarily globally convex. A landscape can be smooth at coarse scale while still
having local structure.

This is measured primarily by autocorrelation length and slope dispersion statistics.

### Autocorrelation length (`autocorrelation_length`)
From a random walk in decision space, let `r(1)` be lag-1 autocorrelation in objective values.
The code uses:

`ell = -1 / ln(|r(1)|)`

Higher `ell` means slower decorrelation (smoother trend); lower `ell` means ruggedness.

Key reference: Weinberger (1990) random-walk autocorrelation analysis.

### LON nodes (`lon_nodes`)
`lon_nodes` is the number of distinct local optima found by repeated local descent from
multiple starts and clustered into attractors. Larger node count typically indicates more
multimodal structure.

## Implemented metrics and outputs

Saved per problem in `<id>_landscape_metrics.json`:

- `autocorrelation_length`
- `information_content_H_eps005`
- `information_content_M_eps005`
- `lon_nodes`, `lon_edge_density`, `lon_basin_entropy`
- `basin_width_mean_norm`, `basin_width_median_norm`, `basin_width_q10_norm`
- `classification_*`
- `pso_recommendation`
- `analysis_threads`
- `cache_hits`, `cache_misses`, `cache_total_queries`, `cache_unique_evals`, `cache_hit_rate`

### Performance instrumentation (new)

The analyzer now includes deterministic memoization of objective evaluations and
reports cache utilization for each problem. This helps quantify repeated probes
from local descent, LON perturbation, and basin-map computations.

Interpretation:

- higher `cache_hit_rate` means stronger reuse and lower repeated FE solves
- `cache_unique_evals` approximates effective expensive evaluations actually computed
- `cache_total_queries` tracks total evaluator calls after all algorithm stages

## Score system and classification mapping

Scoring is implemented in `classify_landscape(...)`.

### 1) Multimodal score (`0..4`)
Increment by 1 for each true condition:

- `lon_nodes >= 4`
- `lon_entropy >= 1.0`
- `info_h >= 0.55`
- `lon_density >= 0.25`

Label rule:

- if `multimodal_score >= 2` -> add label `multimodal`

### 2) Smooth score (`0..3`)
Increment by 1 for each true condition:

- `ac_len >= 6.0`
- `info_h <= 0.45`
- `slope_q90 <= 6 * slope_median`

Label rule:

- if `smooth_score >= 2` -> add label `smooth-macro`

### 3) Narrow score (`0..3`)
Increment by 1 for each true condition:

- `basin_width_median_norm <= 0.10`
- `basin_width_q10_norm <= 0.04`
- `slope_q90 >= 3 * slope_median`

Label rule:

- if `narrow_score >= 2` -> add label `narrow-basin`

Fallback:

- if no label is triggered -> `mixed/uncertain`

## PSO hyperparameter deduction

Implemented in `recommend_pso_coefficients(...)`.

### Fixed mode coefficients

Base values:

- `w = 0.68`
- `c1 = 1.35`
- `c2 = 1.55`

Rule-based adjustments:

- if `multimodal`: `c2 += 0.15`, `c1 -= 0.10`
- if `narrow-basin`: `w -= 0.08`, `c1 += 0.10`
- if `smooth-macro` and not narrow: `w += 0.05`
- if `lon_nodes >= 25` and `lon_density < 0.15`: `c2 += 0.10`
- if `ac_len < 5`: `w -= 0.03`
- if `info_h > 0.70`: `c2 += 0.05`
- if `basin_width_median_norm < 0.04`: `w -= 0.03`, `c1 += 0.05`

Bounds and normalization:

- `w in [0.50, 0.78]`
- `c1 in [0.90, 2.20]`
- `c2 in [1.10, 2.40]`
- normalize `(c1 + c2)` to stay in `[2.0, 3.0]`

The resulting fixed recommendation is stored as:

- `pso_recommendation.recommended`

### Two-phase mode coefficients

Derived directly from fixed recommendation:

- phase 1 (explore):
  - `w1 = min(0.75, w + 0.05)`
  - `c11 = max(0.95, c1 - 0.10)`
  - `c21 = min(2.30, c2 + 0.10)`
- phase 2 (refine):
  - `w2 = max(0.52, w - 0.08)`
  - `c12 = min(2.00, c1 + 0.10)`
  - `c22 = max(1.20, c2 - 0.10)`
- switch point:
  - `switch_fraction_of_iters = 0.60`

This schedule is stored as:

- `pso_recommendation.schedule`

### Swarm size and iteration deduction

Also produced by landscape analysis:

- `swarm_base = 10 * dim`
- `swarm_size = clip(swarm_base + 5*multimodal_score + 5*narrow_score + (10 if lon_nodes>=20 else 0), 30, 150)`

- `iters_base = 200`
- `n_iters = clip(iters_base + 20*multimodal_score + 30*narrow_score + (50 if lon_nodes>=30 else 0) + (50 if ac_len<3 else 0), 150, 600)`

These are written to:

- `pso_recommendation.recommended_swarm_size`
- `pso_recommendation.recommended_iters`

## How to run

From repository root:

```bash
python TrussLandscapeAnalysis/run_all_truss_landscapes.py
```

Run one problem only:

```bash
python TrussLandscapeAnalysis/run_all_truss_landscapes.py --problems truss200_continuous
```

Important options:

- `--problems` comma-separated IDs
- `--seed`
- `--n-ref`
- `--walk-steps`
- `--lon-starts`
- `--basin-grid`
- `--jobs` process count across problems (`1` = sequential, `0` = auto CPU count)
- `--threads-per-problem` thread count within each problem analysis (`0` = auto, default)

## Parallel execution and tuning

Two levels of parallelism are available:

- process-level: run different problems concurrently (`--jobs`)
- thread-level: run parts of a single problem analysis concurrently (`--threads-per-problem`)

Examples:

```bash
# Process parallelism only
python TrussLandscapeAnalysis/run_all_truss_landscapes.py --jobs 0

# Intra-problem threading only (explicit)
python TrussLandscapeAnalysis/run_all_truss_landscapes.py --jobs 1 --threads-per-problem 4

# Default behavior (auto threads per problem)
python TrussLandscapeAnalysis/run_all_truss_landscapes.py --jobs 1

# Hybrid (use with care to avoid oversubscription)
python TrussLandscapeAnalysis/run_all_truss_landscapes.py --jobs 2 --threads-per-problem 2
```

Practical tuning rule:

- keep roughly `jobs * threads_per_problem <= physical_cores`
- in auto mode (`--threads-per-problem 0`), thread count is `max(cpu_count // jobs, 1)`
- if FE solve dominates and releases GIL well, increasing `threads-per-problem` can help
- otherwise prefer higher `jobs` and lower `threads-per-problem`

## Outputs

Per-problem folder: `TrussLandscapeAnalysis/results/<problem_id>/`

- `<id>_landscape_metrics.json`
- `<id>_landscape_report.txt`
- `<id>_autocorr.png`
- `<id>_information_content.png`
- `<id>_basins_map.png`
- `<id>_lon_basins.png`

Comparative files in `TrussLandscapeAnalysis/results/`:

- `all_landscape_metrics.json`
- `comparative_landscape_report.md`
- `comparison_landscape_scores.png`
- `comparison_ruggedness_lon.png`

The comparative markdown report now also includes cache columns (`Cache Hit %`, `Unique Evals`).

## Using recommendations in PSO runs

The PSO runner (`PSO FEA/run_pso_fea.py`) loads these landscape outputs automatically.

Fixed mode:

```bash
python "PSO FEA/run_pso_fea.py" --problem truss200_continuous --coeff-mode fixed
```

Two-phase mode:

```bash
python "PSO FEA/run_pso_fea.py" --problem truss200_continuous --coeff-mode two-phase
```

## Literature references

[1] S. A. Kauffman, The Origins of Order, Oxford University Press, 1993.

[2] E. D. Weinberger, "Correlated and uncorrelated fitness landscapes and how to tell the difference," Biological Cybernetics, 63, 325-336, 1990.

[3] V. K. Vassilev, T. C. Fogarty, J. F. Miller, "Information characteristics and the structure of landscapes," Evolutionary Computation, 8(1), 31-60, 2000.

[4] G. Ochoa, M. Tomassini, S. Verel, C. Darabos, "A study of NK landscapes' basins and local optima networks," GECCO, 2008.

[5] M. Tomassini, S. Verel, G. Ochoa, "Complex-network analysis of combinatorial spaces: The NK landscape case," Physical Review E, 78, 066114, 2008.

[6] F. Daolio, S. Verel, G. Ochoa, M. Tomassini, "Local optima networks of the quadratic assignment problem," IEEE CEC, 2010.

[7] J. Kennedy, R. Eberhart, "Particle swarm optimization," IEEE ICNN, 1995.

[8] Y. Shi, R. Eberhart, "A modified particle swarm optimizer," IEEE ICEC, 1998.

[9] M. Clerc, J. Kennedy, "The particle swarm - explosion, stability, and convergence in a multidimensional complex space," IEEE TEC, 6(1), 58-73, 2002.

Note:
- This implementation combines multiple practical landscape proxies
  (autocorrelation, information-content curves, LON structure, and basin-width probes)
  rather than reproducing only one canonical protocol.
