# TrussLandscapeAnalysis

Objective-landscape analysis for all four truss structural-optimisation problems in this
repository.  The package measures the ruggedness, multimodality, and basin structure of each
problem's fitness landscape, then derives problem-specific PSO hyperparameter recommendations
(inertia weight `w`, cognitive/social coefficients `c1`/`c2`, swarm size, and iteration count).

---

## Package structure

```
TrussLandscapeAnalysis/
├── landscape_core.py          # Core analysis engine (metrics, classification, recommendation)
├── problem_adapters.py        # Wraps each truss model as a LandscapeProblem
├── run_all_truss_landscapes.py# Full analysis runner + comparative report generator
├── rebuild_comparative_report.py # Fast rebuild from saved per-problem JSON (no FEA re-run)
└── results/
    ├── comparative_landscape_report.md   # Cross-problem summary report
    ├── all_landscape_metrics.json        # All four metrics in one file
    ├── comparison_landscape_scores.png   # Bar chart of classification scores
    ├── comparison_ruggedness_lon.png     # Ruggedness vs LON node count
    ├── truss10_continuous/
    │   ├── truss10_continuous_landscape_metrics.json
    │   ├── truss10_continuous_landscape_report.txt
    │   ├── truss10_continuous_autocorr.png
    │   ├── truss10_continuous_information_content.png
    │   ├── truss10_continuous_basins_map.png
    │   └── truss10_continuous_lon_basins.png
    ├── truss10_discrete/       (same structure)
    ├── truss72_continuous/     (same structure)
    └── truss72_discrete/       (same structure)
```

---

## Metrics produced

| Metric | Key in JSON | Interpretation |
|---|---|---|
| Autocorrelation length | `autocorrelation_length` | Steps until walk correlation drops below 1/e.  Shorter → more rugged. |
| Information content H(ε=0.05) | `information_content_H_eps005` | Entropy of objective-value changes along random walks.  Higher → more complex/varied landscape. |
| LON node count | `lon_nodes` | Number of distinct local optima found by the Local Optima Network construction. |
| LON edge density | `lon_edge_density` | Fraction of possible LON edges present.  Low = fragmented attractor landscape. |
| LON basin entropy | `lon_basin_entropy` | Diversity of basin sizes (higher = more even distribution of local optima). |
| Basin width median (normalised) | `basin_width_median_norm` | Median width of attraction basins relative to the search-space extent.  Smaller → narrower basins. |
| Classification labels | `classification_labels` | Comma-separated tags assigned to the problem (see below). |
| Multimodal / smooth / narrow scores | `classification_multimodal_score` etc. | Integer severity scores 0–3 for each classification axis. |

### Classification labels

| Label | Meaning |
|---|---|
| `multimodal` | Multiple local optima detected via LON |
| `smooth-macro` | Long autocorrelation length — smooth macro-scale funnel structure |
| `narrow-basin` | Median attraction-basin width is small — precise convergence needed |

---

## PSO hyperparameter recommendation

`recommend_pso_coefficients()` in `landscape_core.py` maps the measured metrics to concrete PSO
settings.  The full recommendation is stored in each problem's metrics JSON at key
`pso_recommendation`:

```json
{
  "recommended": {"w": 0.54, "c1": 1.292, "c2": 1.708},
  "recommended_sum_c1_c2": 3.0,
  "recommended_swarm_size": 140,
  "recommended_iters": 400,
  "schedule": {
    "phase_1_explore": {"w": 0.59, "c1": 1.192, "c2": 1.808},
    "phase_2_refine":  {"w": 0.52, "c1": 1.392, "c2": 1.608},
    "switch_fraction_of_iters": 0.6
  },
  "rationale": ["..."]
}
```

### Swarm size heuristic

```
swarm_size = clip(10×dim + 5×multimodal_score + 5×narrow_score [+ 10 if LON≥20], 30, 150)
```

### Iteration count heuristic

```
n_iters = clip(200 + 20×multimodal_score + 30×narrow_score [+ 50 if LON≥30] [+ 50 if AC<3], 150, 600)
```

Both values plus the `rationale` list explaining each decision are written to the per-problem JSON.

---

## How to run

### Full analysis (all four problems)

```bash
cd TrussLandscapeAnalysis
python run_all_truss_landscapes.py
```

This runs the expensive FEA-backed landscape analysis for all problems and saves results to
`results/`.  For the 72-bar problems the run uses reduced sampling to remain tractable (~5–10 min
total on a modern machine).

**Options**

| Flag | Default | Description |
|---|---|---|
| `--problems` | *(all)* | Comma-separated subset, e.g. `truss10_continuous,truss10_discrete` |
| `--out-dir` | `TrussLandscapeAnalysis/results` | Output directory |
| `--seed` | `2025` | Random seed |
| `--walk-steps` | `2000` | Random-walk length per problem |
| `--lon-starts` | `40` | LON construction start points |

### Rebuild the comparative report only (fast, no FEA)

```bash
python rebuild_comparative_report.py
```

Reads the saved per-problem JSON files, patches any missing `recommended_swarm_size` /
`recommended_iters` fields, regenerates comparative plots and `comparative_landscape_report.md`.

---

## Output files

### Per-problem

| File | Contents |
|---|---|
| `<id>_landscape_metrics.json` | All numeric metrics + PSO recommendation (machine-readable) |
| `<id>_landscape_report.txt` | Human-readable summary with classification, coefficients, rationale |
| `<id>_autocorr.png` | Autocorrelation function of objective along random walks |
| `<id>_information_content.png` | Information content H(ε) and M(ε) curves |
| `<id>_basins_map.png` | 2-D basin-of-attraction slice around best local optimum |
| `<id>_lon_basins.png` | LON basin-size distribution histogram |

### Comparative (in `results/`)

| File | Contents |
|---|---|
| `comparative_landscape_report.md` | Cross-problem markdown summary table + per-problem analysis |
| `all_landscape_metrics.json` | All four problems' metrics in a single JSON array |
| `comparison_landscape_scores.png` | Side-by-side classification scores |
| `comparison_ruggedness_lon.png` | Autocorrelation length vs LON node count |

---

## Per-problem landscape findings

| Problem | Classification | AC Length | H(ε=0.05) | LON Nodes | Basin Width (med, norm) | w | c1 | c2 | Swarm | Iters |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10-Bar Continuous | multimodal, narrow-basin | 4.46 | 0.752 | 30 | 0.0177 | 0.540 | 1.292 | 1.708 | 140 | 400 |
| 10-Bar Discrete   | multimodal              | 13.04 | 0.736 | 29 | 0.1192 | 0.680 | 1.210 | 1.790 | 130 | 290 |
| 72-Bar Continuous | multimodal, smooth-macro, narrow-basin | 486.1 | 0.706 | 14 | 0.0431 | 0.600 | 1.306 | 1.694 | 150 | 320 |
| 72-Bar Discrete   | multimodal, smooth-macro | 22.08 | 0.803 | 8  | 0.0875 | 0.730 | 1.250 | 1.750 | 150 | 290 |

### Key observations

- **10-bar continuous** is the most challenging: short autocorrelation length (rugged), narrow
  basins, and many local optima → lowest recommended inertia, strongest cognitive pull, largest
  swarm (140 particles) and most iterations (400).
- **10-bar discrete** is less rugged than continuous (discretisation regularises the landscape)
  but still multimodal.  Larger basin width allows a slightly smaller swarm.
- **72-bar continuous** has an extremely long autocorrelation length (smooth macro-funnel) yet
  still contains narrow basins — reflecting the high-dimensional, grouped design variable
  structure.  A moderately large swarm (150) covers the 16-dimensional space adequately.
- **72-bar discrete** is the smoothest overall.  Fewer LON nodes and wider basins make it the
  easiest to optimise — lower required iteration count.
- All four problems are **multimodal** → `c2 > c1` is consistently recommended to prevent swarm
  fragmentation across local basins.

---

## Integration with PSO FEA

The `PSO FEA/` folder loads these landscape metrics automatically via
`PSO FEA/problem_adapters.py`.  When you call `run_pso_fea.py` or `run_batch_pso_fea.py`
**without** explicit `--swarm-size` or `--iters` flags, the landscape-recommended values are
used:

```bash
# Uses landscape-recommended swarm size and iterations for each problem
python run_pso_fea.py --problem truss10_continuous --coeff-mode two-phase

# Override only iterations, keep landscape-recommended swarm size
python run_pso_fea.py --problem truss72_discrete --iters 500
```

The print output will indicate the source of each setting:

```
Swarm size: 140 (landscape-recommended)
Iterations: 400 (landscape-recommended)
```

See [`PSO FEA/README.md`](../PSO%20FEA/README.md) for full PSO runner documentation.
