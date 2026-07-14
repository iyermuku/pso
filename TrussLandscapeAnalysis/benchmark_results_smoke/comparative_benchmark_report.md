# Comparative Benchmark Objective Landscape Report

## Scope
This report compares objective landscapes for 36 benchmark functions using:
- Autocorrelation length
- Information content
- Local Optima Network (LON) structure
- Basin-of-attraction mapping
- Smoothness and narrow-basin diagnostics

## Summary Table

| Problem | Class | AC Length | H(eps=0.05) | LON Nodes | LON Density | Basin Width Median | Time (s) | Cache Hit % | Recommended (w,c1,c2) | Swarm Size | Iterations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Sphere | multimodal, smooth-macro | 15.010 | 0.678 | 5 | 0.200 | 0.5000 | 2.36 | 7.0 | (0.730, 1.250, 1.700) | 115 | 260 |
| Easom | multimodal, smooth-macro | 999999.500 | -0.000 | 5 | 0.050 | 0.5000 | 1.65 | 5.6 | (0.730, 1.250, 1.700) | 30 | 240 |

## Per-Problem Interpretation and PSO Settings

### Sphere
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.700**
- Recommended swarm size: **115**
- Recommended iterations: **260**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.800) -> phase-2(w=0.650, c1=1.350, c2=1.600)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - Swarm size 115: 10×dim=100 base +15 multimodal, clipped to [30, 150].
  - Iterations 260: 200 base +60 multimodal, clipped to [150, 600].

### Easom
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.700**
- Recommended swarm size: **30**
- Recommended iterations: **240**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.800) -> phase-2(w=0.650, c1=1.350, c2=1.600)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - Swarm size 30: 10×dim=20 base +10 multimodal, clipped to [30, 150].
  - Iterations 240: 200 base +40 multimodal, clipped to [150, 600].

## Cross-Function Takeaways
- 2/36 functions were classified as multimodal.
- 2/36 functions were classified as smooth-macro.
- 0/36 functions were classified as narrow-basin.
- Multimodal problems generally push the recommendation toward higher social pressure.
- Narrow basins generally reduce inertia and increase cognitive guidance.
- Smooth unimodal problems usually get a moderate inertia plus a more balanced cognitive/social split.
- Suggested baseline for unknown benchmark functions: w=0.62, c1=1.35, c2=1.60.

## Outputs
- Per-problem reports and metrics are in `results/<problem_id>/`.
- Comparative figures:
  - `benchmark_landscape_scores.png`
  - `benchmark_ruggedness_lon.png`