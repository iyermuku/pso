# Comparative Truss Objective Landscape Report

## Scope
This report compares objective landscapes for all truss problems in the repository using:
- Autocorrelation length
- Information content
- Local Optima Network (LON) structure
- Basin-of-attraction mapping
- Smoothness and narrow-basin diagnostics

## Summary Table

| Problem | Class | AC Length | H(eps=0.05) | LON Nodes | LON Density | Basin Width Median | Recommended (w,c1,c2) | Swarm Size | Iterations |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 200-Bar Planar Truss (Continuous) | multimodal | 29.101 | 0.870 | 14 | 0.071 | 0.1192 | (0.680, 1.250, 1.750) | 150 | 290 |

## Per-Problem Interpretation and PSO Settings

### 200-Bar Planar Truss (Continuous)
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.680, c1=1.250, c2=1.750**
- Recommended swarm size: **150**
- Recommended iterations: **290**
- 2-phase schedule: phase-1(w=0.730, c1=1.150, c2=1.850) -> phase-2(w=0.600, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 150: 10×dim=290 base +15 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 290: 200 base +60 multimodal +30 narrow, clipped to [150, 600].

## Cross-Problem Takeaways
- Problems with higher LON node counts and higher information content are more multimodal/rugged.
- Problems with smaller normalized basin width benefit from lower inertia and stronger local refinement.
- Suggested default when uncertain: w=0.62, c1=1.35, c2=1.65; then adapt per problem diagnostics.

## Outputs
- Per-problem reports and metrics are in `results/<problem_id>/`.
- Comparative figures:
  - `comparison_landscape_scores.png`
  - `comparison_ruggedness_lon.png`