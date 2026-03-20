# Comparative Truss Objective Landscape Report

## Scope
This report compares objective landscapes for all truss problems in the repository using:
- Autocorrelation length
- Information content
- Local Optima Network (LON) structure
- Basin-of-attraction mapping
- Smoothness and narrow-basin diagnostics

## Summary Table

| Problem | Class | AC Length | H(eps=0.05) | LON Nodes | LON Density | Basin Width Median | Time (s) | Cache Hit % | Unique Evals | Recommended (w,c1,c2) | Swarm Size | Iterations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 10-Bar Truss (Continuous) | multimodal, narrow-basin | 1.327 | 0.991 | 6 | 0.233 | 0.0177 | 7.83 | 28.2 | 24774 | (0.540, 1.333, 1.667) | 130 | 400 |

## Per-Problem Interpretation and PSO Settings

### 10-Bar Truss (Continuous)
- Landscape class: **multimodal, narrow-basin**
- Recommended fixed coefficients: **w=0.540, c1=1.333, c2=1.667**
- Recommended swarm size: **130**
- Recommended iterations: **400**
- 2-phase schedule: phase-1(w=0.590, c1=1.233, c2=1.767) -> phase-2(w=0.520, c1=1.433, c2=1.567)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 130: 10×dim=100 base +15 multimodal +15 narrow, clipped to [30, 150].
  - Iterations 400: 200 base +60 multimodal +90 narrow +50 ruggedness, clipped to [150, 600].

## Cross-Problem Takeaways
- Problems with higher LON node counts and higher information content are more multimodal/rugged.
- Problems with smaller normalized basin width benefit from lower inertia and stronger local refinement.
- Suggested default when uncertain: w=0.62, c1=1.35, c2=1.65; then adapt per problem diagnostics.
- Higher cache-hit % indicates more repeated landscape probes and stronger memoization payoff.

## Outputs
- Per-problem reports and metrics are in `results/<problem_id>/`.
- Comparative figures:
  - `comparison_landscape_scores.png`
  - `comparison_ruggedness_lon.png`