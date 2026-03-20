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
| 10-Bar Truss (Continuous) | multimodal, narrow-basin | 4.655 | 0.858 | 5 | 0.200 | 0.0431 | 5.05 | 33.6 | 16154 | (0.570, 1.306, 1.694) | 130 | 350 |

## Per-Problem Interpretation and PSO Settings

### 10-Bar Truss (Continuous)
- Landscape class: **multimodal, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.306, c2=1.694**
- Recommended swarm size: **130**
- Recommended iterations: **350**
- 2-phase schedule: phase-1(w=0.620, c1=1.206, c2=1.794) -> phase-2(w=0.520, c1=1.406, c2=1.594)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 130: 10×dim=100 base +15 multimodal +15 narrow, clipped to [30, 150].
  - Iterations 350: 200 base +60 multimodal +90 narrow, clipped to [150, 600].

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