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
| 25-Bar Space Truss (Discrete Sections) | multimodal, narrow-basin | 1.454 | 0.408 | 30 | 0.093 | 0.0304 | (0.540, 1.313, 1.688) | 115 | 430 |

## Per-Problem Interpretation and PSO Settings

### 25-Bar Space Truss (Discrete Sections)
- Landscape class: **multimodal, narrow-basin**
- Recommended fixed coefficients: **w=0.540, c1=1.313, c2=1.688**
- Recommended swarm size: **115**
- Recommended iterations: **430**
- 2-phase schedule: phase-1(w=0.590, c1=1.213, c2=1.788) -> phase-2(w=0.520, c1=1.413, c2=1.587)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - Many weakly connected attractors: increase social attraction to reduce swarm fragmentation.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 115: 10×dim=80 base +10 multimodal +15 narrow +10 LON, clipped to [30, 150].
  - Iterations 430: 200 base +40 multimodal +90 narrow +50 LON +50 ruggedness, clipped to [150, 600].

## Cross-Problem Takeaways
- Problems with higher LON node counts and higher information content are more multimodal/rugged.
- Problems with smaller normalized basin width benefit from lower inertia and stronger local refinement.
- Suggested default when uncertain: w=0.62, c1=1.35, c2=1.65; then adapt per problem diagnostics.

## Outputs
- Per-problem reports and metrics are in `results/<problem_id>/`.
- Comparative figures:
  - `comparison_landscape_scores.png`
  - `comparison_ruggedness_lon.png`