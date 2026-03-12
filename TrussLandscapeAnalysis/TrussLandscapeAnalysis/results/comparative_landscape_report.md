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
| 10-Bar Truss (Continuous) | multimodal, narrow-basin | 4.460 | 0.752 | 30 | 0.098 | 0.0177 | (0.540, 1.292, 1.708) | 140 | 400 |
| 10-Bar Truss (Discrete Sections) | multimodal | 13.038 | 0.736 | 29 | 0.084 | 0.1192 | (0.680, 1.210, 1.790) | 130 | 290 |
| 72-Bar Truss (Continuous) | multimodal, smooth-macro, narrow-basin | 486.050 | 0.706 | 14 | 0.071 | 0.0431 | (0.600, 1.306, 1.694) | 150 | 320 |
| 72-Bar Truss (Discrete Sections) | multimodal, smooth-macro, narrow-basin | 184.482 | 0.793 | 14 | 0.082 | 0.0367 | (0.570, 1.333, 1.667) | 150 | 320 |

## Per-Problem Interpretation and PSO Settings

### 10-Bar Truss (Continuous)
- Landscape class: **multimodal, narrow-basin**
- Recommended fixed coefficients: **w=0.540, c1=1.292, c2=1.708**
- Recommended swarm size: **140**
- Recommended iterations: **400**
- 2-phase schedule: phase-1(w=0.590, c1=1.192, c2=1.808) -> phase-2(w=0.520, c1=1.392, c2=1.608)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - Many weakly connected attractors: increase social attraction to reduce swarm fragmentation.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 140: 10×dim=100 base +15 multimodal +15 narrow +10 LON, clipped to [30, 150].
  - Iterations 400: 200 base +60 multimodal +90 narrow +50 LON, clipped to [150, 600].

### 10-Bar Truss (Discrete Sections)
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.680, c1=1.210, c2=1.790**
- Recommended swarm size: **130**
- Recommended iterations: **290**
- 2-phase schedule: phase-1(w=0.730, c1=1.110, c2=1.890) -> phase-2(w=0.600, c1=1.310, c2=1.690)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Many weakly connected attractors: increase social attraction to reduce swarm fragmentation.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 130: 10×dim=100 base +15 multimodal +5 narrow +10 LON, clipped to [30, 150].
  - Iterations 290: 200 base +60 multimodal +30 narrow, clipped to [150, 600].

### 72-Bar Truss (Continuous)
- Landscape class: **multimodal, smooth-macro, narrow-basin**
- Recommended fixed coefficients: **w=0.600, c1=1.306, c2=1.694**
- Recommended swarm size: **150**
- Recommended iterations: **320**
- 2-phase schedule: phase-1(w=0.650, c1=1.206, c2=1.794) -> phase-2(w=0.520, c1=1.406, c2=1.594)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 150: 10×dim=160 base +15 multimodal +10 narrow, clipped to [30, 150].
  - Iterations 320: 200 base +60 multimodal +60 narrow, clipped to [150, 600].

### 72-Bar Truss (Discrete Sections)
- Landscape class: **multimodal, smooth-macro, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.333, c2=1.667**
- Recommended swarm size: **150**
- Recommended iterations: **320**
- 2-phase schedule: phase-1(w=0.620, c1=1.233, c2=1.767) -> phase-2(w=0.520, c1=1.433, c2=1.567)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 150: 10×dim=160 base +15 multimodal +10 narrow, clipped to [30, 150].
  - Iterations 320: 200 base +60 multimodal +60 narrow, clipped to [150, 600].

## Cross-Problem Takeaways
- Problems with higher LON node counts and higher information content are more multimodal/rugged.
- Problems with smaller normalized basin width benefit from lower inertia and stronger local refinement.
- Suggested default when uncertain: w=0.62, c1=1.35, c2=1.65; then adapt per problem diagnostics.

## Outputs
- Per-problem reports and metrics are in `results/<problem_id>/`.
- Comparative figures:
  - `comparison_landscape_scores.png`
  - `comparison_ruggedness_lon.png`