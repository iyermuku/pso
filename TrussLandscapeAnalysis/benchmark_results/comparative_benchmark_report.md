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
| Sphere | multimodal, smooth-macro | 15.114 | 0.708 | 5 | 0.200 | 0.5000 | 1.84 | 7.2 | (0.730, 1.250, 1.750) | 115 | 260 |
| Ellipsoid | multimodal, smooth-macro, narrow-basin | 6.393 | 0.801 | 5 | 0.200 | 0.0050 | 2.39 | 7.2 | (0.570, 1.333, 1.667) | 130 | 350 |
| Sum of Different Powers | multimodal, smooth-macro | 10.457 | 0.730 | 4 | 0.583 | 0.5000 | 2.02 | 19.6 | (0.730, 1.250, 1.750) | 120 | 280 |
| Zakharov | multimodal, narrow-basin | 3.235 | 0.750 | 5 | 0.300 | 0.0938 | 2.25 | 12.4 | (0.570, 1.306, 1.694) | 130 | 340 |
| Rosenbrock | multimodal, smooth-macro | 12.810 | 0.772 | 5 | 0.350 | 0.2208 | 2.06 | 6.1 | (0.730, 1.250, 1.750) | 120 | 280 |
| Step | multimodal, smooth-macro | 6.965 | 0.829 | 5 | 0.500 | 0.5000 | 1.71 | 3.9 | (0.730, 1.250, 1.750) | 120 | 280 |
| Quartic | multimodal, smooth-macro | 9.612 | 0.680 | 3 | 0.333 | 0.5000 | 2.03 | 7.1 | (0.730, 1.250, 1.700) | 115 | 260 |
| Schwefel 2.22 | multimodal | 1.979 | 0.923 | 5 | 0.200 | 0.4048 | 1.84 | 7.0 | (0.650, 1.250, 1.750) | 120 | 340 |
| Schwefel 1.2 | multimodal, narrow-basin | 5.020 | 0.857 | 5 | 0.250 | 0.0304 | 1.94 | 19.3 | (0.570, 1.333, 1.667) | 135 | 370 |
| Schwefel 2.21 | multimodal | 4.927 | 0.699 | 5 | 0.200 | 0.5000 | 1.87 | 25.4 | (0.650, 1.250, 1.700) | 115 | 260 |
| Rastrigin | multimodal | 0.489 | 0.754 | 5 | 0.400 | 0.1065 | 1.78 | 5.3 | (0.650, 1.250, 1.750) | 125 | 360 |
| Ackley | multimodal | 3.007 | 0.766 | 5 | 0.250 | 0.5000 | 2.07 | 7.2 | (0.650, 1.250, 1.750) | 120 | 280 |
| Griewank | multimodal, smooth-macro | 15.116 | 0.708 | 5 | 0.200 | 0.5000 | 2.00 | 36.5 | (0.730, 1.250, 1.750) | 115 | 260 |
| Levy | multimodal | 3.438 | 0.760 | 5 | 0.300 | 0.5000 | 2.31 | 5.9 | (0.650, 1.250, 1.750) | 120 | 280 |
| Michalewicz | multimodal | 0.837 | 0.556 | 5 | 0.200 | 0.5000 | 2.31 | 6.3 | (0.650, 1.250, 1.700) | 115 | 310 |
| Alpine 1 | multimodal | 1.777 | 0.781 | 5 | 0.250 | 0.5000 | 1.83 | 6.8 | (0.650, 1.250, 1.750) | 120 | 330 |
| Alpine 2 | multimodal, narrow-basin | 0.708 | 0.965 | 5 | 0.000 | 0.0304 | 1.81 | 9.4 | (0.540, 1.333, 1.667) | 130 | 400 |
| Bent Cigar | multimodal, smooth-macro, narrow-basin | 9.133 | 0.706 | 5 | 0.200 | 0.0050 | 1.81 | 7.5 | (0.570, 1.333, 1.667) | 125 | 320 |
| Discus | multimodal, smooth-macro, narrow-basin | 17.236 | 0.816 | 5 | 0.400 | 0.0050 | 1.80 | 7.5 | (0.570, 1.333, 1.667) | 135 | 370 |
| Weierstrass | multimodal | 3.270 | 0.812 | 5 | 0.250 | 0.5000 | 4.54 | 19.4 | (0.650, 1.250, 1.750) | 120 | 280 |
| HappyCat | multimodal, smooth-macro | 13.164 | 0.781 | 5 | 0.350 | 0.5000 | 1.92 | 5.0 | (0.730, 1.250, 1.750) | 120 | 280 |
| HGBat | multimodal, smooth-macro | 15.027 | 0.708 | 5 | 0.500 | 0.4746 | 1.94 | 5.0 | (0.730, 1.250, 1.750) | 120 | 280 |
| Qing | multimodal, smooth-macro, narrow-basin | 11.599 | 0.676 | 5 | 0.200 | 0.0050 | 2.46 | 38.3 | (0.570, 1.355, 1.645) | 125 | 320 |
| Salomon | multimodal | 3.352 | 0.657 | 5 | 0.400 | 0.5000 | 2.25 | 14.1 | (0.650, 1.250, 1.700) | 120 | 280 |
| Bohachevsky | multimodal, smooth-macro | 9.572 | 0.762 | 5 | 0.550 | 0.0431 | 1.80 | 45.0 | (0.730, 1.250, 1.750) | 45 | 310 |
| Booth | multimodal, smooth-macro | 9.255 | 0.763 | 3 | 0.667 | 0.2144 | 1.90 | 22.0 | (0.730, 1.250, 1.750) | 40 | 290 |
| Matyas | smooth-macro | 8.154 | 0.795 | 1 | 0.000 | 0.5000 | 1.80 | 20.3 | (0.730, 1.350, 1.600) | 30 | 250 |
| Three-hump Camel | multimodal | 3.718 | 0.827 | 2 | 1.000 | 0.5000 | 1.82 | 15.9 | (0.650, 1.250, 1.750) | 35 | 270 |
| Six-hump Camel | multimodal | 4.390 | 0.817 | 3 | 0.667 | 0.3223 | 1.77 | 17.5 | (0.650, 1.250, 1.750) | 40 | 290 |
| Goldstein-Price | multimodal | 7.955 | 0.880 | 5 | 0.250 | 0.1065 | 1.84 | 23.5 | (0.680, 1.250, 1.750) | 45 | 310 |
| Branin | multimodal | 3.145 | 0.877 | 4 | 0.083 | 0.5000 | 2.06 | 19.9 | (0.650, 1.250, 1.750) | 35 | 260 |
| Shubert | multimodal, narrow-basin | 0.550 | 0.819 | 5 | 0.400 | 0.0177 | 1.89 | 16.9 | (0.540, 1.333, 1.667) | 55 | 420 |
| Himmelblau | multimodal | 4.280 | 0.807 | 5 | 0.150 | 0.1319 | 1.80 | 16.6 | (0.650, 1.250, 1.750) | 40 | 290 |
| Easom | multimodal, smooth-macro | 999999.500 | -0.000 | 5 | 0.000 | 0.5000 | 1.65 | 6.1 | (0.730, 1.250, 1.700) | 30 | 240 |
| Cross-in-Tray | multimodal | 0.883 | 0.815 | 4 | 0.583 | 0.5000 | 1.93 | 12.9 | (0.650, 1.250, 1.750) | 45 | 360 |
| Holder Table | multimodal | 0.740 | 0.726 | 5 | 0.300 | 0.5000 | 1.86 | 26.1 | (0.650, 1.250, 1.750) | 45 | 360 |

## Per-Problem Interpretation and PSO Settings

### Sphere
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **115**
- Recommended iterations: **260**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 115: 10×dim=100 base +15 multimodal, clipped to [30, 150].
  - Iterations 260: 200 base +60 multimodal, clipped to [150, 600].

### Ellipsoid
- Landscape class: **multimodal, smooth-macro, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.333, c2=1.667**
- Recommended swarm size: **130**
- Recommended iterations: **350**
- 2-phase schedule: phase-1(w=0.620, c1=1.233, c2=1.767) -> phase-2(w=0.520, c1=1.433, c2=1.567)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 130: 10×dim=100 base +15 multimodal +15 narrow, clipped to [30, 150].
  - Iterations 350: 200 base +60 multimodal +90 narrow, clipped to [150, 600].

### Sum of Different Powers
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### Zakharov
- Landscape class: **multimodal, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.306, c2=1.694**
- Recommended swarm size: **130**
- Recommended iterations: **340**
- 2-phase schedule: phase-1(w=0.620, c1=1.206, c2=1.794) -> phase-2(w=0.520, c1=1.406, c2=1.594)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 130: 10×dim=100 base +20 multimodal +10 narrow, clipped to [30, 150].
  - Iterations 340: 200 base +80 multimodal +60 narrow, clipped to [150, 600].

### Rosenbrock
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### Step
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### Quartic
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

### Schwefel 2.22
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **340**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +15 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 340: 200 base +60 multimodal +30 narrow +50 ruggedness, clipped to [150, 600].

### Schwefel 1.2
- Landscape class: **multimodal, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.333, c2=1.667**
- Recommended swarm size: **135**
- Recommended iterations: **370**
- 2-phase schedule: phase-1(w=0.620, c1=1.233, c2=1.767) -> phase-2(w=0.520, c1=1.433, c2=1.567)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 135: 10×dim=100 base +20 multimodal +15 narrow, clipped to [30, 150].
  - Iterations 370: 200 base +80 multimodal +90 narrow, clipped to [150, 600].

### Schwefel 2.21
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.700**
- Recommended swarm size: **115**
- Recommended iterations: **260**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.800) -> phase-2(w=0.570, c1=1.350, c2=1.600)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - Swarm size 115: 10×dim=100 base +15 multimodal, clipped to [30, 150].
  - Iterations 260: 200 base +60 multimodal, clipped to [150, 600].

### Rastrigin
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **125**
- Recommended iterations: **360**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 125: 10×dim=100 base +20 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 360: 200 base +80 multimodal +30 narrow +50 ruggedness, clipped to [150, 600].

### Ackley
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### Griewank
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **115**
- Recommended iterations: **260**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 115: 10×dim=100 base +15 multimodal, clipped to [30, 150].
  - Iterations 260: 200 base +60 multimodal, clipped to [150, 600].

### Levy
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### Michalewicz
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.700**
- Recommended swarm size: **115**
- Recommended iterations: **310**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.800) -> phase-2(w=0.570, c1=1.350, c2=1.600)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - Swarm size 115: 10×dim=100 base +15 multimodal, clipped to [30, 150].
  - Iterations 310: 200 base +60 multimodal +50 ruggedness, clipped to [150, 600].

### Alpine 1
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **330**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 330: 200 base +80 multimodal +50 ruggedness, clipped to [150, 600].

### Alpine 2
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

### Bent Cigar
- Landscape class: **multimodal, smooth-macro, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.333, c2=1.667**
- Recommended swarm size: **125**
- Recommended iterations: **320**
- 2-phase schedule: phase-1(w=0.620, c1=1.233, c2=1.767) -> phase-2(w=0.520, c1=1.433, c2=1.567)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 125: 10×dim=100 base +15 multimodal +10 narrow, clipped to [30, 150].
  - Iterations 320: 200 base +60 multimodal +60 narrow, clipped to [150, 600].

### Discus
- Landscape class: **multimodal, smooth-macro, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.333, c2=1.667**
- Recommended swarm size: **135**
- Recommended iterations: **370**
- 2-phase schedule: phase-1(w=0.620, c1=1.233, c2=1.767) -> phase-2(w=0.520, c1=1.433, c2=1.567)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 135: 10×dim=100 base +20 multimodal +15 narrow, clipped to [30, 150].
  - Iterations 370: 200 base +80 multimodal +90 narrow, clipped to [150, 600].

### Weierstrass
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### HappyCat
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### HGBat
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### Qing
- Landscape class: **multimodal, smooth-macro, narrow-basin**
- Recommended fixed coefficients: **w=0.570, c1=1.355, c2=1.645**
- Recommended swarm size: **125**
- Recommended iterations: **320**
- 2-phase schedule: phase-1(w=0.620, c1=1.255, c2=1.745) -> phase-2(w=0.520, c1=1.455, c2=1.545)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 125: 10×dim=100 base +15 multimodal +10 narrow, clipped to [30, 150].
  - Iterations 320: 200 base +60 multimodal +60 narrow, clipped to [150, 600].

### Salomon
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.700**
- Recommended swarm size: **120**
- Recommended iterations: **280**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.800) -> phase-2(w=0.570, c1=1.350, c2=1.600)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - Swarm size 120: 10×dim=100 base +20 multimodal, clipped to [30, 150].
  - Iterations 280: 200 base +80 multimodal, clipped to [150, 600].

### Bohachevsky
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **45**
- Recommended iterations: **310**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 45: 10×dim=20 base +20 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 310: 200 base +80 multimodal +30 narrow, clipped to [150, 600].

### Booth
- Landscape class: **multimodal, smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.250, c2=1.750**
- Recommended swarm size: **40**
- Recommended iterations: **290**
- 2-phase schedule: phase-1(w=0.750, c1=1.150, c2=1.850) -> phase-2(w=0.650, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 40: 10×dim=20 base +15 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 290: 200 base +60 multimodal +30 narrow, clipped to [150, 600].

### Matyas
- Landscape class: **smooth-macro**
- Recommended fixed coefficients: **w=0.730, c1=1.350, c2=1.600**
- Recommended swarm size: **30**
- Recommended iterations: **250**
- 2-phase schedule: phase-1(w=0.750, c1=1.250, c2=1.700) -> phase-2(w=0.650, c1=1.450, c2=1.500)
- Rationale:
  - Smooth macro-landscape: slightly higher inertia supports broader traversal.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 30: 10×dim=20 base +5 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 250: 200 base +20 multimodal +30 narrow, clipped to [150, 600].

### Three-hump Camel
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **35**
- Recommended iterations: **270**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 35: 10×dim=20 base +10 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 270: 200 base +40 multimodal +30 narrow, clipped to [150, 600].

### Six-hump Camel
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **40**
- Recommended iterations: **290**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 40: 10×dim=20 base +15 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 290: 200 base +60 multimodal +30 narrow, clipped to [150, 600].

### Goldstein-Price
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.680, c1=1.250, c2=1.750**
- Recommended swarm size: **45**
- Recommended iterations: **310**
- 2-phase schedule: phase-1(w=0.730, c1=1.150, c2=1.850) -> phase-2(w=0.600, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 45: 10×dim=20 base +20 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 310: 200 base +80 multimodal +30 narrow, clipped to [150, 600].

### Branin
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **35**
- Recommended iterations: **260**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 35: 10×dim=20 base +15 multimodal, clipped to [30, 150].
  - Iterations 260: 200 base +60 multimodal, clipped to [150, 600].

### Shubert
- Landscape class: **multimodal, narrow-basin**
- Recommended fixed coefficients: **w=0.540, c1=1.333, c2=1.667**
- Recommended swarm size: **55**
- Recommended iterations: **420**
- 2-phase schedule: phase-1(w=0.590, c1=1.233, c2=1.767) -> phase-2(w=0.520, c1=1.433, c2=1.567)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Narrow basin: reduce inertia and retain cognitive pull for local refinement.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Very narrow basin estimate: damp momentum and increase pbest guidance.
  - Swarm size 55: 10×dim=20 base +20 multimodal +15 narrow, clipped to [30, 150].
  - Iterations 420: 200 base +80 multimodal +90 narrow +50 ruggedness, clipped to [150, 600].

### Himmelblau
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **40**
- Recommended iterations: **290**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 40: 10×dim=20 base +15 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 290: 200 base +60 multimodal +30 narrow, clipped to [150, 600].

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

### Cross-in-Tray
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **45**
- Recommended iterations: **360**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 45: 10×dim=20 base +20 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 360: 200 base +80 multimodal +30 narrow +50 ruggedness, clipped to [150, 600].

### Holder Table
- Landscape class: **multimodal**
- Recommended fixed coefficients: **w=0.650, c1=1.250, c2=1.750**
- Recommended swarm size: **45**
- Recommended iterations: **360**
- 2-phase schedule: phase-1(w=0.700, c1=1.150, c2=1.850) -> phase-2(w=0.570, c1=1.350, c2=1.650)
- Rationale:
  - Multimodality: raise social pull for swarm consensus across local basins.
  - Short autocorrelation length: lower inertia to avoid overshoot.
  - High information content: increase exploitation pressure after discovery.
  - Swarm size 45: 10×dim=20 base +20 multimodal +5 narrow, clipped to [30, 150].
  - Iterations 360: 200 base +80 multimodal +30 narrow +50 ruggedness, clipped to [150, 600].

## Cross-Function Takeaways
- 35/36 functions were classified as multimodal.
- 16/36 functions were classified as smooth-macro.
- 8/36 functions were classified as narrow-basin.
- Multimodal problems generally push the recommendation toward higher social pressure.
- Narrow basins generally reduce inertia and increase cognitive guidance.
- Smooth unimodal problems usually get a moderate inertia plus a more balanced cognitive/social split.
- Suggested baseline for unknown benchmark functions: w=0.62, c1=1.35, c2=1.60.

## Outputs
- Per-problem reports and metrics are in `results/<problem_id>/`.
- Comparative figures:
  - `benchmark_landscape_scores.png`
  - `benchmark_ruggedness_lon.png`