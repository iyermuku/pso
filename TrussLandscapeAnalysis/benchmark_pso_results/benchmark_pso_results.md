# Benchmark PSO Results

| Problem | Dim | Best Value | w | c1 | c2 | Swarm Size | Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sphere | 2 | 7.12872E-28 | 0.730 | 1.250 | 1.750 | 115 | 260 |
| Ellipsoid | 2 | 1.11764E-63 | 0.570 | 1.333 | 1.667 | 130 | 350 |
| Sum of Different Powers | 2 | 1.66761E-34 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Zakharov | 2 | 1.03299E-64 | 0.570 | 1.306 | 1.694 | 130 | 340 |
| Rosenbrock | 2 | 2.35924E-19 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Step | 2 | 0 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Quartic | 2 | 8.03931E-59 | 0.730 | 1.250 | 1.700 | 115 | 260 |
| Schwefel 2.22 | 2 | 1.58846E-24 | 0.650 | 1.250 | 1.750 | 120 | 340 |
| Schwefel 1.2 | 2 | 7.18153E-69 | 0.570 | 1.333 | 1.667 | 135 | 370 |
| Schwefel 2.21 | 2 | 1.59255E-19 | 0.650 | 1.250 | 1.700 | 115 | 260 |
| Rastrigin | 2 | 0 | 0.650 | 1.250 | 1.750 | 125 | 360 |
| Ackley | 2 | 4.44089E-16 | 0.650 | 1.250 | 1.750 | 120 | 280 |
| Griewank | 2 | 0 | 0.730 | 1.250 | 1.750 | 115 | 260 |
| Levy | 2 | 1.49976E-32 | 0.650 | 1.250 | 1.750 | 120 | 280 |
| Michalewicz | 2 | -1.8013 | 0.650 | 1.250 | 1.700 | 115 | 310 |
| Alpine 1 | 2 | 0 | 0.650 | 1.250 | 1.750 | 120 | 330 |
| Alpine 2 | 2 | -6.1295 | 0.540 | 1.333 | 1.667 | 130 | 400 |
| Bent Cigar | 3 | 3.00386E-51 | 0.570 | 1.333 | 1.667 | 125 | 320 |
| Discus | 3 | 2.71427E-60 | 0.570 | 1.333 | 1.667 | 135 | 370 |
| Weierstrass | 4 | 0 | 0.650 | 1.250 | 1.750 | 120 | 280 |
| HappyCat | 4 | 0.0058408 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| HGBat | 4 | 0.0302281 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Qing | 4 | 3.9443E-31 | 0.570 | 1.355 | 1.645 | 125 | 320 |
| Salomon | 5 | 0.0998733 | 0.650 | 1.250 | 1.700 | 120 | 280 |
| Bohachevsky | 5 | 0 | 0.730 | 1.250 | 1.750 | 45 | 310 |
| Booth | 5 | 8.38401E-27 | 0.730 | 1.250 | 1.750 | 40 | 290 |
| Matyas | 10 | 1.8631E-23 | 0.730 | 1.350 | 1.600 | 30 | 250 |
| Three-hump Camel | 10 | 1.51236E-39 | 0.650 | 1.250 | 1.750 | 35 | 270 |
| Six-hump Camel | 10 | -1.03163 | 0.650 | 1.250 | 1.750 | 40 | 290 |
| Goldstein-Price | 10 | 3 | 0.680 | 1.250 | 1.750 | 45 | 310 |
| Branin | 30 | 0.397887 | 0.650 | 1.250 | 1.750 | 35 | 260 |
| Shubert | 30 | -186.731 | 0.540 | 1.333 | 1.667 | 55 | 420 |
| Himmelblau | 30 | 0 | 0.650 | 1.250 | 1.750 | 40 | 290 |
| Easom | 30 | -1 | 0.730 | 1.250 | 1.700 | 30 | 240 |
| Cross-in-Tray | 30 | -2.06261 | 0.650 | 1.250 | 1.750 | 45 | 360 |
| Holder Table | 30 | -19.2085 | 0.650 | 1.250 | 1.750 | 45 | 360 |

Results were generated with the benchmark-landscape summary settings in [TrussLandscapeAnalysis/benchmark_landscape_summary.md](../benchmark_landscape_summary.md).

## Particle Evolution

The particle paths for two representative 2D benchmark functions are rendered in:

- [particle_evolution_two_benchmarks.png](particle_evolution_two_benchmarks.png)

The figure uses Sphere and Rastrigin, with trajectories sampled from the tracked PSO history so the swarm motion stays readable.

## Short Interpretation

- Results reflect the corrected benchmark dimensions and a full rerun of all 36 functions.
- Most functions reached their expected baselines (for example, Sphere, Rastrigin, Ackley, Griewank, Easom, and Holder Table).
- Several higher-dimensional cases remain challenging and may benefit from restarts or larger swarms if tighter targets are required.
