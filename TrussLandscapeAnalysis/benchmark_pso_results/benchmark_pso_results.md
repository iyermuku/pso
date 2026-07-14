# Benchmark PSO Results

| Problem | Best Value | w | c1 | c2 | Swarm Size | Iterations |
|---|---:|---:|---:|---:|---:|---:|
| Sphere | 1.22838e-13 | 0.730 | 1.250 | 1.750 | 115 | 260 |
| Ellipsoid | 5.5419e-36 | 0.570 | 1.333 | 1.667 | 130 | 350 |
| Sum of Different Powers | 4.02084e-28 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Zakharov | 2.30925e-25 | 0.570 | 1.306 | 1.694 | 130 | 340 |
| Rosenbrock | 2.77888 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Step | 0 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Quartic | 1.52088e-27 | 0.730 | 1.250 | 1.700 | 115 | 260 |
| Schwefel 2.22 | 1.08508e-13 | 0.650 | 1.250 | 1.750 | 120 | 340 |
| Schwefel 1.2 | 6.3641e-24 | 0.570 | 1.333 | 1.667 | 135 | 370 |
| Schwefel 2.21 | 1.09724e-07 | 0.650 | 1.250 | 1.700 | 115 | 260 |
| Rastrigin | 8.95463 | 0.650 | 1.250 | 1.750 | 125 | 360 |
| Ackley | 4.77063e-11 | 0.650 | 1.250 | 1.750 | 120 | 280 |
| Griewank | 0.110692 | 0.730 | 1.250 | 1.750 | 115 | 260 |
| Levy | 1.21834e-21 | 0.650 | 1.250 | 1.750 | 120 | 280 |
| Michalewicz | -7.92752 | 0.650 | 1.250 | 1.700 | 115 | 310 |
| Alpine 1 | 4.76066e-13 | 0.650 | 1.250 | 1.750 | 120 | 330 |
| Alpine 2 | -14320.1 | 0.540 | 1.333 | 1.667 | 130 | 400 |
| Bent Cigar | 2.6166e-30 | 0.570 | 1.333 | 1.667 | 125 | 320 |
| Discus | 1.93789e-39 | 0.570 | 1.333 | 1.667 | 135 | 370 |
| Weierstrass | 7.2345e-09 | 0.650 | 1.250 | 1.750 | 120 | 280 |
| HappyCat | 0.0508788 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| HGBat | 0.18776 | 0.730 | 1.250 | 1.750 | 120 | 280 |
| Qing | 9.0719e-30 | 0.570 | 1.355 | 1.645 | 125 | 320 |
| Salomon | 0.199873 | 0.650 | 1.250 | 1.700 | 120 | 280 |
| Bohachevsky | 0 | 0.730 | 1.250 | 1.750 | 45 | 310 |
| Booth | 4.21116e-25 | 0.730 | 1.250 | 1.750 | 40 | 290 |
| Matyas | 7.67533e-23 | 0.730 | 1.350 | 1.600 | 30 | 250 |
| Three-hump Camel | 1.87847e-37 | 0.650 | 1.250 | 1.750 | 35 | 270 |
| Six-hump Camel | -1.03163 | 0.650 | 1.250 | 1.750 | 40 | 290 |
| Goldstein-Price | 3 | 0.680 | 1.250 | 1.750 | 45 | 310 |
| Branin | 0.397887 | 0.650 | 1.250 | 1.750 | 35 | 260 |
| Shubert | -186.731 | 0.540 | 1.333 | 1.667 | 55 | 420 |
| Himmelblau | 0 | 0.650 | 1.250 | 1.750 | 40 | 290 |
| Easom | -1 | 0.730 | 1.250 | 1.700 | 30 | 240 |
| Cross-in-Tray | -2.06261 | 0.650 | 1.250 | 1.750 | 45 | 360 |
| Holder Table | -19.2085 | 0.650 | 1.250 | 1.750 | 45 | 360 |

Results were generated with the benchmark-landscape summary settings in [TrussLandscapeAnalysis/benchmark_landscape_summary.md](../benchmark_landscape_summary.md).

## Short Interpretation

- The summary-driven settings solved the easier bowl-shaped and narrow-basin cases very well, including Sphere, Ellipsoid, Ackley, Levy, Easom, and Holder Table.
- The harder multimodal cases were Rosenbrock, Rastrigin, Griewank, Salomon, and Goldstein-Price; these are the ones most likely to benefit from extra restarts, a larger swarm, or a longer run.
- The benchmark scan suggests a practical default around `w=0.62, c1=1.35, c2=1.60`, with higher social pressure for multimodal problems and lower inertia for narrow basins.