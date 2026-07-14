# Benchmark Landscape Summary

This table condenses the 36 benchmark landscape analyses into one place.

| Problem | Class | Recommended (w,c1,c2) | Swarm Size | Iterations |
|---|---|---|---:|---:|
| Sphere | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 115 | 260 |
| Ellipsoid | multimodal, smooth-macro, narrow-basin | (0.570, 1.333, 1.667) | 130 | 350 |
| Sum of Different Powers | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 120 | 280 |
| Zakharov | multimodal, narrow-basin | (0.570, 1.306, 1.694) | 130 | 340 |
| Rosenbrock | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 120 | 280 |
| Step | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 120 | 280 |
| Quartic | multimodal, smooth-macro | (0.730, 1.250, 1.700) | 115 | 260 |
| Schwefel 2.22 | multimodal | (0.650, 1.250, 1.750) | 120 | 340 |
| Schwefel 1.2 | multimodal, narrow-basin | (0.570, 1.333, 1.667) | 135 | 370 |
| Schwefel 2.21 | multimodal | (0.650, 1.250, 1.700) | 115 | 260 |
| Rastrigin | multimodal | (0.650, 1.250, 1.750) | 125 | 360 |
| Ackley | multimodal | (0.650, 1.250, 1.750) | 120 | 280 |
| Griewank | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 115 | 260 |
| Levy | multimodal | (0.650, 1.250, 1.750) | 120 | 280 |
| Michalewicz | multimodal | (0.650, 1.250, 1.700) | 115 | 310 |
| Alpine 1 | multimodal | (0.650, 1.250, 1.750) | 120 | 330 |
| Alpine 2 | multimodal, narrow-basin | (0.540, 1.333, 1.667) | 130 | 400 |
| Bent Cigar | multimodal, smooth-macro, narrow-basin | (0.570, 1.333, 1.667) | 125 | 320 |
| Discus | multimodal, smooth-macro, narrow-basin | (0.570, 1.333, 1.667) | 135 | 370 |
| Weierstrass | multimodal | (0.650, 1.250, 1.750) | 120 | 280 |
| HappyCat | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 120 | 280 |
| HGBat | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 120 | 280 |
| Qing | multimodal, smooth-macro, narrow-basin | (0.570, 1.355, 1.645) | 125 | 320 |
| Salomon | multimodal | (0.650, 1.250, 1.700) | 120 | 280 |
| Bohachevsky | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 45 | 310 |
| Booth | multimodal, smooth-macro | (0.730, 1.250, 1.750) | 40 | 290 |
| Matyas | smooth-macro | (0.730, 1.350, 1.600) | 30 | 250 |
| Three-hump Camel | multimodal | (0.650, 1.250, 1.750) | 35 | 270 |
| Six-hump Camel | multimodal | (0.650, 1.250, 1.750) | 40 | 290 |
| Goldstein-Price | multimodal | (0.680, 1.250, 1.750) | 45 | 310 |
| Branin | multimodal | (0.650, 1.250, 1.750) | 35 | 260 |
| Shubert | multimodal, narrow-basin | (0.540, 1.333, 1.667) | 55 | 420 |
| Himmelblau | multimodal | (0.650, 1.250, 1.750) | 40 | 290 |
| Easom | multimodal, smooth-macro | (0.730, 1.250, 1.700) | 30 | 240 |
| Cross-in-Tray | multimodal | (0.650, 1.250, 1.750) | 45 | 360 |
| Holder Table | multimodal | (0.650, 1.250, 1.750) | 45 | 360 |

## Baseline Recommendation

For an unknown benchmark function, start with:

| w | c1 | c2 | Notes |
|---|---:|---:|---|
| 0.62 | 1.35 | 1.60 | Good general baseline from the landscape scan |
