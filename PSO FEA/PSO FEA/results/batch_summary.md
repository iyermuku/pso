# PSO FEA Batch Summary

- coefficient mode: `two-phase`
- swarm size: `50`
- iterations: `200`
- seed: `2026`

| Problem | Mode | Best Mass | Max Disp | Max Stress | Final Feasible Fraction |
|---|---|---:|---:|---:|---:|
| 10-Bar Truss (Continuous) | two-phase | 5084.719130 | 1.999984 | 24.723662 | 0.960 |
- Design variables (truss10_continuous): `[30.9801,  0.1327, 23.0472, 16.0115,  0.101 ,  0.5472,  7.6067, 21.538 ,
 20.4771,  0.1745]`
| 10-Bar Truss (Discrete Sections) | two-phase | 5560.223347 | 1.995821 | 14.439188 | 1.000 |
- Design variables (truss10_discrete): `[30.  ,  1.62, 30.  , 15.5 ,  1.62,  1.62,  7.97, 22.9 , 19.9 ,  1.62]`
| 72-Bar Truss (Continuous) | two-phase | 385.056532 | 0.249987 | 24.847301 | 0.660 |
- Design variables (truss72_continuous): `[0.148 , 0.5752, 0.4813, 0.5268, 0.5179, 0.491 , 0.1052, 0.2671, 1.1411,
 0.5015, 0.1272, 0.1029, 1.9667, 0.4953, 0.1033, 0.1005]`
| 72-Bar Truss (Discrete Sections) | two-phase | 396.188550 | 0.249726 | 24.999463 | 0.860 |
- Design variables (truss72_discrete): `[0.141, 0.563, 0.391, 0.563, 0.563, 0.563, 0.25 , 0.307, 1.266, 0.602,
 0.111, 0.111, 1.563, 0.442, 0.111, 0.111]`