# PSO FEA Batch Summary

- coefficient mode: `two-phase`
- swarm size: `landscape-recommended (per problem)`
- iterations: `landscape-recommended (per problem)`
- seed: `2026`

| Problem | Mode | Swarm | Iters | Best Mass | Max Disp | Max Stress | Final Feasible Fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| 10-Bar Truss (Continuous) | two-phase | 140 | 400 | 5068.001730 | 1.999995 | 24.999689 | 0.936 |
- Design variables (truss10_continuous): `[30.7497,  0.1045, 23.3359, 13.9191,  0.1001,  0.4209,  7.5252, 21.5508,
 21.8385,  0.1017]`
| 10-Bar Truss (Discrete Sections) | two-phase | 130 | 290 | 5490.737892 | 1.998943 | 14.196928 | 1.000 |
- Design variables (truss10_discrete): `[33.5 ,  1.62, 22.9 , 14.2 ,  1.62,  1.62,  7.97, 22.9 , 22.  ,  1.62]`
| 72-Bar Truss (Continuous) | two-phase | 150 | 320 | 381.751037 | 0.249967 | 24.951718 | 0.640 |
- Design variables (truss72_continuous): `[0.1572, 0.5311, 0.3979, 0.5803, 0.5228, 0.5148, 0.1032, 0.1036, 1.487 ,
 0.5145, 0.1126, 0.1082, 1.8784, 0.4891, 0.103 , 0.1005]`
| 72-Bar Truss (Discrete Sections) | two-phase | 150 | 290 | 389.334170 | 0.249817 | 20.755116 | 1.000 |
- Design variables (truss72_discrete): `[0.196, 0.563, 0.391, 0.563, 0.563, 0.442, 0.111, 0.111, 1.228, 0.563,
 0.111, 0.111, 1.99 , 0.563, 0.111, 0.111]`