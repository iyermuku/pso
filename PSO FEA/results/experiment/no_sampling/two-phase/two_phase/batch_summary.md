# PSO FEA Batch Summary

- coefficient mode: `two-phase`
- swarm size: `landscape-recommended (per problem)`
- iterations: `landscape-recommended (per problem)`
- seed: `2026`
- seed remembered optima up to: `0.00%` of swarm

| Problem | Mode | Swarm | Iters | Seeded | Seeded->gbest | Time (s) | Best Mass | Max Disp | Max Stress | Final Feasible Fraction |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10-Bar Truss (Continuous) | two-phase | 140 | 400 | 0 | False | 21.59 | 5068.001730 | 1.999995 | 24.999689 | 0.936 |
- Final gbest particle (truss10_continuous): `lhs_particle_057`
- Design variables (truss10_continuous): `[30.7497,  0.1045, 23.3359, 13.9191,  0.1001,  0.4209,  7.5252, 21.5508,
 21.8385,  0.1017]`
| 10-Bar Truss (Discrete Sections) | two-phase | 130 | 290 | 0 | False | 11.50 | 5490.737892 | 1.998943 | 14.196928 | 1.000 |
- Final gbest particle (truss10_discrete): `lhs_particle_000`
- Design variables (truss10_discrete): `[33.5 ,  1.62, 22.9 , 14.2 ,  1.62,  1.62,  7.97, 22.9 , 22.  ,  1.62]`
| 72-Bar Truss (Continuous) | two-phase | 150 | 320 | 0 | False | 410.24 | 381.751037 | 0.249967 | 24.951718 | 0.640 |
- Final gbest particle (truss72_continuous): `lhs_particle_077`
- Design variables (truss72_continuous): `[0.1572, 0.5311, 0.3979, 0.5803, 0.5228, 0.5148, 0.1032, 0.1036, 1.487 ,
 0.5145, 0.1126, 0.1082, 1.8784, 0.4891, 0.103 , 0.1005]`
| 72-Bar Truss (Discrete Sections) | two-phase | 150 | 320 | 0 | False | 290.12 | 389.827793 | 0.249940 | 20.759327 | 1.000 |
- Final gbest particle (truss72_discrete): `lhs_particle_000`
- Design variables (truss72_discrete): `[0.196, 0.602, 0.442, 0.602, 0.442, 0.563, 0.111, 0.111, 1.228, 0.563,
 0.111, 0.111, 1.8  , 0.442, 0.111, 0.111]`
| 200-Bar Planar Truss (Continuous) | two-phase | 150 | 290 | 0 | False | 403.34 | 4169.782416 | 3.999781 | 90.585635 | 0.947 |
- Final gbest particle (truss200_continuous): `lhs_particle_111`
- Design variables (truss200_continuous): `[0.1001, 0.4262, 0.101 , 0.1018, 0.6138, 0.102 , 0.1001, 0.7344, 0.1007,
 0.8713, 0.1007, 0.101 , 0.9315, 0.1011, 1.0796, 0.1004, 0.1006, 1.2803,
 0.1006, 1.3231, 0.101 , 0.1022, 1.4209, 0.1012, 1.5382, 0.1465, 0.8358,
 1.523 , 1.8295]`
| 25-Bar Space Truss (Continuous) | two-phase | 115 | 430 | 0 | False | 71.53 | 479.704472 | 0.349998 | 6.224468 | 0.991 |
- Final gbest particle (truss25_continuous): `lhs_particle_001`
- Design variables (truss25_continuous): `[0.0107, 0.5286, 3.3991, 0.0135, 1.9358, 0.9807, 0.3452, 3.3999]`
| 25-Bar Space Truss (Discrete Sections) | two-phase | 115 | 430 | 0 | False | 98.39 | 546.277346 | 0.349992 | 7.318267 | 0.991 |
- Final gbest particle (truss25_discrete): `lhs_particle_054`
- Design variables (truss25_discrete): `[0.024, 1.837, 3.234, 0.017, 0.013, 0.711, 1.704, 2.559]`