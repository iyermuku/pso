# PSO FEA Batch Summary

- coefficient mode: `fixed`
- swarm size: `landscape-recommended (per problem)`
- iterations: `landscape-recommended (per problem)`
- seed: `2026`
- seed remembered optima up to: `10.00%` of swarm

| Problem | Mode | Swarm | Iters | Seeded | Seeded->gbest | Time (s) | Best Mass | Max Disp | Max Stress | Final Feasible Fraction |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10-Bar Truss (Continuous) | fixed | 140 | 400 | 14 | False | 20.32 | 5068.500837 | 1.999960 | 24.914820 | 0.936 |
- Final gbest particle (truss10_continuous): `lhs_particle_109`
- Design variables (truss10_continuous): `[29.1728,  0.1094, 24.3507, 15.3574,  0.1007,  0.6   ,  7.409 , 21.6432,
 21.1216,  0.1021]`
| 10-Bar Truss (Discrete Sections) | fixed | 130 | 290 | 13 | True | 11.50 | 5490.737892 | 1.998943 | 14.196928 | 1.000 |
- Final gbest particle (truss10_discrete): `seed_optimum_001`
- Design variables (truss10_discrete): `[33.5 ,  1.62, 22.9 , 14.2 ,  1.62,  1.62,  7.97, 22.9 , 22.  ,  1.62]`
| 72-Bar Truss (Continuous) | fixed | 150 | 320 | 14 | False | 390.11 | 382.009533 | 0.250000 | 24.929444 | 0.607 |
- Final gbest particle (truss72_continuous): `lhs_particle_094`
- Design variables (truss72_continuous): `[0.1564, 0.5403, 0.3829, 0.4901, 0.5825, 0.5145, 0.1096, 0.1004, 1.3246,
 0.5488, 0.1047, 0.1014, 1.8045, 0.5215, 0.1119, 0.1046]`
| 72-Bar Truss (Discrete Sections) | fixed | 150 | 320 | 14 | True | 265.05 | 390.352403 | 0.249483 | 20.621302 | 1.000 |
- Final gbest particle (truss72_discrete): `seed_optimum_001`
- Design variables (truss72_discrete): `[0.196, 0.563, 0.391, 0.563, 0.563, 0.442, 0.111, 0.141, 1.228, 0.563,
 0.111, 0.111, 1.99 , 0.563, 0.111, 0.111]`
| 200-Bar Planar Truss (Continuous) | fixed | 150 | 290 | 14 | False | 400.80 | 4169.225652 | 3.999746 | 97.264544 | 0.993 |
- Final gbest particle (truss200_continuous): `lhs_particle_081`
- Design variables (truss200_continuous): `[0.1014, 0.4256, 0.1015, 0.1015, 0.6078, 0.1019, 0.102 , 0.7607, 0.1061,
 0.8926, 0.1009, 0.1014, 1.0406, 0.1007, 1.1126, 0.1   , 0.1   , 1.2125,
 0.1002, 1.3215, 0.1017, 0.1001, 1.4158, 0.107 , 1.4646, 0.1315, 0.8654,
 1.4992, 1.859 ]`
| 25-Bar Space Truss (Continuous) | fixed | 115 | 430 | 11 | False | 70.46 | 479.590195 | 0.349998 | 6.200408 | 0.983 |
- Final gbest particle (truss25_continuous): `lhs_particle_041`
- Design variables (truss25_continuous): `[0.01  , 0.4685, 3.3999, 0.0101, 1.9851, 0.984 , 0.3739, 3.3999]`
| 25-Bar Space Truss (Discrete Sections) | fixed | 115 | 430 | 11 | False | 98.90 | 545.562204 | 0.349955 | 7.178000 | 0.983 |
- Final gbest particle (truss25_discrete): `lhs_particle_109`
- Design variables (truss25_discrete): `[0.029, 2.084, 2.972, 0.011, 0.024, 0.697, 1.605, 2.665]`