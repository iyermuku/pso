# PSO FEA Batch Summary

- coefficient mode: `fixed`
- swarm size: `landscape-recommended (per problem)`
- iterations: `landscape-recommended (per problem)`
- seed: `2026`
- seed remembered optima up to: `0.00%` of swarm

| Problem | Mode | Swarm | Iters | Seeded | Seeded->gbest | Time (s) | Best Mass | Max Disp | Max Stress | Final Feasible Fraction |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10-Bar Truss (Continuous) | fixed | 140 | 400 | 0 | False | 25.25 | 5067.876552 | 1.999993 | 24.836745 | 0.936 |
- Final gbest particle (truss10_continuous): `lhs_particle_119`
- Design variables (truss10_continuous): `[30.1603,  0.1062, 23.5441, 14.9128,  0.1054,  0.545 ,  7.5524, 21.9394,
 20.8912,  0.1048]`
| 10-Bar Truss (Discrete Sections) | fixed | 130 | 290 | 0 | False | 12.86 | 5628.623347 | 1.996541 | 14.877869 | 1.000 |
- Final gbest particle (truss10_discrete): `lhs_particle_000`
- Design variables (truss10_discrete): `[30.  ,  1.62, 33.5 , 13.9 ,  1.62,  1.62,  7.97, 19.9 , 22.9 ,  1.62]`
| 72-Bar Truss (Continuous) | fixed | 150 | 320 | 0 | False | 456.38 | 380.360066 | 0.249996 | 24.977126 | 0.627 |
- Final gbest particle (truss72_continuous): `lhs_particle_026`
- Design variables (truss72_continuous): `[0.1552, 0.577 , 0.3976, 0.5771, 0.5106, 0.5185, 0.1022, 0.1043, 1.2556,
 0.5077, 0.1014, 0.1025, 1.8542, 0.5012, 0.1041, 0.1011]`
| 72-Bar Truss (Discrete Sections) | fixed | 150 | 320 | 0 | False | 307.05 | 390.180956 | 0.249928 | 21.037538 | 1.000 |
- Final gbest particle (truss72_discrete): `lhs_particle_000`
- Design variables (truss72_discrete): `[0.196, 0.563, 0.391, 0.602, 0.391, 0.563, 0.111, 0.111, 1.266, 0.563,
 0.111, 0.111, 1.563, 0.563, 0.111, 0.111]`
| 200-Bar Planar Truss (Continuous) | fixed | 150 | 290 | 0 | False | 636.83 | 4176.120798 | 3.999651 | 91.590628 | 0.980 |
- Final gbest particle (truss200_continuous): `lhs_particle_041`
- Design variables (truss200_continuous): `[0.111 , 0.4277, 0.1092, 0.1053, 0.6605, 0.1003, 0.1009, 0.7563, 0.1005,
 0.8968, 0.105 , 0.1036, 0.9825, 0.1082, 1.0678, 0.1054, 0.1064, 1.1884,
 0.1007, 1.3364, 0.1135, 0.1017, 1.404 , 0.1012, 1.5391, 0.1572, 0.7709,
 1.4164, 1.9886]`
| 25-Bar Space Truss (Continuous) | fixed | 115 | 430 | 0 | False | 86.15 | 479.610480 | 0.349998 | 6.249485 | 0.991 |
- Final gbest particle (truss25_continuous): `lhs_particle_059`
- Design variables (truss25_continuous): `[0.0122, 0.5214, 3.3998, 0.0112, 2.0555, 0.9845, 0.3207, 3.3996]`
| 25-Bar Space Truss (Discrete Sections) | fixed | 115 | 430 | 0 | False | 114.69 | 545.554435 | 0.349988 | 7.177837 | 0.974 |
- Final gbest particle (truss25_discrete): `lhs_particle_051`
- Design variables (truss25_discrete): `[0.016, 2.099, 2.948, 0.013, 0.02 , 0.646, 1.605, 2.741]`