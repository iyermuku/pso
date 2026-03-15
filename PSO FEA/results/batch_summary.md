# PSO FEA Batch Summary

- coefficient mode: `two-phase`
- swarm size: `landscape-recommended (per problem)`
- iterations: `landscape-recommended (per problem)`
- seed: `2026`
- seed remembered optima up to: `10.00%` of swarm

| Problem | Mode | Swarm | Iters | Seeded | Seeded->gbest | Time (s) | Best Mass | Max Disp | Max Stress | Final Feasible Fraction |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10-Bar Truss (Continuous) | two-phase | 140 | 400 | 14 | False | 22.07 | 5071.491438 | 1.999999 | 24.713177 | 0.964 |
- Final gbest particle (truss10_continuous): `lhs_particle_136`
- Design variables (truss10_continuous): `[31.3309,  0.1135, 21.7851, 14.9871,  0.1027,  0.5052,  7.7768, 21.8671,
 21.1999,  0.1035]`
| 10-Bar Truss (Discrete Sections) | two-phase | 130 | 290 | 13 | True | 10.80 | 5490.737892 | 1.998943 | 14.196928 | 1.000 |
- Final gbest particle (truss10_discrete): `seed_optimum_001`
- Design variables (truss10_discrete): `[33.5 ,  1.62, 22.9 , 14.2 ,  1.62,  1.62,  7.97, 22.9 , 22.  ,  1.62]`
| 72-Bar Truss (Continuous) | two-phase | 150 | 320 | 14 | False | 404.07 | 381.181192 | 0.249943 | 24.769642 | 0.647 |
- Final gbest particle (truss72_continuous): `lhs_particle_080`
- Design variables (truss72_continuous): `[0.1572, 0.5472, 0.4227, 0.5767, 0.5264, 0.5229, 0.1058, 0.1073, 1.2131,
 0.5102, 0.1011, 0.1205, 1.8927, 0.5097, 0.1017, 0.1032]`
| 72-Bar Truss (Discrete Sections) | two-phase | 150 | 320 | 14 | True | 289.43 | 393.554944 | 0.249454 | 24.998739 | 1.000 |
- Final gbest particle (truss72_discrete): `seed_optimum_001`
- Design variables (truss72_discrete): `[0.141, 0.602, 0.391, 0.602, 0.785, 0.563, 0.25 , 0.141, 1.228, 0.442,
 0.111, 0.111, 1.99 , 0.442, 0.111, 0.111]`
| 200-Bar Planar Truss (Continuous) | two-phase | 150 | 290 | 14 | False | 405.37 | 4167.233284 | 3.999913 | 100.252939 | 0.967 |
- Final gbest particle (truss200_continuous): `lhs_particle_101`
- Design variables (truss200_continuous): `[0.1014, 0.4315, 0.1004, 0.1001, 0.6393, 0.1003, 0.1026, 0.7353, 0.1002,
 0.873 , 0.1087, 0.1008, 0.9974, 0.1005, 1.1425, 0.1003, 0.1002, 1.2067,
 0.1023, 1.3401, 0.1016, 0.1002, 1.4235, 0.1017, 1.5148, 0.129 , 0.8271,
 1.4519, 1.9601]`
| 25-Bar Space Truss (Continuous) | two-phase | 115 | 430 | 11 | False | 71.84 | 479.616323 | 0.349996 | 6.249064 | 0.991 |
- Final gbest particle (truss25_continuous): `lhs_particle_012`
- Design variables (truss25_continuous): `[0.0136, 0.5254, 3.3993, 0.0109, 2.0703, 0.9783, 0.321 , 3.4   ]`
| 25-Bar Space Truss (Discrete Sections) | two-phase | 115 | 430 | 11 | False | 100.72 | 545.275666 | 0.349994 | 7.196683 | 0.983 |
- Final gbest particle (truss25_discrete): `lhs_particle_065`
- Design variables (truss25_discrete): `[0.013, 2.013, 3.074, 0.015, 0.011, 0.666, 1.617, 2.678]`