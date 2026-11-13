# Rodriguez 11-Bar Truss Feasibility Study

This document describes `TrussRodriguez.py`, a feasibility-search program for the 11-bar, 6-node planar truss.

## Model

The model keeps the original truss connectivity, supports, material properties, and loads from `truss11.py`:

- 6 nodes: A, B, C, D, E, F
- 11 bars
- Supports at nodes A and D
- 200 cm horizontal spacing between adjacent columns
- 100 kN downward loads at nodes B and C
- Young's modulus: `E = 20,000 kN/cm^2`
- Density: `RHO = 0.00785 kg/cm^3`

The 14 design variables are:

```text
[A1, A2, ..., A11, y4, y5, y6]
```

where `Ai` is a member area in cm^2 and `y4`, `y5`, and `y6` are the absolute vertical coordinates of nodes D, E, and F in cm.

The design bounds are:

```text
0.10 <= Ai <= 30.0 cm^2
5.0 <= y4, y5, y6 <= 300.0 cm
```

Changing `y4`, `y5`, or `y6` changes the lengths and direction cosines of every non-horizontal member connected to those nodes. The horizontal members A-B and B-C remain 200 cm long.

## Stress Constraint

`TrussRodriguez.py` uses the ASTM A36/S235 allowable absolute axial stress of
150 MPa:

```text
150 MPa = 15 kN/cm^2
```

## Steepest-Descent Optimization

`TrussRodriguez.py` also includes a bounded steepest-descent search over the
14 normalized design variables:

```text
[A1, ..., A11, y4, y5, y6]
```

It starts from `A1=...=A11=0.10 cm^2` and
`y4=y5=y6=200.0 cm`. The objective is:

```text
mass / 1000 + 100 * (stress_violation / 15)^2
```

The method uses bound-aware finite-difference gradients, projected-gradient
convergence testing, and Armijo backtracking. It validates area and height
bounds and rejects designs with `cond(Kff) > 1e8`.

The captured run reached a feasible design:

```text
areas:
[0.100000, 11.335279, 28.522638, 15.404083, 13.708046,
 19.028415, 15.774317, 8.129115, 11.010426, 1.628092, 16.300988]

y4, y5, y6 (cm):
[171.791360, 126.655508, 5.013261]

mass: 183.757819 kg
maximum absolute stress: 14.981445 kN/cm^2
maximum-stress bar: F10
cond(Kff): 4.457228e3
stress violation: 0.0 kN/cm^2
feasible: True
iterations: 200
evaluations: 7226
elapsed time: reported in seconds by the program
projected-gradient norm: 1.596062e-1
```

The design is feasible, but the projected-gradient norm remains above `1e-5`;
the run reached its iteration limit before full derivative-based convergence.

Buckling is ignored. Both tension and compression are checked through

```text
abs(stress_i) <= 15.0 kN/cm^2
```

A candidate is feasible only when every member satisfies the stress limit and all three height variables satisfy their bounds.

## What the Code Does

`TrussRodriguez.py` performs the following steps:

1. Creates a reproducible random generator using seed `20261113`.
2. Samples 5,000 area and height designs uniformly within the design bounds.
3. Evaluates each design using the finite-element solver in `truss11.py`.
4. Computes mass, signed member stresses, maximum absolute stress, and the member carrying that stress.
5. Classifies a design as feasible when all stress and height constraints pass.
6. Sorts feasible candidates by mass.
7. If fewer than five feasible designs are found, selects the five designs with the smallest stress violation.
8. Plots the selected truss configurations and highlights each maximum-stress bar in red.
9. Monitors the reduced free-free stiffness matrix condition number `cond(Kff)`.
10. Rejects designs with non-finite or excessive conditioning.
11. Writes the selected design variables and metrics to CSV.

The design evaluator validates areas and heights against their bounds instead
of clipping them. The current numerical safety threshold is:

```text
cond(Kff) <= 1e8
```

`Kff` is the stiffness matrix for the free degrees of freedom after applying
the supports at A and D. Conditioning is recorded for every candidate and is
included in the CSV and plot labels.

Run the study with:

```text
python TrussRodriguez.py
```

The command also creates one 8 FPS animation for each optimization method:

```text
TrussRodriguez_optimization_steepest_descent.gif
TrussRodriguez_optimization_conjugate_gradient.gif
TrussRodriguez_optimization_marquardt.gif
TrussRodriguez_optimization_dfp.gif
```

Each iteration displays the current truss configuration, highlights the
maximum-stress bar in red, and reports all areas, `y4`, `y5`, `y6`, objective,
mass, maximum stress, critical bar, stress violation, `cond(Kff)`,
projected-gradient norm, evaluations, and the iteration number. The final
frame is held for two seconds and each animation does not loop.

Animation controls are configurable from the command line:

```text
python TrussRodriguez.py --fps 8
python TrussRodriguez.py --fps 12 --animation-output run12fps.gif
python TrussRodriguez.py --no-animation
```

The outputs are:

- `TrussRodriguez_candidates.png`: five configuration panels
- `TrussRodriguez_candidates.csv`: design variables and feasibility metrics
- `TrussRodriguez_optimization_*.gif`: one optimization animation per method
- `TrussRodriguez_optimization.csv`: optimizer comparison table

To compare the final PSO and steepest-descent designs in one image, run:

```text
python TrussRodriguez_compare.py
```

This uses 30 PSO particles and 100 iterations and creates
`TrussRodriguez_PSO_vs_steepest.png`. PSO is shown in blue and
steepest descent in orange; each method's maximum-stress bar is shown in a
darker accent color. The panels also report mass, maximum stress, critical bar,
conditioning, and feasibility.

## Particle-Swarm Optimization

`TrussRodriguezPSO.py` provides a particle-swarm optimizer for the same 14
design variables and constraints. Each particle contains 11 areas and the
three upper-node heights. Positions are initialized uniformly within the
explicit design bounds, and velocities are limited to 25% of each variable's
range per update.

The PSO objective is:

```text
mass / 1000 + 100 * (stress_violation / 15)^2
```

The optimizer rejects designs with non-finite or excessive `cond(Kff)`, tracks
both the global-best particle and the lowest-mass feasible particle, and uses
the feasible particle for the final output when one exists.

Run it with:

```text
python TrussRodriguezPSO.py
```

Available controls are:

```text
python TrussRodriguezPSO.py --particles 30 --iterations 100 --fps 8
python TrussRodriguezPSO.py --seed 20261113 --output custom_pso.gif
```

The outputs are:

- `PSO_Rodriguez_final.csv`: final feasible design, or the global best if no feasible particle was found
- `PSO_Rodriguez_evolution.gif`: global-best particle evolution by iteration

Each animation frame shows the current global-best truss, highlights its
maximum-stress bar, and reports all design variables, mass, objective, stress,
stress violation, conditioning, feasible-particle count, and evaluation count.

A short validation run using 10 particles and 20 iterations found a feasible
design with mass `198.136970 kg`, maximum absolute stress
`14.667528 kN/cm^2`, critical bar `F3`, and `cond(Kff)=77.7176`.

## Optimization Method Comparison

The program compares four derivative-based methods from the same initial
design (`Ai=0.10 cm^2`, `y4=y5=y6=200 cm`):

- Steepest descent
- Conjugate gradient
- Marquardt
- DFP

The comparison table contains, for every method:

```text
objective, mass, maximum stress, critical bar, stress violation,
cond(Kff), projected-gradient norm, iterations, evaluations,
elapsed seconds, feasibility
```

The default production budgets are 200 iterations for steepest descent,
conjugate gradient, and DFP, and 1,000 iterations for Marquardt. The table is
printed to the console and written to `TrussRodriguez_optimization.csv`.
The steepest-descent run supplies the iteration history used by the GIF.

## Feasibility Study Result

The 5,000-sample run produced:

```text
Samples: 5000
Allowable absolute axial stress: 15.000 kN/cm^2 (150 MPa)
Feasible samples found: 311
Maximum accepted cond(Kff): 1.0e8
```

The five selected candidates are the lowest-mass feasible samples:

```text
Candidate 1: mass=213.373384 kg, max stress=14.178387 kN/cm^2, bar=F6, cond=2.431920e2
Candidate 2: mass=226.658140 kg, max stress=13.370688 kN/cm^2, bar=F4, cond=2.337480e3
Candidate 3: mass=237.621059 kg, max stress=14.471153 kN/cm^2, bar=F4, cond=6.107145e1
Candidate 4: mass=242.065309 kg, max stress=12.632970 kN/cm^2, bar=F4, cond=9.209904e1
Candidate 5: mass=243.156933 kg, max stress=12.134067 kN/cm^2, bar=F8, cond=1.488423e2
```

The feasible-sample count is empirical and depends on the sample count and
random seed; it is not a proof that every feasible region has been found.

## Specific Design Verification

The following configuration was evaluated:

```text
Areas (cm^2):
A1  = 0.12
A2  = 0.98
A3  = 8.35
A4  = 9.69
A5  = 0.18
A6  = 8.53
A7  = 7.73
A8  = 6.99
A9  = 1.88
A10 = 0.19
A11 = 5.71

y4 = 296.54 cm
y5 = 195.37 cm
y6 = 6.59 cm
```

All areas and heights are within their design bounds. The computed mass is:

```text
82.546006 kg
```

The signed member stresses are:

```text
F1    0.000000 kN/cm^2
F2    2.825808 kN/cm^2
F3   11.673590 kN/cm^2
F4  -17.560801 kN/cm^2
F5  -14.363174 kN/cm^2
F6   17.675629 kN/cm^2
F7   17.751018 kN/cm^2
F8   17.345321 kN/cm^2
F9  -23.920699 kN/cm^2
F10  19.022073 kN/cm^2
F11 -17.484876 kN/cm^2
```

The maximum absolute stress is:

```text
23.920699 kN/cm^2 = 239.207 MPa
```

It occurs in member F9. The allowable value is `15.0 kN/cm^2`, so the stress violation is:

```text
23.920699 - 15.0 = 8.920699 kN/cm^2
```

Therefore, this specific design is **not feasible** under the 150 MPa limit.
The violating members are F4, F6, F7, F8, F9, F10, and F11.

## Units

The code uses:

- Length: cm
- Area: cm^2
- Force: kN
- Stress: kN/cm^2
- Mass: kg

The conversion used for stress is:

```text
1 kN/cm^2 = 10 MPa
150 MPa = 15 kN/cm^2
```

## Steepest-Descent Monte Carlo Robustness Study

`TrussRodriguez_monte_carlo.py` tests the captured steepest-descent design under independent
uniform perturbations:

```text
Ai: +/-0.04 cm^2
y4, y5, y6: +/-0.2 cm
load at B and C: 100--110 kN
allowable stress: 150--157.5 MPa
```

Each sample is checked for area bounds, height bounds, `cond(Kff) <= 1e8`, and
absolute axial stress below the sampled allowable value. Run it with:

```text
python TrussRodriguez_monte_carlo.py
```

The default 10,000-sample run used seed `20261113` and the steepest-descent
design documented above. It produced:

```text
Passing designs: 662 / 10000
Pass rate: 6.620%
Area-bound passes: 4988 / 10000
Height-bound passes: 5276 / 10000
Conditioning passes: 10000 / 10000
Stress passes: 2475 / 10000
Maximum sampled stress: 16.569046 kN/cm^2
Maximum sampled cond(Kff): 4.659068e3
```

Stress was the only failing constraint in this run. Per-sample results are
saved to `TrussRodriguez_monte_carlo_summary.csv`.
