# PSO Single-Run Algorithm (10-bar Truss)

This document describes the `pso_single_run` implementation in `pso.py` and
explains how the standard Particle Swarm Optimization algorithm has been
extended with variable coefficients.

## Overview of the Algorithm

`pso_single_run` performs a classic PSO optimization for the 10-bar truss
design problem with 13 decision variables per particle: ten cross-sectional
areas plus three PSO coefficients (`w`, `c1`, `c2`).  The objective is to
minimize the structural mass while satisfying displacement and stress
constraints.

1. **Initialization**
   - A Latin Hypercube Sample (LHS) is used to seed the swarm in the
     full 13-dimensional space.  This ensures a stratified, space-filling
     start for both the design variables (areas) and the PSO parameters.
   - Velocities are initialized randomly in [-1,1] for every dimension.
   - Particle positions are then projected into their allowable ranges: areas
     within `[Amin,Amax]`, inertia weight `w` in `[0.5,0.95]`, and acceleration
     coefficients `c1`/`c2` in `[0.3,2.7]`.  Each particle's `w,c1,c2` values
     are corrected by `project_params` to respect any additional parameter
     constraints.

2. **Evaluation**
   - Each particle's area vector is passed through a finite-element solver
     (`solve_displacements`) to obtain nodal displacements.
   - Constraint violations (displacement and stress) are aggregated into a
     vector; the sum of its positive parts constitutes the constraint violation
     (CV) measure.
   - A penalized objective function (`penalized_objective`) blends mass and
     CV information, using swarm-average mass and violation for normalization.

3. **Personal and Global Bests**
   - Personal bests (`pbest`) are recorded based on Deb's feasibility rule: a
     feasible solution always beats an infeasible one, and among equal
     feasibility status the lower penalized objective is preferred.
   - The global best (`gbest`) is chosen from the personal bests using the same
     rule, allowing a local/lexicographic comparison of constraint violation
     and objective.

4. **Velocity & Position Updates**
   - The PSO velocity update uses each particle's own `w`, `c1`, and `c2`
     coefficients when computing the new velocity:
     ```python
     V = (
         w_vec[:, None] * V
         + c1_vec[:, None] * r1 * (pbest_X - X)
         + c2_vec[:, None] * r2 * (gbest_X - X)
     )
     ```
   - Positions are updated by `X = X + V` and then clamped to feasible
     ranges with reflective boundary handling.
   - After moving, the areas are clipped to `[Amin,Amax]`, and the updated
     `w,c1,c2` entries are re-projected to ensure they remain valid.

5. **History Tracking**
   - During the run, the code collects histories for mass, displacement,
     feasibility fraction, and the evolving coefficients (`w_hist`,
     `c1_hist`, `c2_hist`).  These are returned with the result object for
     plotting and analysis.

6. **Stalling and Restarts**
   - A simple stall counter detects when the global best fails to improve for a
     specified window.  If enough restarts remain, the swarm is reinitialized
     around the current global best with jitter in both areas and coefficients.

## Evolving PSO Coefficients (`w`, `c1`, `c2`)

Unlike a standard PSO where inertia and acceleration constants are fixed,
`pso_single_run` treats these three parameters as additional dimensions in the
search space.  This adds two important behaviours:

- **Self‑adaptation:**  each particle carries its own set of coefficients and
  therefore explores different exploration/exploitation balances.  Good
  coefficient sets can propagate through the swarm when those particles also
  have low objective values.

- **Tracking & Visualization:**  histories of `gbest` values for `w`, `c1`, and
  `c2` are recorded at every iteration (including across restarts).  In
  `main.py` a dedicated plot (`pso_single_params_vs_iteration.png`) shows how
  the best particle's coefficients evolve over time.  The animation and
  weight/displacement plot also indirectly reflect this behaviour via the
  changing velocities and constraint satisfaction.

The evolved coefficients are passed back with the solution in the result
dictionary under keys `'gbest_w'`, `'gbest_c1'`, `'gbest_c2'` and likewise for
best-ever and best-feasible solutions.

By treating the PSO parameters as decision variables, the algorithm effectively
performs a co‑optimization of the search strategy alongside the truss
design.  This can lead to adaptive behaviour where, e.g., inertia decreases as
particles converge or acceleration constants shift depending on the landscape.

---

To reproduce, simply run `python main.py --mode single ...` and inspect the
output plots; the README single_run file serves as a reference for the
algorithmic choices and their implementation details.