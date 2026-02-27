"""Parameter study for pso_single_run: vary swarm size and iterations."""
import numpy as np
import logging
from pso import pso_best_of_seeds

# small study grid (adjust as desired)
swarm_sizes = [20, 40, 60]
iters_list = [100, 200, 400]

results = []
for swarm in swarm_sizes:
    for iters in iters_list:
        best, runs = pso_best_of_seeds(num_runs=3, swarm_size=swarm, iters=iters,
                                       max_restarts=1, stall_window=25,
                                       base_seed=2026, robust=False)
        # choose best feasible across runs
        best_feas_mass = np.inf
        for res in runs:
            if res.get('best_feas_mass', np.nan) < best_feas_mass:
                best_feas_mass = res['best_feas_mass']
        results.append((swarm, iters, best_feas_mass))
        logging.info("swarm=%d iters=%d best_feas_mass=%.4f", swarm, iters, best_feas_mass)

print("Study results:")
for swarm, iters, mass in results:
    print(f"swarm={swarm:3d}, iters={iters:4d} -> best feasible mass {mass:.4f}")
