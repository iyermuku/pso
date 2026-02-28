# Michaelewicz Function PSO Optimization

This directory contains an implementation of Particle Swarm Optimization applied to the **Michaelewicz function**, a highly multimodal benchmark problem.

## The Michaelewicz Function

The objective function is:

```
f(x) = - ∑ᵢ sin(xᵢ) · sin^(2m)(i·xᵢ²/π)
```

with domain **0 ≤ xᵢ ≤ π** and shape parameter **m** (typically 10).

This function is known for its many local minima and a single global minimum, making it an excellent test for optimization algorithms.

## Results for 2D with m=10

The PSO optimizer typically finds the global minimum near:

- **Position**: x₁ ≈ 2.20, x₂ ≈ 1.57
- **Value**: ≈ −1.801

## Files

### Core Implementation

- **`michaelewicz_pso.py`**  
  Main PSO implementation with:
  - `michaelewicz(x, m)` – evaluates the objective function
  - `pso_minimize(...)` – generic PSO minimizer for arbitrary dimensions
  - Configurable swarm size, iterations, inertia, and acceleration coefficients
  - Optional history tracking for visualization
  - Uses Latin Hypercube Sampling (LHS) for initialization

### Testing and Visualization

- **`test_michaelewicz.py`**  
  Complete test & visualization pipeline:
  - Runs 2D optimization with 50 particles and 200 iterations
  - Prints the minimum value and coordinates
  - Generates multiple visualizations:
    - **`particle_positions.png`** – shows initial (blue) and final (red) particle locations with best marked as gold star
    - **`gbest_evolution.png`** – tracks global-best value across iterations
    - **`trajectory.gif`** – animated visualization of particles converging to global minimum (red dot)

### Parameter Study

- **`parameter_study.py`**  
  Comprehensive parameter tuning study:
  - Tests 1,170 combinations of inertia (w), c1, and c2 values
  - Inertia range: 0.4 to 0.9 (steps of 0.1)
  - c1, c2 range: 0.3 to 2.7 (steps of 0.1) with constraint 2.0 ≤ c1+c2 ≤ 3.0
  - Identifies parameters that reach 99.7% of best solution in fewest iterations
  - Outputs:
    - **`parameter_study_convergence.png`** – comparison plots of fastest vs. overall best convergence
    - **`parameter_study_results.csv`** – detailed results for all combinations
  - **Key findings**:
    - Fastest to 99.7%: w=0.5, c1=1.3, c2=0.7 (1 iteration!)
    - Overall best: w=0.4, c1=0.3, c2=1.7 (3 iterations)
    - The Michaelewicz function converges much faster than multimodal functions due to its smoother landscape

## Usage

### Run Basic Test

Run from the `MichaelewiczFunction` directory:

```bash
python test_michaelewicz.py
```

### Run Parameter Study

Find optimal PSO parameters:

```bash
python parameter_study.py
```

Or use the optimizer directly:

```python
from michaelewicz_pso import pso_minimize, michaelewicz
import numpy as np

bounds = [(0.0, np.pi)] * 2
best_pos, best_val, history = pso_minimize(
    lambda x: michaelewicz(x, m=10),
    bounds,
    n_dim=2,
    swarm_size=100,
    iters=500,
    track_history=True
)
print(f"Minimum: {best_val} at {best_pos}")
```

## Customization

Adjust PSO parameters in the test script:
- `swarm_size` – number of particles (default: 50)
- `iters` – optimization iterations (default: 200)
- `seed` – random seed for reproducibility
- `anim_interval` – animation frame interval in ms (lower = faster)

You can also extend to higher dimensions by increasing `n` and adjusting `bounds`.

---

For comparison with multimodal function results, see the `MultimodalFunctionTest` folder.
