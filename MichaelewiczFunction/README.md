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

- **`michaelewicz_pso.py`**  
  Main PSO implementation with:
  - `michaelewicz(x, m)` – evaluates the objective function
  - `pso_minimize(...)` – generic PSO minimizer for arbitrary dimensions
  - Configurable swarm size, iterations, inertia, and acceleration coefficients
  - Optional history tracking for visualization

- **`test_michaelewicz.py`**  
  Complete test & visualization pipeline:
  - Runs 2D optimization with 50 particles and 200 iterations
  - Prints the minimum value and coordinates
  - **`particle_positions.png`** – shows initial (blue) and final (red) particle locations
  - **`gbest_evolution.png`** – tracks global-best value across iterations
  - **`trajectory.gif`** – animated visualization of particles converging to global minimum (red dot)

## Usage

Run from the `MichaelewiczFunction` directory:

```bash
python test_michaelewicz.py
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
