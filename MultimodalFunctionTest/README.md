# Multimodal Function PSO

This directory contains a Particle Swarm Optimization (PSO) implementation applied to the multimodal test function from Xin‑She Yang's *Engineering Optimization*, chapter 15 (equation 15.12).

## Objective Function

The multimodal objective function is:

```text
f(x) = (sum_i |x_i|) * exp(-sum_i |sin(x_i^2)|)
```

with each coordinate bounded in [-2π, 2π]. This function has multiple local optima, making it a good test case for PSO performance.

## Scripts

### Core Implementation

- **`multimodal_pso.py`**: Main PSO implementation with Latin Hypercube Sampling (LHS) for initialization
  - Generic PSO routine for arbitrary dimensions
  - History tracking for visualization
  - Parameters: swarm_size (default 30), iters (default 100), w (inertia), c1/c2 (cognitive/social)
  - Returns: best_pos, best_val, history (if track_history=True)

### Testing and Visualization

- **`test_multimodal_pso.py`**: Comprehensive test script with multiple visualizations
  - Runs 2D PSO optimization (50 particles, 200 iterations)
  - Generates initial/final particle position plots
  - Creates global best evolution plot
  - Produces 2D trajectory animation showing particle movement
  - Creates 3D surface animation with particles moving over the function landscape
  - Outputs: `particle_positions.png`, `gbest_evolution.png`, `trajectory.gif`, `surface_trajectory.gif`

### Analysis Scripts

- **`optima_check.py`**: Identifies all unique final particle positions and evaluates their objective values
  - Useful for understanding convergence behavior
  - Shows which optima particles settled on

- **`particle_trajectory_check.py`**: Prints the complete trajectory of the "stuck" particle
  - Identifies the particle that converged to a secondary (non-global) optimum
  - Displays position and objective value at each iteration
  - Helps understand why particles get trapped in local optima

### Animation Scripts

- **`particle_trajectory_anim.py`**: 2D animation comparing stuck particle with global best
  - Shows stuck particle (red), global best (blue), and personal best (green)
  - Displays trajectory path with dashed line
  - Output: `stuck_particle_vs_gbest.gif`

- **`particle_trajectory_3d_anim.py`**: 3D surface animation (non-interactive)
  - Plots the multimodal function surface
  - Shows stuck particle, global best, and personal best moving over the surface
  - Includes trajectory path in 3D space
  - Output: `stuck_particle_vs_gbest_3d.gif`

- **`particle_trajectory_3d_interactive.py`**: Interactive 3D animation
  - Same as 3D animation but with interactive controls
  - Press SPACEBAR to pause/resume animation
  - Use mouse to rotate and zoom the 3D view
  - Ideal for detailed analysis of particle behavior

### Parameter Study

- **`parameter_study.py`**: Comprehensive parameter tuning study
  - Tests 1,170 combinations of inertia (w), c1, and c2 values
  - Inertia range: 0.4 to 0.9 (steps of 0.1)
  - c1, c2 range: 0.3 to 2.7 (steps of 0.1) with constraint 2.0 ≤ c1+c2 ≤ 3.0
  - Identifies parameters that reach 99.7% of best solution in fewest iterations
  - Outputs: `parameter_study_convergence.png`, `parameter_study_results.csv`
  - **Key finding**: w=0.6, c1=0.6, c2=2.3 reaches 99.7% in just 3 iterations

## Usage Examples

### Basic PSO Run

```python
from multimodal_pso import pso_maximize, multimodal_15_12
import numpy as np

bounds = [(-2*np.pi, 2*np.pi)] * 2
best_pos, best_val, _ = pso_maximize(multimodal_15_12, bounds, n_dim=2, swarm_size=50, iters=200)
print(f"Best position: {best_pos}, Best value: {best_val}")
```

### Generate All Visualizations

```bash
python test_multimodal_pso.py
```

### Analyze Stuck Particle

```bash
python particle_trajectory_check.py      # Text output
python particle_trajectory_anim.py       # 2D animation
python particle_trajectory_3d_anim.py    # 3D animation (GIF)
python particle_trajectory_3d_interactive.py  # Interactive 3D
```

### Run Parameter Study

```bash
python parameter_study.py                # Test 1,170 parameter combinations
```

## Key Findings

- The PSO algorithm successfully finds the global optimum at approximately [6.01126, 6.01126] with value ≈88.83
- In typical runs, 49 out of 50 particles converge to the global optimum
- One particle often gets "stuck" at a secondary peak around [5.81, 5.63] with value ≈4.33
- The stuck particle demonstrates the challenge of multimodal optimization where local optima can trap particles

## Parameters

Adjust these in the PSO call for experimentation:
- `swarm_size`: Number of particles (default 30, tests use 50)
- `iters`: Number of iterations (default 100, tests use 200)
- `w`: Inertia weight (default 0.5)
- `c1`, `c2`: Cognitive and social coefficients (default 1.5 each)
- `seed`: Random seed for reproducibility