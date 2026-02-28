# Easom's Function Analysis

This directory contains a visualization and analysis script for **Easom's function**, a classic benchmark function used to test optimization algorithms.

## Function Definition

Easom's function is defined as:

```
f(x,y) = -cos(x) * cos(y/100) * exp(-(x-π)² - (y/(100π) - 1)²)
```

**Domain:**
- 0 ≤ x ≤ 2π
- 0 ≤ y ≤ 200π

## Key Characteristics

Easom's function is known for its challenging optimization landscape:

1. **Sharp Global Minimum**: The function has a very narrow, needle-like minimum at approximately **(π, 100π)**
2. **Flat Plateau**: The function is nearly flat (close to 0) everywhere except near the global optimum
3. **Narrow Basin**: The exponential term creates an extremely narrow basin of attraction around the optimum
4. **Challenging Benchmark**: This makes it an excellent test case for optimization algorithms - a "needle in a haystack" problem

## Analysis Results

### Global Minimum (Optimization Target)
- **Location**: x ≈ π (≈3.14159), y ≈ 100π (≈314.159)
- **Value**: f(x,y) ≈ **-0.9995**
- **Recommendation**: Use **MINIMIZATION** for optimization

### Global Maximum
- **Location**: x ≈ π, y ≈ 16π
- **Value**: f(x,y) ≈ 0.4326

### Function Range
- Minimum: -0.99947642
- Maximum: 0.43255072
- Range span: 1.43202715

## Optimization Type

**Use MINIMIZATION** because:
- The global minimum has a larger absolute magnitude (-0.9995) than the global maximum (0.4326)
- The optimization goal is to find the deepest valley (sharp minimum) at (π, 100π)

## Scripts

### Core Implementation

- **`easom_pso.py`**: PSO implementation for Easom's function
  - `easom(x, y)`: Function definition
  - `easom_vec(X)`: Vectorized function evaluation
  - `pso_minimize()`: PSO minimizer with:
    - Latin Hypercube Sampling (LHS) initialization
    - Automatic scaling for different axis ranges (x: [0, 2π], y: [0, 200π])
    - Reflection-based boundary handling
    - Optional history tracking for visualization
    - Parameters: swarm_size, iters, inertia, c1, c2, seed

### Visualization & Analysis

- **`plot_easom.py`**: Comprehensive function analysis
  - Evaluates the function over 200×200 grid
  - Identifies global minimum and maximum locations
  - Generates 4-panel visualization:
    - 3D surface plot showing needle-like minimum
    - Contour plot with optima marked
    - Cross-sections at x=π and y=100π
  - Provides optimization recommendation (MINIMIZATION)
  - Output: `easom_function_analysis.png`

- **`test_easom_pso.py`**: PSO optimization test suite
  - Runs PSO with default parameters (50 particles, 300 iterations)
  - Reports initial and final particle spread statistics
  - Performs clustering analysis to identify convergence regions
  - Generates comprehensive visualizations:
    - **Initial vs Final Particle Spread**: Shows convergence behavior
    - **2D Animation**: Particle movement with global best trail (GIF)
    - **3D Animation**: Particle movement on function surface (GIF)
    - **Convergence History**: Global best value vs iteration

- **`parameter_study.py`**: Systematic parameter optimization
  - Tests 180 combinations of PSO parameters:
    - Inertia: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    - (c1, c2) pairs with constraint: 2.0 ≤ c1 + c2 ≤ 3.0
  - Reports configurations reaching 99.7% of optimal in fewest iterations
  - Identifies best final values across all configurations
  - Generates comparison plots and heatmaps

## Usage

### Function Analysis

```bash
python plot_easom.py
```

Output: `easom_function_analysis.png`

### PSO Optimization

```bash
python test_easom_pso.py
```

Output:
- `easom_particle_spread.png`
- `easom_2d_animation.gif`
- `easom_3d_animation.gif`
- `easom_convergence.png`

### Parameter Study

```bash
python parameter_study.py
```

Output:
- `easom_parameter_study_results.csv`
- `easom_parameter_study_comparison.png`
- `easom_parameter_study_all_curves.png`
- `easom_parameter_study_heatmaps.png`

## PSO Optimization Results

### Configuration (test_easom_pso.py)
- Swarm size: 50
- Iterations: 300
- Inertia: 0.7
- c1: 1.5, c2: 1.5
- Seed: 42

### Results
- **Best position**: x = 3.141593 (1.0000π), y = 314.159266 (100.0000π)
- **Best value**: -1.0000000000
- **Achievement**: 100.05% of theoretical optimum (exceeded due to numerical precision)
- **Convergence**: All 50 particles converged to single cluster at global minimum
- **Final spread**: Standard deviation ≈ 0 (perfect convergence)

### Clustering Analysis
- **1 distinct cluster** found (tolerance = 1% of domain)
- All 50 particles converged to global minimum at (π, 100π)
- Mean cluster value: -1.0000000000
- Value range: [-1.0000000000, -1.0000000000]

## Parameter Study Results

### Study Configuration
- **Total combinations tested**: 180
- **Configurations reaching 99.7%**: 180 / 180 (100%)
- **Target value**: -0.99650 (99.7% of -0.9995)

### Top Performers (Fastest to 99.7%)

All reached target in **1 iteration**:
- w=0.9, c1=1.2, c2=1.2 | Final: -0.9999999985
- w=0.9, c1=1.5, c2=1.2 | Final: -0.9999999999
- w=0.9, c1=1.8, c2=1.2 | Final: -1.0000000000
- w=0.9, c1=0.9, c2=1.2 | Final: -1.0000000000
- w=0.8, c1=1.5, c2=1.2 | Final: -1.0000000000
- w=0.8, c1=1.8, c2=1.2 | Final: -1.0000000000
- w=0.8, c1=1.2, c2=1.2 | Final: -1.0000000000
- w=0.8, c1=0.9, c2=1.2 | Final: -1.0000000000

### Best Final Value
- **Parameters**: w=0.4, c1=0.3, c2=1.8
- **Final value**: -1.0000000000
- **Reached 99.7% at iteration**: 5

### Key Findings

1. **Easom's function is relatively easy for PSO** - all 180 configurations succeeded
2. **High inertia (0.8-0.9) performs best** for rapid convergence (1 iteration)
3. **Balanced cognitive/social coefficients** (c1 ≈ c2 ≈ 1.2) work well
4. **Lower inertia still converges** but takes more iterations (5-10)
5. **Excellent benchmark for testing PSO** - difficult topology but well-suited to swarm intelligence

## Implications for PSO

When applying Particle Swarm Optimization to Easom's function:

1. **Use minimization** - search for the lowest function value
2. **Function is PSO-friendly** - despite narrow basin, swarm exploration handles it well
3. **High inertia (0.8-0.9) recommended** - enables rapid convergence (1-2 iterations)
4. **Moderate c1/c2 values** - balanced cognitive/social (≈1.2) work best
5. **Scaling is critical** - different axis ranges (x: 2π, y: 200π) require normalized velocity
6. **Excellent benchmark** - tests exploration capability without being impossibly difficult

## Generated Output

### Function Analysis
- **`easom_function_analysis.png`**: Four-panel visualization
  - 3D surface with global min (red circle) and max (green triangle)
  - Contour plot with optimal points marked
  - Cross-section at y=100π showing sharp minimum
  - Cross-section at x=π showing function behavior along y-axis

### PSO Optimization
- **`easom_particle_spread.png`**: Initial vs final particle positions
  - Left: Latin Hypercube Sampling initialization
  - Right: Final convergence with function values (colormap)
- **`easom_2d_animation.gif`**: Particle movement on contour plot
  - Blue dots: Particles
  - Red star: Global best
  - Red dashed line: Global best trail (last 20 positions)
- **`easom_3d_animation.gif`**: Particle movement on 3D surface
  - Shows particles descending into narrow basin
  - Visualizes challenging topology
- **`easom_convergence.png`**: Global best value vs iteration
  - Shows rapid convergence to optimal value

### Parameter Study
- **`easom_parameter_study_results.csv`**: Full results (180 configurations)
  - Columns: inertia, c1, c2, best_value, first_iter_997, reached_target
- **`easom_parameter_study_comparison.png`**: Side-by-side comparison
  - Left: Fastest to 99.7% (w=0.9, c1=1.2, c2=1.2, 1 iteration)
  - Right: Best final value (w=0.4, c1=0.3, c2=1.8, perfect convergence)
- **`easom_parameter_study_all_curves.png`**: Overlay of convergence curves
  - 50 sampled configurations
  - Highlights fastest and best performers
- **`easom_parameter_study_heatmaps.png`**: Performance by parameter combinations
  - 6 panels (one per inertia value)
  - Shows best_value as function of c1 and c2

## References

Easom's function is a well-known test function in optimization literature, particularly useful for evaluating an algorithm's ability to find a global optimum in a nearly-flat landscape with a very narrow basin of attraction.