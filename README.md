# Particle Swarm Optimization (PSO) Repository

A comprehensive collection of PSO implementations and benchmark function analyses demonstrating various applications of particle swarm optimization algorithms.

## Repository Structure

### Benchmark Functions

#### 1. **MultimodalFunctionTest/**
- **Function**: Custom multimodal function with two peaks
- **Type**: Maximization
- **Challenge**: Multiple local optima (88.83 and 4.33)
- **PSO Success Rate**: 43% of configurations reach 99.7% target
- **Key Finding**: Social-heavy strategy (w=0.6, c1=0.6, c2=2.3) reaches target in 3 iterations
- **Features**:
  - Particle trajectory animations (2D and 3D)
  - Interactive 3D visualization with pause control
  - Parameter study (1,170 combinations)
  - Stuck particle analysis

#### 2. **MichaelewiczFunction/**
- **Function**: Michaelewicz function (m=10)
- **Type**: Minimization
- **Challenge**: Steep valleys, sharp features
- **PSO Success Rate**: 94% of configurations successful
- **Key Finding**: Cognitive-heavy strategy (w=0.5, c1=1.3, c2=0.7) reaches target in 1 iteration
- **Features**:
  - Parameter study (1,170 combinations)
  - Convergence visualizations
  - Fastest convergence among all benchmarks

#### 3. **EasomsFunction/**
- **Function**: Easom's function
- **Type**: Minimization
- **Challenge**: Needle-in-haystack, narrow basin at (π, 100π)
- **PSO Success Rate**: 100% of configurations successful
- **Key Finding**: Balanced strategy with high inertia (w=0.9, c1=1.2, c2=1.2) reaches target in 1 iteration
- **Features**:
  - Axis scaling for different ranges (x: 2π, y: 200π)
  - 2D and 3D particle animations
  - Parameter study (180 combinations)
  - Clustering analysis
  - Complete function topology analysis

### Structural Optimization

#### 4. **PSO10BarTruss/v10-good/**
- **Application**: 10-bar truss structural optimization
- **Objective**: Minimize weight subject to stress and displacement constraints
- **Constraint Handling**: Deb's feasibility rule
- **Features**:
  - Multiple PSO variants (standard, conservative, aggressive)
  - Swarm size and iteration studies
  - Perturbation analysis
  - Comprehensive constraint checking

#### 5. **PSO10BarDiscreeteSectionTruss/**
- Discrete section optimization for 10-bar truss
- Limited catalog of available cross-sections

#### 6. **PSO72BarTruss/v2/**
- Large-scale 72-bar truss optimization
- Benchmark comparisons with published results
- Constraint checking and validation

#### 7. **PSO72BarDiscreteSectionsTruss/**
- Discrete section optimization for 72-bar truss
- Robust and single-run variants

## Parameter Study Comparison

A comprehensive comparison across three benchmark functions (Multimodal, Michaelewicz, Easom) analyzing optimal PSO parameter selection.

**📊 See: [PARAMETER_STUDY_COMPARISON.md](PARAMETER_STUDY_COMPARISON.md)**

### Key Findings Summary

| Function | Success Rate | Fastest Convergence | Optimal Strategy |
|----------|--------------|---------------------|------------------|
| **Multimodal** | 43% | 3 iterations | Social-heavy (c2 > c1) |
| **Michaelewicz** | 94% | 1 iteration | Cognitive-heavy (c1 > c2) |
| **Easom** | 100% | 1 iteration | Balanced (c1 ≈ c2) + High inertia |

### Universal Recommendations

For **unknown functions**, start with:
- **Inertia**: w = 0.6-0.7 (moderate)
- **Cognitive**: c1 = 1.2
- **Social**: c2 = 1.3
- **Total**: c1 + c2 ≈ 2.5

**Strategy Selection**:
- Multiple optima → High social (c2 = 2.0-2.5)
- Smooth unimodal → High cognitive (c1 = 1.3-1.8)
- Narrow basin → Balanced + High inertia (w = 0.8-0.9)

## Comparison Visualizations

Generated comparison plots:

1. **[parameter_study_comparison_summary.png](parameter_study_comparison_summary.png)**
   - Success rates across functions
   - Convergence speed comparison

2. **[parameter_study_comparison_detailed.png](parameter_study_comparison_detailed.png)**
   - Optimal parameter values by function
   - Cognitive/social balance analysis
   - Difficulty ranking

3. **[parameter_study_strategies.png](parameter_study_strategies.png)**
   - Strategy recommendations by function type
   - Decision matrix for parameter selection

## Common Features

All PSO implementations include:

- **Latin Hypercube Sampling (LHS)** for initial particle distribution
- **History tracking** for visualization and analysis
- **Boundary handling** via reflection or clamping
- **Comprehensive testing** with reproducible seeds
- **Detailed documentation** with usage examples

## Algorithm Parameters

### Standard PSO Configuration

```python
swarm_size = 50        # Number of particles
iterations = 300       # Maximum iterations
inertia = 0.7         # Inertia weight (w)
c1 = 1.5              # Cognitive coefficient
c2 = 1.5              # Social coefficient
```

### Velocity Update Equation

```
V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
X[i] = X[i] + V[i]
```

Where:
- `V[i]` = velocity of particle i
- `X[i]` = position of particle i
- `pbest[i]` = personal best position of particle i
- `gbest` = global best position
- `r1, r2` = random numbers in [0, 1]

## Quick Start

### Benchmark Functions

```bash
# Multimodal function
cd MultimodalFunctionTest
python test_multimodal_pso.py
python parameter_study.py

# Michaelewicz function
cd MichaelewiczFunction
python test_michaelewicz.py
python parameter_study.py

# Easom's function
cd EasomsFunction
python test_easom_pso.py
python parameter_study.py
```

### Structural Optimization

```bash
# 10-bar truss
cd PSO10BarTruss/v10-good
python main.py

# 72-bar truss
cd PSO72BarTruss/v2
python pso72_main.py
```

## Key Results

### Difficulty Ranking (Easiest to Hardest)

1. **Easom** (100% success) - Surprisingly PSO-friendly despite narrow basin
2. **Michaelewicz** (94% success) - Smooth landscape enables fast convergence
3. **Multimodal** (43% success) - Multiple optima challenge algorithm

### Convergence Insights

- **Multiple optima are harder than narrow basins** - Easom's needle-in-haystack is easier than Multimodal's dual peaks
- **Smooth landscapes enable 1-iteration convergence** - Michaelewicz and Easom both achieve this
- **Social influence critical for multimodal** - High c2 helps swarm reach consensus
- **Cognitive exploration better for unimodal** - High c1 allows individual particles to find optimum

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Pandas (for parameter studies)
- SciPy (optional, for some analyses)

## Installation

```bash
pip install numpy matplotlib pandas scipy
```

## Generated Outputs

Each implementation generates:
- **Figures**: PNG files showing convergence, particle distributions, animations
- **Animations**: GIF files of particle movement (2D and 3D)
- **Data**: CSV files with parameter study results
- **Reports**: Console output with detailed analysis

## Research Applications

This repository demonstrates PSO applications in:

1. **Benchmark testing** - standardized function evaluation
2. **Parameter tuning** - systematic optimization of PSO parameters
3. **Visualization** - understanding swarm behavior through animation
4. **Structural engineering** - real-world truss optimization
5. **Constraint handling** - practical constraint satisfaction techniques

## References

### Benchmark Functions

- **Multimodal**: Custom test function with known optima
- **Michaelewicz**: Michaelewicz, Z. (1996). Genetic Algorithms + Data Structures = Evolution Programs
- **Easom**: Easom, E. E. (1990). A survey of global optimization techniques

### PSO Algorithm

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization
- Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer
- Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence

### Constraint Handling

- Deb, K. (2000). An efficient constraint handling method for genetic algorithms

## Contributing

Each directory contains its own README with detailed documentation for:
- Function definitions
- Algorithm implementations
- Usage examples
- Expected outputs
- Analysis methodologies

## License

This is a research and educational repository demonstrating PSO implementations.

---

**Last Updated**: February 27, 2026

**Quick Links**:
- 📊 [Parameter Study Comparison](PARAMETER_STUDY_COMPARISON.md)
- 🔍 [Multimodal Function](MultimodalFunctionTest/README.md)
- 🔍 [Michaelewicz Function](MichaelewiczFunction/README.md)
- 🔍 [Easom's Function](EasomsFunction/README.md)
- 🏗️ [10-Bar Truss Optimization](PSO10BarTruss/v10-good/README.md)
