# Overview
This project focuses on the PSO (Particle Swarm Optimization) algorithm applied to the Bar Truss problem. It aims to find the optimal design of a truss structure through simulation.

# Dependencies
- Python 3.6 or higher
- NumPy
- Matplotlib
- SciPy

# Step-by-Step Setup
1. **Clone the repository**  
   To get started, clone the repository by running:
   ```bash
   git clone https://github.com/iyermuku/pso.git
   cd pso/v10-good
   ```

2. **Create a virtual environment (optional)**  
   It is recommended to create a virtual environment. You can create one using:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. **Install dependencies**  
   Use pip to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

# Command-Line Arguments
To run the PSO algorithm, use the following command:  
```bash
python pso.py --iterations <number_of_iterations> --population <population_size> --output <output_file>
```

| Argument         | Description                              |
|------------------|------------------------------------------|
| `--iterations`   | Number of iterations for the algorithm   |
| `--population`    | Size of the swarm (number of particles) |
| `--output`       | Name of the output file                  |

# Expected Output
The expected output is a text file containing:
- The best designs obtained  
- Performance metrics of the run, including the best fitness values recorded.

# Runtime Information
- Each run's duration can vary based on the parameters set, average runtime is approximately X minutes for Y iterations and Z population size.

# Common Issues
1. **ImportError**: Make sure all dependencies are correctly installed. Refer to the requirements.txt for the list of necessary packages.
2. **FileNotFoundError**: Check if the output file path is correct or accessible.
3. **Performance Degradation**: If you notice a slowdown, try reducing the population size or iterations.
