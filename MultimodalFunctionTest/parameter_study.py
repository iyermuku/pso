"""
Parameter study for PSO: inertia, c1, c2
Find parameters that achieve 99.7% of best solution in fewest iterations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multimodal_pso import pso_maximize, multimodal_15_12

bounds = [(-2 * np.pi, 2 * np.pi)] * 2
n_dim = 2
swarm_size = 50
max_iters = 200
seed_base = 42

# Generate parameter grid
inertias = np.arange(0.4, 1.0, 0.1)  # 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
c_values = np.arange(0.3, 2.8, 0.1)  # 0.3 to 2.7

# Filter valid c1, c2 combinations: 2.0 <= c1 + c2 <= 3.0
valid_c_pairs = []
for c1 in c_values:
    for c2 in c_values:
        c_sum = c1 + c2
        if 2.0 <= c_sum <= 3.0:
            valid_c_pairs.append((c1, c2))

print(f"Inertia values: {inertias}")
print(f"Valid (c1, c2) pairs: {len(valid_c_pairs)}")
print(f"Total combinations: {len(inertias) * len(valid_c_pairs)}")

# Run parameter study
results = []
overall_best_val = -np.inf
overall_best_params = None
overall_best_history = None

for i, w in enumerate(inertias):
    print(f"\n--- Inertia {w} ({i+1}/{len(inertias)}) ---")
    for j, (c1, c2) in enumerate(valid_c_pairs):
        # Run PSO with these parameters
        best_pos, best_val, history = pso_maximize(
            multimodal_15_12,
            bounds,
            n_dim=n_dim,
            swarm_size=swarm_size,
            iters=max_iters,
            inertia=w,
            c1=c1,
            c2=c2,
            seed=seed_base,
            track_history=True,
        )
        
        # Find iteration when 99.7% of best is reached
        target_val = best_val * 0.997
        gbest_history = history['gbest_history']
        iters_to_target = None
        for iter_idx, val in enumerate(gbest_history):
            if val >= target_val:
                iters_to_target = iter_idx
                break
        
        if iters_to_target is None:
            iters_to_target = max_iters
        
        results.append({
            'w': w,
            'c1': c1,
            'c2': c2,
            'c_sum': c1 + c2,
            'best_val': best_val,
            'iters_to_99_7': iters_to_target,
            'gbest_history': gbest_history,
        })
        
        # Track overall best
        if best_val > overall_best_val:
            overall_best_val = best_val
            overall_best_params = {'w': w, 'c1': c1, 'c2': c2}
            overall_best_history = gbest_history
        
        if (j + 1) % 10 == 0:
            print(f"  Completed {j+1}/{len(valid_c_pairs)} combinations")

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(f"\n\nTotal runs completed: {len(df)}")

# Find the fastest to reach 99.7%
fastest_idx = df['iters_to_99_7'].idxmin()
fastest_params = df.iloc[fastest_idx]
fastest_history = fastest_params['gbest_history']

print(f"\n=== FASTEST TO 99.7% ===")
print(f"Parameters: w={fastest_params['w']:.1f}, c1={fastest_params['c1']:.1f}, c2={fastest_params['c2']:.1f}")
print(f"Best value found: {fastest_params['best_val']:.6f}")
print(f"Iterations to 99.7%: {fastest_params['iters_to_99_7']}")

print(f"\n=== OVERALL BEST ===")
print(f"Parameters: w={overall_best_params['w']:.1f}, c1={overall_best_params['c1']:.1f}, c2={overall_best_params['c2']:.1f}")
print(f"Best value found: {overall_best_val:.6f}")

# Check if they're different
is_different = (fastest_params['w'] != overall_best_params['w'] or 
                fastest_params['c1'] != overall_best_params['c1'] or 
                fastest_params['c2'] != overall_best_params['c2'])

# Plot objective evolution
fig, axes = plt.subplots(1, 2 if is_different else 1, figsize=(12 if is_different else 6, 5))

if is_different:
    ax1, ax2 = axes
else:
    ax1 = axes

# Plot fastest to 99.7%
ax1.plot(fastest_history, marker='.', linewidth=1.5)
ax1.axhline(y=fastest_params['best_val'] * 0.997, color='r', linestyle='--', label='99.7% of best')
ax1.set_title(f"Fastest to 99.7%\nw={fastest_params['w']:.1f}, c1={fastest_params['c1']:.1f}, c2={fastest_params['c2']:.1f}")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Best Objective Value")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot overall best if different
if is_different:
    ax2.plot(overall_best_history, marker='.', linewidth=1.5, color='orange')
    ax2.axhline(y=overall_best_val * 0.997, color='r', linestyle='--', label='99.7% of best')
    ax2.set_title(f"Overall Best\nw={overall_best_params['w']:.1f}, c1={overall_best_params['c1']:.1f}, c2={overall_best_params['c2']:.1f}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Objective Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

plt.tight_layout()
plt.savefig("parameter_study_convergence.png", dpi=100)
print("\nSaved: parameter_study_convergence.png")
plt.close()

# Save detailed results to CSV
df_export = df.copy()
df_export['gbest_history'] = df_export['gbest_history'].apply(lambda x: str(x.tolist())[:50])  # truncate for CSV
df_export.to_csv("parameter_study_results.csv", index=False)
print("Saved: parameter_study_results.csv")

# Print top 10 fastest combinations
print("\n=== TOP 10 FASTEST TO 99.7% ===")
top10 = df.nsmallest(10, 'iters_to_99_7')[['w', 'c1', 'c2', 'best_val', 'iters_to_99_7']]
print(top10.to_string(index=False))

# Print sorted by best value
print("\n=== TOP 10 BY BEST VALUE ===")
top10_val = df.nlargest(10, 'best_val')[['w', 'c1', 'c2', 'best_val', 'iters_to_99_7']]
print(top10_val.to_string(index=False))
