"""
Parameter study for Easom's function PSO optimization
Tests different combinations of inertia, c1, and c2 parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easom_pso import pso_minimize, easom_vec
from itertools import product

print("=" * 70)
print("PSO PARAMETER STUDY FOR EASOM'S FUNCTION")
print("=" * 70)
print()

# Configuration
bounds = [(0, 2*np.pi), (0, 200*np.pi)]
n_dim = 2
swarm_size = 50
iters = 300
seed = 42

# Parameter ranges
inertia_vals = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
c_vals = np.arange(0.3, 3.0, 0.3)  # 0.3 to 2.7

# Generate valid (c1, c2) combinations with constraint 2.0 <= c1 + c2 <= 3.0
valid_c_pairs = []
for c1 in c_vals:
    for c2 in c_vals:
        if 2.0 <= (c1 + c2) <= 3.0:
            valid_c_pairs.append((c1, c2))

print(f"Testing {len(inertia_vals)} inertia values: {inertia_vals}")
print(f"Testing {len(valid_c_pairs)} valid (c1, c2) combinations")
print(f"Constraint: 2.0 <= c1 + c2 <= 3.0")
print(f"Total combinations: {len(inertia_vals) * len(valid_c_pairs)}")
print()
print(f"Configuration:")
print(f"  Swarm size: {swarm_size}")
print(f"  Iterations: {iters}")
print(f"  Seed: {seed}")
print()

# Target: 99.7% of optimal (-0.9995)
target_value = -0.9995 * 0.997

print(f"Target value (99.7% of optimal): {target_value:.10f}")
print()
print("Running parameter study...")

# Store results
results = []

# Run all combinations
total_runs = len(inertia_vals) * len(valid_c_pairs)
run_count = 0

for w in inertia_vals:
    for c1, c2 in valid_c_pairs:
        run_count += 1
        if run_count % 50 == 0:
            print(f"Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")
        
        # Run PSO
        best_pos, best_val, history = pso_minimize(
            easom_vec, 
            bounds, 
            n_dim,
            swarm_size=swarm_size,
            iters=iters,
            inertia=w,
            c1=c1,
            c2=c2,
            seed=seed,
            track_history=True
        )
        
        # Find first iteration that reaches target
        gbest_history = history['gbest_history']
        reaches_target = gbest_history <= target_value
        
        if np.any(reaches_target):
            first_iter = np.argmax(reaches_target)
        else:
            first_iter = iters  # Did not reach target
        
        results.append({
            'inertia': w,
            'c1': c1,
            'c2': c2,
            'best_value': best_val,
            'first_iter_997': first_iter,
            'reached_target': np.any(reaches_target),
            'history': gbest_history
        })

print(f"Progress: {total_runs}/{total_runs} (100.0%)")
print()

# Convert to DataFrame
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'history'} for r in results])

# Find best configurations
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()

# Configurations that reached target
reached_df = df[df['reached_target']]
print(f"Configurations that reached 99.7%: {len(reached_df)} / {len(df)}")
print()

if len(reached_df) > 0:
    # Sort by first iteration to reach target
    reached_df_sorted = reached_df.sort_values('first_iter_997')
    
    print("TOP 10 FASTEST TO REACH 99.7%:")
    print("-" * 70)
    for idx, row in reached_df_sorted.head(10).iterrows():
        print(f"w={row['inertia']:.1f}, c1={row['c1']:.1f}, c2={row['c2']:.1f} | "
              f"Iter: {int(row['first_iter_997']):<3} | "
              f"Final: {row['best_value']:.10f}")
    print()
    
    # Best final value
    best_final_idx = df['best_value'].idxmin()
    best_final = df.loc[best_final_idx]
    print("BEST FINAL VALUE:")
    print("-" * 70)
    print(f"w={best_final['inertia']:.1f}, c1={best_final['c1']:.1f}, c2={best_final['c2']:.1f}")
    print(f"Final value: {best_final['best_value']:.10f}")
    print(f"Reached 99.7% at iteration: {int(best_final['first_iter_997'])}")
    print()

else:
    print("WARNING: No configuration reached 99.7% target!")
    print()
    
    # Show best performers
    df_sorted = df.sort_values('best_value')
    print("TOP 10 BEST FINAL VALUES:")
    print("-" * 70)
    for idx, row in df_sorted.head(10).iterrows():
        print(f"w={row['inertia']:.1f}, c1={row['c1']:.1f}, c2={row['c2']:.1f} | "
              f"Final: {row['best_value']:.10f}")
    print()

# Save results
df.to_csv('easom_parameter_study_results.csv', index=False)
print("Results saved to: easom_parameter_study_results.csv")
print()

# =============================================================================
# VISUALIZATION: Convergence Comparison
# =============================================================================
print("Generating comparison plots...")

if len(reached_df) > 0:
    # Get the fastest to 99.7%
    fastest_idx = reached_df_sorted.index[0]
    fastest_result = results[fastest_idx]
    
    # Get the best final value
    best_final_result = results[best_final_idx]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Fastest to 99.7%
    ax1.plot(fastest_result['history'], 'b-', linewidth=2, label='Global Best')
    ax1.axhline(target_value, color='red', linestyle='--', linewidth=1, 
                label=f'99.7% target ({target_value:.6f})')
    ax1.axhline(-0.9995, color='green', linestyle=':', linewidth=1, 
                label=f'Optimal (-0.9995)')
    ax1.axvline(fastest_result['first_iter_997'], color='red', linestyle='--', 
                linewidth=1, alpha=0.5)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Global Best Value', fontsize=12)
    ax1.set_title(f"Fastest to 99.7%\nw={fastest_result['inertia']:.1f}, "
                  f"c1={fastest_result['c1']:.1f}, c2={fastest_result['c2']:.1f} | "
                  f"Reached at iter {fastest_result['first_iter_997']}", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([min(fastest_result['history'].min(), -1.0), 0.5])
    
    # Plot 2: Best final value
    ax2.plot(best_final_result['history'], 'b-', linewidth=2, label='Global Best')
    ax2.axhline(target_value, color='red', linestyle='--', linewidth=1, 
                label=f'99.7% target ({target_value:.6f})')
    ax2.axhline(-0.9995, color='green', linestyle=':', linewidth=1, 
                label=f'Optimal (-0.9995)')
    if best_final_result['first_iter_997'] < iters:
        ax2.axvline(best_final_result['first_iter_997'], color='red', linestyle='--', 
                    linewidth=1, alpha=0.5)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Global Best Value', fontsize=12)
    ax2.set_title(f"Best Final Value\nw={best_final_result['inertia']:.1f}, "
                  f"c1={best_final_result['c1']:.1f}, c2={best_final_result['c2']:.1f} | "
                  f"Final: {best_final_result['best_value']:.8f}", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([min(best_final_result['history'].min(), -1.0), 0.5])
    
    plt.tight_layout()
    plt.savefig('easom_parameter_study_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: easom_parameter_study_comparison.png")
    plt.close()
    
    # Additional plot: All convergence curves (sample)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot sample of convergence curves
    sample_size = min(50, len(results))
    sample_indices = np.linspace(0, len(results)-1, sample_size, dtype=int)
    
    for i in sample_indices:
        r = results[i]
        alpha = 0.3 if not r['reached_target'] else 0.7
        color = 'gray' if not r['reached_target'] else 'blue'
        ax.plot(r['history'], color=color, alpha=alpha, linewidth=0.5)
    
    # Highlight best performers
    ax.plot(fastest_result['history'], 'r-', linewidth=2, 
            label=f"Fastest (w={fastest_result['inertia']:.1f}, "
                  f"c1={fastest_result['c1']:.1f}, c2={fastest_result['c2']:.1f})")
    
    if fastest_idx != best_final_idx:
        ax.plot(best_final_result['history'], 'g-', linewidth=2, 
                label=f"Best final (w={best_final_result['inertia']:.1f}, "
                      f"c1={best_final_result['c1']:.1f}, c2={best_final_result['c2']:.1f})")
    
    ax.axhline(target_value, color='orange', linestyle='--', linewidth=1, 
               label=f'99.7% target')
    ax.axhline(-0.9995, color='purple', linestyle=':', linewidth=1, 
               label=f'Optimal')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Global Best Value', fontsize=12)
    ax.set_title(f'Convergence Comparison ({sample_size} sampled configurations)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([df['best_value'].min() - 0.1, 0.5])
    
    plt.tight_layout()
    plt.savefig('easom_parameter_study_all_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: easom_parameter_study_all_curves.png")
    plt.close()

else:
    # No configuration reached target - just plot best few
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_sorted = df.sort_values('best_value')
    top_n = min(10, len(df_sorted))
    
    for i, idx in enumerate(df_sorted.head(top_n).index):
        r = results[idx]
        ax.plot(r['history'], linewidth=2, 
                label=f"w={r['inertia']:.1f}, c1={r['c1']:.1f}, c2={r['c2']:.1f}")
    
    ax.axhline(-0.9995, color='red', linestyle='--', linewidth=1, 
               label=f'Optimal (-0.9995)')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Global Best Value', fontsize=12)
    ax.set_title(f'Top {top_n} Configurations by Final Value', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('easom_parameter_study_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: easom_parameter_study_comparison.png")
    plt.close()

# =============================================================================
# VISUALIZATION: Heatmaps
# =============================================================================
print("Generating heatmap analysis...")

# Create heatmaps for each inertia value
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, w in enumerate(inertia_vals):
    ax = axes[i]
    
    # Filter results for this inertia
    w_df = df[df['inertia'] == w].copy()
    
    # Create pivot table for heatmap
    pivot = w_df.pivot_table(values='best_value', index='c2', columns='c1', aggfunc='mean')
    
    # Plot heatmap
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f'{c:.1f}' for c in pivot.columns], fontsize=8)
    ax.set_yticklabels([f'{c:.1f}' for c in pivot.index], fontsize=8)
    ax.set_xlabel('c1', fontsize=10)
    ax.set_ylabel('c2', fontsize=10)
    ax.set_title(f'w = {w:.1f}', fontsize=12)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)

plt.suptitle('Best Final Values for Different Parameter Combinations', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('easom_parameter_study_heatmaps.png', dpi=150, bbox_inches='tight')
print("Saved: easom_parameter_study_heatmaps.png")
plt.close()

print()
print("=" * 70)
print("PARAMETER STUDY COMPLETE")
print("=" * 70)
print("Generated files:")
print("  - easom_parameter_study_results.csv")
print("  - easom_parameter_study_comparison.png")
if len(reached_df) > 0:
    print("  - easom_parameter_study_all_curves.png")
print("  - easom_parameter_study_heatmaps.png")
