"""
Generate comparison visualizations across all three parameter studies
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from parameter studies
functions = ['Multimodal', 'Michaelewicz', 'Easom']

# Success rates
success_rates = [43, 94, 100]  # percentage
total_configs = [1170, 1170, 180]
successful_configs = [int(sr * tc / 100) for sr, tc in zip(success_rates, total_configs)]

# Fastest to 99.7%
fastest_iters = [3, 1, 1]
fastest_params = [
    'w=0.6\nc1=0.6\nc2=2.3',
    'w=0.5\nc1=1.3\nc2=0.7',
    'w=0.9\nc1=1.2\nc2=1.2'
]

# Best final value parameters
best_params = [
    'w=0.6\nc1=0.6\nc2=2.3',
    'w=0.4\nc1=0.3\nc2=1.7',
    'w=0.4\nc1=0.3\nc2=1.8'
]
best_iters = [3, 3, 5]

# Coefficient patterns
c1_fastest = [0.6, 1.3, 1.2]
c2_fastest = [2.3, 0.7, 1.2]
w_fastest = [0.6, 0.5, 0.9]

# =============================================================================
# Figure 1: Success Rates
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot of success rates
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(functions, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_title('PSO Success Rate by Function\n(Configurations Reaching 99.7% Target)', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 110])
ax1.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{rate}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add configuration counts
for i, (func, succ, total) in enumerate(zip(functions, successful_configs, total_configs)):
    ax1.text(i, 5, f'{succ}/{total}', ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')

# Iterations to 99.7%
bars2 = ax2.bar(functions, fastest_iters, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Iterations', fontsize=12)
ax2.set_title('Fastest Convergence to 99.7% Target', fontsize=14, fontweight='bold')
ax2.set_ylim([0, max(fastest_iters) + 1])
ax2.grid(True, alpha=0.3, axis='y')

# Add iteration labels
for bar, iters, params in zip(bars2, fastest_iters, fastest_params):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{iters}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    # Add parameters below bar
    ax2.text(bar.get_x() + bar.get_width()/2., -0.3,
             params, ha='center', va='top', fontsize=8)

ax2.set_ylim([-0.5, max(fastest_iters) + 0.5])

plt.tight_layout()
plt.savefig('parameter_study_comparison_summary.png', dpi=150, bbox_inches='tight')
print("Saved: parameter_study_comparison_summary.png")
plt.close()

# =============================================================================
# Figure 2: Parameter Patterns
# =============================================================================
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Inertia weights
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(functions, w_fastest, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Inertia (w)', fontsize=11)
ax1.set_title('Optimal Inertia Weight', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, w_fastest):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 2: c1 coefficients
ax2 = plt.subplot(2, 3, 2)
bars = ax2.bar(functions, c1_fastest, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('c1 (Cognitive)', fontsize=11)
ax2.set_title('Optimal Cognitive Coefficient', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 2.5])
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, c1_fastest):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 3: c2 coefficients
ax3 = plt.subplot(2, 3, 3)
bars = ax3.bar(functions, c2_fastest, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('c2 (Social)', fontsize=11)
ax3.set_title('Optimal Social Coefficient', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 2.5])
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, c2_fastest):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 4: c1 + c2 total
ax4 = plt.subplot(2, 3, 4)
totals = [c1 + c2 for c1, c2 in zip(c1_fastest, c2_fastest)]
bars = ax4.bar(functions, totals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.axhline(2.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Min constraint (2.0)')
ax4.axhline(3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max constraint (3.0)')
ax4.set_ylabel('c1 + c2', fontsize=11)
ax4.set_title('Total Acceleration Coefficient', fontsize=12, fontweight='bold')
ax4.set_ylim([1.5, 3.5])
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(fontsize=8)
for bar, val in zip(bars, totals):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 5: c1/c2 ratio (cognitive/social balance)
ax5 = plt.subplot(2, 3, 5)
ratios = [c1/c2 for c1, c2 in zip(c1_fastest, c2_fastest)]
bars = ax5.bar(functions, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax5.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Balanced (c1=c2)')
ax5.set_ylabel('c1 / c2 Ratio', fontsize=11)
ax5.set_title('Cognitive/Social Balance', fontsize=12, fontweight='bold')
ax5.set_ylim([0, 2.0])
ax5.grid(True, alpha=0.3, axis='y')
ax5.legend(fontsize=8)
for bar, val, f in zip(bars, ratios, functions):
    height = bar.get_height()
    if val < 1:
        label = 'Social-heavy'
        offset = -0.15
        va = 'top'
    elif val > 1:
        label = 'Cognitive-heavy'
        offset = 0.05
        va = 'bottom'
    else:
        label = 'Balanced'
        offset = 0.05
        va = 'bottom'
    ax5.text(bar.get_x() + bar.get_width()/2., height + offset,
             f'{val:.2f}\n{label}', ha='center', va=va, fontsize=9, fontweight='bold')

# Subplot 6: Difficulty ranking
ax6 = plt.subplot(2, 3, 6)
difficulty_scores = [100 - sr for sr in success_rates]  # Invert success rate
bars = ax6.barh(functions, difficulty_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_xlabel('Difficulty Score\n(100 - Success Rate)', fontsize=11)
ax6.set_title('Function Difficulty for PSO', fontsize=12, fontweight='bold')
ax6.set_xlim([0, 100])
ax6.grid(True, alpha=0.3, axis='x')
for bar, score, func in zip(bars, difficulty_scores, functions):
    width = bar.get_width()
    if score == 0:
        label = 'EASY'
        color = 'green'
    elif score < 30:
        label = 'EASY-MODERATE'
        color = 'blue'
    else:
        label = 'MODERATE-HIGH'
        color = 'red'
    ax6.text(width + 2, bar.get_y() + bar.get_height()/2.,
             f'{score} ({label})', ha='left', va='center', fontsize=10, fontweight='bold', color=color)

plt.suptitle('PSO Parameter Comparison Across Benchmark Functions', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('parameter_study_comparison_detailed.png', dpi=150, bbox_inches='tight')
print("Saved: parameter_study_comparison_detailed.png")
plt.close()

# =============================================================================
# Figure 3: Strategy Recommendations
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Create strategy table
strategies = [
    ['Multimodal\n(Multiple Optima)', 'Social-Heavy', 'c1 < c2', 'Moderate w\n(0.6)', 
     'Slow convergence\n(3+ iters)', 'Convergence\nto consensus'],
    ['Michaelewicz\n(Smooth Unimodal)', 'Cognitive-Heavy', 'c1 > c2', 'Moderate w\n(0.5)', 
     'Fast convergence\n(1 iter)', 'Individual\nexploration'],
    ['Easom\n(Narrow Basin)', 'Balanced', 'c1 = c2', 'High w\n(0.9)', 
     'Very fast\n(1 iter)', 'Momentum-driven\nconvergence']
]

col_labels = ['Function\nType', 'Strategy', 'Balance', 'Inertia', 'Speed', 'Mechanism']
row_labels = ['', '', '']

# Create table
table = ax.table(cellText=strategies, colLabels=col_labels, rowLabels=row_labels,
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 3)

# Color code rows
row_colors = ['#FFE5E5', '#E5F5F5', '#E5E5FF']
for i, color in enumerate(row_colors):
    for j in range(len(col_labels)):
        cell = table[(i+1, j)]
        cell.set_facecolor(color)
        cell.set_text_props(weight='bold')

# Color header
for j in range(len(col_labels)):
    cell = table[(0, j)]
    cell.set_facecolor('#CCCCCC')
    cell.set_text_props(weight='bold', size=11)

ax.axis('off')
ax.set_title('PSO Strategy Recommendations by Function Type', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('parameter_study_strategies.png', dpi=150, bbox_inches='tight')
print("Saved: parameter_study_strategies.png")
plt.close()

# =============================================================================
# Summary Statistics
# =============================================================================
print()
print("=" * 70)
print("PARAMETER STUDY COMPARISON SUMMARY")
print("=" * 70)
print()

for i, func in enumerate(functions):
    print(f"{func.upper()}")
    print("-" * 70)
    print(f"  Success Rate: {success_rates[i]}% ({successful_configs[i]}/{total_configs[i]} configs)")
    print(f"  Fastest to 99.7%: {fastest_iters[i]} iterations")
    print(f"  Optimal Parameters: w={w_fastest[i]:.1f}, c1={c1_fastest[i]:.1f}, c2={c2_fastest[i]:.1f}")
    print(f"  c1/c2 Ratio: {ratios[i]:.2f} ({['Social-heavy', 'Balanced', 'Cognitive-heavy'][np.argmax([ratios[i] < 0.8, abs(ratios[i] - 1.0) < 0.2, ratios[i] > 1.2])]})")
    print(f"  Difficulty: {difficulty_scores[i]}/100")
    print()

print("=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print()
print("1. DIFFICULTY RANKING (easiest to hardest):")
print(f"   Easom ({success_rates[2]}%) > Michaelewicz ({success_rates[1]}%) > Multimodal ({success_rates[0]}%)")
print()
print("2. CONVERGENCE SPEED:")
print(f"   Fastest: Michaelewicz & Easom (1 iteration)")
print(f"   Slowest: Multimodal (3 iterations)")
print()
print("3. STRATEGY PATTERNS:")
print("   - Multiple optima → Social-heavy (high c2)")
print("   - Smooth unimodal → Cognitive-heavy (high c1)")
print("   - Narrow basin → Balanced + High inertia")
print()
print("4. UNIVERSAL RECOMMENDATION for unknown functions:")
print("   w=0.6-0.7, c1=1.2, c2=1.3 (total ≈ 2.5)")
print()
print("=" * 70)
print("VISUALIZATIONS GENERATED")
print("=" * 70)
print("  - parameter_study_comparison_summary.png")
print("  - parameter_study_comparison_detailed.png")
print("  - parameter_study_strategies.png")
