"""
Test and visualization script for Easom's function PSO optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from easom_pso import pso_minimize, easom_vec

# Set random seed for reproducibility
SEED = 42

# Run PSO optimization
print("=" * 60)
print("PSO OPTIMIZATION OF EASOM'S FUNCTION")
print("=" * 60)
print()

bounds = [(0, 2*np.pi), (0, 200*np.pi)]
n_dim = 2
swarm_size = 50
iters = 300
inertia = 0.7
c1 = 1.5
c2 = 1.5

print(f"Configuration:")
print(f"  Swarm size: {swarm_size}")
print(f"  Iterations: {iters}")
print(f"  Inertia: {inertia}")
print(f"  c1: {c1}")
print(f"  c2: {c2}")
print(f"  Seed: {SEED}")
print()

best_pos, best_val, history = pso_minimize(
    easom_vec, 
    bounds, 
    n_dim, 
    swarm_size=swarm_size, 
    iters=iters,
    inertia=inertia,
    c1=c1,
    c2=c2,
    seed=SEED,
    track_history=True
)

print(f"Optimization Results:")
print(f"  Best position: x = {best_pos[0]:.6f} ({best_pos[0]/np.pi:.4f}*pi)")
print(f"                 y = {best_pos[1]:.6f} ({best_pos[1]/np.pi:.4f}*pi)")
print(f"  Best value: {best_val:.10f}")
print(f"  Expected minimum: -0.9995 at (pi, 100*pi)")
print(f"  Achieved percentage: {(best_val / -0.9995) * 100:.4f}%")
print()

# Extract history
X_history = history['X_history']
gbest_history = history['gbest_history']
gbest_X_history = history['gbest_X_history']

# Initial and final particle positions
X_initial = X_history[0]
X_final = X_history[-1]

print("=" * 60)
print("INITIAL PARTICLE SPREAD")
print("=" * 60)
print(f"X range: [{X_initial[:, 0].min():.4f}, {X_initial[:, 0].max():.4f}]")
print(f"Y range: [{X_initial[:, 1].min():.4f}, {X_initial[:, 1].max():.4f}]")
print(f"Mean position: x = {X_initial[:, 0].mean():.4f}, y = {X_initial[:, 1].mean():.4f}")
print(f"Std deviation: x = {X_initial[:, 0].std():.4f}, y = {X_initial[:, 1].std():.4f}")
print()

print("=" * 60)
print("FINAL PARTICLE SPREAD")
print("=" * 60)
print(f"X range: [{X_final[:, 0].min():.4f}, {X_final[:, 0].max():.4f}]")
print(f"Y range: [{X_final[:, 1].min():.4f}, {X_final[:, 1].max():.4f}]")
print(f"Mean position: x = {X_final[:, 0].mean():.4f}, y = {X_final[:, 1].mean():.4f}")
print(f"Std deviation: x = {X_final[:, 0].std():.4f}, y = {X_final[:, 1].std():.4f}")
print()

# Evaluate final values
final_vals = np.array([easom_vec(x) for x in X_final])
print(f"Final function values:")
print(f"  Best: {final_vals.min():.10f}")
print(f"  Worst: {final_vals.max():.10f}")
print(f"  Mean: {final_vals.mean():.10f}")
print(f"  Std: {final_vals.std():.10f}")
print()

# Check for overlapping regions
print("=" * 60)
print("CLUSTERING ANALYSIS")
print("=" * 60)

# Use scaled coordinates for clustering (normalize to [0, 1])
X_final_scaled = np.zeros_like(X_final)
X_final_scaled[:, 0] = (X_final[:, 0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
X_final_scaled[:, 1] = (X_final[:, 1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0])

# Find unique positions (tolerance-based clustering)
tol = 0.01  # 1% of domain in scaled space
unique_positions = []
clusters = []

for i, pos in enumerate(X_final_scaled):
    found = False
    for j, unique_pos in enumerate(unique_positions):
        dist = np.linalg.norm(pos - unique_pos)
        if dist < tol:
            clusters[j].append(i)
            found = True
            break
    if not found:
        unique_positions.append(pos)
        clusters.append([i])

print(f"Found {len(unique_positions)} distinct cluster(s) with tolerance = {tol}")
print()

# Report each cluster
for cluster_id, particle_indices in enumerate(clusters):
    print(f"Cluster {cluster_id + 1}: {len(particle_indices)} particles")
    cluster_positions = X_final[particle_indices]
    cluster_values = final_vals[particle_indices]
    
    mean_x = cluster_positions[:, 0].mean()
    mean_y = cluster_positions[:, 1].mean()
    mean_val = cluster_values.mean()
    
    print(f"  Mean position: x = {mean_x:.6f} ({mean_x/np.pi:.4f}*pi)")
    print(f"                 y = {mean_y:.6f} ({mean_y/np.pi:.4f}*pi)")
    print(f"  Mean value: {mean_val:.10f}")
    print(f"  Value range: [{cluster_values.min():.10f}, {cluster_values.max():.10f}]")
    if len(particle_indices) <= 10:
        print(f"  Particle indices: {particle_indices}")
    else:
        print(f"  Particle indices: {particle_indices[:10]}... (showing first 10)")
    print()

# =============================================================================
# VISUALIZATION 1: Initial and Final Particle Spread
# =============================================================================
print("Generating initial and final particle spread plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Initial spread
ax = axes[0]
ax.scatter(X_initial[:, 0]/np.pi, X_initial[:, 1]/np.pi, c='blue', alpha=0.6, s=50)
ax.set_xlabel('x / pi')
ax.set_ylabel('y / pi')
ax.set_title('Initial Particle Spread (LHS)')
ax.grid(True, alpha=0.3)
ax.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Global min x')
ax.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Global min y')
ax.legend()

# Final spread
ax = axes[1]
scatter = ax.scatter(X_final[:, 0]/np.pi, X_final[:, 1]/np.pi, 
                     c=final_vals, cmap='viridis', alpha=0.8, s=50)
ax.scatter(best_pos[0]/np.pi, best_pos[1]/np.pi, c='red', marker='*', s=300, 
           edgecolors='black', linewidths=1.5, label=f'Best: {best_val:.6f}')
ax.set_xlabel('x / pi')
ax.set_ylabel('y / pi')
ax.set_title('Final Particle Spread')
ax.grid(True, alpha=0.3)
ax.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.legend()
plt.colorbar(scatter, ax=ax, label='Function Value')

plt.tight_layout()
plt.savefig('easom_particle_spread.png', dpi=150, bbox_inches='tight')
print("Saved: easom_particle_spread.png")
plt.close()

# =============================================================================
# VISUALIZATION 2: 2D Animation of Particle Movement
# =============================================================================
print("Generating 2D animation...")

fig, ax = plt.subplots(figsize=(10, 8))

# Create contour background
x_grid = np.linspace(bounds[0][0], bounds[0][1], 100)
y_grid = np.linspace(bounds[1][0], bounds[1][1], 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
Z_mesh = easom_vec(np.stack([X_mesh.ravel(), Y_mesh.ravel()], axis=1)).reshape(X_mesh.shape)

contour = ax.contourf(X_mesh/np.pi, Y_mesh/np.pi, Z_mesh, levels=20, cmap='viridis', alpha=0.3)
ax.contour(X_mesh/np.pi, Y_mesh/np.pi, Z_mesh, levels=10, colors='black', alpha=0.2, linewidths=0.5)

# Initialize scatter plots
particles = ax.scatter([], [], c='blue', alpha=0.6, s=30, label='Particles')
gbest_point = ax.scatter([], [], c='red', marker='*', s=200, 
                         edgecolors='black', linewidths=1, label='Global Best')
gbest_trail, = ax.plot([], [], 'r--', linewidth=1, alpha=0.5, label='Best Trail')

ax.set_xlabel('x / pi')
ax.set_ylabel('y / pi')
ax.set_title('PSO Optimization - Iteration 0')
ax.grid(True, alpha=0.3)
ax.axvline(1, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(100, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.legend()
plt.colorbar(contour, ax=ax, label='Function Value')

def animate_2d(frame):
    X = X_history[frame]
    gbest_X = gbest_X_history[frame]
    
    particles.set_offsets(np.column_stack([X[:, 0]/np.pi, X[:, 1]/np.pi]))
    gbest_point.set_offsets([[gbest_X[0]/np.pi, gbest_X[1]/np.pi]])
    
    # Update trail (show last 20 positions)
    trail_start = max(0, frame - 20)
    trail_X = gbest_X_history[trail_start:frame+1]
    gbest_trail.set_data(trail_X[:, 0]/np.pi, trail_X[:, 1]/np.pi)
    
    ax.set_title(f'PSO Optimization - Iteration {frame} | Best: {gbest_history[frame]:.6f}')
    return particles, gbest_point, gbest_trail

# Create animation - sample every 5 frames for speed
frame_skip = 5
frames = range(0, len(X_history), frame_skip)
anim = FuncAnimation(fig, animate_2d, frames=frames, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim.save('easom_2d_animation.gif', writer=writer)
print("Saved: easom_2d_animation.gif")
plt.close()

# =============================================================================
# VISUALIZATION 3: 3D Animation
# =============================================================================
print("Generating 3D animation (this may take a while)...")

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create 3D surface (use coarser grid for performance)
x_grid_3d = np.linspace(bounds[0][0], bounds[0][1], 50)
y_grid_3d = np.linspace(bounds[1][0], bounds[1][1], 50)
X_mesh_3d, Y_mesh_3d = np.meshgrid(x_grid_3d, y_grid_3d)
Z_mesh_3d = easom_vec(np.stack([X_mesh_3d.ravel(), Y_mesh_3d.ravel()], axis=1)).reshape(X_mesh_3d.shape)

# Plot surface
surf = ax.plot_surface(X_mesh_3d/np.pi, Y_mesh_3d/np.pi, Z_mesh_3d, 
                       cmap='viridis', alpha=0.3, edgecolor='none')

# Initialize scatter plots
particles_3d = ax.scatter([], [], [], c='blue', alpha=0.7, s=30, label='Particles')
gbest_point_3d = ax.scatter([], [], [], c='red', marker='*', s=200,
                            edgecolors='black', linewidths=1, label='Global Best')

ax.set_xlabel('x / pi')
ax.set_ylabel('y / pi')
ax.set_zlabel('Function Value')
ax.set_title('PSO Optimization 3D - Iteration 0')
ax.legend()

# Set view angle
ax.view_init(elev=20, azim=45)

def animate_3d(frame):
    X = X_history[frame]
    gbest_X = gbest_X_history[frame]
    
    # Evaluate function values
    Z = np.array([easom_vec(x) for x in X])
    Z_gbest = easom_vec(gbest_X)
    
    particles_3d._offsets3d = (X[:, 0]/np.pi, X[:, 1]/np.pi, Z)
    gbest_point_3d._offsets3d = ([gbest_X[0]/np.pi], [gbest_X[1]/np.pi], [Z_gbest])
    
    ax.set_title(f'PSO Optimization 3D - Iteration {frame} | Best: {gbest_history[frame]:.6f}')
    
    return particles_3d, gbest_point_3d

# Create animation - sample every 10 frames for speed
frame_skip_3d = 10
frames_3d = range(0, len(X_history), frame_skip_3d)
anim_3d = FuncAnimation(fig, animate_3d, frames=frames_3d, interval=100, blit=False)

writer_3d = PillowWriter(fps=10)
anim_3d.save('easom_3d_animation.gif', writer=writer_3d)
print("Saved: easom_3d_animation.gif")
plt.close()

# =============================================================================
# VISUALIZATION 4: Convergence History
# =============================================================================
print("Generating convergence plot...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(gbest_history, 'b-', linewidth=2)
ax.axhline(-0.9995, color='red', linestyle='--', linewidth=1, label='Expected minimum')
ax.set_xlabel('Iteration')
ax.set_ylabel('Global Best Value')
ax.set_title('PSO Convergence History')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('easom_convergence.png', dpi=150, bbox_inches='tight')
print("Saved: easom_convergence.png")
plt.close()

print()
print("=" * 60)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 60)
print("Generated files:")
print("  - easom_particle_spread.png")
print("  - easom_2d_animation.gif")
print("  - easom_3d_animation.gif")
print("  - easom_convergence.png")
