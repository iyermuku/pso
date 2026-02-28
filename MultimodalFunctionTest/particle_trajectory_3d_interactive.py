import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from multimodal_pso import pso_maximize, multimodal_15_12

n = 2
bounds = [(-2 * np.pi, 2 * np.pi)] * n
best_pos, best_val, history = pso_maximize(multimodal_15_12, bounds, n_dim=n, swarm_size=50, iters=200, seed=123, track_history=True)
Xhist = history["X_history"]
gbest_X_hist = history["gbest_X_history"]
final = Xhist[-1]
values = np.array([multimodal_15_12(pt) for pt in final])
stuck_idx = np.argmin(values)  # index of stuck particle

# Get personal best trajectory for stuck particle
pbest_hist = []
pbest_val = -float('inf')  # for maximization
pbest_pos = Xhist[0, stuck_idx].copy()
for i in range(Xhist.shape[0]):
    pos = Xhist[i, stuck_idx]
    val = multimodal_15_12(pos)
    if val > pbest_val:
        pbest_val = val
        pbest_pos = pos.copy()
    pbest_hist.append(pbest_pos.copy())
pbest_hist = np.array(pbest_hist)

# Create surface grid
grid_n = 50
xs = np.linspace(bounds[0][0], bounds[0][1], grid_n)
ys = np.linspace(bounds[1][0], bounds[1][1], grid_n)
Xg, Yg = np.meshgrid(xs, ys)
Zg = np.array([
    multimodal_15_12(np.array([x, y]))
    for x, y in zip(Xg.ravel(), Yg.ravel())
]).reshape(Xg.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.6, linewidth=0, antialiased=False)

# Initialize markers
traj = Xhist[:, stuck_idx, :]
particle_z = [multimodal_15_12(traj[0])]
gbest_z = [multimodal_15_12(gbest_X_hist[0])]
pbest_z = [multimodal_15_12(pbest_hist[0])]

particle_dot = ax.scatter([], [], [], c='red', s=100, marker='o', label='Stuck Particle', edgecolors='black', linewidths=2)
gbest_dot = ax.scatter([], [], [], c='blue', s=100, marker='*', label='Global Best', edgecolors='black', linewidths=2)
pbest_dot = ax.scatter([], [], [], c='green', s=100, marker='^', label='Personal Best', edgecolors='black', linewidths=2)
traj_line, = ax.plot([], [], [], 'r--', alpha=0.7, linewidth=2, label='Trajectory')

ax.set_xlabel('x1', fontsize=10)
ax.set_ylabel('x2', fontsize=10)
ax.set_zlabel('f(x)', fontsize=10)
ax.set_title('3D Trajectory: Stuck Particle vs Global Best (Interactive)', fontsize=12)
ax.legend(loc='upper left')

anim_interval = 150
paused = False

def update(frame):
    # Update particle position
    particle_z = multimodal_15_12(traj[frame])
    particle_dot._offsets3d = ([traj[frame, 0]], [traj[frame, 1]], [particle_z])
    
    # Update global best position
    gbest_z = multimodal_15_12(gbest_X_hist[frame])
    gbest_dot._offsets3d = ([gbest_X_hist[frame, 0]], [gbest_X_hist[frame, 1]], [gbest_z])
    
    # Update personal best position
    pbest_z = multimodal_15_12(pbest_hist[frame])
    pbest_dot._offsets3d = ([pbest_hist[frame, 0]], [pbest_hist[frame, 1]], [pbest_z])
    
    # Update trajectory line
    traj_z = [multimodal_15_12(pt) for pt in traj[:frame+1]]
    traj_line.set_data(traj[:frame+1, 0], traj[:frame+1, 1])
    traj_line.set_3d_properties(traj_z)
    
    ax.set_title(f'3D Trajectory: Stuck Particle vs Global Best - Iteration {frame+1}/{Xhist.shape[0]}', fontsize=12)
    return particle_dot, gbest_dot, pbest_dot, traj_line

def on_click(event):
    global paused
    if event.key == ' ':
        if paused:
            anim.resume()
            paused = False
            print("Animation resumed")
        else:
            anim.pause()
            paused = True
            print("Animation paused (press SPACE to resume)")

fig.canvas.mpl_connect('key_press_event', on_click)

print("Interactive 3D Animation:")
print("- Press SPACE to pause/resume")
print("- Use mouse to rotate the 3D view")
print("- Close window when done")

anim = FuncAnimation(fig, update, frames=Xhist.shape[0], interval=anim_interval, blit=False, repeat=True)
plt.show()
