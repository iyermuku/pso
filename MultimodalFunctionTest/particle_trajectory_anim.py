import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
pbest_val = float('inf')
pbest_pos = None
for i in range(Xhist.shape[0]):
    pos = Xhist[i, stuck_idx]
    val = multimodal_15_12(pos)
    if val > pbest_val:
        pbest_hist.append(pbest_pos)
    else:
        pbest_val = val
        pbest_pos = pos.copy()
        pbest_hist.append(pbest_pos)
pbest_hist = np.array(pbest_hist)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_title("Trajectory of Stuck Particle vs Global Best")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

particle_dot, = ax.plot([], [], "ro", label="Particle")
gbest_dot, = ax.plot([], [], "bo", label="Global Best")
pbest_dot, = ax.plot([], [], "go", label="Personal Best")
traj_line, = ax.plot([], [], "r--", alpha=0.5)

ax.legend()

anim_interval = 120

# Prepare trajectory for stuck particle
traj = Xhist[:, stuck_idx, :]

def update(frame):
    particle_dot.set_data([traj[frame, 0]], [traj[frame, 1]])
    gbest_dot.set_data([gbest_X_hist[frame, 0]], [gbest_X_hist[frame, 1]])
    pbest_dot.set_data([pbest_hist[frame, 0]], [pbest_hist[frame, 1]])
    traj_line.set_data(traj[:frame+1, 0], traj[:frame+1, 1])
    ax.set_title(f"Iteration {frame+1}")
    return particle_dot, gbest_dot, pbest_dot, traj_line

anim = FuncAnimation(fig, update, frames=Xhist.shape[0], interval=anim_interval, blit=True)
anim.save("stuck_particle_vs_gbest.gif", writer="pillow")
plt.close(fig)
print("Animation saved as stuck_particle_vs_gbest.gif")
