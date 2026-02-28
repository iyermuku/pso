import numpy as np
from multimodal_pso import pso_maximize, multimodal_15_12

n = 2
bounds = [(-2 * np.pi, 2 * np.pi)] * n
best_pos, best_val, history = pso_maximize(multimodal_15_12, bounds, n_dim=n, swarm_size=50, iters=200, seed=123, track_history=True)
Xhist = history["X_history"]
final = Xhist[-1]
# Find the index of the particle stuck at the secondary peak (value ~4.33)
values = np.array([multimodal_15_12(pt) for pt in final])
stuck_idx = np.argmin(values)  # the lowest value is the secondary peak
print(f"Particle index stuck at secondary peak: {stuck_idx}")
print(f"Final position: {final[stuck_idx]}, value: {values[stuck_idx]}")
print("Trajectory over all iterations:")
for i in range(Xhist.shape[0]):
    pos = Xhist[i, stuck_idx]
    val = multimodal_15_12(pos)
    print(f"Iter {i:3d}: pos = {pos}, value = {val}")
