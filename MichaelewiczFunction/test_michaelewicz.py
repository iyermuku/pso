"""Test and visualization for Michaelewicz PSO optimization."""

from michaelewicz_pso import pso_minimize, michaelewicz
import numpy as np
import matplotlib.pyplot as plt


def run_michaelewicz_optimization():
    """Run PSO on 2D Michaelewicz function with visualizations."""
    print("Running PSO optimization on 2D Michaelewicz function (m=10)")
    
    n = 2
    m = 10
    bounds = [(0.0, np.pi)] * n
    
    # Run PSO with history tracking
    best_pos, best_val, history = pso_minimize(
        lambda x: michaelewicz(x, m=m),
        bounds,
        n_dim=n,
        swarm_size=50,
        iters=200,
        seed=123,
        track_history=True,
    )
    
    print(f"Global minimum found:")
    print(f"  Position: x1={best_pos[0]:.6f}, x2={best_pos[1]:.6f}")
    print(f"  Value: {best_val:.6f}")
    
    # scatter initial and final particle locations
    Xhist = history["X_history"]  # shape (iters+1, swarm, 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(Xhist[0, :, 0], Xhist[0, :, 1], c="blue", alpha=0.6)
    axes[0].set_title("Initial particle positions")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_xlim(bounds[0])
    axes[0].set_ylim(bounds[1])
    
    axes[1].scatter(Xhist[-1, :, 0], Xhist[-1, :, 1], c="red", alpha=0.6)
    axes[1].scatter(best_pos[0], best_pos[1], c="gold", s=200, marker="*", 
                    edgecolors="black", linewidths=2, label="Global minimum")
    axes[1].set_title("Final particle positions")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].set_xlim(bounds[0])
    axes[1].set_ylim(bounds[1])
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig("particle_positions.png", dpi=100)
    print("Saved: particle_positions.png")
    plt.show()
    
    # plot gbest evolution
    plt.figure()
    plt.plot(history["gbest_history"], marker=".", linewidth=1)
    plt.title("Global best value per iteration (Michaelewicz m=10)")
    plt.xlabel("iteration")
    plt.ylabel("best objective value")
    plt.grid(True, alpha=0.3)
    plt.savefig("gbest_evolution.png", dpi=100)
    print("Saved: gbest_evolution.png")
    plt.show()
    
    # create animation of particle movement with current global best highlighted
    from matplotlib.animation import FuncAnimation
    
    gbest_X_hist = history["gbest_X_history"]
    iters_plus = Xhist.shape[0]
    fig2, ax2 = plt.subplots()
    scat = ax2.scatter(Xhist[0, :, 0], Xhist[0, :, 1], c="blue", alpha=0.6)
    best_pt, = ax2.plot(gbest_X_hist[0, 0], gbest_X_hist[0, 1], "ro", markersize=8)
    ax2.set_xlim(bounds[0])
    ax2.set_ylim(bounds[1])
    ax2.set_title("Particle trajectories (global minimum in red)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    
    anim_interval = 100  # ms per frame; adjust to speed up/slow down
    
    def _update(frame):
        scat.set_offsets(Xhist[frame])
        # set_data expects sequences
        best_pt.set_data([gbest_X_hist[frame, 0]], [gbest_X_hist[frame, 1]])
        return scat, best_pt
    
    anim = FuncAnimation(fig2, _update, frames=range(iters_plus), 
                         interval=anim_interval, blit=True)
    # try to write a gif; fallback if ffmpeg not available
    try:
        anim.save("trajectory.mp4", fps=1000/anim_interval)
        print("Saved: trajectory.mp4")
    except Exception:
        anim.save("trajectory.gif", writer="pillow", fps=1000/anim_interval)
        print("Saved: trajectory.gif")
    
    plt.show()


if __name__ == "__main__":
    run_michaelewicz_optimization()
