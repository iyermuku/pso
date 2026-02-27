"""Test and visualization for Michaelewicz PSO optimization."""

from michaelewicz_pso import pso_minimize, michaelewicz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


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
    gbest_X_hist = history["gbest_X_history"]
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

    # 3D surface view with per-iteration controls
    # Lets the user step through iterations and inspect particle locations.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig3 = plt.figure(figsize=(10, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2)

    grid_n = 80
    xs = np.linspace(bounds[0][0], bounds[0][1], grid_n)
    ys = np.linspace(bounds[1][0], bounds[1][1], grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = np.array(
        [
            michaelewicz(np.array([x, y]), m=m)
            for x, y in zip(Xg.ravel(), Yg.ravel())
        ]
    ).reshape(Xg.shape)

    ax3.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.65, linewidth=0, antialiased=False)
    z_pad = 0.05 * (float(np.max(Zg)) - float(np.min(Zg)))
    ax3.set_zlim(float(np.min(Zg)) - z_pad, float(np.max(Zg)) + z_pad)
    ax3.set_xlim(bounds[0])
    ax3.set_ylim(bounds[1])
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("f(x)")

    iters_plus = Xhist.shape[0]
    init_pts = Xhist[0]
    init_z = [michaelewicz(pt, m=m) for pt in init_pts]
    scat3 = ax3.scatter(init_pts[:, 0], init_pts[:, 1], init_z, c="red", s=24, alpha=0.9)
    best_z0 = michaelewicz(gbest_X_hist[0], m=m)
    best3, = ax3.plot(
        [gbest_X_hist[0, 0]],
        [gbest_X_hist[0, 1]],
        [best_z0],
        marker="*",
        markersize=14,
        color="gold",
        markeredgecolor="black",
    )

    def set_frame(frame):
        pts = Xhist[frame]
        zs = [michaelewicz(pt, m=m) for pt in pts]
        scat3._offsets3d = (pts[:, 0], pts[:, 1], zs)
        gbest = gbest_X_hist[frame]
        gbest_z = michaelewicz(gbest, m=m)
        best3.set_data_3d([gbest[0]], [gbest[1]], [gbest_z])
        ax3.set_title(
            f"Michaelewicz surface - iteration {frame}/{iters_plus - 1} "
            f"(best={history['gbest_history'][frame]:.6f})"
        )
        fig3.canvas.draw_idle()

    set_frame(0)

    slider_ax = fig3.add_axes([0.15, 0.08, 0.7, 0.03])
    iter_slider = Slider(
        slider_ax,
        "Iter",
        0,
        iters_plus - 1,
        valinit=0,
        valstep=1,
    )

    prev_ax = fig3.add_axes([0.03, 0.06, 0.08, 0.06])
    next_ax = fig3.add_axes([0.88, 0.06, 0.08, 0.06])
    prev_btn = Button(prev_ax, "Prev")
    next_btn = Button(next_ax, "Next")

    def _on_slider(val):
        set_frame(int(val))

    def _on_prev(_event):
        new_val = max(0, int(iter_slider.val) - 1)
        iter_slider.set_val(new_val)

    def _on_next(_event):
        new_val = min(iters_plus - 1, int(iter_slider.val) + 1)
        iter_slider.set_val(new_val)

    def _on_key(event):
        if event.key == "left":
            _on_prev(None)
        elif event.key == "right":
            _on_next(None)

    iter_slider.on_changed(_on_slider)
    prev_btn.on_clicked(_on_prev)
    next_btn.on_clicked(_on_next)
    fig3.canvas.mpl_connect("key_press_event", _on_key)

    fig3.savefig("surface_iteration.png", dpi=120)
    print("Saved: surface_iteration.png")
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
