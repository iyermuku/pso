"""Simple sanity test for multimodal PSO implementation."""

from multimodal_pso import pso_maximize, multimodal_15_12
import numpy as np


def test_2d_search():
    n = 2
    bounds = [(-2 * np.pi, 2 * np.pi)] * n
    best_pos, best_val, _ = pso_maximize(multimodal_15_12, bounds, n_dim=n, swarm_size=50, iters=200, seed=123)
    # Make sure the returned value is finite and within expected bounds
    assert np.isfinite(best_val)
    assert best_val > 0  # should find a positive peak
    assert len(best_pos) == n


if __name__ == "__main__":
    print("Running manual PSO for 2D function")
    # run search and show results explicitly, tracking history for plotting
    import matplotlib.pyplot as plt

    n = 2
    bounds = [(-2 * np.pi, 2 * np.pi)] * n
    best_pos, best_val, history = pso_maximize(
        multimodal_15_12, bounds, n_dim=n, swarm_size=50, iters=200, seed=123,
        track_history=True,
    )
    print("best position:", best_pos)
    print("best value:", best_val)

    # scatter initial and final particle locations
    Xhist = history["X_history"]  # shape (iters+1, swarm, 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(Xhist[0, :, 0], Xhist[0, :, 1], c="blue", label="start")
    axes[0].set_title("Initial particle positions")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[1].scatter(Xhist[-1, :, 0], Xhist[-1, :, 1], c="red", label="end")
    axes[1].set_title("Final particle positions")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    plt.tight_layout()
    fig.savefig("particle_positions.png")
    plt.show()

    # plot gbest evolution
    plt.figure()
    plt.plot(history["gbest_history"], marker=".")
    plt.title("Global best value per iteration")
    plt.xlabel("iteration")
    plt.ylabel("best objective")
    plt.grid(True)
    plt.savefig("gbest_evolution.png")
    plt.show()

    # create animation of particle movement with current global best highlighted
    from matplotlib.animation import FuncAnimation

    gbest_X_hist = history["gbest_X_history"]
    iters_plus = Xhist.shape[0]
    fig2, ax2 = plt.subplots()
    scat = ax2.scatter(Xhist[0, :, 0], Xhist[0, :, 1], c="blue")
    best_pt, = ax2.plot(gbest_X_hist[0, 0], gbest_X_hist[0, 1], "ro", markersize=8)
    ax2.set_xlim(bounds[0][0], bounds[0][1])
    ax2.set_ylim(bounds[1][0], bounds[1][1])
    ax2.set_title("Particle trajectories (global best in red)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    anim_interval = 100  # ms per frame; adjust to speed up/slow down

    def _update(frame):
        scat.set_offsets(Xhist[frame])
        # set_data expects sequences
        best_pt.set_data([gbest_X_hist[frame, 0]], [gbest_X_hist[frame, 1]])
        return scat, best_pt

    anim = FuncAnimation(fig2, _update, frames=range(iters_plus), interval=anim_interval, blit=True)
    # try to write a gif; fallback if ffmpeg not available
    try:
        anim.save("trajectory.mp4", fps=1000/anim_interval)
    except Exception:
        anim.save("trajectory.gif", writer="pillow", fps=1000/anim_interval)
    plt.show()

    test_2d_search()
    print("Test completed")
