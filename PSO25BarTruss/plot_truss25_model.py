"""Create a labeled 3D plot of the 25-bar truss model.

Outputs:
  PSO25BarTruss/truss25_model_plot.png

The plot includes:
- Node markers with node IDs
- Element lines with element IDs
- Global coordinate system arrows (x, y, z)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import truss25 as T


def _set_equal_axes_3d(ax, points: np.ndarray) -> None:
    """Set equal scaling for x, y, z axes for a true geometric view."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 1e-9:
        span = 1.0
    half = 0.6 * span

    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def make_plot(save_path: Path) -> None:
    nodes = T.node_coordinates()
    elems = T.element_connectivity()

    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Draw elements and element labels.
    for e_id, (n1, n2) in enumerate(elems, start=1):
        p1 = nodes[n1 - 1]
        p2 = nodes[n2 - 1]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color="black",
            linewidth=1.2,
            alpha=0.95,
        )
        mid = 0.5 * (p1 + p2)
        ax.text(mid[0], mid[1], mid[2], str(e_id), color="crimson", fontsize=8)

    # Draw nodes and node labels.
    free_nodes = [i for i in range(1, T.N_NODES + 1) if i not in T.FIXED_NODES]
    fixed_nodes = list(T.FIXED_NODES)

    p_free = nodes[np.array(free_nodes) - 1]
    p_fix = nodes[np.array(fixed_nodes) - 1]

    ax.scatter(p_free[:, 0], p_free[:, 1], p_free[:, 2], c="forestgreen", s=36, label="Free nodes")
    ax.scatter(p_fix[:, 0], p_fix[:, 1], p_fix[:, 2], c="royalblue", marker="^", s=52, label="Pinned nodes")

    for n_id, p in enumerate(nodes, start=1):
        ax.text(p[0], p[1], p[2] + 6.0, str(n_id), color="darkgreen", fontsize=9)

    # Draw global coordinate system arrows near one base corner.
    origin = np.array([-130.0, -130.0, 0.0])
    L = 65.0
    ax.quiver(origin[0], origin[1], origin[2], L, 0.0, 0.0, color="tab:red", arrow_length_ratio=0.12, linewidth=2.0)
    ax.quiver(origin[0], origin[1], origin[2], 0.0, L, 0.0, color="tab:orange", arrow_length_ratio=0.12, linewidth=2.0)
    ax.quiver(origin[0], origin[1], origin[2], 0.0, 0.0, L, color="tab:purple", arrow_length_ratio=0.12, linewidth=2.0)
    ax.text(origin[0] + L + 6, origin[1], origin[2], "x", color="tab:red", fontsize=11)
    ax.text(origin[0], origin[1] + L + 6, origin[2], "y", color="tab:orange", fontsize=11)
    ax.text(origin[0], origin[1], origin[2] + L + 8, "z", color="tab:purple", fontsize=11)

    # Styling.
    ax.set_title("25-Bar Space Truss Model (Nodes, Elements, Labels, Coordinate System)")
    ax.set_xlabel("X (in)")
    ax.set_ylabel("Y (in)")
    ax.set_zlabel("Z (in)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=18, azim=-58)

    _set_equal_axes_3d(ax, nodes)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "truss25_model_plot.png"
    make_plot(out)
    print(f"Saved: {out}")
