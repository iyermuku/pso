"""Create a labeled 3D plot of the 72-bar truss model.

Outputs:
  PSO72BarTruss/v2/truss72_model_plot.png

The plot shows:
- nodes and element connectivity
- pinned boundary conditions at the support nodes
- applied load arrows for both benchmark load cases
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import truss72 as T


def _set_equal_axes_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span <= 1e-9:
        span = 1.0
    half = 0.62 * span
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def _draw_axes_triad(ax, origin: np.ndarray, length: float) -> None:
    ax.quiver(origin[0], origin[1], origin[2], length, 0.0, 0.0, color="tab:red", arrow_length_ratio=0.12, linewidth=2.0)
    ax.quiver(origin[0], origin[1], origin[2], 0.0, length, 0.0, color="tab:orange", arrow_length_ratio=0.12, linewidth=2.0)
    ax.quiver(origin[0], origin[1], origin[2], 0.0, 0.0, length, color="tab:purple", arrow_length_ratio=0.12, linewidth=2.0)
    ax.text(origin[0] + length + 6, origin[1], origin[2], "x", color="tab:red", fontsize=10)
    ax.text(origin[0], origin[1] + length + 6, origin[2], "y", color="tab:orange", fontsize=10)
    ax.text(origin[0], origin[1], origin[2] + length + 8, "z", color="tab:purple", fontsize=10)


def _load_cases() -> list[tuple[str, np.ndarray]]:
    cases = []
    for idx, load_case in enumerate(T.LOAD_CASES, start=1):
        F = np.zeros(3 * T.GEOM.nodes.shape[0], dtype=float)
        for node_id, vec in load_case.forces.items():
            base = 3 * (node_id - 1)
            F[base : base + 3] += vec
        cases.append((f"Load Case {idx}", F))
    return cases


def _draw_model(ax, title: str, loads: np.ndarray, nodes: np.ndarray, elems: np.ndarray, fixed_nodes: list[int]) -> None:
    span = float(np.max(nodes.max(axis=0) - nodes.min(axis=0)))
    if span <= 1e-9:
        span = 1.0

    # Elements and labels.
    for e_id, (n1, n2) in enumerate(elems, start=1):
        p1 = nodes[n1 - 1]
        p2 = nodes[n2 - 1]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color="#243248",
            linewidth=1.1,
            alpha=0.95,
        )
        mid = 0.5 * (p1 + p2)
        ax.text(mid[0], mid[1], mid[2], str(e_id), color="crimson", fontsize=6)

    # Nodes.
    free_nodes = [i for i in range(1, T.GEOM.nodes.shape[0] + 1) if i not in fixed_nodes]
    p_free = nodes[np.array(free_nodes) - 1]
    p_fix = nodes[np.array(fixed_nodes) - 1]
    ax.scatter(p_free[:, 0], p_free[:, 1], p_free[:, 2], c="forestgreen", s=30, label="Free nodes")
    ax.scatter(p_fix[:, 0], p_fix[:, 1], p_fix[:, 2], c="royalblue", marker="^", s=52, label="Pinned nodes")

    for n_id, p in enumerate(nodes, start=1):
        ax.text(p[0], p[1], p[2] + 6.0, str(n_id), color="darkgreen", fontsize=8)

    # Supports.
    support_offset = 0.08 * span
    for n_id in fixed_nodes:
        p = nodes[n_id - 1]
        ax.text(p[0], p[1], p[2] - 0.12 * span, "▴", color="royalblue", fontsize=16, ha="center", va="center")
        ax.text(p[0], p[1], p[2] - 0.16 * span, f"N{n_id}", color="royalblue", fontsize=8, ha="center")

    # Loads.
    load_magnitudes = []
    for node_id in range(1, nodes.shape[0] + 1):
        base = 3 * (node_id - 1)
        vec = loads[base : base + 3]
        mag = float(np.linalg.norm(vec))
        if mag > 0:
            load_magnitudes.append(mag)
    max_load = max(load_magnitudes) if load_magnitudes else 1.0
    load_scale = 0.20 * span / max_load

    for node_id in range(1, nodes.shape[0] + 1):
        base = 3 * (node_id - 1)
        vec = loads[base : base + 3]
        if np.allclose(vec, 0.0):
            continue
        p = nodes[node_id - 1]
        dx, dy, dz = vec * load_scale
        ax.quiver(
            p[0],
            p[1],
            p[2],
            dx,
            dy,
            dz,
            color="darkorange",
            arrow_length_ratio=0.14,
            linewidth=2.2,
        )
        label_pos = p + 1.05 * np.array([dx, dy, dz], dtype=float)
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            f"({vec[0]:.0f}, {vec[1]:.0f}, {vec[2]:.0f})",
            color="saddlebrown",
            fontsize=7,
        )

    origin = np.min(nodes, axis=0) - np.array([0.18 * span, 0.18 * span, 0.0])
    _draw_axes_triad(ax, origin, 0.22 * span)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (in)")
    ax.set_ylabel("Y (in)")
    ax.set_zlabel("Z (in)")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=18, azim=-58)
    _set_equal_axes_3d(ax, nodes)


def make_plot(save_path: Path) -> None:
    nodes = T.GEOM.nodes
    elems = T.GEOM.elems
    fixed_nodes = list(T.GEOM.fixed_nodes)
    cases = _load_cases()

    fig = plt.figure(figsize=(16, 8), dpi=150)

    for idx, (title, loads) in enumerate(cases, start=1):
        ax = fig.add_subplot(1, len(cases), idx, projection="3d")
        _draw_model(ax, title, loads, nodes, elems, fixed_nodes)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            framealpha=0.95,
        )

    fig.suptitle("72-Bar Space Truss Model", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "truss72_model_plot.png"
    make_plot(out)
    print(f"Saved: {out}")