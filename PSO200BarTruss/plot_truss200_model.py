"""Create a labeled 2D plot of the 200-bar truss model.

Outputs:
  PSO200BarTruss/truss200_model_plot.png

The plot includes:
- node markers with node IDs
- element lines with element IDs
- applied load arrows from the model load vector
- support symbols at the fixed nodes
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import truss200 as T


def _load_vectors() -> dict[int, np.ndarray]:
    load_map: dict[int, np.ndarray] = {}
    loads = T.applied_loads()
    for node_id in range(1, T.N_NODES + 1):
        base = 2 * (node_id - 1)
        load_map[node_id] = np.array([loads[base], loads[base + 1]], dtype=float)
    return load_map


def draw_truss_model(output_path: str | Path = "truss200_model_plot.png") -> Path:
    nodes = T.node_coordinates()
    elements = T.element_connectivity()
    load_map = _load_vectors()
    fixed_nodes = list(T.GEOMETRY.supports)

    x_span = float(np.ptp(nodes[:, 0]))
    y_span = float(np.ptp(nodes[:, 1]))
    span = max(x_span, y_span, 1.0)

    fig, ax = plt.subplots(figsize=(18, 8), dpi=180)

    # Elements and labels.
    for elem_id, (node_i, node_j) in enumerate(elements, start=1):
        p1 = nodes[node_i - 1]
        p2 = nodes[node_j - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#273449", linewidth=0.8, alpha=0.95, zorder=1)
        mid = 0.5 * (p1 + p2)
        ax.text(
            mid[0],
            mid[1] + 0.006 * span,
            str(elem_id),
            color="crimson",
            fontsize=5,
            ha="center",
            va="bottom",
            zorder=3,
        )

    # Nodes.
    free_nodes = [node_id for node_id in range(1, T.N_NODES + 1) if node_id not in fixed_nodes]
    p_free = nodes[np.array(free_nodes) - 1]
    p_fix = nodes[np.array(fixed_nodes) - 1]
    ax.scatter(p_free[:, 0], p_free[:, 1], c="#1f8b2f", s=28, label="Free nodes", zorder=4)
    ax.scatter(p_fix[:, 0], p_fix[:, 1], c="royalblue", marker="^", s=70, label="Supports", zorder=5)

    for node_id, p in enumerate(nodes, start=1):
        ax.text(p[0] + 0.008 * span, p[1] + 0.008 * span, str(node_id), color="#14532d", fontsize=7, zorder=6)

    # Supports.
    support_drop = 0.05 * span
    for node_id in fixed_nodes:
        p = nodes[node_id - 1]
        support = plt.Polygon(
            [
                (p[0] - 0.02 * span, p[1] - support_drop),
                (p[0] + 0.02 * span, p[1] - support_drop),
                (p[0], p[1]),
            ],
            closed=True,
            facecolor="#3b82f6",
            edgecolor="#1d4ed8",
            alpha=0.75,
            zorder=2,
        )
        ax.add_patch(support)

    # Load arrows.
    load_magnitudes = [float(np.linalg.norm(vec)) for vec in load_map.values() if not np.allclose(vec, 0.0)]
    max_load = max(load_magnitudes) if load_magnitudes else 1.0
    load_scale = 0.12 * span / max_load

    for node_id, vec in load_map.items():
        if np.allclose(vec, 0.0):
            continue
        p = nodes[node_id - 1]
        dx, dy = vec * load_scale
        ax.arrow(
            p[0],
            p[1],
            dx,
            dy,
            head_width=0.012 * span,
            head_length=0.018 * span,
            length_includes_head=True,
            linewidth=2.0,
            color="darkorange",
            zorder=6,
        )
        ax.text(
            p[0] + 1.05 * dx,
            p[1] + 1.05 * dy,
            f"({vec[0]:.0f}, {vec[1]:.0f})",
            color="#9a3412",
            fontsize=6,
            ha="center",
            va="center",
            zorder=7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("200-Bar Truss Model")
    ax.set_xlabel("x (in)")
    ax.set_ylabel("y (in)")
    ax.grid(True, alpha=0.22)

    pad_x = 0.05 * span
    pad_y = 0.10 * span
    ax.set_xlim(nodes[:, 0].min() - pad_x, nodes[:, 0].max() + pad_x)
    ax.set_ylim(nodes[:, 1].min() - pad_y, nodes[:, 1].max() + pad_y)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f8b2f", markersize=6, label="Free nodes"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="royalblue", markersize=7, label="Supports"),
        plt.Line2D([0], [0], color="#273449", linewidth=1.2, label="Elements"),
        plt.Line2D([0], [0], color="darkorange", linewidth=2.0, label="Applied loads"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.95, fontsize=9)

    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    output_file = draw_truss_model()
    print(f"Saved truss model figure to {output_file.resolve()}")


if __name__ == "__main__":
    main()