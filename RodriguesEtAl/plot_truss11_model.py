"""Create a labeled 2D plot of the 11-bar truss benchmark.

Outputs:
  RodriguesEtAl/truss11_model_plot.png

The plot includes:
- node markers with node IDs and paper labels A-F
- element lines with element IDs
- applied downward loads at nodes B and C
- pinned support symbols at nodes A and D
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import truss11 as T


def _load_vectors() -> dict[int, np.ndarray]:
    load_map: dict[int, np.ndarray] = {}
    loads = T.applied_loads()
    for node_id in range(1, T.N_NODES + 1):
        ux_idx, uy_idx = T.dof_index(node_id)
        load_map[node_id] = np.array([loads[ux_idx], loads[uy_idx]], dtype=float)
    return load_map


def draw_truss_model(
    output_path: str | Path = "truss11_model_plot.png",
    shape_y: tuple[float, float, float] = (200.0, 200.0, 200.0),
) -> Path:
    nodes = T.node_coordinates(*shape_y)
    elements = T.element_connectivity()
    load_map = _load_vectors()
    fixed_nodes = list(T.GEOMETRY.supports)

    x_span = float(np.ptp(nodes[:, 0]))
    y_span = float(np.ptp(nodes[:, 1]))
    span = max(x_span, y_span, 1.0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    # Elements and element IDs.
    for elem_id, (node_i, node_j) in enumerate(elements, start=1):
        p1 = nodes[node_i - 1]
        p2 = nodes[node_j - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#283448", linewidth=2.0, zorder=1)
        mid = 0.5 * (p1 + p2)
        ax.text(
            mid[0],
            mid[1] + 0.04 * span,
            str(elem_id),
            color="crimson",
            fontsize=9,
            ha="center",
            va="bottom",
            zorder=4,
        )

    # Nodes and labels.
    free_nodes = [node_id for node_id in range(1, T.N_NODES + 1) if node_id not in fixed_nodes]
    p_free = nodes[np.array(free_nodes) - 1]
    p_fix = nodes[np.array(fixed_nodes) - 1]
    ax.scatter(p_free[:, 0], p_free[:, 1], c="#166534", s=70, label="Free nodes", zorder=5)
    ax.scatter(p_fix[:, 0], p_fix[:, 1], c="#1d4ed8", marker="^", s=120, label="Supports", zorder=6)

    label_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}
    for node_id, p in enumerate(nodes, start=1):
        ax.text(
            p[0] + 0.02 * span,
            p[1] + 0.02 * span,
            f"{label_map[node_id]} ({node_id})",
            color="#0f172a",
            fontsize=10,
            weight="bold",
            zorder=7,
        )

    # Supports as filled triangles under A and D.
    support_drop = 0.06 * span
    support_size = 0.06 * span
    for node_id in fixed_nodes:
        p = nodes[node_id - 1]
        support = plt.Polygon(
            [
                (p[0] - support_size, p[1] - support_drop),
                (p[0] + support_size, p[1] - support_drop),
                (p[0], p[1]),
            ],
            closed=True,
            facecolor="#60a5fa",
            edgecolor="#1d4ed8",
            alpha=0.75,
            zorder=3,
        )
        ax.add_patch(support)

    # Loads.
    load_magnitudes = [float(np.linalg.norm(vec)) for vec in load_map.values() if not np.allclose(vec, 0.0)]
    max_load = max(load_magnitudes) if load_magnitudes else 1.0
    load_scale = 0.18 * span / max_load

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
            head_width=0.03 * span,
            head_length=0.04 * span,
            length_includes_head=True,
            linewidth=2.2,
            color="darkorange",
            zorder=8,
        )
        ax.text(
            p[0] + 1.06 * dx,
            p[1] + 1.06 * dy,
            f"{vec[1]:.0f} kN",
            color="#9a3412",
            fontsize=9,
            ha="center",
            va="center",
            zorder=9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("11-Bar Truss Benchmark (Rodrigues da Silva et al., 2024)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.grid(True, alpha=0.25)

    pad_x = 0.08 * span
    pad_y = 0.18 * span
    ax.set_xlim(nodes[:, 0].min() - pad_x, nodes[:, 0].max() + pad_x)
    ax.set_ylim(nodes[:, 1].min() - pad_y, nodes[:, 1].max() + pad_y)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#166534", markersize=7, label="Free nodes"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#1d4ed8", markersize=8, label="Supports"),
        plt.Line2D([0], [0], color="#283448", linewidth=2.0, label="Elements"),
        plt.Line2D([0], [0], color="darkorange", linewidth=2.2, label="Applied loads"),
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