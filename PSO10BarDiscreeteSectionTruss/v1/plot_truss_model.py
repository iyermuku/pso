#!/usr/bin/env python3
"""
Draw the 10-bar truss model and export it as a PNG.

The figure shows nodes, element connectivity, downward point loads,
and fixed boundary conditions.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from truss_model import F_base, fixed_dofs, members, nodes, dof_index


def _node_array() -> tuple[np.ndarray, list[int]]:
    node_ids = sorted(nodes)
    coords = np.array([nodes[node_id] for node_id in node_ids], dtype=float)
    return coords, node_ids


def _load_vectors() -> dict[int, np.ndarray]:
    load_map: dict[int, np.ndarray] = {}
    for node_id in sorted(nodes):
        ux_idx, uy_idx = dof_index(node_id)
        load_map[node_id] = np.array([F_base[ux_idx], F_base[uy_idx]], dtype=float)
    return load_map


def draw_truss_model(output_path: str | Path = "truss_model.png") -> Path:
    coords, node_ids = _node_array()
    load_map = _load_vectors()

    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    span = max(x_span, y_span, 1.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for member_id, (i_node, j_node) in members.items():
        xi, yi = nodes[i_node]
        xj, yj = nodes[j_node]
        ax.plot([xi, xj], [yi, yj], color="#2f3b52", linewidth=2.5, zorder=1)
        mid_x = 0.5 * (xi + xj)
        mid_y = 0.5 * (yi + yj)
        ax.text(
            mid_x,
            mid_y + 0.04 * span,
            f"m{member_id}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f2937",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4,
        )

    ax.scatter(coords[:, 0], coords[:, 1], s=85, color="#111827", zorder=5)

    node_offset = 0.03 * span
    for node_id in node_ids:
        x, y = nodes[node_id]
        ax.text(
            x + node_offset,
            y + node_offset,
            f"N{node_id}",
            fontsize=10,
            weight="bold",
            color="#111827",
            zorder=6,
        )

    arrow_length = 0.18 * span
    for node_id, force in load_map.items():
        if np.allclose(force, 0.0):
            continue
        x, y = nodes[node_id]
        fx, fy = force
        magnitude = float(np.linalg.norm(force))
        dx = arrow_length * (fx / magnitude)
        dy = arrow_length * (fy / magnitude)
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.05 * span,
            head_length=0.07 * span,
            length_includes_head=True,
            linewidth=2.0,
            color="#d97706",
            zorder=3,
        )
        ax.text(
            x + dx * 1.05,
            y + dy * 1.05,
            f"{force[1]:.0f} kip",
            fontsize=9,
            color="#b45309",
            ha="center",
            va="center",
            zorder=6,
        )

    support_size = 0.12 * span
    base_drop = 0.12 * span

    fixed_nodes = sorted({(dof // 2) + 1 for dof in fixed_dofs})
    for idx, node_id in enumerate(fixed_nodes):
        x, y = nodes[node_id]
        triangle = plt.Polygon(
            [
                (x - support_size * 0.45, y - base_drop),
                (x + support_size * 0.45, y - base_drop),
                (x, y),
            ],
            closed=True,
            facecolor="#16a34a",
            edgecolor="#166534",
            linewidth=1.2,
            alpha=0.75,
            zorder=2,
        )
        ax.add_patch(triangle)
        if idx == 0:
            ax.text(x - support_size, y - 0.18 * span, "Fixed support", fontsize=9, color="#166534")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (in)")
    ax.set_ylabel("y (in)")
    ax.set_title("10-Bar Truss Model")
    ax.grid(True, alpha=0.25)

    pad = 0.18 * span
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - 0.28 * span, coords[:, 1].max() + 0.18 * span)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markersize=7, label="Node"),
        plt.Line2D([0], [0], color="#2f3b52", linewidth=2.5, label="Element"),
        plt.Line2D([0], [0], color="#d97706", linewidth=2.0, label="Applied load"),
        plt.Line2D([0], [0], color="#16a34a", linewidth=8, label="Boundary condition"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)

    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    output_file = draw_truss_model()
    print(f"Saved truss model figure to {output_file.resolve()}")


if __name__ == "__main__":
    main()