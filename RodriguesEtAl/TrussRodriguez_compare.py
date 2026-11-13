"""Compare final PSO and steepest-descent Rodriguez truss designs.

Run with:

    python TrussRodriguez_compare.py

The output image uses blue for PSO and orange for steepest descent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import TrussRodriguez as R
import TrussRodriguezPSO as P


OUTPUT_IMAGE = Path("TrussRodriguez_PSO_vs_steepest.png")
PSO_PARTICLES = 30
PSO_ITERATIONS = 100


def run_steepest() -> R.OptimizationResult:
    return R.steepest_descent()


def run_pso(particles: int, iterations: int, seed: int) -> P.Candidate:
    _, feasible, _, _ = P.particle_swarm(
        particle_count=particles,
        iterations=iterations,
        seed=seed,
    )
    if feasible is None:
        raise RuntimeError("PSO did not find a feasible design")
    return feasible


def draw_design(axis: plt.Axes, design: np.ndarray, candidate: object, color: str, label: str) -> None:
    heights = design[R.T.N_ELEMS :]
    nodes = R.T.node_coordinates(*heights)
    max_bar = int(candidate.max_bar)
    for bar_index, (node_i, node_j) in enumerate(R.T.ELEMENTS, start=1):
        first = nodes[node_i - 1]
        second = nodes[node_j - 1]
        is_maximum = bar_index == max_bar
        axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color="#9a3412" if is_maximum and label == "Steepest descent" else (
                "#1e3a8a" if is_maximum else color
            ),
            linewidth=3.5 if is_maximum else 2.0,
        )
        midpoint = 0.5 * (first + second)
        axis.text(midpoint[0], midpoint[1] + 5.0, str(bar_index), fontsize=8)

    axis.scatter(nodes[:, 0], nodes[:, 1], color=color, s=45, zorder=3)
    for node_label, point in zip(("A", "B", "C", "D", "E", "F"), nodes):
        axis.text(point[0] + 5.0, point[1] + 5.0, node_label, weight="bold")

    axis.set_title(
        f"{label}\n"
        f"mass={candidate.mass:.2f} kg, max stress={candidate.max_stress:.3f} kN/cm$^2$\n"
        f"critical bar=F{candidate.max_bar}, cond(Kff)={candidate.condition_number:.2e}\n"
        f"feasible={candidate.feasible}",
        fontsize=10,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-35.0, 435.0)
    axis.set_ylim(-25.0, 335.0)
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)")
    axis.grid(alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PSO and steepest descent final designs")
    parser.add_argument("--particles", type=int, default=PSO_PARTICLES)
    parser.add_argument("--iterations", type=int, default=PSO_ITERATIONS)
    parser.add_argument("--seed", type=int, default=P.DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_IMAGE)
    args = parser.parse_args()

    steepest_result = run_steepest()
    steepest_design = np.concatenate(
        (
            steepest_result.candidate.areas,
            steepest_result.candidate.heights,
        )
    )
    pso_candidate = run_pso(args.particles, args.iterations, args.seed)

    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    draw_design(axes[0], pso_candidate.design, pso_candidate, "#2563eb", "PSO")
    draw_design(axes[1], steepest_design, steepest_result.candidate, "#f59e0b", "Steepest descent")
    figure.suptitle(
        f"Rodriguez truss final designs | PSO: {args.particles} particles, "
        f"{args.iterations} iterations",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved comparison image: {args.output.resolve()}")
    print(f"PSO configuration: {args.particles} particles, {args.iterations} iterations, seed={args.seed}")
    print(f"PSO mass={pso_candidate.mass:.6f} kg, max stress={pso_candidate.max_stress:.6f} kN/cm^2")
    print(f"Steepest descent mass={steepest_result.candidate.mass:.6f} kg, max stress={steepest_result.candidate.max_stress:.6f} kN/cm^2")


if __name__ == "__main__":
    main()
