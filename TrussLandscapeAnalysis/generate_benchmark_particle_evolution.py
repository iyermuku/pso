from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from benchmark_adapters import get_all_benchmark_problems
from benchmark_pso import pso_minimize


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "benchmark_pso_results"


DEFAULT_PROBLEMS = ["Sphere", "Rastrigin"]
DEFAULT_PARAMS = {
    "Sphere": (0.730, 1.250, 1.750, 115, 260),
    "Rastrigin": (0.650, 1.250, 1.750, 125, 360),
}


def _objective(problem):
    def fn(x: np.ndarray) -> float:
        value, _, _ = problem.evaluate(x)
        return float(value)

    return fn


def _bounds(problem) -> list[tuple[float, float]]:
    return [(float(lo), float(hi)) for lo, hi in zip(problem.lo, problem.hi)]


def _resolve_problems(selected: list[str]):
    problems = get_all_benchmark_problems()
    lookup = {problem.label: problem for problem in problems}
    lookup.update({problem.problem_id: problem for problem in problems})

    resolved = []
    for name in selected:
        if name not in lookup:
            raise KeyError(f"Unknown benchmark problem '{name}'")
        resolved.append(lookup[name])
    return resolved


def _plot_problem(ax, problem, seed: int):
    params = DEFAULT_PARAMS.get(problem.label)
    if params is None:
        raise KeyError(f"No default PSO parameters for {problem.label}")

    w, c1, c2, swarm_size, iters = params
    result = pso_minimize(
        func=_objective(problem),
        bounds=_bounds(problem),
        swarm_size=swarm_size,
        iters=iters,
        inertia=w,
        c1=c1,
        c2=c2,
        seed=seed,
        track_history=True,
    )

    history = result.history
    if history is None:
        raise RuntimeError(f"History was not captured for {problem.label}")

    x_hist = history["X_history"]
    gbest_hist = history["gbest_X_history"]

    lo = np.asarray(problem.lo, dtype=float)
    hi = np.asarray(problem.hi, dtype=float)
    span = hi - lo
    pad = 0.08 * span
    x_min = lo[0] - pad[0]
    x_max = hi[0] + pad[0]
    y_min = lo[1] - pad[1]
    y_max = hi[1] + pad[1]

    grid_n = 180
    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    xg, yg = np.meshgrid(xs, ys)
    zg = np.array([problem.evaluate(np.array([x, y]))[0] for x, y in zip(xg.ravel(), yg.ravel())], dtype=float).reshape(xg.shape)

    contour = ax.contourf(xg, yg, zg, levels=28, cmap="viridis", alpha=0.85)
    ax.contour(xg, yg, zg, levels=14, colors="white", linewidths=0.3, alpha=0.22)

    sample_count = min(24, x_hist.shape[1])
    sample_idx = np.linspace(0, x_hist.shape[1] - 1, sample_count, dtype=int)
    trajectory_subset = x_hist[:, sample_idx, :]

    time_values = np.arange(trajectory_subset.shape[0])
    norm = plt.Normalize(time_values.min(), time_values.max())
    for particle_traj in trajectory_subset.transpose(1, 0, 2):
        segments = np.stack([particle_traj[:-1], particle_traj[1:]], axis=1)
        line = LineCollection(segments, cmap="plasma", norm=norm, linewidths=1.0, alpha=0.38)
        line.set_array(time_values[:-1])
        ax.add_collection(line)

        ax.scatter(
            particle_traj[0, 0],
            particle_traj[0, 1],
            s=16,
            facecolors="none",
            edgecolors="#f3f3f3",
            linewidths=0.8,
            zorder=4,
        )
        ax.scatter(
            particle_traj[-1, 0],
            particle_traj[-1, 1],
            s=20,
            c="#ffd166",
            edgecolors="#111111",
            linewidths=0.4,
            zorder=5,
        )

    ax.plot(gbest_hist[:, 0], gbest_hist[:, 1], color="#ff4d4d", linewidth=2.0, label="Global best path", zorder=6)
    ax.scatter(
        gbest_hist[0, 0],
        gbest_hist[0, 1],
        s=40,
        c="#ff4d4d",
        marker="o",
        edgecolors="#111111",
        linewidths=0.5,
        zorder=7,
    )
    ax.scatter(
        result.best_position[0],
        result.best_position[1],
        s=90,
        c="#ffffff",
        marker="*",
        edgecolors="#111111",
        linewidths=0.9,
        zorder=8,
        label="Final best",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{problem.label}  |  w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    return contour


def main() -> None:
    parser = argparse.ArgumentParser(description="Render particle evolution for two benchmark PSO problems")
    parser.add_argument(
        "--problems",
        default=",".join(DEFAULT_PROBLEMS),
        help="Comma-separated benchmark labels or problem IDs (default: Sphere,Rastrigin)",
    )
    parser.add_argument(
        "--output",
        default=str(OUT_DIR / "particle_evolution_two_benchmarks.png"),
        help="Output image path",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = [item.strip() for item in args.problems.split(",") if item.strip()]
    problems = _resolve_problems(selected)

    if len(problems) != 2:
        raise ValueError("Please provide exactly two benchmark problems")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    contours = []
    for index, (ax, problem) in enumerate(zip(axes, problems, strict=True)):
        contours.append(_plot_problem(ax, problem, seed=args.seed + index))

    cbar = fig.colorbar(contours[0], ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Objective value")
    fig.suptitle("Particle evolution on two benchmark functions", fontsize=14)

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved particle evolution figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()