"""Particle-swarm optimization for the variable-height Rodriguez truss.

Run with:

    python TrussRodriguezPSO.py

The program optimizes 11 member areas and the three upper-node heights, writes
PSO_Rodriguez_final.csv, and saves PSO_Rodriguez_evolution.gif.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import truss11 as T


N_VARIABLES = T.N_ELEMS + 3
LOWER = np.concatenate((np.full(T.N_ELEMS, 0.10), np.full(3, 5.0)))
UPPER = np.concatenate((np.full(T.N_ELEMS, 30.0), np.full(3, 300.0)))
ALLOWABLE_STRESS = 15.0  # 150 MPa = 15 kN/cm^2
CONDITION_NUMBER_LIMIT = 1.0e8
PENALTY_WEIGHT = 100.0
DEFAULT_PARTICLES = 30
DEFAULT_ITERATIONS = 100
DEFAULT_SEED = 20261113
DEFAULT_FPS = 8
FINAL_HOLD_SECONDS = 2.0


@dataclass(frozen=True)
class Candidate:
    design: np.ndarray
    mass: float
    stresses: np.ndarray
    max_stress: float
    max_bar: int
    stress_violation: float
    condition_number: float
    objective: float
    feasible: bool


@dataclass(frozen=True)
class SwarmFrame:
    iteration: int
    candidate: Candidate
    evaluations: int
    feasible_count: int


def evaluate_design(design: np.ndarray) -> Candidate:
    """Evaluate one physical 14-variable design without clipping it."""
    design = np.asarray(design, dtype=float).reshape(-1)
    if design.size != N_VARIABLES:
        raise ValueError(f"Expected {N_VARIABLES} design variables, got {design.size}")
    if np.any(design < LOWER) or np.any(design > UPPER):
        raise ValueError("Design is outside the configured bounds")

    areas = design[: T.N_ELEMS]
    heights = design[T.N_ELEMS :]
    stiffness = T.assemble_global_stiffness(areas, shape_y=heights)
    kff = stiffness[np.ix_(T.FREE_DOFS, T.FREE_DOFS)]
    condition_number = float(np.linalg.cond(kff))
    if not np.isfinite(condition_number) or condition_number > CONDITION_NUMBER_LIMIT:
        return Candidate(
            design=design.copy(),
            mass=float("inf"),
            stresses=np.full(T.N_ELEMS, np.nan),
            max_stress=float("inf"),
            max_bar=0,
            stress_violation=float("inf"),
            condition_number=condition_number,
            objective=float("inf"),
            feasible=False,
        )

    result = T.evaluate(areas, shape_y=heights)
    stresses = np.asarray(result["stresses"], dtype=float)
    absolute_stresses = np.abs(stresses)
    max_index = int(np.argmax(absolute_stresses))
    max_stress = float(absolute_stresses[max_index])
    stress_violation = max(0.0, max_stress - ALLOWABLE_STRESS)
    objective = float(result["mass"] / 1000.0 + PENALTY_WEIGHT * (stress_violation / ALLOWABLE_STRESS) ** 2)
    return Candidate(
        design=design.copy(),
        mass=float(result["mass"]),
        stresses=stresses.copy(),
        max_stress=max_stress,
        max_bar=max_index + 1,
        stress_violation=stress_violation,
        condition_number=condition_number,
        objective=objective,
        feasible=stress_violation <= 0.0,
    )


def better_feasible(current: Candidate | None, candidate: Candidate) -> Candidate | None:
    """Keep the lower-mass feasible candidate."""
    if not candidate.feasible:
        return current
    if current is None or candidate.mass < current.mass:
        return candidate
    return current


def particle_swarm(
    particle_count: int = DEFAULT_PARTICLES,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
    callback=None,
) -> tuple[Candidate, Candidate | None, list[SwarmFrame], int]:
    """Run bounded PSO and return global best, best feasible, history, evaluations."""
    if particle_count <= 0 or iterations <= 0:
        raise ValueError("particle_count and iterations must be positive")
    rng = np.random.default_rng(seed)
    span = UPPER - LOWER
    positions = rng.uniform(LOWER, UPPER, size=(particle_count, N_VARIABLES))
    velocities = rng.uniform(-0.15 * span, 0.15 * span, size=(particle_count, N_VARIABLES))
    personal_best_positions = positions.copy()
    personal_best_candidates = [evaluate_design(position) for position in positions]
    evaluations = particle_count
    global_best = min(personal_best_candidates, key=lambda candidate: candidate.objective)
    best_feasible = None
    for candidate in personal_best_candidates:
        best_feasible = better_feasible(best_feasible, candidate)

    history: list[SwarmFrame] = []
    for iteration in range(iterations + 1):
        feasible_count = sum(candidate.feasible for candidate in personal_best_candidates)
        frame = SwarmFrame(iteration, global_best, evaluations, feasible_count)
        history.append(frame)
        if callback is not None:
            callback(frame)
        if iteration == iterations:
            break

        random_cognitive = rng.random((particle_count, N_VARIABLES))
        random_social = rng.random((particle_count, N_VARIABLES))
        velocities = (
            inertia * velocities
            + cognitive * random_cognitive * (personal_best_positions - positions)
            + social * random_social * (global_best.design - positions)
        )
        velocity_limit = 0.25 * span
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        positions = np.clip(positions + velocities, LOWER, UPPER)

        for index in range(particle_count):
            candidate = evaluate_design(positions[index])
            evaluations += 1
            if candidate.objective < personal_best_candidates[index].objective:
                personal_best_candidates[index] = candidate
                personal_best_positions[index] = positions[index].copy()
            best_feasible = better_feasible(best_feasible, candidate)
        global_best = min(personal_best_candidates, key=lambda candidate: candidate.objective)

    return global_best, best_feasible, history, evaluations


def animate_history(
    history: list[SwarmFrame],
    output_path: Path,
    fps: int = DEFAULT_FPS,
    final_hold_seconds: float = FINAL_HOLD_SECONDS,
) -> Path:
    """Save the global-best PSO evolution as a non-looping GIF."""
    if not history:
        raise ValueError("PSO history is empty")
    hold_count = max(0, int(round(final_hold_seconds * fps)))
    frames = history + [history[-1]] * hold_count
    figure, axis = plt.subplots(figsize=(11, 7))

    def update(index: int) -> None:
        frame = frames[index]
        axis.clear()
        design = frame.candidate.design
        nodes = T.node_coordinates(*design[T.N_ELEMS :])
        for bar_index, (node_i, node_j) in enumerate(T.ELEMENTS, start=1):
            first = nodes[node_i - 1]
            second = nodes[node_j - 1]
            maximum = bar_index == frame.candidate.max_bar
            axis.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                color="crimson" if maximum else "#334155",
                linewidth=3.0 if maximum else 1.5,
            )
            midpoint = 0.5 * (first + second)
            axis.text(midpoint[0], midpoint[1] + 5.0, str(bar_index), fontsize=8)
        axis.scatter(nodes[:, 0], nodes[:, 1], color="#166534", zorder=3)
        for label, point in zip(("A", "B", "C", "D", "E", "F"), nodes):
            axis.text(point[0] + 5.0, point[1] + 5.0, label, weight="bold")

        areas = ", ".join(f"A{i}={value:.3f}" for i, value in enumerate(design[: T.N_ELEMS], 1))
        heights = ", ".join(f"y{i}={value:.3f}" for i, value in zip((4, 5, 6), design[T.N_ELEMS :]))
        metrics = (
            f"iteration={frame.iteration}   evaluations={frame.evaluations}\n"
            f"objective={frame.candidate.objective:.6f}   mass={frame.candidate.mass:.3f} kg\n"
            f"max |stress|={frame.candidate.max_stress:.4f} kN/cm^2   bar=F{frame.candidate.max_bar}\n"
            f"violation={frame.candidate.stress_violation:.4f}   cond(Kff)={frame.candidate.condition_number:.3e}\n"
            f"feasible particles={frame.feasible_count}\n"
            f"areas: {areas}\nheights: {heights}"
        )
        axis.text(1.02, 0.98, metrics, transform=axis.transAxes, va="top", family="monospace", fontsize=8,
                  bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#94a3b8"})
        axis.set_title("TrussRodriguez particle-swarm optimization")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-35.0, 435.0)
        axis.set_ylim(-25.0, 335.0)
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)")
        axis.grid(alpha=0.25)

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000.0 / fps, repeat=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output_path


def write_result(result: Candidate, elapsed_seconds: float, output_path: Path) -> None:
    """Write the final feasible PSO design to CSV."""
    headers = [*(f"A{i}" for i in range(1, T.N_ELEMS + 1)), "y4", "y5", "y6", "mass_kg", "max_stress_kN_cm2", "max_bar", "stress_violation_kN_cm2", "condition_number", "feasible", "elapsed_seconds"]
    values = [*result.design, result.mass, result.max_stress, result.max_bar, result.stress_violation, result.condition_number, result.feasible, elapsed_seconds]
    with output_path.open("w", encoding="ascii", newline="") as handle:
        handle.write(",".join(headers) + "\n")
        handle.write(",".join(str(value) for value in values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Particle-swarm optimization for the Rodriguez truss")
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--output", type=Path, default=Path("PSO_Rodriguez_evolution.gif"))
    args = parser.parse_args()

    start_time = time.perf_counter()
    global_best, best_feasible, history, evaluations = particle_swarm(
        particle_count=args.particles,
        iterations=args.iterations,
        seed=args.seed,
    )
    elapsed_seconds = time.perf_counter() - start_time
    selected = best_feasible if best_feasible is not None else global_best
    write_result(selected, elapsed_seconds, Path("PSO_Rodriguez_final.csv"))
    animate_history(history, args.output, fps=args.fps)

    print(f"Particles: {args.particles}")
    print(f"Iterations: {args.iterations}")
    print(f"Evaluations: {evaluations}")
    print(f"Elapsed PSO time: {elapsed_seconds:.6f} s")
    print(f"Feasible design found: {best_feasible is not None}")
    print(f"Objective: {selected.objective:.6f}")
    print(f"Mass: {selected.mass:.6f} kg")
    print(f"Maximum absolute stress: {selected.max_stress:.6f} kN/cm^2")
    print(f"Maximum-stress bar: F{selected.max_bar}")
    print(f"Condition number: {selected.condition_number:.6e}")
    print("Areas:", np.array2string(selected.design[: T.N_ELEMS], precision=6))
    print("Heights [y4, y5, y6]:", np.array2string(selected.design[T.N_ELEMS :], precision=6))
    print(f"Saved final design: {Path('PSO_Rodriguez_final.csv').resolve()}")
    print(f"Saved animation: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
