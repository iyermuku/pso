"""Random feasibility search for the variable-height 11-bar truss.

The 14 design variables are the 11 member areas followed by the absolute
vertical coordinates y4, y5, and y6 of nodes D, E, and F. A design is feasible
when every absolute axial stress is at most 150 MPa and every height is within
5--300 cm. Buckling is not considered.

Run from the repository directory with:

    python TrussRodriguez.py

The script prints the best five feasible samples, or the five closest samples
when fewer than five feasible designs are found, and saves a configuration plot
as TrussRodriguez_candidates.png.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import truss11 as T


SAMPLE_COUNT = 5000
RANDOM_SEED = 20261113
AREA_BOUNDS = (0.10, 30.0)
HEIGHT_BOUNDS = (5.0, 300.0)
ALLOWABLE_STRESS = 15.0  # 150 MPa = 15 kN/cm^2
CONDITION_NUMBER_LIMIT = 1.0e8
OPTIMIZATION_MAX_ITERATIONS = 200
MARQUARDT_MAX_ITERATIONS = 1000
OPTIMIZATION_GRADIENT_TOLERANCE = 1.0e-5
FINITE_DIFFERENCE_STEP = 1.0e-5
ARMIJO_CONSTANT = 1.0e-4
OUTPUT_PLOT = Path("TrussRodriguez_candidates.png")
OUTPUT_TABLE = Path("TrussRodriguez_candidates.csv")
OUTPUT_OPTIMIZATION_TABLE = Path("TrussRodriguez_optimization.csv")
OUTPUT_ANIMATION = Path("TrussRodriguez_optimization.gif")
ANIMATION_FPS = 8
ANIMATION_FINAL_HOLD_SECONDS = 2.0


@dataclass(frozen=True)
class Candidate:
    areas: np.ndarray
    heights: np.ndarray
    mass: float
    stresses: np.ndarray
    max_stress: float
    max_bar: int
    stress_violation: float
    condition_number: float
    feasible: bool


@dataclass(frozen=True)
class OptimizationResult:
    design: np.ndarray
    objective: float
    projected_gradient_norm: float
    iterations: int
    evaluations: int
    elapsed_seconds: float
    candidate: Candidate


@dataclass(frozen=True)
class OptimizationFrame:
    iteration: int
    design: np.ndarray
    objective: float
    projected_gradient_norm: float
    evaluations: int
    candidate: Candidate


def sample_design(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw one area and upper-node-height design from the requested bounds."""
    areas = rng.uniform(AREA_BOUNDS[0], AREA_BOUNDS[1], T.N_ELEMS)
    heights = rng.uniform(HEIGHT_BOUNDS[0], HEIGHT_BOUNDS[1], 3)
    return areas, heights


def evaluate_design(areas: np.ndarray, heights: np.ndarray) -> Candidate:
    """Evaluate one sampled design and classify its stress feasibility."""
    areas = np.asarray(areas, dtype=float).reshape(-1)
    heights = np.asarray(heights, dtype=float).reshape(-1)
    if areas.size != T.N_ELEMS:
        raise ValueError(f"Expected {T.N_ELEMS} areas, got {areas.size}")
    if heights.size != 3:
        raise ValueError(f"Expected three heights, got {heights.size}")
    if np.any((areas < AREA_BOUNDS[0]) | (areas > AREA_BOUNDS[1])):
        raise ValueError("Member areas are outside AREA_BOUNDS")
    if np.any((heights < HEIGHT_BOUNDS[0]) | (heights > HEIGHT_BOUNDS[1])):
        raise ValueError("Node heights are outside HEIGHT_BOUNDS")

    stiffness = T.assemble_global_stiffness(areas, shape_y=heights)
    kff = stiffness[np.ix_(T.FREE_DOFS, T.FREE_DOFS)]
    condition_number = float(np.linalg.cond(kff))
    if not np.isfinite(condition_number) or condition_number > CONDITION_NUMBER_LIMIT:
        return Candidate(
            areas=areas.copy(),
            heights=heights.copy(),
            mass=float("inf"),
            stresses=np.full(T.N_ELEMS, np.nan),
            max_stress=float("inf"),
            max_bar=0,
            stress_violation=float("inf"),
            condition_number=condition_number,
            feasible=False,
        )

    result = T.evaluate(areas, shape_y=heights)
    stresses = np.asarray(result["stresses"], dtype=float)
    absolute_stresses = np.abs(stresses)
    max_index = int(np.argmax(absolute_stresses))
    max_stress = float(absolute_stresses[max_index])
    stress_violation = max(0.0, max_stress - ALLOWABLE_STRESS)
    feasible = stress_violation <= 0.0
    return Candidate(
        areas=np.asarray(areas, dtype=float).copy(),
        heights=np.asarray(heights, dtype=float).copy(),
        mass=float(result["mass"]),
        stresses=stresses.copy(),
        max_stress=max_stress,
        max_bar=max_index + 1,
        stress_violation=stress_violation,
        condition_number=condition_number,
        feasible=feasible,
    )


def _design_from_normalized(normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a normalized 14-variable vector to physical areas and heights."""
    values = np.asarray(normalized, dtype=float).reshape(-1)
    if values.size != T.N_ELEMS + 3:
        raise ValueError(f"Expected {T.N_ELEMS + 3} normalized variables, got {values.size}")
    areas = AREA_BOUNDS[0] + values[: T.N_ELEMS] * (AREA_BOUNDS[1] - AREA_BOUNDS[0])
    heights = HEIGHT_BOUNDS[0] + values[T.N_ELEMS :] * (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0])
    return areas, heights


def _optimization_value(
    normalized: np.ndarray,
    evaluations: list[int],
) -> tuple[float, Candidate]:
    """Return mass plus a quadratic stress penalty for one normalized design."""
    areas, heights = _design_from_normalized(normalized)
    candidate = evaluate_design(areas, heights)
    evaluations[0] += 1
    if not np.isfinite(candidate.condition_number):
        return 1.0e12, candidate
    normalized_violation = candidate.stress_violation / ALLOWABLE_STRESS
    value = candidate.mass / 1000.0 + 100.0 * normalized_violation**2
    if candidate.condition_number > CONDITION_NUMBER_LIMIT:
        value += 1.0e6 * (candidate.condition_number / CONDITION_NUMBER_LIMIT) ** 2
    return float(value), candidate


def _finite_difference_gradient(normalized: np.ndarray, evaluations: list[int]) -> np.ndarray:
    """Estimate the objective gradient with bound-aware finite differences."""
    gradient = np.zeros_like(normalized)
    step = FINITE_DIFFERENCE_STEP
    for index in range(normalized.size):
        plus = normalized.copy()
        minus = normalized.copy()
        plus[index] = min(1.0, plus[index] + step)
        minus[index] = max(0.0, minus[index] - step)
        denominator = plus[index] - minus[index]
        if denominator <= 0.0:
            continue
        plus_value, _ = _optimization_value(plus, evaluations)
        minus_value, _ = _optimization_value(minus, evaluations)
        gradient[index] = (plus_value - minus_value) / denominator
    return gradient


def _projected_gradient(normalized: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Return the projected gradient for normalized bounds [0, 1]."""
    return normalized - np.clip(normalized - gradient, 0.0, 1.0)


def _initial_optimization_state() -> tuple[np.ndarray, list[int], float, Candidate, float]:
    """Create the common initial state used by every optimization method."""
    initial_areas = np.full(T.N_ELEMS, AREA_BOUNDS[0])
    initial_heights = np.full(3, 200.0)
    normalized = np.concatenate(
        (
            (initial_areas - AREA_BOUNDS[0]) / (AREA_BOUNDS[1] - AREA_BOUNDS[0]),
            (initial_heights - HEIGHT_BOUNDS[0]) / (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0]),
        )
    )
    evaluations = [0]
    objective, candidate = _optimization_value(normalized, evaluations)
    return normalized, evaluations, objective, candidate, time.perf_counter()


def _armijo_step(
    normalized: np.ndarray,
    direction: np.ndarray,
    objective: float,
    gradient: np.ndarray,
    evaluations: list[int],
) -> tuple[np.ndarray, float, Candidate, bool]:
    """Take a projected Armijo backtracking step."""
    slope = float(gradient @ direction)
    if slope >= 0.0:
        direction = -gradient
        slope = -float(gradient @ gradient)
    step = 1.0
    while step >= 1.0e-10:
        trial = np.clip(normalized + step * direction, 0.0, 1.0)
        trial_value, trial_candidate = _optimization_value(trial, evaluations)
        if trial_value <= objective + ARMIJO_CONSTANT * step * slope:
            return trial, trial_value, trial_candidate, True
        step *= 0.5
    return normalized, objective, _optimization_value(normalized, evaluations)[1], False


def steepest_descent(
    initial_areas: np.ndarray | None = None,
    initial_heights: np.ndarray | None = None,
    iteration_callback: Callable[[OptimizationFrame], None] | None = None,
) -> OptimizationResult:
    """Run bounded steepest descent from a physical initial design."""
    if initial_areas is None:
        initial_areas = np.full(T.N_ELEMS, AREA_BOUNDS[0])
    if initial_heights is None:
        initial_heights = np.full(3, 200.0)
    initial_areas = np.asarray(initial_areas, dtype=float).reshape(-1)
    initial_heights = np.asarray(initial_heights, dtype=float).reshape(-1)
    if initial_areas.size != T.N_ELEMS or initial_heights.size != 3:
        raise ValueError("Initial design must contain 11 areas and 3 heights")
    normalized = np.concatenate(
        (
            (initial_areas - AREA_BOUNDS[0]) / (AREA_BOUNDS[1] - AREA_BOUNDS[0]),
            (initial_heights - HEIGHT_BOUNDS[0]) / (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0]),
        )
    )
    if np.any((normalized < 0.0) | (normalized > 1.0)):
        raise ValueError("Initial design is outside the configured bounds")

    start_time = time.perf_counter()
    evaluations = [0]
    objective, candidate = _optimization_value(normalized, evaluations)
    for iteration in range(1, OPTIMIZATION_MAX_ITERATIONS + 1):
        gradient = _finite_difference_gradient(normalized, evaluations)
        projected = _projected_gradient(normalized, gradient)
        projected_norm = float(np.linalg.norm(projected))
        if iteration_callback is not None:
            iteration_callback(
                OptimizationFrame(
                    iteration=iteration - 1,
                    design=normalized.copy(),
                    objective=objective,
                    projected_gradient_norm=projected_norm,
                    evaluations=evaluations[0],
                    candidate=candidate,
                )
            )
        if projected_norm <= OPTIMIZATION_GRADIENT_TOLERANCE:
            return OptimizationResult(normalized, objective, projected_norm, iteration - 1, evaluations[0], time.perf_counter() - start_time, candidate)

        normalized, objective, candidate, accepted = _armijo_step(
            normalized, -gradient, objective, gradient, evaluations
        )
        if not accepted:
            return OptimizationResult(normalized, objective, projected_norm, iteration, evaluations[0], time.perf_counter() - start_time, candidate)

    gradient = _finite_difference_gradient(normalized, evaluations)
    projected_norm = float(np.linalg.norm(_projected_gradient(normalized, gradient)))
    return OptimizationResult(normalized, objective, projected_norm, OPTIMIZATION_MAX_ITERATIONS, evaluations[0], time.perf_counter() - start_time, candidate)


def conjugate_gradient(
    iteration_callback: Callable[[OptimizationFrame], None] | None = None,
) -> OptimizationResult:
    """Run projected nonlinear conjugate gradient with Armijo backtracking."""
    normalized, evaluations, objective, candidate, start_time = _initial_optimization_state()
    gradient = _finite_difference_gradient(normalized, evaluations)
    direction = -gradient
    for iteration in range(1, OPTIMIZATION_MAX_ITERATIONS + 1):
        projected = _projected_gradient(normalized, gradient)
        projected_norm = float(np.linalg.norm(projected))
        if iteration_callback is not None:
            iteration_callback(OptimizationFrame(iteration - 1, normalized.copy(), objective, projected_norm, evaluations[0], candidate))
        if projected_norm <= OPTIMIZATION_GRADIENT_TOLERANCE:
            return OptimizationResult(normalized, objective, projected_norm, iteration - 1, evaluations[0], time.perf_counter() - start_time, candidate)
        old_gradient = gradient
        normalized, objective, candidate, accepted = _armijo_step(normalized, direction, objective, gradient, evaluations)
        if not accepted:
            return OptimizationResult(normalized, objective, projected_norm, iteration, evaluations[0], time.perf_counter() - start_time, candidate)
        gradient = _finite_difference_gradient(normalized, evaluations)
        beta = max(0.0, float(gradient @ (gradient - old_gradient)) / max(float(old_gradient @ old_gradient), 1.0e-30))
        direction = -gradient + beta * direction
    return OptimizationResult(normalized, objective, projected_norm, OPTIMIZATION_MAX_ITERATIONS, evaluations[0], time.perf_counter() - start_time, candidate)


def marquardt(
    iteration_callback: Callable[[OptimizationFrame], None] | None = None,
) -> OptimizationResult:
    """Run a damped quasi-Newton Marquardt search."""
    normalized, evaluations, objective, candidate, start_time = _initial_optimization_state()
    hessian = np.eye(normalized.size)
    damping = 1.0
    gradient = _finite_difference_gradient(normalized, evaluations)
    for iteration in range(1, MARQUARDT_MAX_ITERATIONS + 1):
        projected_norm = float(np.linalg.norm(_projected_gradient(normalized, gradient)))
        if iteration_callback is not None:
            iteration_callback(OptimizationFrame(iteration - 1, normalized.copy(), objective, projected_norm, evaluations[0], candidate))
        if projected_norm <= OPTIMIZATION_GRADIENT_TOLERANCE:
            return OptimizationResult(normalized, objective, projected_norm, iteration - 1, evaluations[0], time.perf_counter() - start_time, candidate)
        try:
            direction = -np.linalg.solve(hessian + damping * np.eye(normalized.size), gradient)
        except np.linalg.LinAlgError:
            hessian = np.eye(normalized.size)
            damping *= 10.0
            continue
        old_gradient = gradient
        old_normalized = normalized.copy()
        normalized, objective, candidate, accepted = _armijo_step(normalized, direction, objective, gradient, evaluations)
        if not accepted:
            damping *= 10.0
            continue
        gradient = _finite_difference_gradient(normalized, evaluations)
        delta_design = normalized - old_normalized
        delta_gradient = gradient - old_gradient
        curvature = float(delta_design @ delta_gradient)
        if curvature > 1.0e-12:
            hessian += np.outer(delta_gradient, delta_gradient) / curvature
            hessian = 0.5 * (hessian + hessian.T)
            eigenvalues = np.linalg.eigvalsh(hessian)
            if not np.all(np.isfinite(hessian)) or eigenvalues[0] <= 0.0:
                hessian = np.eye(normalized.size)
        damping = max(damping * 0.7, 1.0e-8)
    return OptimizationResult(normalized, objective, projected_norm, MARQUARDT_MAX_ITERATIONS, evaluations[0], time.perf_counter() - start_time, candidate)


def dfp(
    iteration_callback: Callable[[OptimizationFrame], None] | None = None,
) -> OptimizationResult:
    """Run projected DFP using an inverse-Hessian approximation."""
    normalized, evaluations, objective, candidate, start_time = _initial_optimization_state()
    inverse_hessian = np.eye(normalized.size)
    gradient = _finite_difference_gradient(normalized, evaluations)
    for iteration in range(1, OPTIMIZATION_MAX_ITERATIONS + 1):
        projected_norm = float(np.linalg.norm(_projected_gradient(normalized, gradient)))
        if iteration_callback is not None:
            iteration_callback(OptimizationFrame(iteration - 1, normalized.copy(), objective, projected_norm, evaluations[0], candidate))
        if projected_norm <= OPTIMIZATION_GRADIENT_TOLERANCE:
            return OptimizationResult(normalized, objective, projected_norm, iteration - 1, evaluations[0], time.perf_counter() - start_time, candidate)
        direction = -inverse_hessian @ gradient
        old_normalized = normalized.copy()
        old_gradient = gradient
        normalized, objective, candidate, accepted = _armijo_step(normalized, direction, objective, gradient, evaluations)
        if not accepted:
            return OptimizationResult(normalized, objective, projected_norm, iteration, evaluations[0], time.perf_counter() - start_time, candidate)
        gradient = _finite_difference_gradient(normalized, evaluations)
        delta_design = normalized - old_normalized
        delta_gradient = gradient - old_gradient
        curvature = float(delta_design @ delta_gradient)
        denominator = float(delta_gradient @ inverse_hessian @ delta_gradient)
        if curvature > 1.0e-12 and denominator > 1.0e-12:
            first = np.outer(delta_design, delta_design) / curvature
            second = np.outer(inverse_hessian @ delta_gradient, inverse_hessian @ delta_gradient) / denominator
            inverse_hessian += first - second
            inverse_hessian = 0.5 * (inverse_hessian + inverse_hessian.T)
            if not np.all(np.isfinite(inverse_hessian)):
                inverse_hessian = np.eye(normalized.size)
    return OptimizationResult(normalized, objective, projected_norm, OPTIMIZATION_MAX_ITERATIONS, evaluations[0], time.perf_counter() - start_time, candidate)


def animate_optimization(
    frames: list[OptimizationFrame],
    method_name: str,
    output_path: Path = OUTPUT_ANIMATION,
    fps: int = ANIMATION_FPS,
    final_hold_seconds: float = ANIMATION_FINAL_HOLD_SECONDS,
) -> Path:
    """Save an iteration animation and hold the final frame before stopping."""
    if not frames:
        raise ValueError("At least one optimization frame is required")
    if fps <= 0:
        raise ValueError("fps must be positive")

    hold_count = max(0, int(round(final_hold_seconds * fps)))
    render_frames = frames + [frames[-1]] * hold_count
    figure, axis = plt.subplots(figsize=(11, 7))

    def update(frame_index: int) -> None:
        frame = render_frames[frame_index]
        axis.clear()
        nodes = T.node_coordinates(*frame.candidate.heights)
        for bar_index, (node_i, node_j) in enumerate(T.ELEMENTS, start=1):
            first = nodes[node_i - 1]
            second = nodes[node_j - 1]
            is_maximum = bar_index == frame.candidate.max_bar
            axis.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                color="crimson" if is_maximum else "#334155",
                linewidth=3.0 if is_maximum else 1.5,
            )
            midpoint = 0.5 * (first + second)
            axis.text(midpoint[0], midpoint[1] + 5.0, str(bar_index), fontsize=8)

        axis.scatter(nodes[:, 0], nodes[:, 1], color="#166534", zorder=3)
        for label, point in zip(("A", "B", "C", "D", "E", "F"), nodes):
            axis.text(point[0] + 5.0, point[1] + 5.0, label, weight="bold")

        area_text = ", ".join(f"A{i}={value:.3f}" for i, value in enumerate(frame.candidate.areas, 1))
        height_text = ", ".join(f"y{i}={value:.3f}" for i, value in zip((4, 5, 6), frame.candidate.heights))
        metrics = (
            f"iteration={frame.iteration}   evaluations={frame.evaluations}\n"
            f"objective={frame.objective:.6f}   mass={frame.candidate.mass:.3f} kg\n"
            f"max |stress|={frame.candidate.max_stress:.4f} kN/cm^2   "
            f"bar=F{frame.candidate.max_bar}   violation={frame.candidate.stress_violation:.4f}\n"
            f"cond(Kff)={frame.candidate.condition_number:.3e}   "
            f"projected |gradient|={frame.projected_gradient_norm:.3e}\n"
            f"areas: {area_text}\n"
            f"heights: {height_text}"
        )
        axis.text(
            1.02,
            0.98,
            metrics,
            transform=axis.transAxes,
            va="top",
            family="monospace",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#94a3b8"},
        )
        axis.set_title(f"TrussRodriguez {method_name} optimization")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-35.0, 435.0)
        axis.set_ylim(-25.0, 335.0)
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)")
        axis.grid(alpha=0.25)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(render_frames),
        interval=1000.0 / fps,
        repeat=False,
    )
    output_path = Path(output_path)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output_path


def _method_animation_path(base_path: Path, method_name: str) -> Path:
    """Create a method-specific animation filename from a base path."""
    slug = method_name.lower().replace(" ", "_")
    return base_path.with_name(f"{base_path.stem}_{slug}{base_path.suffix or '.gif'}")


def search_candidates(sample_count: int = SAMPLE_COUNT) -> tuple[list[Candidate], list[Candidate]]:
    """Return all feasible candidates and the closest candidates found."""
    rng = np.random.default_rng(RANDOM_SEED)
    feasible: list[Candidate] = []
    all_candidates: list[Candidate] = []
    for _ in range(sample_count):
        areas, heights = sample_design(rng)
        candidate = evaluate_design(areas, heights)
        all_candidates.append(candidate)
        if candidate.feasible:
            feasible.append(candidate)

    feasible.sort(key=lambda candidate: candidate.mass)
    closest = sorted(
        all_candidates,
        key=lambda candidate: (candidate.stress_violation, candidate.mass, candidate.condition_number),
    )
    return feasible, closest


def _candidate_label(candidate: Candidate, index: int, fallback: bool) -> str:
    status = "near-feasible" if fallback else "feasible"
    return (
        f"{status} {index}\n"
        f"mass={candidate.mass:.2f} kg, max |stress|={candidate.max_stress:.3f} kN/cm$^2$\n"
        f"bar {candidate.max_bar}, cond={candidate.condition_number:.2e}, "
        f"y=[{candidate.heights[0]:.1f}, "
        f"{candidate.heights[1]:.1f}, {candidate.heights[2]:.1f}] cm"
    )


def plot_candidates(candidates: list[Candidate], fallback: bool, output_path: Path = OUTPUT_PLOT) -> None:
    """Plot up to five sampled truss configurations with their metrics."""
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    axes_flat = axes.ravel()
    node_labels = ("A", "B", "C", "D", "E", "F")

    for axis, candidate, index in zip(axes_flat, candidates, range(1, len(candidates) + 1)):
        nodes = T.node_coordinates(*candidate.heights)
        for bar_index, (node_i, node_j) in enumerate(T.ELEMENTS, start=1):
            first = nodes[node_i - 1]
            second = nodes[node_j - 1]
            color = "crimson" if bar_index == candidate.max_bar else "#334155"
            width = 3.0 if bar_index == candidate.max_bar else 1.5
            axis.plot([first[0], second[0]], [first[1], second[1]], color=color, linewidth=width)
            midpoint = 0.5 * (first + second)
            axis.text(midpoint[0], midpoint[1] + 5.0, str(bar_index), fontsize=7, color=color)

        axis.scatter(nodes[:, 0], nodes[:, 1], color="#166534", zorder=3)
        for label, point in zip(node_labels, nodes):
            axis.text(point[0] + 5.0, point[1] + 5.0, label, fontsize=9, weight="bold")
        axis.set_title(_candidate_label(candidate, index, fallback), fontsize=9)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-35.0, 435.0)
        axis.set_ylim(-25.0, 335.0)
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)")
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(candidates) :]:
        axis.axis("off")
    figure.suptitle(
        "TrussRodriguez sampled designs: "
        + ("feasible candidates" if not fallback else "closest candidates; no feasible sample found"),
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_candidate_table(candidates: list[Candidate], output_path: Path = OUTPUT_TABLE) -> None:
    """Write design variables and constraint metrics to CSV."""
    with output_path.open("w", encoding="ascii", newline="") as handle:
        headers = [*(f"A{i}" for i in range(1, T.N_ELEMS + 1)), "y4", "y5", "y6", "mass_kg", "max_stress_kN_cm2", "max_bar", "stress_violation_kN_cm2", "condition_number", "feasible"]
        handle.write(",".join(headers) + "\n")
        for candidate in candidates:
            values = [
                *candidate.areas,
                *candidate.heights,
                candidate.mass,
                candidate.max_stress,
                candidate.max_bar,
                candidate.stress_violation,
                candidate.condition_number,
                candidate.feasible,
            ]
            handle.write(",".join(str(value) for value in values) + "\n")


def write_optimization_table(
    runs: list[tuple[str, OptimizationResult]],
    output_path: Path = OUTPUT_OPTIMIZATION_TABLE,
) -> None:
    """Write a comparison table for all derivative-based methods."""
    with output_path.open("w", encoding="ascii", newline="") as handle:
        headers = [
            "method",
            "objective",
            "mass_kg",
            "max_stress_kN_cm2",
            "max_bar",
            "stress_violation_kN_cm2",
            "condition_number",
            "projected_gradient_norm",
            "iterations",
            "evaluations",
            "elapsed_seconds",
            "feasible",
        ]
        handle.write(",".join(headers) + "\n")
        for method, result in runs:
            values = [
                method,
                result.objective,
                result.candidate.mass,
                result.candidate.max_stress,
                result.candidate.max_bar,
                result.candidate.stress_violation,
                result.candidate.condition_number,
                result.projected_gradient_norm,
                result.iterations,
                result.evaluations,
                result.elapsed_seconds,
                result.candidate.feasible,
            ]
            handle.write(",".join(str(value) for value in values) + "\n")


def print_candidates(candidates: list[Candidate], fallback: bool) -> None:
    """Print the selected candidates and their design variables."""
    label = "closest" if fallback else "feasible"
    print(f"Selected {label} candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        print(f"\nCandidate {index} ({'FEASIBLE' if candidate.feasible else 'near-feasible'})")
        print("  areas:", np.array2string(candidate.areas, precision=6))
        print("  heights [y4, y5, y6] cm:", np.array2string(candidate.heights, precision=6))
        print(f"  mass: {candidate.mass:.6f} kg")
        print(f"  maximum absolute axial stress: {candidate.max_stress:.6f} kN/cm^2")
        print(f"  maximum-stress bar: F{candidate.max_bar}")
        print(f"  cond(Kff): {candidate.condition_number:.6e}")
        print(f"  stress violation: {candidate.stress_violation:.6f} kN/cm^2")
        print(f"  stresses: {np.array2string(candidate.stresses, precision=6)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and optimize the Rodriguez truss")
    parser.add_argument("--fps", type=int, default=ANIMATION_FPS, help="Animation frame rate")
    parser.add_argument(
        "--animation-output",
        type=Path,
        default=OUTPUT_ANIMATION,
        help="Output GIF path",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip creation of the steepest-descent animation",
    )
    args = parser.parse_args()

    feasible, closest = search_candidates()
    selected = feasible[:5]
    fallback = len(selected) < 5
    if fallback:
        selected = closest[:5]

    print(f"Samples: {SAMPLE_COUNT}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Allowable absolute axial stress: {ALLOWABLE_STRESS:.3f} kN/cm^2 (150 MPa)")
    print(f"Maximum accepted cond(Kff): {CONDITION_NUMBER_LIMIT:.3e}")
    print(f"Feasible samples found: {len(feasible)}")
    print_candidates(selected, fallback)
    write_candidate_table(selected)
    plot_candidates(selected, fallback)
    methods: list[tuple[str, Callable[..., OptimizationResult]]] = [
        ("Steepest descent", lambda **kwargs: steepest_descent(**kwargs)),
        ("Conjugate gradient", lambda **kwargs: conjugate_gradient(**kwargs)),
        ("Marquardt", lambda **kwargs: marquardt(**kwargs)),
        ("DFP", lambda **kwargs: dfp(**kwargs)),
    ]
    runs: list[tuple[str, OptimizationResult]] = []
    histories: dict[str, list[OptimizationFrame]] = {}
    print("\nOptimization comparison")
    print("method,objective,mass_kg,max_stress,bar,stress_violation,cond(Kff),projected_gradient,iterations,evaluations,seconds,feasible")
    for method_name, method in methods:
        history: list[OptimizationFrame] = []
        histories[method_name] = history
        callback = history.append
        result = method(iteration_callback=callback)
        runs.append((method_name, result))
        print(
            f"{method_name},{result.objective:.6f},{result.candidate.mass:.6f},"
            f"{result.candidate.max_stress:.6f},F{result.candidate.max_bar},"
            f"{result.candidate.stress_violation:.6f},{result.candidate.condition_number:.6e},"
            f"{result.projected_gradient_norm:.6e},{result.iterations},{result.evaluations},"
            f"{result.elapsed_seconds:.6f},{result.candidate.feasible}"
        )
    write_optimization_table(runs)
    if not args.no_animation:
        for method_name, _ in runs:
            animation_path = animate_optimization(
                histories[method_name],
                method_name=method_name,
                output_path=_method_animation_path(args.animation_output, method_name),
                fps=args.fps,
            )
            print(f"Saved {method_name} animation: {animation_path.resolve()}")
    print(f"Saved optimization table: {OUTPUT_OPTIMIZATION_TABLE.resolve()}")
    print(f"Saved plot: {OUTPUT_PLOT.resolve()}")
    print(f"Saved table: {OUTPUT_TABLE.resolve()}")


if __name__ == "__main__":
    main()
