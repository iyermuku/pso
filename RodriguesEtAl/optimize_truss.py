"""Compare four bounded optimization methods on the 11-bar truss.

The objective is minimum structural mass with penalties for member stress and
for deformed vertical coordinates of nodes D, E, and F outside 5--300 cm.
The 11 member areas are the only design variables; the undeformed geometry is
fixed at y4=y5=y6=200 cm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import truss11 as T


N_DESIGN = T.N_ELEMS
LOWER = np.full(N_DESIGN, T.A_MIN)
UPPER = np.full(N_DESIGN, T.A_MAX)
REQUESTED_INITIAL = np.full(N_DESIGN, 0.1)
INITIAL_TOP_HEIGHTS = np.full(3, 200.0)
DEFORMED_HEIGHT_BOUNDS = np.array([5.0, 300.0])
DISPLACEMENT_SCALE = DEFORMED_HEIGHT_BOUNDS[1] - DEFORMED_HEIGHT_BOUNDS[0]
MASS_SCALE = 1.0e3
PENALTY = 100.0
MAX_ITERATIONS = 100
MARQUARDT_MAX_ITERATIONS = 1000
GRAD_TOLERANCE = 1.0e-5


@dataclass
class Evaluator:
    """Evaluate mass plus a quadratic penalty for stress violations.

    The penalty weight controls how strongly the optimizer prioritizes the
    stress limit relative to mass. A zero weight ignores stress; increasing
    the weight makes violations increasingly expensive.
    """

    penalty: float = PENALTY
    evaluations: int = 0

    def physical_design(self, z: np.ndarray) -> np.ndarray:
        return LOWER + np.clip(z, 0.0, 1.0) * (UPPER - LOWER)

    def components(self, z: np.ndarray) -> tuple[float, float, float, float]:
        """Return mass, stress penalty, displacement penalty, and total."""
        design = self.physical_design(z)
        result = T.evaluate(design)
        tensile_stress = float(np.max(np.asarray(result["stresses"])[np.asarray(result["stresses"]) > 0.0], initial=0.0))
        violation = max(0.0, tensile_stress - T.S_ALLOW)
        displacements = np.asarray(result["U"], dtype=float)
        deformed_heights = INITIAL_TOP_HEIGHTS + displacements[[7, 9, 11]]
        lower_violation = np.maximum(0.0, DEFORMED_HEIGHT_BOUNDS[0] - deformed_heights)
        upper_violation = np.maximum(0.0, deformed_heights - DEFORMED_HEIGHT_BOUNDS[1])
        displacement_violation = lower_violation + upper_violation
        self.evaluations += 1
        normalized_violation = violation / T.S_ALLOW
        mass_term = float(result["mass"] / MASS_SCALE)
        stress_penalty_term = float(self.penalty * normalized_violation**2)
        displacement_penalty_term = float(
            self.penalty * np.sum((displacement_violation / DISPLACEMENT_SCALE) ** 2)
        )
        total = mass_term + stress_penalty_term + displacement_penalty_term
        return mass_term, stress_penalty_term, displacement_penalty_term, total

    def value(self, z: np.ndarray) -> float:
        """Return the scalar objective minimized by all four methods."""
        return self.components(z)[3]

    def gradient(self, z: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(z)
        step = 1.0e-5
        for index in range(z.size):
            plus = z.copy()
            minus = z.copy()
            plus[index] = min(1.0, plus[index] + step)
            minus[index] = max(0.0, minus[index] - step)
            denominator = plus[index] - minus[index]
            if denominator > 0.0:
                gradient[index] = (self.value(plus) - self.value(minus)) / denominator
        return gradient


def line_search(evaluator: Evaluator, z: np.ndarray, direction: np.ndarray, value: float, gradient: np.ndarray) -> tuple[np.ndarray, float]:
    slope = float(gradient @ direction)
    if slope >= 0.0:
        direction = -gradient
        slope = -float(gradient @ gradient)
    step = 1.0
    while step >= 1.0e-10:
        candidate = np.clip(z + step * direction, 0.0, 1.0)
        candidate_value = evaluator.value(candidate)
        if candidate_value <= value + 1.0e-4 * step * slope:
            return candidate, candidate_value
        step *= 0.5
    return z, value


def projected_gradient(z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Return the projected gradient for the normalized bounds 0 <= z <= 1."""
    return z - np.clip(z - gradient, 0.0, 1.0)


def steepest_descent(evaluator: Evaluator, z: np.ndarray) -> tuple[np.ndarray, int]:
    gradient = evaluator.gradient(z)
    for iteration in range(1, MAX_ITERATIONS + 1):
        if np.linalg.norm(projected_gradient(z, gradient)) <= GRAD_TOLERANCE:
            return z, iteration - 1
        z_new, value_new = line_search(evaluator, z, -gradient, evaluator.value(z), gradient)
        if np.array_equal(z_new, z):
            return z, iteration
        z = z_new
        gradient = evaluator.gradient(z)
    return z, MAX_ITERATIONS


def conjugate_gradient(evaluator: Evaluator, z: np.ndarray) -> tuple[np.ndarray, int]:
    gradient = evaluator.gradient(z)
    direction = -gradient
    for iteration in range(1, MAX_ITERATIONS + 1):
        if np.linalg.norm(projected_gradient(z, gradient)) <= GRAD_TOLERANCE:
            return z, iteration - 1
        old_gradient = gradient
        z_new, value_new = line_search(evaluator, z, direction, evaluator.value(z), gradient)
        if np.array_equal(z_new, z):
            return z, iteration
        z = z_new
        gradient = evaluator.gradient(z)
        beta = max(0.0, float(gradient @ (gradient - old_gradient)) / max(float(old_gradient @ old_gradient), 1.0e-30))
        direction = -gradient + beta * direction
    return z, MAX_ITERATIONS


def marquardt(evaluator: Evaluator, z: np.ndarray) -> tuple[np.ndarray, int]:
    hessian = np.eye(z.size)
    damping = 1.0
    gradient = evaluator.gradient(z)
    for iteration in range(1, MARQUARDT_MAX_ITERATIONS + 1):
        if np.linalg.norm(projected_gradient(z, gradient)) <= GRAD_TOLERANCE:
            return z, iteration - 1
        direction = -np.linalg.solve(hessian + damping * np.eye(z.size), gradient)
        z_new, value_new = line_search(evaluator, z, direction, evaluator.value(z), gradient)
        if np.array_equal(z_new, z):
            damping *= 10.0
            continue
        old_gradient = gradient
        gradient = evaluator.gradient(z_new)
        delta_z = z_new - z
        delta_gradient = gradient - old_gradient
        curvature = float(delta_z @ delta_gradient)
        if curvature > 1.0e-12:
            hessian += np.outer(delta_gradient, delta_gradient) / curvature - (hessian @ np.outer(delta_z, delta_z) @ hessian) / max(float(delta_z @ hessian @ delta_z), 1.0e-12)
            hessian = 0.5 * (hessian + hessian.T)
            if not np.all(np.isfinite(hessian)) or np.min(np.linalg.eigvalsh(hessian)) <= 0.0:
                hessian = np.eye(z.size)
        damping = max(damping * 0.7, 1.0e-8)
        z = z_new
    return z, MARQUARDT_MAX_ITERATIONS


def dfp(evaluator: Evaluator, z: np.ndarray) -> tuple[np.ndarray, int]:
    inverse_hessian = np.eye(z.size)
    gradient = evaluator.gradient(z)
    for iteration in range(1, MAX_ITERATIONS + 1):
        if np.linalg.norm(projected_gradient(z, gradient)) <= GRAD_TOLERANCE:
            return z, iteration - 1
        direction = -inverse_hessian @ gradient
        z_new, value_new = line_search(evaluator, z, direction, evaluator.value(z), gradient)
        if np.array_equal(z_new, z):
            return z, iteration
        old_gradient = gradient
        gradient = evaluator.gradient(z_new)
        delta_z = z_new - z
        delta_gradient = gradient - old_gradient
        first = np.outer(delta_z, delta_z) / max(float(delta_z @ delta_gradient), 1.0e-12)
        second_denominator = float(delta_gradient @ inverse_hessian @ delta_gradient)
        if second_denominator > 1.0e-12:
            second = np.outer(inverse_hessian @ delta_gradient, inverse_hessian @ delta_gradient) / second_denominator
            inverse_hessian += first - second
        z = z_new
    return z, MAX_ITERATIONS


METHODS: dict[str, Callable[[Evaluator, np.ndarray], tuple[np.ndarray, int]]] = {
    "Steepest descent": steepest_descent,
    "Conjugate gradient": conjugate_gradient,
    "Marquardt": marquardt,
    "DFP": dfp,
}


def report(method_name: str, method: Callable[[Evaluator, np.ndarray], tuple[np.ndarray, int]]) -> dict[str, float | int | str]:
    evaluator = Evaluator()
    initial = np.clip((REQUESTED_INITIAL - LOWER) / (UPPER - LOWER), 0.0, 1.0)
    design, iterations = method(evaluator, initial)
    physical = evaluator.physical_design(design)
    mass_term, stress_penalty_term, displacement_penalty_term, final_value = evaluator.components(design)
    gradient = evaluator.gradient(design)
    gradient_norm = float(np.linalg.norm(projected_gradient(design, gradient)))
    return {
        "method": method_name,
        "objective": final_value,
        "mass_term": mass_term,
        "stress_penalty_term": stress_penalty_term,
        "displacement_penalty_term": displacement_penalty_term,
        "projected_gradient_norm": gradient_norm,
        "evaluations": evaluator.evaluations,
        "iterations": iterations,
        "areas": physical[: T.N_ELEMS],
    }


def main() -> None:
    print("Initial design: A1,...,A11=0.1; undeformed y4=y5=y6=200.0 cm.")
    print("Objective: mass_term + stress penalty + displacement penalty")
    print(f"Penalty weight: {PENALTY:g}")
    print("  mass_term = mass / 1000")
    print(f"  tensile stress penalty = penalty * (max(0, max_tensile_stress - {T.S_ALLOW:g}) / {T.S_ALLOW:g})^2")
    print("  deformed heights = [200, 200, 200] + [v4, v5, v6]")
    print("  deformed-height bounds: 5.0 <= y4, y5, y6 <= 300.0 cm")
    print(f"Iteration limits: other methods={MAX_ITERATIONS}, Marquardt={MARQUARDT_MAX_ITERATIONS}")
    print()
    for method_name, method in METHODS.items():
        result = report(method_name, method)
        print(
            f"{result['method']:>20}: objective={result['objective']:.6f}, "
            f"mass_term={result['mass_term']:.6f}, "
            f"stress_penalty={result['stress_penalty_term']:.6f}, "
            f"displacement_penalty={result['displacement_penalty_term']:.6f}, "
            f"||projected gradient||={result['projected_gradient_norm']:.3e}, "
            f"evaluations={result['evaluations']}, iterations={result['iterations']}"
        )
        print("  areas:", np.array2string(result["areas"], precision=5))


if __name__ == "__main__":
    main()
