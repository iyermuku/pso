from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmark_adapters import get_all_benchmark_problems
from benchmark_pso import pso_minimize


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "benchmark_landscape_summary.md"
OUT_DIR = ROOT / "benchmark_pso_results"


PARAMS = {
    "Sphere": (0.730, 1.250, 1.750, 115, 260),
    "Ellipsoid": (0.570, 1.333, 1.667, 130, 350),
    "Sum of Different Powers": (0.730, 1.250, 1.750, 120, 280),
    "Zakharov": (0.570, 1.306, 1.694, 130, 340),
    "Rosenbrock": (0.730, 1.250, 1.750, 120, 280),
    "Step": (0.730, 1.250, 1.750, 120, 280),
    "Quartic": (0.730, 1.250, 1.700, 115, 260),
    "Schwefel 2.22": (0.650, 1.250, 1.750, 120, 340),
    "Schwefel 1.2": (0.570, 1.333, 1.667, 135, 370),
    "Schwefel 2.21": (0.650, 1.250, 1.700, 115, 260),
    "Rastrigin": (0.650, 1.250, 1.750, 125, 360),
    "Ackley": (0.650, 1.250, 1.750, 120, 280),
    "Griewank": (0.730, 1.250, 1.750, 115, 260),
    "Levy": (0.650, 1.250, 1.750, 120, 280),
    "Michalewicz": (0.650, 1.250, 1.700, 115, 310),
    "Alpine 1": (0.650, 1.250, 1.750, 120, 330),
    "Alpine 2": (0.540, 1.333, 1.667, 130, 400),
    "Bent Cigar": (0.570, 1.333, 1.667, 125, 320),
    "Discus": (0.570, 1.333, 1.667, 135, 370),
    "Weierstrass": (0.650, 1.250, 1.750, 120, 280),
    "HappyCat": (0.730, 1.250, 1.750, 120, 280),
    "HGBat": (0.730, 1.250, 1.750, 120, 280),
    "Qing": (0.570, 1.355, 1.645, 125, 320),
    "Salomon": (0.650, 1.250, 1.700, 120, 280),
    "Bohachevsky": (0.730, 1.250, 1.750, 45, 310),
    "Booth": (0.730, 1.250, 1.750, 40, 290),
    "Matyas": (0.730, 1.350, 1.600, 30, 250),
    "Three-hump Camel": (0.650, 1.250, 1.750, 35, 270),
    "Six-hump Camel": (0.650, 1.250, 1.750, 40, 290),
    "Goldstein-Price": (0.680, 1.250, 1.750, 45, 310),
    "Branin": (0.650, 1.250, 1.750, 35, 260),
    "Shubert": (0.540, 1.333, 1.667, 55, 420),
    "Himmelblau": (0.650, 1.250, 1.750, 40, 290),
    "Easom": (0.730, 1.250, 1.700, 30, 240),
    "Cross-in-Tray": (0.650, 1.250, 1.750, 45, 360),
    "Holder Table": (0.650, 1.250, 1.750, 45, 360),
}


def _objective(problem):
    def fn(x: np.ndarray) -> float:
        value, _, _ = problem.evaluate(x)
        return float(value)

    return fn


def _bounds(problem) -> list[tuple[float, float]]:
    return [(float(lo), float(hi)) for lo, hi in zip(problem.lo, problem.hi)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems = get_all_benchmark_problems()
    by_label = {p.label: p for p in problems}

    rows = []
    for problem in problems:
        if problem.label not in PARAMS:
            raise KeyError(f"No benchmark parameters for {problem.label}")
        w, c1, c2, swarm_size, iters = PARAMS[problem.label]
        result = pso_minimize(
            func=_objective(problem),
            bounds=_bounds(problem),
            swarm_size=swarm_size,
            iters=iters,
            inertia=w,
            c1=c1,
            c2=c2,
            seed=2026,
            track_history=False,
        )
        rows.append(
            {
                "problem_id": problem.problem_id,
                "label": problem.label,
                "w": w,
                "c1": c1,
                "c2": c2,
                "swarm_size": swarm_size,
                "iters": iters,
                "best_value": float(result.best_value),
                "best_position": result.best_position.tolist(),
            }
        )
        print(f"{problem.label}: best_value={result.best_value:.6g} | w={w:.3f} c1={c1:.3f} c2={c2:.3f} | swarm={swarm_size} iters={iters}")

    (OUT_DIR / "benchmark_pso_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (OUT_DIR / "benchmark_pso_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["problem_id", "label", "w", "c1", "c2", "swarm_size", "iters", "best_value", "best_position"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
