"""Monte Carlo robustness study around the steepest-descent Rodriguez design.

The study perturbs the stored PSO design using:

- area perturbations: +/-0.04 cm^2
- height perturbations: +/-0.2 cm
- downward load: 100--110 kN at nodes B and C
- allowable absolute stress: 150--157.5 MPa

Run with:

    python TrussRodriguez_monte_carlo.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import truss11 as T


OUTPUT_SUMMARY = Path("TrussRodriguez_monte_carlo_summary.csv")
DEFAULT_SAMPLES = 10_000
DEFAULT_SEED = 20261113
STEEPEST_AREAS = np.array(
    [0.1791889884925987,2.340502109572624,16.106470369420805,30.0,2.334866468437772,
     14.489835510828478,10.638385106912803,7.391595016091477,0.1,2.72489437366327,9.655736500952086],
#    [0.100000, 11.335279, 28.522638, 15.404083, 13.708046,
#     19.028415, 15.774317, 8.129115, 11.010426, 1.628092, 16.300988],
    dtype=float,
)
STEEPEST_HEIGHTS = np.array([253.4295736224215,117.25356577258349,21.93471430648415],
#    [171.791360, 126.655508, 5.013261], 
    dtype=float)
AREA_PERTURBATION = 0.04
HEIGHT_PERTURBATION = 0.2
LOAD_BOUNDS = (100.0, 110.0)
ALLOWABLE_STRESS_BOUNDS = (15.0, 15.75)  # kN/cm^2 = 150--157.5 MPa


def load_steepest_design() -> tuple[np.ndarray, np.ndarray]:
    """Return the captured steepest-descent design from the optimizer run."""
    return STEEPEST_AREAS.copy(), STEEPEST_HEIGHTS.copy()


def evaluate_scenario(
    areas: np.ndarray,
    heights: np.ndarray,
    load_magnitude: float,
) -> tuple[np.ndarray, float, float]:
    """Solve one load scenario and return stresses, mass, and cond(Kff)."""
    stiffness = T.assemble_global_stiffness(areas, shape_y=heights)
    kff = stiffness[np.ix_(T.FREE_DOFS, T.FREE_DOFS)]
    condition_number = float(np.linalg.cond(kff))
    loads = np.zeros(T.N_DOF, dtype=float)
    loads[T.dof_index(2)[1]] = -load_magnitude
    loads[T.dof_index(3)[1]] = -load_magnitude
    free_indices = np.asarray(T.FREE_DOFS, dtype=int)
    displacements = np.zeros(T.N_DOF, dtype=float)
    displacements[free_indices] = np.linalg.solve(kff, loads[free_indices])
    stresses = T.member_stresses(displacements, shape_y=heights)
    mass = T.mass_from_areas(areas, shape_y=heights)
    return stresses, float(mass), condition_number


def run_monte_carlo(sample_count: int, seed: int) -> dict[str, object]:
    """Run the robustness study and return aggregate results."""
    base_areas, base_heights = load_steepest_design()
    rng = np.random.default_rng(seed)
    passed = 0
    condition_passed = 0
    stress_passed = 0
    area_bound_passed = 0
    height_bound_passed = 0
    max_stress_values: list[float] = []
    condition_values: list[float] = []
    mass_values: list[float] = []
    records: list[dict[str, float | bool]] = []

    for _ in range(sample_count):
        areas = base_areas + rng.uniform(-AREA_PERTURBATION, AREA_PERTURBATION, T.N_ELEMS)
        heights = base_heights + rng.uniform(-HEIGHT_PERTURBATION, HEIGHT_PERTURBATION, 3)
        load = rng.uniform(*LOAD_BOUNDS)
        allowable_stress = rng.uniform(*ALLOWABLE_STRESS_BOUNDS)

        area_ok = bool(np.all((areas >= T.A_MIN) & (areas <= T.A_MAX)))
        height_ok = bool(np.all((heights >= 5.0) & (heights <= 300.0)))
        stresses, mass, condition_number = evaluate_scenario(areas, heights, load)
        max_stress = float(np.max(np.abs(stresses)))
        condition_ok = bool(np.isfinite(condition_number) and condition_number <= 1.0e8)
        stress_ok = bool(max_stress <= allowable_stress)
        feasible = area_ok and height_ok and condition_ok and stress_ok

        area_bound_passed += area_ok
        height_bound_passed += height_ok
        condition_passed += condition_ok
        stress_passed += stress_ok
        passed += feasible
        max_stress_values.append(max_stress)
        condition_values.append(condition_number)
        mass_values.append(mass)
        records.append(
            {
                "load_kN": load,
                "allowable_stress_kN_cm2": allowable_stress,
                "max_stress_kN_cm2": max_stress,
                "mass_kg": mass,
                "condition_number": condition_number,
                "area_bounds_ok": area_ok,
                "height_bounds_ok": height_ok,
                "condition_ok": condition_ok,
                "stress_ok": stress_ok,
                "feasible": feasible,
            }
        )

    return {
        "base_areas": base_areas,
        "base_heights": base_heights,
        "passed": passed,
        "sample_count": sample_count,
        "area_bound_passed": area_bound_passed,
        "height_bound_passed": height_bound_passed,
        "condition_passed": condition_passed,
        "stress_passed": stress_passed,
        "max_stress_min": min(max_stress_values),
        "max_stress_max": max(max_stress_values),
        "condition_max": max(condition_values),
        "mass_min": min(mass_values),
        "mass_max": max(mass_values),
        "records": records,
    }


def write_summary(result: dict[str, object], output_path: Path = OUTPUT_SUMMARY) -> None:
    """Write per-sample Monte Carlo outcomes to CSV."""
    records = result["records"]
    if not records:
        return
    with output_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo robustness study around the PSO design")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_SUMMARY)
    args = parser.parse_args()

    result = run_monte_carlo(args.samples, args.seed)
    write_summary(result, args.output)
    print(f"Samples: {result['sample_count']}")
    print(f"Seed: {args.seed}")
    print(f"Perturbations: areas +/-{AREA_PERTURBATION:g} cm^2; heights +/-{HEIGHT_PERTURBATION:g} cm")
    print("Baseline: steepest-descent design")
    print("Load range: 100--110 kN")
    print("Allowable stress range: 150--157.5 MPa")
    print(f"Passing designs: {result['passed']} / {result['sample_count']}")
    print(f"Pass rate: {100.0 * result['passed'] / result['sample_count']:.3f}%")
    print(f"Area-bound passes: {result['area_bound_passed']} / {result['sample_count']}")
    print(f"Height-bound passes: {result['height_bound_passed']} / {result['sample_count']}")
    print(f"Conditioning passes: {result['condition_passed']} / {result['sample_count']}")
    print(f"Stress passes: {result['stress_passed']} / {result['sample_count']}")
    print(f"Maximum sampled stress range: {result['max_stress_min']:.6f}--{result['max_stress_max']:.6f} kN/cm^2")
    print(f"Maximum sampled cond(Kff): {result['condition_max']:.6e}")
    print(f"Saved samples: {args.output.resolve()}")


if __name__ == "__main__":
    main()
