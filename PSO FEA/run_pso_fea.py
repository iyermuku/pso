from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_pso import run_fixed_coeff_pso
from problem_adapters import get_problem, list_problem_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common PSO-FEA runner for truss problems")
    parser.add_argument("--problem", type=str, required=True, choices=list_problem_ids())
    parser.add_argument(
        "--swarm-size",
        type=int,
        default=None,
        help="Swarm size (default: landscape-recommended value for the problem)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of iterations (default: landscape-recommended value for the problem)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="PSO FEA/results")
    parser.add_argument("--coeff-mode", type=str, default="fixed", choices=["fixed", "two-phase"])
    parser.add_argument(
        "--seed-optima-pct",
        type=float,
        default=0.0,
        help="Use up to this percent of swarm as seeded particles from remembered detected optima",
    )
    return parser.parse_args()


def save_convergence_plots(result: dict, out_dir: Path) -> None:
    obj_hist = np.asarray(result["objective_history"])
    mass_hist = np.asarray(result["mass_history"])
    cv_hist = np.asarray(result["cv_history"])
    feas_hist = np.asarray(result["feasible_fraction_history"])
    prefix = result["problem_id"]

    plt.figure(figsize=(9, 5))
    plt.plot(obj_hist, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best penalized objective")
    plt.title(f"{result['label']} - Objective Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_objective_convergence.png", dpi=170)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(mass_hist, color="tab:blue", linewidth=2, label="Best mass")
    ax2.plot(feas_hist, color="tab:green", linestyle="--", linewidth=2, label="Feasible fraction")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best mass", color="tab:blue")
    ax2.set_ylabel("Feasible fraction", color="tab:green")
    ax1.set_title(f"{result['label']} - Mass and Feasibility")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_mass_feasibility.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(cv_hist, linewidth=2, color="tab:red")
    plt.xlabel("Iteration")
    plt.ylabel("Best constraint violation")
    plt.title(f"{result['label']} - Constraint Violation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_constraint_violation.png", dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    problem = get_problem(args.problem)
    t0 = time.perf_counter()
    result = run_fixed_coeff_pso(
        problem=problem,
        swarm_size=args.swarm_size,
        iters=args.iters,
        seed=args.seed,
        v_frac=0.20,
        reflection_on_violation=True,
        coeff_mode=args.coeff_mode,
        seed_optima_pct=args.seed_optima_pct,
    )
    pso_runtime_seconds = float(time.perf_counter() - t0)

    out_dir = out_root / problem.problem_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "problem_id": result["problem_id"],
        "label": result["label"],
        "recommended_coefficients": result["recommended_coefficients"],
        "coefficient_mode": result["coefficient_mode"],
        "recommended_schedule": result["recommended_schedule"],
        "swarm_size": result["swarm_size"],
        "iters": result["iters"],
        "seed_optima_pct": result["seed_optima_pct"],
        "seeded_particles_requested_max": result["seeded_particles_requested_max"],
        "seeded_particles_count": result["seeded_particles_count"],
        "seeded_particles": result["seeded_particles"],
        "gbest_particle_name": result["gbest_particle_name"],
        "seeded_particle_reached_gbest": result["seeded_particle_reached_gbest"],
        "pso_runtime_seconds": pso_runtime_seconds,
        "design_variables": result["best_design_variables"].tolist(),
        "best_mass": float(result["gbest_mass"]),
        "max_displacement": float(result["best_max_disp"]),
        "max_stress": float(result["best_max_stress"]),
        "final_feasible_fraction": float(result["final_feasible_fraction"]),
    }
    (out_dir / f"{problem.problem_id}_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    save_convergence_plots(result, out_dir)

    print(problem.label)
    print(f"Coefficient mode: {args.coeff_mode}")
    print(f"Recommended coeffs (phase-1/fixed): w={result['recommended_coefficients']['w']:.3f}, c1={result['recommended_coefficients']['c1']:.3f}, c2={result['recommended_coefficients']['c2']:.3f}")
    if args.coeff_mode == "two-phase":
        phase2 = result["recommended_schedule"]["phase_2_refine"]
        print(f"Phase-2 coeffs: w={phase2['w']:.3f}, c1={phase2['c1']:.3f}, c2={phase2['c2']:.3f}")
    swarm_src = "user-specified" if args.swarm_size is not None else "landscape-recommended"
    iters_src = "user-specified" if args.iters is not None else "landscape-recommended"
    print(f"Swarm size: {result['swarm_size']} ({swarm_src})")
    print(f"Iterations: {result['iters']} ({iters_src})")
    print(
        f"Seeded optima: {result['seeded_particles_count']} particles "
        f"(requested up to {result['seeded_particles_requested_max']} from {result['seed_optima_pct']:.2f}% of swarm)"
    )
    print(f"Final gbest particle: {result['gbest_particle_name']}")
    print(f"Seeded particle reached gbest: {result['seeded_particle_reached_gbest']}")
    print(f"PSO runtime: {pso_runtime_seconds:.2f} s")
    print(f"Best mass: {result['gbest_mass']:.6f}")
    print(f"Max displacement (best design): {result['best_max_disp']:.6f}")
    print(f"Max stress (best design): {result['best_max_stress']:.6f}")
    print(f"Design variables: {np.array2string(result['best_design_variables'], precision=4, separator=', ')}")
    print(f"Final feasible fraction: {result['final_feasible_fraction']:.3f}")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
