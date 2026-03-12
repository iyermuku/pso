from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common_pso import run_fixed_coeff_pso
from problem_adapters import get_problem, list_problem_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for common PSO-FEA across truss problems")
    parser.add_argument(
        "--problems",
        type=str,
        default="all",
        help="Comma-separated problem IDs or 'all'",
    )
    parser.add_argument(
        "--swarm-size",
        type=int,
        default=None,
        help="Swarm size (default: landscape-recommended per problem)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of iterations (default: landscape-recommended per problem)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="PSO FEA/results")
    parser.add_argument("--coeff-mode", type=str, default="fixed", choices=["fixed", "two-phase"])
    return parser.parse_args()


def _resolve_problems(raw: str) -> list[str]:
    available = set(list_problem_ids())
    if raw.strip().lower() == "all":
        return list_problem_ids()
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in selected if x not in available]
    if unknown:
        raise ValueError(f"Unknown problems: {unknown}. Available: {sorted(available)}")
    return selected


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    problem_ids = _resolve_problems(args.problems)
    summaries = []

    for problem_id in problem_ids:
        problem = get_problem(problem_id)
        result = run_fixed_coeff_pso(
            problem=problem,
            swarm_size=args.swarm_size,
            iters=args.iters,
            seed=args.seed,
            v_frac=0.20,
            reflection_on_violation=True,
            coeff_mode=args.coeff_mode,
        )

        p_out = out_root / problem.problem_id
        p_out.mkdir(parents=True, exist_ok=True)

        payload = {
            "problem_id": result["problem_id"],
            "label": result["label"],
            "coefficient_mode": result["coefficient_mode"],
            "recommended_coefficients": result["recommended_coefficients"],
            "recommended_schedule": result["recommended_schedule"],
            "swarm_size": result["swarm_size"],
            "iters": result["iters"],
            "design_variables": result["best_design_variables"].tolist(),
            "best_mass": float(result["gbest_mass"]),
            "max_displacement": float(result["best_max_disp"]),
            "max_stress": float(result["best_max_stress"]),
            "final_feasible_fraction": float(result["final_feasible_fraction"]),
        }
        (p_out / f"{problem.problem_id}_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        summaries.append(
            {
                "problem_id": result["problem_id"],
                "label": result["label"],
                "coeff_mode": result["coefficient_mode"],
                "swarm_size": result["swarm_size"],
                "iters": result["iters"],
                "best_mass": float(result["gbest_mass"]),
                "max_displacement": float(result["best_max_disp"]),
                "max_stress": float(result["best_max_stress"]),
                "design_variables": result["best_design_variables"].tolist(),
                "final_feasible_fraction": float(result["final_feasible_fraction"]),
            }
        )

        print(
            f"{result['problem_id']}: mode={result['coefficient_mode']} | "
            f"mass={result['gbest_mass']:.6f} | max_disp={result['best_max_disp']:.6f} | max_stress={result['best_max_stress']:.6f}"
        )

    (out_root / "batch_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    md_lines = [
        "# PSO FEA Batch Summary",
        "",
        f"- coefficient mode: `{args.coeff_mode}`",
        f"- swarm size: `{'landscape-recommended (per problem)' if args.swarm_size is None else args.swarm_size}`",
        f"- iterations: `{'landscape-recommended (per problem)' if args.iters is None else args.iters}`",
        f"- seed: `{args.seed}`",
        "",
        "| Problem | Mode | Swarm | Iters | Best Mass | Max Disp | Max Stress | Final Feasible Fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        md_lines.append(
            f"| {row['label']} | {row['coeff_mode']} | {row['swarm_size']} | {row['iters']} | {row['best_mass']:.6f} | "
            f"{row['max_displacement']:.6f} | {row['max_stress']:.6f} | {row['final_feasible_fraction']:.3f} |"
        )
        md_lines.append(f"- Design variables ({row['problem_id']}): `{np.array2string(np.asarray(row['design_variables']), precision=4, separator=', ')}`")

    (out_root / "batch_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Batch outputs written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
