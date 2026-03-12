from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from landscape_core import analyze_problem
from problem_adapters import get_all_truss_problems


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run comparative landscape analysis for all truss problems")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=str, default="TrussLandscapeAnalysis/results")
    p.add_argument("--n-ref", type=int, default=600)
    p.add_argument("--walk-steps", type=int, default=360)
    p.add_argument("--lon-starts", type=int, default=30)
    p.add_argument("--basin-grid", type=int, default=24)
    p.add_argument(
        "--problems",
        type=str,
        default="",
        help="Comma-separated subset of problem IDs to run (e.g., truss72_discrete).",
    )
    return p.parse_args()


def _comparative_plots(out_dir: Path, metrics: list[dict]) -> None:
    labels = [m["label"] for m in metrics]
    mm = [m["classification_multimodal_score"] for m in metrics]
    ns = [m["classification_narrow_score"] for m in metrics]
    sm = [m["classification_smooth_score"] for m in metrics]
    acl = [m["autocorrelation_length"] for m in metrics]
    lon = [m["lon_nodes"] for m in metrics]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width, mm, width=width, label="Multimodal score")
    plt.bar(x, sm, width=width, label="Smooth score")
    plt.bar(x + width, ns, width=width, label="Narrow score")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Landscape Classification Scores Across Truss Problems")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_landscape_scores.png", dpi=170)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(x, acl, "o-", color="tab:blue", label="Autocorrelation length")
    ax2.plot(x, lon, "s--", color="tab:red", label="LON nodes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Autocorrelation length", color="tab:blue")
    ax2.set_ylabel("LON node count", color="tab:red")
    ax1.set_title("Ruggedness vs Local-Optima Count")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_ruggedness_lon.png", dpi=170)
    plt.close()


def _write_comparative_report(out_dir: Path, metrics: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Comparative Truss Objective Landscape Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("This report compares objective landscapes for all truss problems in the repository using:")
    lines.append("- Autocorrelation length")
    lines.append("- Information content")
    lines.append("- Local Optima Network (LON) structure")
    lines.append("- Basin-of-attraction mapping")
    lines.append("- Smoothness and narrow-basin diagnostics")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Problem | Class | AC Length | H(eps=0.05) | LON Nodes | LON Density | Basin Width Median | Recommended (w,c1,c2) | Swarm Size | Iterations |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for m in metrics:
        rec = m["pso_recommendation"]["recommended"]
        sw = m["pso_recommendation"].get("recommended_swarm_size", "—")
        ni = m["pso_recommendation"].get("recommended_iters", "—")
        lines.append(
            f"| {m['label']} | {m['classification_labels']} | {m['autocorrelation_length']:.3f} | "
            f"{m['information_content_H_eps005']:.3f} | {m['lon_nodes']} | {m['lon_edge_density']:.3f} | "
            f"{m['basin_width_median_norm']:.4f} | ({rec['w']:.3f}, {rec['c1']:.3f}, {rec['c2']:.3f}) | {sw} | {ni} |"
        )
    lines.append("")

    lines.append("## Per-Problem Interpretation and PSO Settings")
    lines.append("")
    for m in metrics:
        rec = m["pso_recommendation"]
        lines.append(f"### {m['label']}")
        lines.append(f"- Landscape class: **{m['classification_labels']}**")
        lines.append(
            f"- Recommended fixed coefficients: **w={rec['recommended']['w']:.3f}, "
            f"c1={rec['recommended']['c1']:.3f}, c2={rec['recommended']['c2']:.3f}**"
        )
        sw = rec.get("recommended_swarm_size", "—")
        ni = rec.get("recommended_iters", "—")
        lines.append(f"- Recommended swarm size: **{sw}**")
        lines.append(f"- Recommended iterations: **{ni}**")
        lines.append(
            f"- 2-phase schedule: phase-1(w={rec['schedule']['phase_1_explore']['w']:.3f}, "
            f"c1={rec['schedule']['phase_1_explore']['c1']:.3f}, c2={rec['schedule']['phase_1_explore']['c2']:.3f}) -> "
            f"phase-2(w={rec['schedule']['phase_2_refine']['w']:.3f}, c1={rec['schedule']['phase_2_refine']['c1']:.3f}, "
            f"c2={rec['schedule']['phase_2_refine']['c2']:.3f})"
        )
        lines.append("- Rationale:")
        for reason in rec["rationale"]:
            lines.append(f"  - {reason}")
        lines.append("")

    lines.append("## Cross-Problem Takeaways")
    lines.append("- Problems with higher LON node counts and higher information content are more multimodal/rugged.")
    lines.append("- Problems with smaller normalized basin width benefit from lower inertia and stronger local refinement.")
    lines.append("- Suggested default when uncertain: w=0.62, c1=1.35, c2=1.65; then adapt per problem diagnostics.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("- Per-problem reports and metrics are in `results/<problem_id>/`.")
    lines.append("- Comparative figures:")
    lines.append("  - `comparison_landscape_scores.png`")
    lines.append("  - `comparison_ruggedness_lon.png`")

    (out_dir / "comparative_landscape_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = get_all_truss_problems()
    if args.problems.strip():
        selected = {x.strip() for x in args.problems.split(",") if x.strip()}
        problems = [p for p in problems if p.problem_id in selected]
        if not problems:
            raise ValueError("No matching problems for --problems filter.")
    all_metrics: list[dict] = []

    for problem in problems:
        p_out = out_dir / problem.problem_id
        if len(problem.lo) >= 16:
            walk_steps = min(args.walk_steps, 220)
            lon_starts = min(args.lon_starts, 14)
            lon_perturb = 2
            basin_grid = min(args.basin_grid, 12)
        else:
            walk_steps = args.walk_steps
            lon_starts = args.lon_starts
            lon_perturb = 4
            basin_grid = args.basin_grid

        metrics = analyze_problem(
            problem=problem,
            out_dir=p_out,
            seed=args.seed,
            n_ref=args.n_ref,
            walk_steps=walk_steps,
            walk_step_frac=0.03,
            lon_starts=lon_starts,
            lon_perturb=lon_perturb,
            basin_grid=basin_grid,
        )
        all_metrics.append(metrics)
        print(
            f"{problem.problem_id}: {metrics['classification_labels']} | "
            f"rec=({metrics['pso_recommendation']['recommended']['w']:.3f},"
            f"{metrics['pso_recommendation']['recommended']['c1']:.3f},"
            f"{metrics['pso_recommendation']['recommended']['c2']:.3f})"
        )

    (out_dir / "all_landscape_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    _comparative_plots(out_dir, all_metrics)
    _write_comparative_report(out_dir, all_metrics)

    print("Comparative landscape analysis complete")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
