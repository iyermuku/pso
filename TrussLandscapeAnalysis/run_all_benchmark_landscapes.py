from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_adapters import get_all_benchmark_problems
from landscape_core import analyze_problem


_WORKER_PROBLEMS: dict[str, object] | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run comparative landscape analysis for benchmark functions")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=str, default="TrussLandscapeAnalysis/benchmark_results")
    p.add_argument("--n-ref", type=int, default=220)
    p.add_argument("--walk-steps", type=int, default=240)
    p.add_argument("--lon-starts", type=int, default=18)
    p.add_argument("--basin-grid", type=int, default=18)
    p.add_argument("--jobs", type=int, default=1, help="Number of worker processes (1 = sequential, 0 = auto cpu count).")
    p.add_argument(
        "--threads-per-problem",
        type=int,
        default=0,
        help="Number of threads used within each benchmark landscape analysis (0 = auto).",
    )
    p.add_argument(
        "--problems",
        type=str,
        default="",
        help="Comma-separated subset of problem IDs to run (e.g., bench01_sphere,bench34_easom).",
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

    plt.figure(figsize=(13, 5.5))
    plt.bar(x - width, mm, width=width, label="Multimodal score")
    plt.bar(x, sm, width=width, label="Smooth score")
    plt.bar(x + width, ns, width=width, label="Narrow score")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Landscape Classification Scores Across Benchmark Functions")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_landscape_scores.png", dpi=170)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(x, acl, "o-", color="tab:blue", label="Autocorrelation length")
    ax2.plot(x, lon, "s--", color="tab:red", label="LON nodes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Autocorrelation length", color="tab:blue")
    ax2.set_ylabel("LON node count", color="tab:red")
    ax1.set_title("Ruggedness vs Local-Optima Count")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_ruggedness_lon.png", dpi=170)
    plt.close()


def _write_comparative_report(out_dir: Path, metrics: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Comparative Benchmark Objective Landscape Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("This report compares objective landscapes for 36 benchmark functions using:")
    lines.append("- Autocorrelation length")
    lines.append("- Information content")
    lines.append("- Local Optima Network (LON) structure")
    lines.append("- Basin-of-attraction mapping")
    lines.append("- Smoothness and narrow-basin diagnostics")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Problem | Class | AC Length | H(eps=0.05) | LON Nodes | LON Density | Basin Width Median | Time (s) | Cache Hit % | Recommended (w,c1,c2) | Swarm Size | Iterations |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for m in metrics:
        rec = m["pso_recommendation"]["recommended"]
        sw = m["pso_recommendation"].get("recommended_swarm_size", "—")
        ni = m["pso_recommendation"].get("recommended_iters", "—")
        t_sec = float(m.get("analysis_runtime_seconds", 0.0))
        cache_hit_pct = 100.0 * float(m.get("cache_hit_rate", 0.0))
        lines.append(
            f"| {m['label']} | {m['classification_labels']} | {m['autocorrelation_length']:.3f} | "
            f"{m['information_content_H_eps005']:.3f} | {m['lon_nodes']} | {m['lon_edge_density']:.3f} | "
            f"{m['basin_width_median_norm']:.4f} | {t_sec:.2f} | {cache_hit_pct:.1f} | "
            f"({rec['w']:.3f}, {rec['c1']:.3f}, {rec['c2']:.3f}) | {sw} | {ni} |"
        )
    lines.append("")
    lines.append("## Per-Problem Interpretation and PSO Settings")
    lines.append("")
    for m in metrics:
        rec = m["pso_recommendation"]
        lines.append(f"### {m['label']}")
        lines.append(f"- Landscape class: **{m['classification_labels']}**")
        lines.append(f"- Recommended fixed coefficients: **w={rec['recommended']['w']:.3f}, c1={rec['recommended']['c1']:.3f}, c2={rec['recommended']['c2']:.3f}**")
        sw = rec.get("recommended_swarm_size", "—")
        ni = rec.get("recommended_iters", "—")
        lines.append(f"- Recommended swarm size: **{sw}**")
        lines.append(f"- Recommended iterations: **{ni}**")
        lines.append(
            f"- 2-phase schedule: phase-1(w={rec['schedule']['phase_1_explore']['w']:.3f}, c1={rec['schedule']['phase_1_explore']['c1']:.3f}, c2={rec['schedule']['phase_1_explore']['c2']:.3f}) -> "
            f"phase-2(w={rec['schedule']['phase_2_refine']['w']:.3f}, c1={rec['schedule']['phase_2_refine']['c1']:.3f}, c2={rec['schedule']['phase_2_refine']['c2']:.3f})"
        )
        lines.append("- Rationale:")
        for reason in rec["rationale"]:
            lines.append(f"  - {reason}")
        lines.append("")

    multimodal = sum(1 for m in metrics if "multimodal" in m["classification_labels"])
    smooth = sum(1 for m in metrics if "smooth-macro" in m["classification_labels"])
    narrow = sum(1 for m in metrics if "narrow-basin" in m["classification_labels"])
    lines.append("## Cross-Function Takeaways")
    lines.append(f"- {multimodal}/36 functions were classified as multimodal.")
    lines.append(f"- {smooth}/36 functions were classified as smooth-macro.")
    lines.append(f"- {narrow}/36 functions were classified as narrow-basin.")
    lines.append("- Multimodal problems generally push the recommendation toward higher social pressure.")
    lines.append("- Narrow basins generally reduce inertia and increase cognitive guidance.")
    lines.append("- Smooth unimodal problems usually get a moderate inertia plus a more balanced cognitive/social split.")
    lines.append("- Suggested baseline for unknown benchmark functions: w=0.62, c1=1.35, c2=1.60.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("- Per-problem reports and metrics are in `results/<problem_id>/`.")
    lines.append("- Comparative figures:")
    lines.append("  - `benchmark_landscape_scores.png`")
    lines.append("  - `benchmark_ruggedness_lon.png`")

    (out_dir / "comparative_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def _resolve_threads_per_problem(requested_threads: int, jobs: int) -> int:
    if requested_threads < 0:
        raise ValueError("--threads-per-problem must be >= 0")
    if requested_threads > 0:
        return requested_threads
    cpu = max(os.cpu_count() or 1, 1)
    return max(cpu // max(jobs, 1), 1)


def _build_task_for_problem(problem, args: argparse.Namespace, out_dir: Path, threads_per_problem: int) -> dict:
    if len(problem.lo) >= 10:
        walk_steps = min(args.walk_steps, 180)
        lon_starts = min(args.lon_starts, 12)
        basin_grid = min(args.basin_grid, 12)
        lon_perturb = 3
    else:
        walk_steps = args.walk_steps
        lon_starts = args.lon_starts
        basin_grid = args.basin_grid
        lon_perturb = 4

    return {
        "problem_id": problem.problem_id,
        "out_dir": str(out_dir / problem.problem_id),
        "seed": int(args.seed),
        "n_ref": int(args.n_ref),
        "walk_steps": int(walk_steps),
        "lon_starts": int(lon_starts),
        "lon_perturb": int(lon_perturb),
        "basin_grid": int(basin_grid),
        "threads_per_problem": int(threads_per_problem),
    }


def _get_worker_problems() -> dict[str, object]:
    global _WORKER_PROBLEMS
    if _WORKER_PROBLEMS is None:
        _WORKER_PROBLEMS = {p.problem_id: p for p in get_all_benchmark_problems()}
    return _WORKER_PROBLEMS


def _run_problem_task(task: dict) -> dict:
    problems = _get_worker_problems()
    problem_id = task["problem_id"]
    if problem_id not in problems:
        raise ValueError(f"Unknown problem id: {problem_id}")

    t0 = time.perf_counter()
    metrics = analyze_problem(
        problem=problems[problem_id],
        out_dir=Path(task["out_dir"]),
        seed=int(task["seed"]),
        n_ref=int(task["n_ref"]),
        walk_steps=int(task["walk_steps"]),
        walk_step_frac=0.03,
        lon_starts=int(task["lon_starts"]),
        lon_perturb=int(task["lon_perturb"]),
        basin_grid=int(task["basin_grid"]),
        n_threads=max(int(task.get("threads_per_problem", 1)), 1),
    )
    metrics["analysis_runtime_seconds"] = float(time.perf_counter() - t0)
    return metrics


def _print_problem_summary(metrics: dict) -> None:
    rec = metrics["pso_recommendation"]["recommended"]
    print(
        f"{metrics['problem_id']}: {metrics['classification_labels']} | "
        f"rec=({rec['w']:.3f},{rec['c1']:.3f},{rec['c2']:.3f}) | "
        f"time={float(metrics.get('analysis_runtime_seconds', 0.0)):.2f}s | "
        f"cache_hit={100.0 * float(metrics.get('cache_hit_rate', 0.0)):.1f}% | "
        f"threads={int(metrics.get('analysis_threads', 1))}"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = get_all_benchmark_problems()
    if args.problems.strip():
        selected = {x.strip() for x in args.problems.split(",") if x.strip()}
        problems = [p for p in problems if p.problem_id in selected]
        if not problems:
            raise ValueError("No matching problems for --problems filter.")

    jobs = int(args.jobs)
    if jobs == 0:
        jobs = max(os.cpu_count() or 1, 1)
    if jobs < 1:
        raise ValueError("--jobs must be >= 0")

    threads_per_problem = _resolve_threads_per_problem(args.threads_per_problem, jobs)

    tasks = []
    problem_order = [p.problem_id for p in problems]
    for problem in problems:
        tasks.append(_build_task_for_problem(problem, args, out_dir, threads_per_problem))

    t_all_start = time.perf_counter()
    all_metrics: list[dict] = []
    if jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            metrics = _run_problem_task(task)
            all_metrics.append(metrics)
            _print_problem_summary(metrics)
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_run_problem_task, task) for task in tasks]
            for fut in cf.as_completed(futures):
                metrics = fut.result()
                all_metrics.append(metrics)
                _print_problem_summary(metrics)

    order = {pid: i for i, pid in enumerate(problem_order)}
    all_metrics.sort(key=lambda m: order[m["problem_id"]])
    total_elapsed = float(time.perf_counter() - t_all_start)

    (out_dir / "all_benchmark_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    _comparative_plots(out_dir, all_metrics)
    _write_comparative_report(out_dir, all_metrics)

    if all_metrics:
        hit_rate = np.mean([float(m.get("cache_hit_rate", 0.0)) for m in all_metrics]) * 100.0
        unique = int(np.sum([int(m.get("cache_unique_evals", 0)) for m in all_metrics]))
        total_q = int(np.sum([int(m.get("cache_total_queries", 0)) for m in all_metrics]))
        print(f"Cache summary: avg_hit_rate={hit_rate:.1f}% | total_unique_evals={unique} | total_queries={total_q}")
    print(f"Total wall time: {total_elapsed:.2f}s (jobs={jobs})")
    print(f"Threading config: threads_per_problem={threads_per_problem}")
    print("Comparative benchmark landscape analysis complete")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
