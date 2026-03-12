from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import the recommendation function so we can patch old JSON files.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from landscape_core import recommend_pso_coefficients


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _patch_swarm_iter(m: dict) -> dict:
    """Add recommended_swarm_size / recommended_iters to pso_recommendation if absent."""
    rec = m.get("pso_recommendation", {})
    if "recommended_swarm_size" in rec and "recommended_iters" in rec:
        return m  # already up to date

    n_dim_lookup = {
        "truss10_continuous": 10,
        "truss10_discrete": 10,
        "truss72_continuous": 16,
        "truss72_discrete": 16,
    }
    n_dim = n_dim_lookup.get(m.get("problem_id", ""), 10)

    classes = {
        "labels": [lbl.strip() for lbl in m.get("classification_labels", "").split(",") if lbl.strip()],
        "multimodal_score": m.get("classification_multimodal_score", 0),
        "smooth_score": m.get("classification_smooth_score", 0),
        "narrow_score": m.get("classification_narrow_score", 0),
        "summary": m.get("classification_summary", ""),
    }
    narrow = {
        "basin_width_median_norm": m.get("basin_width_median_norm", 0.05),
        "basin_width_mean_norm": m.get("basin_width_mean_norm", 0.05),
        "basin_width_q10_norm": m.get("basin_width_q10_norm", 0.01),
        "rise_abs": m.get("rise_abs", 100.0),
    }
    new_rec = recommend_pso_coefficients(
        classes=classes,
        ac_len=m.get("autocorrelation_length", 5.0),
        info_h=m.get("information_content_H_eps005", 0.5),
        lon_nodes=m.get("lon_nodes", 10),
        lon_density=m.get("lon_edge_density", 0.1),
        narrow=narrow,
        n_dim=n_dim,
    )
    # Preserve existing w/c1/c2 from the JSON; only inject new fields.
    rec["recommended_swarm_size"] = new_rec["recommended_swarm_size"]
    rec["recommended_iters"] = new_rec["recommended_iters"]
    # Append swarm/iter rationale lines that are new.
    existing_rationale = rec.get("rationale", [])
    for line in new_rec["rationale"]:
        if line not in existing_rationale:
            existing_rationale.append(line)
    rec["rationale"] = existing_rationale
    m["pso_recommendation"] = rec
    return m


def load_metrics() -> list[dict]:
    names = [
        "truss10_continuous/truss10_continuous_landscape_metrics.json",
        "truss10_discrete/truss10_discrete_landscape_metrics.json",
        "truss72_continuous/truss72_continuous_landscape_metrics.json",
        "truss72_discrete/truss72_discrete_landscape_metrics.json",
    ]
    data = []
    for rel in names:
        path = RESULTS / rel
        m = json.loads(path.read_text(encoding="utf-8"))
        m = _patch_swarm_iter(m)
        # Write patched version back so PSO FEA can read the new fields.
        path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        data.append(m)
    return data


def comparative_plots(metrics: list[dict]) -> None:
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
    plt.savefig(RESULTS / "comparison_landscape_scores.png", dpi=170)
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
    plt.savefig(RESULTS / "comparison_ruggedness_lon.png", dpi=170)
    plt.close()


def write_report(metrics: list[dict]) -> None:
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
    lines.append("## Relative Interpretation")
    lines.append("")
    lines.append("- **Most rugged / locally fragmented**: 10-bar continuous and 10-bar discrete, due to high LON node counts and high information content.")
    lines.append("- **Sharpest/narrowest basin**: 10-bar continuous, with the smallest normalized basin width.")
    lines.append("- **Smoothest macro-scale landscape**: 72-bar continuous, with extremely long autocorrelation length.")
    lines.append("- **Discrete regularization effect**: both discrete formulations are less narrow than their continuous counterparts.")
    lines.append("")
    lines.append("## Per-Problem Interpretation and PSO Settings")
    lines.append("")
    for m in metrics:
        rec = m["pso_recommendation"]
        lines.append(f"### {m['label']}")
        lines.append(f"- Landscape class: **{m['classification_labels']}**")
        lines.append(
            f"- Recommended fixed coefficients: **w={rec['recommended']['w']:.3f}, c1={rec['recommended']['c1']:.3f}, c2={rec['recommended']['c2']:.3f}**"
        )
        sw = rec.get("recommended_swarm_size", "—")
        ni = rec.get("recommended_iters", "—")
        lines.append(f"- Recommended swarm size: **{sw}**")
        lines.append(f"- Recommended iterations: **{ni}**")
        lines.append(
            f"- 2-phase schedule: phase-1(w={rec['schedule']['phase_1_explore']['w']:.3f}, "
            f"c1={rec['schedule']['phase_1_explore']['c1']:.3f}, c2={rec['schedule']['phase_1_explore']['c2']:.3f}) -> "
            f"phase-2(w={rec['schedule']['phase_2_refine']['w']:.3f}, c1={rec['schedule']['phase_2_refine']['c1']:.3f}, c2={rec['schedule']['phase_2_refine']['c2']:.3f})"
        )
        lines.append("- Rationale:")
        for reason in rec["rationale"]:
            lines.append(f"  - {reason}")
        lines.append("")
    lines.append("## Cross-Problem Takeaways")
    lines.append("- 10-bar problems favor stronger damping or refinement because of sharper local basins.")
    lines.append("- 72-bar problems can tolerate larger inertia because their macro-landscape is smoother.")
    lines.append("- All four truss problems show multimodality, so `c2 > c1` is consistently preferred.")
    lines.append("- Suggested global default across truss problems: `w=0.62, c1=1.30, c2=1.70`.")
    (RESULTS / "comparative_landscape_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metrics = load_metrics()
    (RESULTS / "all_landscape_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparative_plots(metrics)
    write_report(metrics)
    print("Rebuilt comparative report from per-problem metrics")


if __name__ == "__main__":
    main()
