#!/usr/bin/env python3
"""Main script for PSO on 72-bar truss.
Usage examples:
    python3 pso72_main.py --mode robust --iters 500 --swarm 40 --seeds 5
    python3 pso72_main.py --mode single --iters 600 --swarm 60 --seeds 3

Produces:
 - INFO logs to console, DEBUG logs to pso72_debug.log
 - In single mode, saves a plot of w,c1,c2 history of the best seed: w_c_history.png
 - Also saves a plot of mass history (gbest per iteration) with threshold markers at 80%, 95%, 99.7% progress.
 - Prints final text summary: max displacement, max member stress, and areas
"""
import argparse
import logging
import sys
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pso72


def setup_logging():
    logger = logging.getLogger('pso72')
    logger.setLevel(logging.DEBUG)
    # Console handler at INFO
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)
    # File handler at DEBUG
    fh = logging.FileHandler('pso72_debug.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)
    return logger


def plot_param_history(w_hist, c1_hist, c2_hist, out_path='w_c_history.png'):
    iters = range(1, len(w_hist)+1)
    plt.figure(figsize=(8,5))
    plt.plot(iters, w_hist, label='w (inertia weight)')
    plt.plot(iters, c1_hist, label='c1 (cognitive)')
    plt.plot(iters, c2_hist, label='c2 (social)')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title('Best particle parameter history (single-run mode)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_mass_history(mass_hist, out_path='mass_history.png'):
    """Plot best-particle mass history and mark thresholds at 80%, 95%, 99.7% of improvement toward lowest mass.
    Thresholds are computed relative to initial mass m0 and final minimum m_min:
        progress(t) = (m0 - m(t)) / (m0 - m_min)
    We mark the earliest iteration where progress >= {0.80, 0.95, 0.997}.
    """
    import numpy as np
    iters = np.arange(1, len(mass_hist)+1)
    m = np.array(mass_hist, dtype=float)
    m0 = m[0]
    m_min = float(np.min(m))
    denom = max(1e-12, (m0 - m_min))
    progress = (m0 - m) / denom

    # Thresholds
    thr_vals = [0.80, 0.95, 0.997]
    thr_labels = ['80%', '95%', '99.7%']
    hit_iters = []
    for tv in thr_vals:
        idx = np.argmax(progress >= tv)
        if progress[idx] >= tv:
            hit_iters.append(int(idx+1))
        else:
            hit_iters.append(None)

    plt.figure(figsize=(9,5))
    plt.plot(iters, m, label='gbest mass', color='#1f77b4')
    plt.xlabel('Iteration')
    plt.ylabel('Mass (lbm)')
    plt.title('Best particle mass history')
    plt.grid(True, alpha=0.3)
    # Mark thresholds
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    for hit, lab, col in zip(hit_iters, thr_labels, colors):
        if hit is not None:
            plt.axvline(hit, color=col, linestyle='--', alpha=0.6)
            plt.scatter([hit], [m[hit-1]], color=col, zorder=5)
            plt.text(hit, m[hit-1], f" {lab}", color=col, va='bottom', ha='left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def final_text_summary(metrics, areas, out_path='pso72_best_summary.txt'):
    lines = []
    lines.append(f"Max displacement: {metrics['max_disp']:.6f} in (allow eff {metrics['disp_allow_eff']:.6f} in)")
    lines.append(f"Max member stress: {metrics['max_stress']:.6f} ksi (allow eff {metrics['stress_allow_eff']:.6f} ksi)")
    lines.append(f"Computed mass: {metrics['mass']:.4f} lbm")
    lines.append("Final 16 area group values (in^2):")
    for i, a in enumerate(areas, start=1):
        lines.append(f"  A{str(i).zfill(2)} = {a:.6f}")
    text = "".join(lines)
    with open(out_path, 'w') as f:
        f.write(text + "")
    return text


def main():
    parser = argparse.ArgumentParser(description='PSO for 72-bar truss (robust constriction or single-run meta search).')
    parser.add_argument('--mode', choices=['robust', 'single'], default='robust', help='robust: 16 areas with constriction; single: 19 vars (areas + w,c1,c2)')
    parser.add_argument('--iters', type=int, default=500, help='iterations per seed')
    parser.add_argument('--swarm', type=int, default=40, help='swarm size per seed')
    parser.add_argument('--seeds', type=int, default=3, help='number of seeds')
    parser.add_argument('--c1', type=float, default=2.05, help='c1 for robust mode')
    parser.add_argument('--c2', type=float, default=2.05, help='c2 for robust mode')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Starting PSO mode={args.mode} iters={args.iters} swarm={args.swarm} seeds={args.seeds}")

    best_overall = None
    best_seed_idx = -1

    for s in range(args.seeds):
        if args.mode == 'robust':
            result = pso72.robust_constriction_pso(seed=s, iters=args.iters, swarm=args.swarm,
                                                   c1=args.c1, c2=args.c2, logger=logger)
        else:
            result = pso72.single_run_pso(seed=s, iters=args.iters, swarm=args.swarm, logger=logger)
        if (best_overall is None) or (result['gbest_J'] < best_overall['gbest_J']):
            best_overall = result
            best_seed_idx = s

    # Output summary and plots
    metrics = best_overall['metrics']
    areas = best_overall['gbest_X'][:16]
    summary_text = final_text_summary(metrics, areas)
    logger.info("=== FINAL BEST SUMMARY ===" + summary_text)

    # Mass history plot for best seed
    mass_hist = best_overall.get('mass_hist', [])
    if mass_hist:
        plot_mass_history(mass_hist)
        logger.info(f"Saved mass history plot: mass_history.png (best seed={best_seed_idx})")

    # Parameter history plot for single mode
    if args.mode == 'single':
        w_hist = best_overall['w_hist']
        c1_hist = best_overall['c1_hist']
        c2_hist = best_overall['c2_hist']
        plot_param_history(w_hist, c1_hist, c2_hist)
        logger.info(f"Saved parameter history plot: w_c_history.png (best seed={best_seed_idx})")

    logger.info("Done.")


if __name__ == '__main__':
    main()
