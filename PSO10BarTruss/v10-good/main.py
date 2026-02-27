
import logging
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse 
from distutils.util import strtobool

from pso import pso_best_of_seeds
from truss_model import U_ALLOW, member_stresses, solve_displacements, solve_stresses

# ------------------------------
# Logging configuration (verbose)
# ------------------------------
log_file = 'pso_run.log'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# File handler (verbose)
f_handler = logging.FileHandler(log_file, mode='w')
f_handler.setLevel(logging.DEBUG)
f_formatter = logging.Formatter(
    fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
f_handler.setFormatter(f_formatter)
logger.addHandler(f_handler)

# Console handler (info)
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setLevel(logging.INFO)
c_formatter = logging.Formatter('%(levelname)s %(message)s')
c_handler.setFormatter(c_formatter)
logger.addHandler(c_handler)

# ------------------------------
# Run and timestamp
# ------------------------------
parser = argparse.ArgumentParser(description='Run PSO and plot results (robust or single_run).')
parser.add_argument('--mode', choices=['robust','single'], default='robust', help='Select PSO mode')
parser.add_argument('--num_runs', type=int, default=25, help='Number of seed runs')
parser.add_argument('--iters', type=int, default=200, help='Iterations per run')
parser.add_argument('--swarm', type=int, default=60, help='Swarm size')
parser.add_argument('--max_restarts', type=int, default=2, help='Max restarts on stall')
parser.add_argument('--stall_window', type=int, default=20, help='Stall window before restart')
args = parser.parse_args()
mode = args.mode
start_ts = datetime.now()
logging.info(f"=== PSO ({'robust lbest' if mode=='robust' else 'single_run'}) with seeds & restarts: START ===")
logging.info("Start timestamp: %s", start_ts.strftime('%Y-%m-%d %H:%M:%S'))

# Parameters used for seed generation
base_seed = 2026

def map_to_bool(s):
    mapping = {'robust': True, 'single': False}
    return mapping.get(s.strip().lower(), True) # Default to False if unknown
runmode = map_to_bool(mode)
#breakpoint()
# Run seeds (enable robust mode)
# best, runs = pso_best_of_seeds(num_runs=25, swarm_size=60, iters=200, max_restarts=2,
                               # stall_window=20, base_seed=base_seed, robust=True)
best, runs = pso_best_of_seeds(args.num_runs, args.swarm, args.iters, args.max_restarts,
                               args.stall_window, base_seed, runmode==True)

# Select the BEST FEASIBLE across all seeds by lowest mass (subject to constraints)
best_feas_across = None
for k, res_k in enumerate(runs):
    # A run is considered feasible if it found at least one feasible solution
    if hasattr(res_k['best_feas_A'], 'size') and res_k['best_feas_A'].size > 0:
        score = (res_k['best_feas_mass'], res_k['best_feas_max_disp'])
        seed_k = base_seed + 97 * k
        if (best_feas_across is None) or (score < best_feas_across['score']):
            best_feas_across = {'score': score, 'res': res_k, 'seed': seed_k, 'run_index': k}

# Fallback: if none of the runs achieved feasibility, use the 'best' returned by pso_best_of_seeds
if best_feas_across is None:
    logging.warning("No feasible solution found across seeds. Reporting the selector's chosen run instead.")
    chosen = best
else:
    chosen = best_feas_across

particle = chosen['run_index'] + 1
res = chosen['res']
A_opt = res['gbest_A']  # optimal areas (in^2) from the chosen run

# Log the chosen run explicitly as BEST FEASIBLE across seeds (if feasible)
if res.get('best_feas_A', np.array([])).size > 0:
    # Compute max stress for the chosen feasible solution
    try:
        U_feas = solve_displacements(res['best_feas_A'])
        max_stress = float(np.max(np.abs(member_stresses(U_feas))))
    except Exception as e:
        max_stress = float('nan')
    logging.info(
        "=== BEST FEASIBLE ACROSS SEEDS === seed=%d particle=%d mass=%.6f lbm max_disp=%.6f in max_stress=%.6f ksi w=%.6f c1=%.6f c2=%.6f",
        chosen['seed'], particle, float(res['best_feas_mass']), float(res['best_feas_max_disp']), max_stress,
        float(res['best_feas_w']), float(res['best_feas_c1']), float(res['best_feas_c2'])
    )
    # ------------------------------
    # Plots (use the chosen run)
    # ------------------------------
    # plt.figure(figsize=(9, 5))
    # plt.plot(res['mass_hist'], label='Best feasible mass so far', color='C0')
    # plt.xlabel('Iteration'); plt.ylabel('Mass (lbm)')
    # plt.title('PSO ({mode}): Best feasible mass vs iteration (chosen run)')
    # plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    # plt.savefig('pso_best_mass_vs_iteration.png', dpi=150)
    plt.figure(figsize=(10, 6))
    mass_hist = np.array(res['mass_hist'], dtype=float)
    iters = np.arange(1, len(mass_hist)+1)
    plt.plot(iters, mass_hist, label=f'Best feasible weight (mass) for particle {particle}', color='C0', lw=2)

    # Compute final best weight (mass) and thresholds (ignore NaNs)
    valid = ~np.isnan(mass_hist)
    final_best = np.nanmin(mass_hist) if np.any(valid) else np.nan
    # thresholds at 68%, 95%, 99.7% of minimum weight
    thresholds = {
        '68%': final_best / 0.68,
        '95%': final_best / 0.95,
        '99.7%': final_best / 0.9997,
    } if np.isfinite(final_best) else {}

    # Draw horizontal lines for those percentages and mark first attainment
    first_hits = {}
    for label, thr in thresholds.items():
        color = {'68%':'#2ca02c','95%':'#ff7f0e','99.7%':'#d62728'}[label]
        plt.axhline(thr, color=color, linestyle='--', lw=1.5, label=f'{label} of best weight ({thr:.2f} lbm)')
        hit_idx = np.where(valid & (mass_hist < thr))[0]
        if hit_idx.size > 0:
            first = int(hit_idx[0])
            first_hits[label] = first+1  # iteration number (1-based)
            plt.axvline(first+1, color=color, alpha=0.3, lw=1)
            plt.scatter([first+1], [mass_hist[first]], color=color, zorder=5)
            plt.annotate(f'{label} at it={first+1}', xy=(first+1, mass_hist[first]), xytext=(5, 10),
                         textcoords='offset points', fontsize=9, color=color)

    plt.xlabel('Iteration'); plt.ylabel('Mass (lbm)')
    plt.title(f'Best feasible weight (mass) vs iteration (chosen run) for particle {particle}')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('pso_best_mass_vs_iteration.png', dpi=150)

    # Print a small console summary of threshold hits
    if thresholds:
        print('\nThreshold attainment (best feasible weight plot):')
        for label in thresholds.keys():
            if label in first_hits:
                print(f'  {label} reached at iteration {first_hits[label]}')
            else:
                print(f'  {label} not reached')


    plt.figure(figsize=(9, 5))
    plt.plot(res['disp_hist'], label='gbest max displacement', color='C1')
    plt.axhline(U_ALLOW, color='red', linestyle='--', linewidth=1.4, label=f'Constraint U \u2264 {U_ALLOW} in')
    plt.xlabel('Iteration'); plt.ylabel('Max displacement (in)')
    plt.title(f'Max displacement vs iteration (chosen run) for particle {particle}')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('pso_best_max_disp_vs_iteration.png', dpi=150)

    plt.figure(figsize=(9, 5))
    plt.plot(res['feas_frac_hist'], label='Swarm feasible fraction', color='C2')
    plt.plot(res['gbest_feas_hist'], label='gbest feasible (0/1)', color='C3', alpha=0.6)
    plt.xlabel('Iteration'); plt.ylabel('Feasible fraction / flag')
    plt.title(f'Feasibility over iterations (chosen run) for particle {particle}')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    # Additional plots: parameter evolution (single_run only)
    if mode == 'single' and all(k in res for k in ['w_hist','c1_hist','c2_hist']):
        iters_params = np.arange(1, len(res['w_hist'])+1)
        plt.figure(figsize=(10, 7))
        ax1 = plt.subplot(3,1,1)
        ax1.plot(iters_params, res['w_hist'], label='gbest w', color='C0')
        if 'w_mean_hist' in res: ax1.plot(iters_params, res['w_mean_hist'], label='swarm mean w', color='C0', linestyle='--', alpha=0.7)
        ax1.set_ylabel('w'); ax1.grid(True, alpha=0.3); ax1.legend()
        ax2 = plt.subplot(3,1,2)
        ax2.plot(iters_params, res['c1_hist'], label='gbest c1', color='C1')
        if 'c1_mean_hist' in res: ax2.plot(iters_params, res['c1_mean_hist'], label='swarm mean c1', color='C1', linestyle='--', alpha=0.7)
        ax2.set_ylabel('c1'); ax2.grid(True, alpha=0.3); ax2.legend()
        ax3 = plt.subplot(3,1,3)
        ax3.plot(iters_params, res['c2_hist'], label='gbest c2', color='C2')
        if 'c2_mean_hist' in res: ax3.plot(iters_params, res['c2_mean_hist'], label='swarm mean c2', color='C2', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Iteration'); ax3.set_ylabel('c2'); ax3.grid(True, alpha=0.3); ax3.legend()
        plt.suptitle(f'Parameter evolution vs iteration (chosen run) for particle {particle}')
        plt.tight_layout()
        plt.savefig('pso_single_params_vs_iteration.png', dpi=150)
        print('Additional plot saved: pso_single_params_vs_iteration.png')
        plt.savefig('pso_best_feasibility_history.png', dpi=150)
    # combined plot of w and displacement constraint over iterations
    if all(k in res for k in ['w_hist','disp_hist']):
        iters_all = np.arange(1, len(res['w_hist'])+1)
        figc, axw = plt.subplots(figsize=(9,5))
        axw.plot(iters_all, res['w_hist'], color='C0', label='gbest w')
        axw.set_xlabel('Iteration')
        axw.set_ylabel('w (inertia)', color='C0')
        axw.tick_params(axis='y', labelcolor='C0')
        axd = axw.twinx()
        axd.plot(iters_all, res['disp_hist'], color='C1', label='gbest max disp')
        axd.axhline(U_ALLOW, color='red', linestyle='--', linewidth=1.4)
        axd.set_ylabel('Max displacement (in)', color='C1')
        axd.tick_params(axis='y', labelcolor='C1')
        plt.title(f'Weight & displacement constraints vs iteration (particle {particle})')
        figc.tight_layout()
        figc.savefig('pso_weight_disp_vs_iteration.png', dpi=150)
        print('Additional plot saved: pso_weight_disp_vs_iteration.png')

    # ------------------------------
    # Console summary (chosen run)
    # ------------------------------
    print('=== PSO (robust lbest) with seeds & restarts ===')
    print(f"Chosen seed (best feasible across seeds): {chosen['seed']}")
    print(f"Best parameters (gbest snapshot): w = {res['gbest_w']:.4f}, c1 = {res['gbest_c1']:.4f}, c2 = {res['gbest_c2']:.4f}")
    print(f"Best mass (gbest snapshot): {res['gbest_mass']:.2f} lbm")
    print(f"Best max U (gbest snapshot): {res['gbest_max_disp']:.6f} in -> {'FEASIBLE' if res['gbest_max_disp']<=U_ALLOW+1e-9 else 'VIOLATION'}")
    print("Optimal areas (in^2) from chosen run:")
    for i, a in enumerate(A_opt, start=1):
        print(f" A[{i}] = {a:.6f}")
    print('Plots saved: pso_best_mass_vs_iteration.png, pso_best_max_disp_vs_iteration.png, pso_best_feasibility_history.png')

    # --- Best-ever objective solution (may be infeasible) ---
    print('--- Best-ever objective solution (over entire run; chosen run) ---')
    print(f"w = {res['best_obj_w']:.6f}, c1 = {res['best_obj_c1']:.6f}, c2 = {res['best_obj_c2']:.6f}")
    print(f"Mass = {res['best_obj_mass']:.6f} lbm; Max displacement = {res['best_obj_max_disp']:.6f} in; J = {res['best_obj_J']:.6f}")
    print("Areas (in^2):")
    for i, a in enumerate(res['best_obj_A'], start=1):
        print(f" A[{i}] = {a:.6f}")

    # --- Best FEASIBLE global solution by objective (constraints satisfied; chosen run) ---
    print('--- Best FEASIBLE global solution by objective (constraints satisfied; chosen run) ---')
    if res['best_feas_A'].size == 0:
        print('No feasible solution was found in the chosen run.')
    else:
        print(f"w = {res['best_feas_w']:.6f}, c1 = {res['best_feas_c1']:.6f}, c2 = {res['best_feas_c2']:.6f}")
        print(f"Mass = {res['best_feas_mass']:.6f} lbm; Max displacement = {res['best_feas_max_disp']:.6f} in; J = {res['best_feas_J']:.6f}")
        print("Areas (in^2):")
        for i, a in enumerate(res['best_feas_A'], start=1):
            print(f" A[{i}] = {a:.6f}")
        stresses = solve_stresses(res['best_feas_A'])
        print("Member Stresses (ksi):")
        for i, a in enumerate(stresses, start=1):
            print(f" stresses[{i}] = {a:.6f}")
        
    # ------------------------------
    # Save summary to file
    # ------------------------------
    with open('pso_best_run_summary.txt','w') as f:
        f.write('=== PSO (robust lbest) with seeds & restarts ===\n')
        f.write(f"Chosen seed (best feasible across seeds): {chosen['seed']}\n")
        f.write(f"Best parameters (gbest snapshot): w = {res['gbest_w']:.4f}, c1 = {res['gbest_c1']:.4f}, c2 = {res['gbest_c2']:.4f}\n")
        f.write(f"Best mass (gbest snapshot): {res['gbest_mass']:.2f} lbm\n")
        f.write(f"Best max U (gbest snapshot): {res['gbest_max_disp']:.6f} in -> {'FEASIBLE' if res['gbest_max_disp']<=U_ALLOW+1e-9 else 'VIOLATION'}\n")
        f.write("Optimal areas (in^2) from chosen run:\n")
        for i, a in enumerate(A_opt, start=1):
            f.write(f" A[{i}] = {a:.6f}\n")
        f.write('Plots saved: pso_best_mass_vs_iteration.png, pso_best_max_disp_vs_iteration.png, pso_best_feasibility_history.png\n')
        # Best-ever objective (chosen run)
        f.write('--- Best-ever objective solution (over entire run; chosen run) ---\n')
        f.write(f"w = {res['best_obj_w']:.6f}, c1 = {res['best_obj_c1']:.6f}, c2 = {res['best_obj_c2']:.6f}\n")
        f.write(f"Mass = {res['best_obj_mass']:.6f} lbm; Max displacement = {res['best_obj_max_disp']:.6f} in; J = {res['best_obj_J']:.6f}\n")
        f.write('Areas (in^2):\n')
        for i, a in enumerate(res['best_obj_A'], start=1):
            f.write(f" A[{i}] = {a:.6f}\n")
        # Best feasible (chosen run)
        f.write('--- Best FEASIBLE global solution by objective (constraints satisfied; chosen run) ---\n')
        if res['best_feas_A'].size == 0:
            f.write('No feasible solution was found in the chosen run.\n')
        else:
            f.write(f"w = {res['best_feas_w']:.6f}, c1 = {res['best_feas_c1']:.6f}, c2 = {res['best_feas_c2']:.6f}\n")
            f.write(f"Mass = {res['best_feas_mass']:.6f} lbm; Max displacement = {res['best_feas_max_disp']:.6f} in; J = {res['best_feas_J']:.6f}\n")
            f.write('Areas (in^2):\n')
            for i, a in enumerate(res['best_feas_A'], start=1):
                f.write(f" A[{i}] = {a:.6f}\n")

    # Save areas to a simple text file (one value per line) for reuse
    np.savetxt('pso_best_areas.txt', A_opt, fmt='%.6f')

else:
    logging.info(
    "=== BEST RUN (NO FEASIBLE FOUND) === seed=%d gbest_mass=%.6f lbm gbest_max_disp=%.6f in",
    chosen['seed'], float(res['gbest_mass']), float(res['gbest_max_disp'])
)

# ------------------------------
# Stop timestamp & duration
# ------------------------------
stop_ts = datetime.now()
duration = (stop_ts - start_ts).total_seconds()
logging.info("Stop timestamp: %s", stop_ts.strftime('%Y-%m-%d %H:%M:%S'))
logging.info("=== PSO: STOP (elapsed %.2f s) ===", duration)
