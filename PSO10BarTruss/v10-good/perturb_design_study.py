"""Perturbation study of design variables for 10-bar truss.

Starting from a baseline vector of areas, each variable is independently
varied ±10% in 10 steps.  For each perturbed design the truss mass,
stresses, and displacements are computed.  A pair of plots is produced for
each variable showing how the weight and constraint violations change.
"""
import numpy as np
import matplotlib.pyplot as plt
from truss_model import solve_displacements, solve_stresses, mass_from_A

# baseline design
base_A = np.array([
    31.278784,
    0.100000,
    22.802556,
    15.128871,
    0.100000,
    0.532462,
    7.513537,
    20.975065,
    21.379619,
    0.100000,
], dtype=float)

# limits for constraint violation
stress_limit = 25.0  # ksi
disp_limit = 2.0     # in

n_steps = 10
perturb_frac = 0.10

for idx in range(len(base_A)):
    var_name = f'A[{idx+1}]'
    values = base_A[idx] * (1 + np.linspace(-perturb_frac, perturb_frac, n_steps))
    weights = []
    stress_viol = []
    disp_viol = []
    for v in values:
        A_mod = base_A.copy()
        A_mod[idx] = v
        U = solve_displacements(A_mod)
        stresses = solve_stresses(A_mod)
        mass = mass_from_A(A_mod)
        # compute max violation amounts
        stress_over = np.maximum(np.abs(stresses) - stress_limit, 0.0)
        disp_over = np.maximum(np.abs(U) - disp_limit, 0.0)
        stress_viol.append(float(np.max(stress_over)) if stress_over.size else 0.0)
        disp_viol.append(float(np.max(disp_over)) if disp_over.size else 0.0)
        weights.append(mass)
    weights = np.array(weights)
    stress_viol = np.array(stress_viol)
    disp_viol = np.array(disp_viol)
    # convert to percentage deviations
    base_mass = mass_from_A(base_A)
    weights = (weights - base_mass) / base_mass * 100.0
    stress_viol = stress_viol / stress_limit * 100.0
    disp_viol = disp_viol / disp_limit * 100.0

    # plotting
    fig, ax_w = plt.subplots(figsize=(8, 5))
    ax_w.plot(values, weights, marker='o', label='weight deviation (%)')
    ax_w.set_xlabel(var_name)
    ax_w.set_ylabel('Weight deviation (%)')
    ax_w.grid(True, alpha=0.3)

    ax2 = ax_w.twinx()
    # plot stress violation as filled bars
    ax2.bar(values, stress_viol, width=(values[1]-values[0])*0.8, color='C1', alpha=0.3,
            label='stress violation (%)')
    # plot displacement violation as dashed line
    ax2.plot(values, disp_viol, marker='^', color='C2', linestyle='--', label='disp violation (%)')
    ax2.set_ylabel('Violation (%)')

    # combine legends: weight on left, violations on right
    lines, labels = ax_w.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_w.legend(lines + lines2, labels + labels2, loc='best')

    plt.title(f'Perturbation study of {var_name}')
    plt.tight_layout()
    outname = f'perturb_{idx+1}.png'
    fig.savefig(outname, dpi=150)
    print(f'Saved {outname}')

print('Study complete.')
