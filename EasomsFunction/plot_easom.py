"""
Plot Easom's function to visualize the optimization landscape
and determine whether optimization should be maximization or minimization.

Easom's function:
f(x,y) = -cos(x) * cos(y/100) * exp(-(x-pi)^2 - (y/(100*pi) - 1)^2)

Domain: 0 <= x <= 2*pi, 0 <= y <= 200*pi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def easom(x, y):
    """
    Easom's function.
    
    Parameters
    ----------
    x : float or array
        First coordinate, domain [0, 2*pi]
    y : float or array
        Second coordinate, domain [0, 200*pi]
    
    Returns
    -------
    float or array
        Function value
    """
    term1 = -np.cos(x)
    term2 = np.cos(y / 100)
    term3 = np.exp(-(x - np.pi)**2 - (y / (100 * np.pi) - 1)**2)
    return term1 * term2 * term3


# Create grid for plotting
n_points = 200
x = np.linspace(0, 2 * np.pi, n_points)
y = np.linspace(0, 200 * np.pi, n_points)
X, Y = np.meshgrid(x, y)
Z = easom(X, Y)

# Find global minimum and maximum
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
max_idx = np.unravel_index(np.argmax(Z), Z.shape)

min_x, min_y = X[min_idx], Y[min_idx]
min_z = Z[min_idx]

max_x, max_y = X[max_idx], Y[max_idx]
max_z = Z[max_idx]

print("=" * 60)
print("EASOM'S FUNCTION ANALYSIS")
print("=" * 60)
print(f"\nFunction: f(x,y) = -cos(x) * cos(y/100) * exp(-(x-pi)^2 - (y/(100*pi)-1)^2)")
print(f"Domain: 0 <= x <= 2*pi, 0 <= y <= 200*pi")
print(f"\nGlobal MINIMUM:")
print(f"  Location: x = {min_x:.6f} (pi = {min_x/np.pi:.4f}*pi)")
print(f"            y = {min_y:.6f} ({min_y/np.pi:.4f}*pi)")
print(f"  Value: f(x,y) = {min_z:.8f}")
print(f"\nGlobal MAXIMUM:")
print(f"  Location: x = {max_x:.6f} (pi = {max_x/np.pi:.4f}*pi)")
print(f"            y = {max_y:.6f} ({max_y/np.pi:.4f}*pi)")
print(f"  Value: f(x,y) = {max_z:.8f}")
print(f"\nFunction range: [{min_z:.8f}, {max_z:.8f}]")
print(f"Range span: {max_z - min_z:.8f}")

# Determine optimization type
print("\n" + "=" * 60)
if abs(min_z) > abs(max_z):
    print("RECOMMENDATION: Use MINIMIZATION")
    print(f"Reason: Global minimum ({min_z:.6f}) has larger magnitude")
    print("        than global maximum ({:.6f})".format(max_z))
else:
    print("RECOMMENDATION: Use MAXIMIZATION")
    print(f"Reason: Global maximum ({max_z:.6f}) has larger magnitude")
    print("        than global minimum ({:.6f})".format(min_z))
print("=" * 60)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. 3D Surface plot
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)
ax1.scatter([min_x], [min_y], [min_z], c='red', s=100, marker='o', edgecolors='black', linewidths=2, label='Global Min')
ax1.scatter([max_x], [max_y], [max_z], c='lime', s=100, marker='^', edgecolors='black', linewidths=2, label='Global Max')
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('f(x,y)', fontsize=10)
ax1.set_title("Easom's Function - 3D Surface", fontsize=12, fontweight='bold')
ax1.legend()
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 2. Contour plot
ax2 = fig.add_subplot(2, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=50, cmap=cm.viridis)
ax2.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5, alpha=0.3)
ax2.scatter([min_x], [min_y], c='red', s=150, marker='o', edgecolors='black', linewidths=2, label='Global Min', zorder=5)
ax2.scatter([max_x], [max_y], c='lime', s=150, marker='^', edgecolors='black', linewidths=2, label='Global Max', zorder=5)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_title("Easom's Function - Contour Plot", fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
fig.colorbar(contour, ax=ax2)

# 3. Cross-section at y = 100*pi (where minimum occurs)
ax3 = fig.add_subplot(2, 2, 3)
y_slice = 100 * np.pi
z_slice = easom(x, y_slice)
ax3.plot(x, z_slice, 'b-', linewidth=2)
ax3.axvline(x=np.pi, color='red', linestyle='--', linewidth=1, label='x = pi')
ax3.axhline(y=min_z, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Min value = {min_z:.6f}')
ax3.scatter([np.pi], [easom(np.pi, y_slice)], c='red', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('f(x, y=100*pi)', fontsize=10)
ax3.set_title("Cross-section at y = 100*pi", fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Cross-section at x = pi (where minimum occurs)
ax4 = fig.add_subplot(2, 2, 4)
x_slice = np.pi
z_slice = easom(x_slice, y)
ax4.plot(y, z_slice, 'b-', linewidth=2)
ax4.axvline(x=100*np.pi, color='red', linestyle='--', linewidth=1, label='y = 100*pi')
ax4.axhline(y=min_z, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Min value = {min_z:.6f}')
ax4.scatter([100*np.pi], [easom(x_slice, 100*np.pi)], c='red', s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
ax4.set_xlabel('y', fontsize=10)
ax4.set_ylabel('f(x=pi, y)', fontsize=10)
ax4.set_title("Cross-section at x = pi", fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig("easom_function_analysis.png", dpi=150, bbox_inches='tight')
print(f"\nPlot saved as: easom_function_analysis.png")
plt.show()

# Additional analysis: Check function behavior
print("\n" + "=" * 60)
print("FUNCTION CHARACTERISTICS:")
print("=" * 60)
print(f"1. The function has a sharp global minimum at (pi, 100*pi)")
print(f"2. The minimum value is approximately {min_z:.6f}")
print(f"3. The function is nearly flat (close to 0) everywhere else")
print(f"4. This is a MINIMIZATION problem - searching for the deepest valley")
print(f"5. The exponential term creates a very narrow basin around the optimum")
print(f"6. This makes it a challenging benchmark for optimization algorithms")
print("=" * 60)
