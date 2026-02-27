#!/usr/bin/env python3
"""
Test script for the 72-bar truss model. Input 16 design variables (areas in in^2)
and print displacements (max per load case), member stress extrema, and objective mass.

Notes:
- This uses the synthetic geometry packaged inside truss72.py so it runs out-of-the-box.
- For benchmark-grade results, swap in the canonical geometry (nodes/connectivity) from
  Sedaghati/CoFE. See the truss72.py header for reference.
"""
import sys
import numpy as np
import truss72 as t72

# Parse 16 areas from CLI or use a default vector
if len(sys.argv) == 17:
    A = np.array([float(x) for x in sys.argv[1:]], dtype=float)
else:
    print('[INFO] No 16 areas provided on command line; using Sedaghati/CoFE example set.')
    A = np.array([0.1565,0.5456,0.4104,0.5697,0.5237,0.5171,0.1,0.1,1.268,0.5117,0.1,0.1,1.886,0.5123,0.1,0.1], dtype=float)

A = np.clip(A, t72.A_MIN, t72.A_MAX)
res = t72.evaluate(A)

print(f'Objective mass (lbm): {res["mass"]:.6f}')
for i,U in enumerate(res['U'], start=1):
    print(f'Load case {i}: max |U| = {np.max(np.abs(U)):.6f} in')
    sig = res['stresses'][i-1]
    print(f'             max |sigma| = {np.max(np.abs(sig)):.6f} ksi; min = {np.min(sig):.6f} ksi')

# Dump first few nodal displacements for inspection
U1 = res['U'][0]
print('\nSample displacements (LC1): node 1 U =', U1[0:3], '; node 2 U =', U1[3:6])
