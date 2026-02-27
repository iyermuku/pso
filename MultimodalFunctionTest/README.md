# Multimodal Function PSO

This directory contains a simple implementation of Particle Swarm Optimization
(PSO) applied to the multimodal test function from Xin‑She Yang's *Engineering
Optimization*, chapter 15 (equation 15.12).

The objective function is

```text
f(x) = (sum_i x_i) * exp(-sum_i sin(x_i^2)),
```

with each coordinate bounded in [-2\pi, 2\pi].  The module `multimodal_pso.py`
implements a generic PSO routine that handles arbitrary dimension `n_dim` and
by default solves a 2‑dimensional problem when the script is run directly.

A lightweight test is included in `test_multimodal_pso.py` which exercises the
2‑D search and verifies a positive value is found.

Usage example from the repository root:

```sh
python -c "import numpy as np; from MultimodalFunctionTest import multimodal_pso as m
; print(m.pso_maximize(m.multimodal_15_12, [(-2*np.pi,2*np.pi)]*2, n_dim=2))"
```

Feel free to adjust `swarm_size`, `iters`, and other PSO parameters for
experimentation.