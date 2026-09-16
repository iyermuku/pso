# 11-Bar Planar Truss Benchmark

This benchmark follows the 11-bar planar truss model defined in `RodriguesEtAl/truss11.py`.

## Problem definition

The truss has 6 nodes and 11 members. The member connectivity is

```python
ELEMENTS = np.array([
    [1, 4],
    [2, 5],
    [3, 6],
    [1, 2],
    [2, 3],
    [4, 5],
    [5, 6],
    [4, 2],
    [1, 5],
    [5, 3],
    [2, 6],
], dtype=int)
```

The global nodal displacement vector is ordered as

```python
[u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6]^T
```

with the support nodes fixed at A and D, corresponding to the restrained DOFs.

## Equilibrium matrix

In the truss formulation used in this benchmark, the internal element forces and applied nodal forces are related by the equilibrium matrix as

```text
M N = P                                                    (7)
```

where:

- `M` is the `nDOF x n_el` equilibrium matrix,
- `N` is the `n_el x 1` vector of member axial forces,
- `P` is the `nDOF x 1` vector of applied nodal forces,
- `nDOF` is the number of global degrees of freedom,
- `n_el` is the number of truss elements.

For this 11-bar planar truss, `nDOF = 12` and `n_el = 11`, so the equilibrium matrix has size `12 x 11`.

For each member, the member-force contribution is built from the direction cosines

```python
c = (xj - xi) / L
s = (yj - yi) / L
```

and assembled into the global matrix `M`.

For the default geometry in this file, the assembled equilibrium matrix is:

```python
M = np.array([
    [ 0.000000,  0.000000,  0.000000, -1.000000,  0.000000,  0.000000,  0.000000,  0.000000, -0.707107,  0.000000,  0.000000],
    [-1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000, -0.707107,  0.000000,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  1.000000, -1.000000,  0.000000,  0.000000,  0.707107,  0.000000,  0.000000, -0.707107],
    [ 0.000000, -1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000, -0.707107,  0.000000,  0.000000, -0.707107],
    [ 0.000000,  0.000000,  0.000000,  0.000000,  1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.707107,  0.000000],
    [ 0.000000,  0.000000, -1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000, -0.707107,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  0.000000,  0.000000, -1.000000,  0.000000, -0.707107,  0.000000,  0.000000,  0.000000],
    [ 1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.707107,  0.000000,  0.000000,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  1.000000, -1.000000,  0.000000,  0.707107, -0.707107,  0.000000],
    [ 0.000000,  1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.707107,  0.707107,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  1.000000,  0.000000,  0.000000,  0.000000,  0.707107],
    [ 0.000000,  0.000000,  1.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.707107],
])
```

This matrix is consistent with the implementation in `truss11.py`, where the element direction vectors are assembled using the nodal coordinates and member connectivity.

## Force vectors

For the 11-bar benchmark, the internal member-force vector is ordered as

```python
N = [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11]^T
```

and the applied nodal-force vector is

```python
P = [F1x, F1y, F2x, F2y, F3x, F3y, F4x, F4y, F5x, F5y, F6x, F6y]^T
```

For the benchmark loading specified in the model, the applied nodal force vector is

```python
P = [0, 0, 0, -100, 0, -100, 0, 0, 0, 0, 0, 0]^T
```

which corresponds to downward 100 kN loads at nodes B and C.

The complete equilibrium relation is therefore

```text
M N = P
```

with

```python
M  : 12 x 11
N  : 11 x 1
P  : 12 x 1
```

## Rank of the equilibrium matrix

Using the exact geometry and connectivity in `truss11.py`, the assembled equilibrium matrix has numerical rank

```text
rank(M) = 9
```

for this 11-bar benchmark. This was verified from the assembled `12 x 11` matrix, and it indicates that the equilibrium matrix is rank-deficient in the full global form.

## Mechanisms and self-stress counts from the matrix dimensions

Let the equilibrium matrix be `M ∈ R^(n_r × n_c)` with `n_r = 12` rows and `n_c = 11` columns, and let `r = rank(M) = 9`.

Then the number of independent zero-energy deformation modes, or mechanisms, is the nullity of the force-to-displacement equilibrium operator:

```text
m = n_c - r = 11 - 9 = 2
```

The degree of static indeterminacy, denoted `gH`, is the number of redundant force states that can exist without any external loading. For this truss it is

```text
gH = n_e - r = 11 - 9 = 2
```

where `n_e = 11` is the number of elements. In structural mechanics, `gH` is also known as the degree of static indeterminacy, or the number of independent states of self-stress.

The left-nullity of the same matrix is also

```text
s = n_r - r = 12 - 9 = 3
```

with `s` interpreted as the number of independent self-stress states in the force-equilibrium space.

So for this benchmark,

```text
m = 2
gH = 2
s = 3
```

meaning that the equilibrium matrix admits two independent mechanism directions in the member-force space, has a degree of static indeterminacy of two, and has a three-dimensional left null space. These values are consistent with the rank deficiency found from the SVD and with the fact that the `12 x 11` equilibrium matrix has a two-dimensional right null space and a three-dimensional left null space.

## SVD decomposition in the form M = U V W^T

For the truss equilibrium matrix, the singular value decomposition is written as

```text
M = U V W^T
```

where:

- `U = [u1, ..., u_r]` is the matrix of left singular vectors, with `U^T U = I`,
- `W = [w1, ..., w_r]` is the matrix of right singular vectors, with `W^T W = I`,
- `V` is a diagonal matrix containing the singular values `σi` on the leading diagonal, i.e. `Vii = σi` and `Vij = 0` for `i ≠ j`.

In the usual SVD terminology, the coefficients `Vii` are the singular values of `M`, and the vectors `ui` and `wi` are, respectively, the ith left singular vector and the ith right singular vector.

For this benchmark,

```text
n_r = 12,   n_e = 11,   r = 9
```

so `M` is `12 x 11`, `U` is `12 x 12`, `W` is `11 x 11`, and `V` is `12 x 11` with its nonzero diagonal entries equal to the nine positive singular values.

The numerically computed singular values are

```python
σ = [2.19610465, 1.95521755, 1.83467311, 1.53818900,
     1.47550816, 1.41421356, 1.27827016, 1.08495361,
     0.79622522]
```

Thus,

```text
Vii = σi,   i = 1, ..., 9
```

and the left/right singular vectors satisfy

```text
M wi = σi ui,
M^T ui = σi wi
```

for each singular triplet `(ui, σi, wi)`.

The corresponding diagonal singular-value matrix is

```python
V = np.zeros((12, 11))
np.fill_diagonal(V, [2.19610465, 1.95521755, 1.83467311, 1.53818900,
                    1.47550816, 1.41421356, 1.27827016, 1.08495361,
                    0.79622522])
```

The associated orthogonal matrices are obtained from the SVD, and the right-singular vector matrix is

```python
W = np.array([
    [ 1.590707e-01,  2.766832e-01,  5.699026e-19],
    [-8.326673e-16,  5.533664e-01,  2.498002e-16],
    [-1.590707e-01,  2.766832e-01, -4.266023e-16],
    [ 3.695014e-01,  8.933426e-02,  4.440369e-01],
    [-3.695014e-01,  8.933426e-02, -4.440369e-01],
    [ 3.695014e-01,  8.933426e-02, -4.440369e-01],
    [-3.695014e-01,  8.933426e-02,  4.440369e-01],
    [ 3.175169e-01,  3.566357e-01,  2.298504e-01],
    [ 3.175169e-01,  3.566357e-01, -2.298504e-01],
    [-3.175169e-01,  3.566357e-01,  2.298504e-01],
    [-3.175169e-01,  3.566357e-01, -2.298504e-01],
])
```

and the last two columns of `W` span the null space because their corresponding singular values are effectively zero.

The null-space basis matrix is therefore

```python
V_null = np.array([
    [-0.288131252193,  0.209677153009],
    [-0.044082244741,  0.469338873226],
    [ 0.244049007452,  0.259661720217],
    [-0.288131252193,  0.209677153009],
    [ 0.244049007452,  0.259661720217],
    [-0.288131252193,  0.209677153009],
    [ 0.244049007452,  0.259661720217],
    [ 0.407479124594, -0.296528273505],
    [ 0.407479124594, -0.296528273505],
    [-0.345137416222, -0.367217126359],
    [-0.345137416222, -0.367217126359],
])
```

and satisfies

```text
M V_null ≈ 0
```

to numerical tolerance.

## Notes

- `M` is `12 x 11` for this benchmark.
- The 12 rows correspond to the global nodal DOFs.
- The 11 columns correspond to the 11 truss members.
- The relation `M N = P` is the equilibrium form used for the force transformation in the truss model.
- The verified rank of the assembled matrix is `9`.
- The nullity is `2`, so the null-space basis matrix has shape `11 x 2`.
