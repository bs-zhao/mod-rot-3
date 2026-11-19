# CR4: exact analytic Hessians (9 x 9, no NP/subjects required)

## Purpose and source

Use **the actual CR4 three-parameter CPT** with `alpha=1`, taking
physical stimulus coordinates from the IID-X generation policy in
[`np/reference/cr4/s0_s1_cr4.py`](../../np/reference/cr4/s0_s1_cr4.py)
and the value/logit formula in
[`np/reference/s0_cr2_design_recovery.py`](../../np/reference/s0_cr2_design_recovery.py).

Unlike previous CR4 datasets, new CR4 generates a different X for
every `(subject,trial)`; its per-item distribution is known, so
analytical derivatives or integration do not require a common X bank,
learned latents, or actual subject data.

Coordinate order is strictly

`W = (lambda, gamma, tau, gain_A, loss_A, p_A, gain_B, loss_B, p_B)`

and Hessian blocks are `TT: 0:3 x 0:3`, `TX: 0:3 x 3:9`, `XX: 3:9 x 3:9`.

## Simplified CR4 equation

For option `j` in `{A,B}` define

- `h_j = log(p_j/(1-p_j))`
- `w_j = sigmoid(gamma*h_j)`
- `m_j = gain_j + lambda*loss_j`
- `c_j = w_j - 1`
- `V_j = (w_j*m_j - lambda*loss_j)/100`

Then `eta = tau*(V_A-V_B)` and `q = sigmoid(eta)` is the
probability of choosing A. Binary choice is a Bernoulli draw, not a smooth
function of continuous W; differentiate eta or q rather than raw 0/1 draws.

Let `a_j = w_j(1-w_j)`, `b_j = 1-2*w_j`,
`h'_j = 1/[p_j(1-p_j)]`, and
`h''_j = (2p_j-1)/[p_j^2(1-p_j)^2]`.

The exact weighting derivatives are

```text
w_p    = a*gamma*h'
w_g    = a*h
w_pp   = a*(gamma*h'' + b*gamma^2*(h')^2)
w_pg   = a*h'*(1+b*gamma*h)
w_gg   = a*b*h^2
```

Here suffix `g` denotes the *parameter gamma* (not payoff gain).

Define `s_A=+1`, `s_B=-1`. Every block is assembled by summing the
following per-option contributions with multiplier `s_j/100`.

**TT (3 x 3, order lambda/gamma/tau)**:

```text
H_TT[lambda,lambda] = 0
H_TT[lambda,gamma]  = tau/100 * sum_j s_j*loss_j*w_g,j
H_TT[lambda,tau]    = 1/100   * sum_j s_j*loss_j*(w_j-1)
H_TT[gamma,gamma]   = tau/100 * sum_j s_j*m_j*w_gg,j
H_TT[gamma,tau]     = 1/100   * sum_j s_j*m_j*w_g,j
H_TT[tau,tau]       = 0
```

The opposite triangle is the transpose.

**TX per option j (3 x 3, columns gain_j/loss_j/p_j)**:

```text
H_TX,j = s_j/100 * [
    [0,       tau*(w_j-1),       tau*loss_j*w_p,j],
    [tau*w_g,j, tau*lambda*w_g,j, tau*m_j*w_pg,j],
    [w_j,     lambda*(w_j-1),    m_j*w_p,j   ]
]
```

**XX within option j (3 x 3, order gain_j/loss_j/p_j)**:

```text
H_XX,j = tau*s_j/100 * [
    [0,       0,          w_p,j],
    [0,       0,          lambda*w_p,j],
    [w_p,j,   lambda*w_p,j, m_j*w_pp,j]
]
```

The full logit `H_XX` is `diag(H_XX,A, H_XX,B)`, with **exactly
zero A-B cross-block**. Thus the exact logit is structurally sparse.

The exact probability gradient and Hessian follow from sigmoid:

```text
grad(q) = q*(1-q)*grad(eta)
H(q) = q*(1-q)*[ H(eta) + (1-2*q)*outer(grad(eta),grad(eta)) ]
```

Even with zero A-B mixed derivatives of eta, `H(q)` usually has
nonzero A-B mixed derivatives owing to the outer-product term.

For observed binary `y`, the **negative-log-likelihood Hessian** is

```text
H_NLL = q*(1-q)*outer(grad(eta),grad(eta)) + (q-y)*H(eta)
```

The **choice-averaged Fisher matrix** is

```text
I(W) = E_y[H_NLL | W] = q*(1-q)*outer(grad(eta),grad(eta))
```

This Fisher matrix is **not** the probability Hessian; it is PSD and
is a different object from H(eta) or H(q).

## The role of scaling

All formulas above are with respect to **physical coordinates**. The
amount derivatives are measured per 1 payoff unit, while p derivatives
are measured per 1 probability unit. Comparing their magnitudes or
sparsity needs justified scaling, not arbitrary raw-unit comparisons.

The helper `standardized_hessian(H)` converts an analytic 9x9 Hessian
to coordinate-wise standardized W (population prior SD for lambda,
gamma and tau; the known uniform distribution SDs for X):

```text
H_standardized = diag(sd_W) @ H_physical @ diag(sd_W)
```

This is exact for the affine change of coordinates. Do not confuse
standardized **true theta** coordinates with learned NP latent Z. A
derivative with respect to latent Z requires a specified map Z -> theta.

## Reproducible commands

Pointwise exact Hessian and Fisher at specified raw coordinates:

```cmd
python math/cr4_analytic_hessian/cr4_hessian.py --mode point --theta 1.98 0.7 2.2 --x 100 40 0.75 80 60 0.35 --output outputs/choice/CR4/analytic_hessian/point.json
```

No subject data: Monte Carlo integration against known P(T) and P(X),
saving **signed mean**, **mean absolute**, and **RMS** matrices for
both logit and choice probability:

```cmd
python math/cr4_analytic_hessian/cr4_hessian.py --mode mc --samples 50000 --seed 20261118 --output outputs/choice/CR4/analytic_hessian/mc50k.json
```

To use physical rather than standardized coordinates, add `--physical`.
The generated script refuses to overwrite an existing output file.

Tests (numeric finite differences of *every* element for 12 random CR4
points, exact CPT reference, symmetry, Fisher identity, coordinate
rescaling, and deterministic MC):

```cmd
python -m unittest discover -s math/cr4_analytic_hessian -p test_cr4_hessian.py -v
```

**Research cautions:** (a) signed averages can cancel curvature of opposite
signs; report RMS and mean absolute curvature alongside means;
(b) probability Hessian can be dense purely due to sigmoid link, so do
not confuse link-function coupling with structural CPT interaction;
(c) CR4 X is independent *between subject/trial draws*, not independent
of Y once conditioned on theta; and (d) averaging exact Hessians over
known P(T)P(X) requires no NP decoder and does not by itself recover
an unknown orthogonal rotation of a learned Z representation.
