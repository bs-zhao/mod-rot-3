# Universal condition-wise dimension sparsity: shared L(R), combined across groups

This is a **new independent package** inside the earlier analytic-only CR3
experiment. It does NOT alter A/B/C, train an NN, modify fixed original
X, resample the choice dataset, or choose parameter names/axes inside L.
It **reuses** `moments.npz` produced by the existing analytic CR3 run
(default: `analytic_cr3_64`), avoiding expensive differentiation.

## Core idea: one L, all trial groups

For each candidate orthogonal R (same R for all four groups) and category
s, use the SAME function `L_s(R)=phi(q_s(R))`, with

```
U = R T_std;             R @ R.T = I
G_s = E_s[grad_T f grad_T f.T]          # 4x4, original zero-amount trials kept
B_s = E_s[H_TT @ H_TT.T]               # 4x4
C_s = sum_j support_sj *
      E_s[UX_:j UX_:j.T | j tangent-valid]   # 4x4
```

The U, UU and UX axis-energy vectors are respectively the diagonals of
`R G_s R.T`, `R B_s R.T`, `R C_s R.T`.

For H, `B_s[a,b]=sum_j E_s[H[a,j] H[b,j]]`, computed exactly via the saved
4th-order moment tensor. This is a **dimension-wise Hessian row energy**,
not a penalty on off-diagonal Hessian entries. Real multiplicative
interactions between genuinely active dimensions are not automatically
considered undesirable.

For UX, use only the original bank's per-coordinate *valid, condition-
preserving X tangent directions*. In particular, the zero amount's
undefined X derivative is never inserted as zero, and gamma-neutral's
p derivatives do not count as tangent to that condition. The valid share
of each X axis is used to weight conditional covariance. The
`--ux-kind ambient` option instead checks any ambient-smooth X axis,
and `finite_tangent` uses the saved finite-contrast effects; keep them
separate (these are different estimands).

Each energy FAMILY is normalized within category:
`q_s^U = diag(R G_s R.T)/tr(G_s)`, etc. A recipe's q is the equally
weighted mean of its nonzero component-family q's. If an entire family
has exactly zero total energy, that family is omitted, not assigned
an arbitrary uniform vector. The same recipe is used in ALL categories.

Three recipes (no knowledge of CR3's parameter count per category):

1. `U`: first-order U-gradient dimension energy.
2. `U_UU`: U plus U-Hessian row energy (family-normalized).
3. `U_UU_UX`: U plus UU plus valid UX dimension-row energy.

Two common *generic* scalar functions phi:

- `lp`: `sum_i q_i**p`, default `p=.5`, smaller = more axis sparse;
- `auto_topk`: `min_{k=1..zdim} [sum(sorted(q)[k:]) + lam*k/zdim]`,
  default `lam=.3`. It automatically selects k; it does NOT assume
  gain-only needs 3 or loss-only needs 3.

These two function definitions apply identically to all s and all
model types with appropriate derivative moments (though this runner
currently loads CR3 only).

## Two genuinely different aggregation orders

Primary target, compute per-group **loss first**, then weighted average:

```
L_stratified(R) = sum_s pi_s L(q_s(R))
```

Control, aggregate raw moments by family then compute the same L:

```
M_pooled = sum_s pi_s M_s
q_pooled = group_q(M_pooled, R)
L_pooled(R) = L(q_pooled)
```

The two need not agree because phi is nonlinear. Compare both for
identical rotations. `L_pooled` normalizes each derivative family *after*
pooling; `L_stratified` normalizes each family *inside* each condition
before applying phi. The distinction is intentional.

Weights `pi_s` may be:

- `--balances 0`: empirical design proportions 60:60:40:340.
- `--balances 1`: each condition weight .25.
- `--balances 0 .25 .5 .75 1`: convex interpolated sensitivity sweep.
  Coefficient `b` yields `pi_s(b)=(1-b)*n_s/500 + b/4`.

All recipes and weights are predetermined. No conditions get special
parameters, no target 'dead axis' is prescribed. Different groups are
allowed to prefer different coordinate subsets; they MUST share R.

## Run on your existing Windows py11 outputs

```cmd
conda activate py11
git pull origin main
python -m pytest experiments/cr3_condition_identifiability/tests -q
```

**Smoke** (reads saved `analytic_cr3_64/logit/moments.npz`):

```cmd
python -m experiments.cr3_condition_identifiability.run_group_sparsity --source-tag analytic_cr3_64 --tag group_sparsity_smoke --channels logit --random-rotations 12 --no-optimize --no-plots
```

**Main** (no additional NN or analytic derivative computation):

```cmd
python -m experiments.cr3_condition_identifiability.run_group_sparsity --source-tag analytic_cr3_64 --tag group_sparsity_v1 --channels logit prob --recipes U U_UU U_UU_UX --metrics lp auto_topk --balances 0 1 --power 0.5 --complexity-penalty 0.3 --random-rotations 128 --restarts 2 --maxiter 80
```

**Sensitivity sweep with a separate tag**:

```cmd
python -m experiments.cr3_condition_identifiability.run_group_sparsity --source-tag analytic_cr3_64 --tag group_sparsity_weights_v1 --channels logit --balances 0 0.25 0.5 0.75 1 --random-rotations 128 --no-optimize
```

If you have only the uploaded ZIP, restore
`outputs/cr3_condition_identifiability/analytic_cr3_64/`
with each channel's `moments.npz`; no `fields.npz` needed.

## Output: new folder, no interference

```
outputs/cr3_condition_identifiability/group_sparsity/<tag>/
  config.json
  REPORT.md
  logit/
    rotation_scores.csv          # all R, all recipes, both aggregation orders
    best_rotations.json          # best blind candidate R plus group-by-group q
    moment_summaries.json        # U,UU,UX traces, domain support, weights
    figures/                     # six pairwise angular curves per configuration
  prob/
    ...
```

Per candidate record both `stratified_score` and `pooled_score`,
individual category losses and effective dimensions. Identity and all
six angle-sweep planes are **post-hoc analytic diagnostics only**;
searched best candidate is chosen from Haar SO(4) rotations and/or
derivative-free optimized rotations with random parameter starts.
Alignment to true axes (signed permutation invariant) is evaluated
after selecting the candidate, never optimized directly.

### Interpretation limits

- This test asks whether a *universal* sparsity preference, aggregated
  across structured trial categories, recovers the generator's true axes.
  It does NOT assume that it must succeed. If the original model really
  depends on all four axes within a group, that group can offer little
  dimensionality-based recovery information.
- Pointwise zero U gradient on all points is enough for a dead axis;
  UU and legal UX are redundant in the exact ideal case, but can change
  robustness or empirical scoring. Their addition does not guarantee
  improvement.
- Near-zero derivative energy is only a proxy for finite-dimensional
  predictive sufficiency. A future predictive drop-k experiment must
  check this independently on held-out subjects.
- Existing `moments.npz` are **analytic true-theta standardized**
  moments, NOT frozen Neural Process derivative matrices. NN fidelity
  is a subsequent separate experiment.
- The new output tag protects different input checksums, priors,
  source directory and search parameters against accidental cache reuse.
