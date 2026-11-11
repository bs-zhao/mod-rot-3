# Minimum variable subsets for UX interactions

AE found no CR3 criterion (best 2/6 planes). AH tests a different sparsity
definition: the smallest subset of U variables that predicts each Xj
omission contrast with a prescribed relative L2 error. It allows arbitrary
nonlinear functions of the retained variables through complete polynomial
blocks plus radial kernels.

Source functions are U-polynomial degree five and U-RBF2 response banks,
fitted to all original WY rows. X omission contrasts are the same frozen
joint-design maps from Z. Whole-U/whole-X means are removed. Each channel/X
contrast is normalized by its own function energy; per-X errors are then
averaged equally or assessed by the worst column.

For each W-only coordinate class, enumerate every nonempty coordinate
subset. Identical projection subspaces are cached using W only. CR3 has
255 distinct subspaces. On each, use a complete degree-five dictionary
plus 128 radial features at lengthscale 2. Ridge is chosen on an inner
subject split, and every one of 5000 subjects receives an out-of-fold
prediction. Choosing the best subset uses the pooled out-of-fold errors;
these are descriptive compression costs, not an unbiased nested-wrapper
test error. All sizes form a nonincreasing prediction-error envelope.

Scores are relative L2 tails at fixed variable budgets, the area of the
budget/error curve, and minimum/fractional variable counts at 1%,3%,10%,30%
relative L2 error. Counts beyond the full dimension are censored. Adequacy
requires full-dimensional crossfit relative MSE <=.1 for EVERY X contrast,
as well as source-function validation NMSE <=.1. Poor full-model
approximation cannot masquerade as a sparsity failure.

No generating function, task-specific X semantics, original weights or
saved rotation matrix is used. No per-dataset signs or selected planes
are allowed. CR3 runs first; later datasets require the same
criterion to survive. A grid win would still need capacity/seed stability,
bootstrap margins and blind rotation minimization.

Before launch, ten conditional/subset tests passed, including a sparse UX
linear example made denser by a rotation, arbitrary orthonormal bases of
the same subspace, and a complete W/Y-only pipeline. The full suite passed
135 tests.

## Actual CR3 outcome

AH completed with no full-plane criterion; the best settings pass 3/6
planes and rank baseline 7/37. Full-U crossfit errors pass the adequacy
guard: <=3.3e-10 for polynomial teacher contrasts and <=.002099 for RBF2.
These are sufficiently resolved failures of the tested compression rule.
Later datasets were skipped.

The unchanged-score AH2 scope audit treats tail-one, budget-curve area and
30% counts as controls. Area can favor a dominant first variable in the
user's example; it is not sparse support. The remaining primary subset
scores obey that example, including six replicated X columns. AH2 still
has zero primary passes. Original launch sources, scores and failed planes
are retained; no numerical losses or signs were changed by this audit.
