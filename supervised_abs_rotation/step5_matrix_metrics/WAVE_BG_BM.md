# Retained approximate optima and further WY-only function complexity

Windows experiments, 2026-11-12, America/New_York. The user's clarification
is applied throughout: neither a fixed .01 prediction-error cutoff nor an
optimum slightly displaced from baseline automatically discards a candidate.
Every scored family uses the same definition and settings across models.
All 5000 subjects are used in the final fits; only actual W/Y, row metadata
and our own derived fit artifacts enter scoring.

## Approximate candidates are first-class recorded results

`record_rotation_optima.py` now records the optimal observed angle and plane,
best interior angle, deviation modulo a 90-degree axis permutation, baseline
gap, complete curve contrast, all numerical ties and the 15/75-degree cases.
Angles are in degrees and score gaps are in relative percent, in separate
fields. Numerical endpoint ties are represented by baseline; raw minima
and raw differences remain saved. This is a numerical convention, not a
fitness cutoff. `write_approximate_shortlist.py` makes a readable shortlist
without selecting different definitions in different models.

A useful retained example is BA2's **logit, GP posterior derivative,
pooled worst-X UX soft support at 3%**:

| Model | Observed grid optimum | Axis deviation modulo permutation | Score change from baseline |
|---|---:|---:|---:|
| CR3 | 0 degrees | 0 degrees | 0% |
| C1 | 0 degrees | 0 degrees | 0% |
| C2 | 75 degrees | 15 degrees | -.01090% |
| C3 | 15 degrees | 15 degrees | -.11706% |

The same frozen radial-GP ordinal-source primary BB2 also has deviations
0/0/15/15 degrees on the four grids, with score gaps 0/0/-.00613%/-.11400%.
These are retained approximate four-grid results. They do not establish
unique unsupervised coordinate recovery: BB2's separately frozen CR3 blind
frame is 1.0587% lower with a maximum matched axis angle of 42.55 degrees.
`record_blind_axis_positions.py` saves sign/permutation-matched angles and
score differences separately. BB's C1/C2 blind optima have maximum matched
angles 1.17/12.06 degrees and score changes -.5093%/-2.0671%, also retained.

## BG: double conditional residualization

The BC feature innovation is paired with response-function targets after
predicting them from the other U coordinates. Nested five-fold polynomial
plus radial regression has two fixed capacities: degree3/64 centers and
degree5/128 centers. Every normalized X/channel function contributes equal
energy to ridge selection. Residuals are not renormalized separately for
each omitted direction; per-X graph support is normalized after assembly.
This is an exploratory conditional-moment adaptation, not a calibrated KCI
test. Raw/diagonal-removed moments and all tails remain saved.

Strict primary combination counts are 4/368/154/24 in CR3/C1/C2/C3. Two
degree5 contextual logit 10% worst-X variants pass all four grids. However,
the previously frozen BC3 frame is still about 12.12% lower in CR3. Better
nuisance removal does not explain away that counterexample.

## BH/BJ: empirical-Bayes sparse function groups

For each X/channel target, complete polynomial features (degrees3/5) plus
a generic radial remainder approximate its U function. Each anonymous U
subset has a coefficient-prior variance learned by expected GP marginal
density. Two training noise starts are fitted for 60 steps; the best
training density per task initializes a 120-step all-subject refit. This
is an expected function-ensemble density, not independent copies for every
retained X-mode. Hyperparameter thresholds are numerical parameter bounds,
not prediction-quality vetoes.

Each graph edge receives the empirical energies of fitted subset functions
containing that Ui, with an additional posterior-uncertainty variant. Small
tails are then normalized per X. Subset components can overlap under
dependent U; this is not a unique causal or independent-input ANOVA graph.
The model evidence and group-function supports are recorded alongside UX
participation support.

BH keeps the initial overlapping group basis. BJ subtracts the common span
of all lower-order groups from higher-order groups, using W only. All
groups at an order use the same lower span, preserving axis permutation
symmetry. The full function span is preserved by a block triangular basis
change before scaling; no response-dependent feature is removed to force a
desired answer. Toy tests check span preservation and actual geometric
symmetries; saved real-data posteriors are independently reconstructed by
`audit_group_gp_reconstruction.py`.

The real-data BJ audit reproduces every checked component energy within
4.3e-15 and the saved per-subject density exactly. The largest normalized
lower-span cross covariance is 1.64e-12, in C3 degree5. These are numerical
integrity checks, not thresholds used to select a scientific candidate.

| Family | CR3 | C1 | C2 | C3 | Identical strict four-grid combinations |
|---|---:|---:|---:|---:|---:|
| BH | 0 | 123 | 22 | 23 | 0 |
| BJ | 12 | 237 | 62 | 0 | 0 |

BJ's degree3 posterior-mean worst-X 10% UX support still gives
6/6, 6/6, 6/6, 12/15; C3's best point is 15 degrees, only .26870% lower.
It is retained as an approximate candidate. The frozen CR3 counterexample
remains about 10.22% lower on this definition.

The small GP target errors are errors in **conditional X response functions**,
not claims of equally accurate predictions on every original WY row.
For example BH's worst heldout degree5 logit errors range from 4.56e-7 in
C1 to 6.43e-5 in CR3. The separate original-row/profile preparation errors
remain in each protocol. Both uncertainty sources must be considered.

## BI: paired subject refits, not just frozen-feature resampling

Twenty-four 80% subject subsets without replacement compare the reference
frame with the already frozen BC3 frame. Each draw recomputes response
centering, direct target SVD and the W-only Ui conditional feature estimates.
CR3's exact observed shared X witnesses and the two frames are fixed.
This is a labelled paired stability audit, not another blind optimization;
subsample quantiles are not population confidence intervals.

The counterexample is lower in **24/24** draws for both the contextual raw
worst-X 10% soft count and 1% fractional support. Relative-margin
5%/median/95% quantiles are respectively:

- 10% soft count: -12.500% / -12.266% / -11.970%.
- 1% fractional support: -.12125% / -.10775% / -.09460%.

The latter discrepancy is small but stable. Its near-grid behavior remains
recorded; it is not silently converted into either a success or a rejection.

## BK/BL: distinguish pure U response from UX variation

BC's double centering excludes the constant-X response component. BK adds
the centered observed-design X mean as one additional normalized function
component. Its score names explicitly include `with_U_main`; pure UX scores
remain separate. There is no per-model mixture weight. On CR3, the old
counterexample becomes 10.36% worse on the contextual raw worst-component
10% score. A new independent 32-proposal/two-start exact-rescored blind
search finds 3.34811220 vs baseline 3.41771473, **2.0365% lower**, with a
maximum matched axis angle of 47.92 degrees. This reduces the score gap,
but does not establish coordinate recovery.

The pure-U component dominates the baseline worst-component score in C1,
C2 and C3; the CR3 baseline is still dominated by a UX component. Thus
BK is not presented as an improved pure-UX sparsity score.

BL instead removes the common affine U projection from the main response
before measuring its nonlinear support. The affine fit is preserved in
the audit and treated as a rotation-invariant free reference; no affine
form is assumed for the remaining function. This does not improve CR3.

| Family | CR3 | C1 | C2 | C3 | Identical strict four-grid combinations |
|---|---:|---:|---:|---:|---:|
| BK | 20 | 204 | 3 | 2 | 0 |
| BL | 0 | 174 | 73 | 5 | 0 |

## BM: radial approximations inside every U subset

BM gives every nonempty U subset its own eight-center radial dictionary,
alongside complete degree3 polynomial support, rather than assigning all
generic nonpolynomial remainder to the full-U group. All subsets use the
same W-only bandwidth/center policy, shared seed and hierarchy rule.
Polynomial and radial subblocks have equal empirical feature trace within
a subset, followed by group normalization. This is a common approximation
policy, not a supplied task equation. The preceding empirical-Bayes fit and
scoring definitions are unchanged. The four-model continuation completes
with 6/117/31/0 strict primary combinations and no common strict metric.
The independent real-data replay matches all checked component energies
within 1.2e-12 and the saved density exactly. C3's normalized residual
lower-span cross covariance is 2.62e-6: this records the finite numerical
projection precision rather than claiming exact orthogonality.

The current full suite passes 237 tests. Actual scores, failed variants,
continuous optima, source quality and read audits are retained locally.
No new approximation error or optimum-position cutoff is used.

Primary references:
[Zhang et al., Kernel-based Conditional Independence Test](https://arxiv.org/abs/1202.3775),
[Tipping, Sparse Bayesian Learning and the Relevance Vector Machine](https://jmlr.org/papers/v1/tipping01a.html).
The grouped function-ensemble criterion and hierarchy here are exploratory
adaptations; those papers' statistical guarantees are not claimed for them.
