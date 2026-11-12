# Conditional kernel UX support and a globally learned response link

Experiments completed on the user's Windows machine, 2026-11-11/12,
America/New_York. Every fit/refit uses all 5000 subjects eventually. Source
inputs are the actual case W matrices, shared Y and row metadata. Prediction
errors and small ranking gaps are continuous diagnostics, without a .01 veto.

## BC / BC2: derivative-free conditional feature moments

For each X column, a W-only conditional kernel feature witness removes
predictability from the other X columns and retains their context. The actual
response table is double centered to isolate UX response variation; each X
target is normalized to unit empirical energy and compressed by direct SVD
with a 1e-8 relative squared-energy budget. CR3 uses its verified shared
observed 500-node X grid exactly. Other models use generic X-profile
interpolation fitted from every original row; that approximation is reported.

For Ui, a marginal midrank RBF embedding is residualized against the other U
coordinates using nested five-fold W-only polynomial/RBF regression. Its
tensor product with constant plus centered other-U radial features retains
context interactions. Squared feature covariances with each X response
target form a Ui-by-Xj graph. Small relative tails and effective support,
including mean/worst-X versions, measure graph sparsity. Overall effect
magnitude is removed before those summaries. Raw and diagonal-removed
moments are retained; diagonal removal is not claimed exactly unbiased
with learned, empirically centered features. This is not an exact Sobol
decomposition or a calibrated conditional-independence test.

The same **logit, contextual Ui innovation, raw moment, worst-X soft count
at 10% relative L2 activation** gives:

| Ui/context centers | CR3 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| 16, BC | 6/6, rank 1 | 6/6, rank 1 | 6/6, rank 1 | 14/15, rank 2 |
| 32, BC2 | 6/6, rank 1 | 6/6, rank 1 | 6/6, rank 1 | 15/15, rank 1 |

BC's C3 lower-scoring grid case improves only .03275% over baseline. BC2
has one identical strict four-grid primary combination. All endpoint errors
are zero. The direction cache now explicitly matches antipodes to avoid
sign flips caused by numerical zeros; it uses no evaluation labels.

## BC3: frozen blind counterexample

Thirty-two random SO frames and two 60-evaluation KNN-proxy Powell searches
propose frames. Eight proposals and 24 Givens refinements are scored with
the **original exact nested-regression feature construction**, rather than
the proxy. The source statistic is independently reproduced first. Scores
and frame hashes freeze before baseline labels are joined.

The 16-center baseline is 3.41811219; the best evaluated frame is 3.00366512,
12.125% lower. Transferring this fixed frame to BC2, without another
optimization, gives 3.00439103 versus 3.41771473, **12.094% lower**.
Increasing the kernel resolution does not remove this counterexample.
Grid success is retained as a useful result, not claimed as global recovery.

All 416 support/tail/channel/estimator variants are also rescored on this
same frozen frame. The logit contextual worst-X fractional 1% support has
grid gaps .02716%, .000315%, .01281%, .01048% in CR3/C1/C2/C3. The fixed
CR3 frame improves .10725%. These small baseline-relative differences do
not automatically disqualify the family. However CR3's entire grid contrast
is only .19759%: the blind difference is about 54.3% of that contrast.
Near saturation must be considered alongside the small numerical gaps.

`report_kernel_continuous_evidence.py` saves every grid gap, full contrast,
gap/contrast ratio and the fixed CR3 frame margin, with plots showing every
plane. The report does not silently select a winner or impose a new cutoff.

## BE: learn one global monotone output map from actual WY

This is a new source fit, unlike AZ's fixed ordinal calibration. A 256-center
joint-W radial GP dictionary has a single isotropic U-block length and a
single X-block length, chosen from W distances. A 33-quantile-knot positive
piecewise linear output map is learned per channel by GP marginal density
**including its Jacobian**. Identity and learned-link models are both
retained. No known link formula is supplied. Sufficient statistics stream
all original rows without saving a huge feature matrix. Train-subject fits
are validated on original heldout Y rows, then both models refit all 5000.
Validation is conditional on knot locations frozen from all original Y.

Original-Y heldout relative MSE (identity -> learned):

| Model | logit_nn | p_nn |
|---|---:|---:|
| CR3 | .05916 -> .06204 | .07127 -> .05196 |
| C1 | .01189 -> .01063 | .01070 -> .00777 |
| C2 | .03891 -> .03796 | .03662 -> .02618 |
| C3 | .10134 -> .10328 | .11511 -> .09174 |

These are diagnostics, including the larger C3 errors. Neither source is
discarded by a fixed prediction threshold. The learned map is applied to
actual Y before profile preparation; predicted GP responses are not used
as replacement observations.

## BF: unchanged kernel support on globally transformed Y

CR3/C1/C2/C3 all complete with the same 32-center scoring rule. They have
24/218/90/14 strict primary combinations, but **zero identical strict
four-model combinations**. The preceding BC2 10% metric becomes 5/6,
6/6, 5/6, 14/15; the lower grid gaps are 1.0257%, 0%, .001206%, .3213%.
The contextual logit fractional 1% variant gives 6/6, 5/6, 5/6, 9/15;
its lower grid gaps are 0%, .000180%, .01976%, .005643%.

The frozen BC3 frame scores 3.18435258 versus BF baseline 3.50867971,
**9.2436% lower** on the 10% metric. The learned global transformation
reduces this discrepancy but does not remove it. Full curves, approximation
errors and near-ties remain available for further investigation.

## Code verification and provenance

Tests cover dense-GP density/coefficients against sufficient statistics,
the Jacobian and monotone invertibility, equivariant dictionary geometry,
actual-row processing with forbidden-file sentinels, packed moments versus
explicit pairwise products, context interactions, configurable kernel
resolution, exact blind-score reproduction and the complete frozen-frame
metric family. The complete suite passes 213 tests. App-level read audits enumerate accesses; they are not an
OS-level sandbox. All new fit artifacts and full scores remain locally
under each source's `step5_matrix_metrics` directory and are not pushed.

Primary references:
[Zhang et al., Kernel-based Conditional Independence Test](https://proceedings.mlr.press/r9/zhang11b.html),
[Snelson et al., Warped Gaussian Processes](https://proceedings.neurips.cc/paper/2003/file/6b5754d737784b51ec5075c0dc437bf0-Paper.pdf).
The conditional moment construction here is an exploratory adaptation;
the papers' calibration guarantees are not asserted for it.
