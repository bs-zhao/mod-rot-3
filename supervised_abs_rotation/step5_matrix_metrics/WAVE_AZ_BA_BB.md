# Ordinal response calibration and actual GP effect support

Prediction error is a continuous diagnostic throughout this continuation.
No fixed .01 MSE cutoff excludes a structural candidate.

## AZ: one global response calibration

Each actual Y channel is transformed once over all original rows by
midrank probability followed by the normal quantile function. This is a
fixed ordinal calibration, not a learned warped-GP likelihood, and does
not assume a Gaussian joint input/output distribution. It is invariant
to strictly monotone output unit changes, subject to the observed ties.
The transformation can change additive-interaction structure; the resulting
scores describe the ordinal response rather than claiming an unchanged
original-scale causal graph.

For unshared X, actual Y rows are calibrated before generic-X-profile
interpolation. On a shared grid every transformed value is checked exactly
against actual transformed Y. GP subject validation is conditional on this
frozen all-data calibration; it is not an independent end-to-end validation
of a transformation estimated without the heldout Y values.

AZ uses the same W-only conditional X witnesses and 128-center GP policy
as AT: three starts, 30 steps on 3750 subjects, 60 further steps on all
5000, 1e-6 output-mode squared-energy tolerance. ARD lengthscale support
fails CR3, best 1/6 planes, rank 7/8. Worst GP heldout relative MSE is
.00018042, refit .00008275. No prediction cutoff explains this failure.

## BA/BA2: evaluate fitted functions, not lengthscale parameters

The saved best GP hyperparameters are reconstructed without optimization.
Analytic derivatives and centered finite changes (.05/.2 common U-scale
units) of the posterior mean provide Ui-by-Xj effect energies. Per-X
normalization precedes effective-support and small-relative-tail summaries;
pointwise subject summaries and pooled effects are both recorded.

BA, using the raw AT response source, has no six-plane CR3 score. Best
pooled worst-X supports reach 5/6, rank 2. Some gaps to the lowest grid
score are only .01%--.12%; these remain near-ties to investigate. Recreated
GP prediction errors agree with the saved fits to 9.12e-18. Function-value
accuracy does not establish derivative accuracy.

BA2 applies the same function-effect scorer to AZ's ordinal source and
has 12 primary CR3 six-plane combinations: pooled worst-X soft support
at 1%/3% thresholds, for derivatives and both finite scales, in the two
highly related output channels. These are correlated exploratory variants.
`run_rank_posterior_wave.py` continues the frozen family through C1/C2/C3,
reporting prediction errors without a rigid veto. The continuation is
currently running; there is no validated universal result.

## BB: one fixed radial GP function per task

BB fits each X/channel function only once with a radial GP. Equal initial
inverse-lengthscales and equal projected parameter gradients constrain
all axes to the same learned lengthscale throughout fitting. Actual
posterior gradient covariance is evaluated at every subject. Every case
then rotates this *same* fixed field exactly; there is no per-case refit.

The primary blind metric is pointwise X-mean, subject-mean soft support
at 10% relative activation. There are 64 random SO rotations and two
100-evaluation blind Powell searches, with labels read after scores freeze.
The raw-response four-model run is in progress. CR3 has no strict grid
candidate and a blind primary loss 2.74354 versus baseline 3.29189. C1/C2
have strong grid signals, with primary optima 1.50595/2.18349 against
baselines 1.51366/2.22958. C3 remains to be assessed.

The preceding full suite passes 194 tests. Four additional radial-GP tests
pass: exact radial constraint/fit equivariance, covariance versus actual
derivative energy, exact field rotations/primary scoring, and a complete
one-fit matrix-only pipeline.

Background on output transformations:
[Snelson et al., Warped Gaussian Processes](https://proceedings.neurips.cc/paper/2003/file/6b5754d737784b51ec5075c0dc437bf0-Paper.pdf).
This implementation uses empirical rank calibration, not that full model.
