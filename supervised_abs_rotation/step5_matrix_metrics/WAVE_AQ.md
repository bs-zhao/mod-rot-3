# GP automatic relevance for UX function contrasts

AQ completed CR3 with zero qualified full-plane criteria; later datasets
were skipped. The best score reaches 5/6 planes, baseline rank 2, for RBF2
logit, worst-U-row fractional support at a 3% relative tail. GP worst heldout
relative MSE is .00167 for poly5 and .00298 for RBF2; all declared fidelity
guards pass. Optimizer and center-count convergence remain unverified.

A Nystrom GP covariance model learns one inverse
lengthscale per U axis for each X-omission function ensemble. Primary
scores are normalized effective support and small relative tails of that
Ui-by-Xj relevance graph. Signal amplitude and residual noise are fitted
separately. Every X target is normalized to unit function energy.

Sources are the verified degree-five polynomial and RBF2 response banks,
fitted to all original WY rows. Source latent reconstruction error is
<=1e-7 relative squared energy per X task; task-specific output modes keep
1-1e-6 of the energy. X uses frozen degree-two omission maps on the observed
joint design. Scalar inputs are not assumed independent.

The objective is expected scalar GP negative log density over the
normalized function ensemble: .5*(logdet C + trace(C^-1 E[g*g^T])).
It averages function scores rather than asserting independent X profiles.
Woodbury evidence and posterior prediction for C=a^2*Phi*Phi^T+sigma^2*I
are checked against explicit dense covariance calculations.

There are 31 W-only signed/permutation classes, 12 X/channel tasks and
three starts: 1116 independent models per source. Inducing centers are 128
training-subject U vectors. Initial inverse lengths .3,1,3 are identical
on every axis and case. Thirty steps fit 75% of subjects, followed by a
heldout-subject fidelity check; sixty further steps optimize all 5000.
All arithmetic is float64. Jitter is 1e-6, Adam rate .03, gradient clipping
is independent per model. Bounds and exact settings are in the protocol.

Qualification requires GP relative MSE <=.01 for every X task/case in both
heldout evaluation and full refit, plus source-function adequacy. Poorly
approximated directions cannot qualify as a sparse explanation. Optimizer,
center-count and restart stability remain unverified after a budgeted run.

ARD and marginal-likelihood context: Rasmussen and Williams, GPML chapter 5,
https://gaussianprocess.org/gpml/chapters/RW5.pdf . This is a numerical prior,
not a supplied behavioral equation. No original neural weights are read.

Six GP tests pass: dense/Woodbury equivalence, independent model gradients,
CPU/CUDA agreement, reconstruction budgets, a W/Y-only complete pipeline,
and a positive sparse-variable example with its rotated version. The
preceding full suite passed 157 tests before that last positive check.
