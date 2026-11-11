# Wave D: unknown monotone response links and additive compression

Protocol: `waveD_unknown_link_seed384`, CR3 first. The hypothesis is that a
numerically learned monotone response transformation might expose simple
composition even when the original response is not additive. This is a
model family being tested, not an assumed behavioral equation.

The numerical idea is motivated by
[Breiman & Friedman, Estimating Optimal Transformations for Multiple
Regression and Correlation](https://www.tandfonline.com/doi/abs/10.1080/01621459.1985.10478157).
The implementation is a finite-dictionary, monotone, regularized variant;
the original ACE convergence guarantees are not asserted here.

## Estimation

Use Wave C's numeric response-bank detection: all recorded common-X
responses if the X matrices repeat exactly; otherwise use the generic
all-trial X profile approximation and report its limitations and error.
All 5,000 subjects and all response columns enter fitting. No known link,
task equation, theta, decoder or saved rotation is read.

For each anonymous case and each Y channel:

1. Build an additive U dictionary: polynomial degree 3/5 or one-dimensional
   Gaussian kernels with 6/10 quantile knots. Center and whiten individual
   groups on training subjects. Fix empirical ridge at 0.001.
2. Start from standardized observed Y. Fit the additive dictionary jointly
   to all stimulus columns.
3. Regress additive predictions isotonicly on the *training* response
   values for each stimulus. Handle ties using weighted means. Standardize
   the monotone transform and mix in 10% standardized identity to prevent
   a flat or non-invertible transformation. Normalize again on training
   subjects. No response formula is supplied.
4. Repeat up to 12 times. Select the iteration using validation prediction
   error measured after inverting the learned link back to **original Y**.
   Test labels are never used to fit the link or choose the iteration.
5. Report original-Y test NMSE. Refit that precise iteration count on all
   subjects to compute component energy tails and direction variation.

Subject splits are fixed 60/20/20. Learned links use piecewise linear
interpolation, with boundary clipping; extrapolation error on held-out
subjects is therefore included in original-Y prediction error. The fixed
identity mixture is a regularizer and sensitivity to it is required for a
surviving candidate. This procedure is not a fully general unknown-link
estimator and does not guarantee recovery of any preferred coordinates.

## Scores and screening

Report unknown-link test NMSE, identity-link test NMSE, per-response
component tail-1/tail-2 L2, and component signed-variation cancellation
along its own U axis. All use one direction, lower-is-simpler, for every
dataset. The structural scores are conditional on the learned additive
approximation; interactions are diagnosed by its predictive failure rather
than asserted absent in the unknown function.

A candidate requires worst-case original-Y test NMSE across cases <=0.10,
an adequate response-profile approximation if used, baseline rank 1 and
strict endpoint preference on every plane. CR3 failure stops the batch.
Polynomial and Gaussian family agreement, resampling, resolution checks
and blind arbitrary rotation optimization remain necessary for any claim
of a robust universal result.

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_unknown_link.py --batch
```
