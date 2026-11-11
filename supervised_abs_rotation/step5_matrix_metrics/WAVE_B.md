# Wave B: empirical additive/pairwise predictive compression

This experiment was frozen after Wave A failed on CR3. Its definition is
the same for all four datasets. CR3 is screened first; an unsuccessful
CR3 screen stops the batch before launching C1/C2/C3.

## Inputs and profile construction

Only actual case W, common Y, and subject/trial layout metadata are read.
No generator, theta, decoder, saved R or preferred angle is a scoring input.
Every subject and every trial contributes to the final fits.

To reduce trial-wise W/Y to a function of subject coordinates, each subject
is fitted using the same generic X feature dictionary: degree-2 polynomial
features and a 64-center RBF Nystrom basis with lengthscale 2. Centers and
scaling use within-subject training trials. Trial indices divisible by 5
are held out to select one shared ridge, from 1e-6, 1e-3 and 0.1, using
mean normalized prediction error over the two Y channels. All trials are
then used to refit every subject. Nothing is specialized to a dataset's
behavioral equation or stimulus semantics.

Profiles are coefficients multiplied by a square root of the pooled
empirical X Gram matrix. Their squared Euclidean distance therefore
approximates a squared response distance over the common empirical X
measure. This is an approximation, and its within-subject held-out error
is reported. It does not establish that every subject shares the full X
support, nor that U and X are independent.

## Projection experiment

A fixed 60/20/20 subject split selects ridge on validation subjects and
reports predictive compression on test subjects. The selected ridge is
also used for final fits on all 5,000 subjects, which determine component
energies. The output metrics distinguish these two uses of the data.

For each anonymous case, fit vector-valued functions of U using:

- Additive bases: sum of functions of one U coordinate.
- Nested pairwise bases: additive terms plus every product of two
  one-dimensional basis groups.
- Four fixed basis resolutions: polynomial degrees 3 and 4, and
  one-dimensional Gaussian dictionaries with 3 and 4 quantile knots.

Groups are empirically centered and whitened on training subjects. U uses
one common scalar scale, preserving the orthogonal geometry. Signed
permutations preserve all score definitions. Any actual endpoint mismatch
is retained as a failure rather than ignored.

Metrics include additive/pairwise test NMSE, their prediction gap, tails
of all component energies, and tails of empirical pair-projection energies.
The latter project a pair dictionary out of the *whole* additive span
before measuring its fit to the residual. Pair dictionaries can still be
correlated with one another. These energies do not sum to a unique
variance decomposition and **are not Sobol indices**.

All metrics have one fixed direction: lower means simpler. A candidate
must satisfy the existing strict baseline-rank and endpoint tests for
every plane. Prediction adequacy requires both profile trial NMSE and the
worst-case pairwise subject test NMSE to be at most 0.10. Basis failure
and ranking failure are reported separately. This exploratory screen is
not a guarantee of identifiability, statistical confirmation, or arbitrary
rotation recovery.

## Run

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_additive.py --batch
```

The sequence is CR3, C1, C2, C3. CR3 failure stops automatically. Use a new
`--run-name` and seed for any changed protocol; old failures are retained.
Four BLAS threads are used by default to avoid excessive overhead in the
many small regressions. All changes in this wave stay in Step 5.

## Methodological context

[Additive Gaussian Processes](https://arxiv.org/abs/1112.4394) motivates
low-dimensional function composition. The present implementation uses
finite regression dictionaries and makes no Bayesian GP uncertainty claim.

[ANOVA kernels and RKHS of zero mean functions](https://arxiv.org/abs/1106.3571)
motivates centering. The present empirical projection is not an
implementation of that paper's exact kernel ANOVA decomposition.

[Generalized Hoeffding-Sobol decomposition for dependent variables](https://arxiv.org/abs/1112.1788)
explains why dependent empirical coordinates need more care than ordinary
independent-input Sobol indices. The present scores remain predictive
projection diagnostics, with no independent-input interpretation.
