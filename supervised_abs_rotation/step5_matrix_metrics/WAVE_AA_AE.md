# Polynomial/kernel approximation followed by UX complexity

These are exploratory WY-only experiments. UX sparsity is the primary
scope. No common criterion has yet satisfied the user's four datasets.
Each batch starts CR3, then C1/C2/C3 only if the same criterion survives.

## Completed CR3 experiments

* `waveZ2_column_budget_seed406`: local support of Z's frozen UX energies,
  normalizing each anchor and X column before pooling. The flattened graph
  tail budget is two entries per X column, rather than a fixed top two for
  an arbitrarily large graph. No full-plane criterion survived. The user's
  two amplitude vectors and six replicated columns are tested explicitly.
* `waveAA_poly5_UX_all_trials`: the complete degree-five joint polynomial
  has 3003 coefficients, fitted to all 2,500,000 recorded rows with CUDA
  float64 sufficient statistics. Subject-validation NMSE is .00439294
  (logit) and .00700272 (probability). Exact all-trial mixed-derivative UX
  support still has no full-plane criterion.
* `waveAD_poly5_UX_term_complexity`: greedily compress only UX polynomial
  terms, with both pure-U and pure-X spans unpenalized. Compression uses
  all 5000 subjects and all 2,500,000 rows. Relative prediction tails and
  required counts, rather than coefficient magnitudes, measure sparsity.
  No full-plane criterion survived.
* `waveAB_poly4_UX_compression_pilot20` and
  `waveAC_rbf2_UX_compression_pilot20`: double-center the fitted polynomial
  or RBF function across the whole U and joint-X empirical blocks, then
  approximate the UX coupling by sums of separable products. Every factor
  is learned; no behavioral equation is supplied. There are 31 W-only
  signed/permutation classes, two channels and two starts. Ranks 1, 2, 4
  use 20 Adam epochs each, traversing all rows each epoch. There is no
  preliminary grid survivor. Rank-four mean relative squared errors remain
  about .351 and .393, respectively. These are under-resolved optimization
  pilots, not evidence that converged higher-rank compression cannot work.

All results and unsuccessful planes remain in their original directories.
No model-specific signs, chosen planes, original weights or saved rotations
were used. Higher capacity must be tested before accepting/rejecting the
separable-compression approach on approximation grounds.

## AE: empirical conditional UX support

`run_conditional_ux.py --batch` implements a distinct conditional-prediction
diagnostic. First fit polynomial/RBF functions to WY; double-center their
whole-U/whole-X coupling. For edge (i,j), remove Ui by cross-fitted prediction
from U without i, and remove Xj with the previously frozen held-out-X
prediction maps. The energy is

    ||R_(U without i) F_UX R_(X without j)^T||_F^2 / (N M).

Every U subject is scored out of fold; every source-fit row is retained.
Complete homogeneous polynomial blocks plus radial features make the
choice of orthonormal complement to Ui irrelevant in exact arithmetic.
The affine complement is unpenalized. Polynomial degrees 3/5 with 64/128
RBF centers and nested ridge selection are prescribed before case reveal.

Output compression balances every channel/X-omission task before PCA and
requires relative squared reconstruction error <=1e-7 for every task.
The final score is effective support or relative-tail support of UX energies;
per-X support is invariant to arbitrary positive scaling of each X column.
Total coupling strength is not the primary objective.

The observed dependence inside U and inside X is preserved. This is not
claimed to be the independent-input Sobol decomposition, or a causal graph.
Generalized dependent-input decompositions motivate the distinction:
https://arxiv.org/abs/1112.1788 . Our cross-fitted deletion estimator is not
claimed to reproduce that paper's algorithm.

Validation: 123 tests pass, including full synthetic W/Y-only pipelines,
free pure-U/X nuisance, exact double centering, conditional affine recovery,
per-X output-reconstruction bounds and the user's scale/sparsity example.

### Numerical audit and fresh run

The initial AE run stopped at the RBF function-compression guard after its
polynomial scores had been retained. A cutoff on the feature covariance
discarded small-variance kernel directions with large fitted coefficients.
It is marked `FAILED_NUMERICAL_COMPRESSION`, not a failed scientific score.
Launch code and partial artifacts are retained in that original directory.

The completed `waveAE_conditional_UX_SVD_seed407` uses direct observed-feature
SVD and measures reconstruction error in actual function values for every
X-omission task. Its RBF1/RBF2 worst relative squared reconstruction errors
are about 9.89e-8 and 9.83e-8, within the prospective 1e-7 budget. A regression
test explicitly keeps feature directions scaled down by 1e-8 when their
coefficients are scaled up by 1e8. The full suite passes 131 tests including
the higher-budget ALS work described in WAVE_AF_AG.md. CR3 has zero adequate
full-plane criteria; the best settings pass only 2/6 planes. C1/C2/C3 were
skipped. Endpoint equality holds (maximum error 2.84e-14), so the failure
is not an endpoint/sign-permutation numerical artifact.
