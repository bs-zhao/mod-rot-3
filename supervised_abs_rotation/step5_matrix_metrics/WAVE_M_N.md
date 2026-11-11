# Waves M/N: empirical prediction compression and mixed-block spectra

Both use an immutable `joint_bank_seed391`, fitted to all original W/Y
rows and all 5,000 subjects with degrees 2/3/4. Subject-held-out response
NMSE selects ridge; final projections use everyone. The bank separately
stores empirical feature Gram matrices on all and held-out rows, with a
SHA256 file manifest. Function preparation creates no case scores.

Run `run_predictive_compression.py --batch` or
`run_joint_separation.py --batch`. Every batch screens CR3 first, then
C1/C2/C3, and stops at an empty intersection of identical estimator,
channel and metric settings. Case names enter evaluation after frozen
anonymous scores. No true parameters, saved R or original decoder is read.

## M: actual-support prediction compression

Profile out the whole X-only polynomial span using training-subject Gram
matrices. Normalize remaining feature columns in the training metric.
Forward orthogonal least squares selects the atom with largest conditional
training-error reduction, refitting all selected coefficients. Duplicate
prediction columns cannot increase support. Compute each selected training
fit's error on held-out subjects against the fixed full-data projection.
This is held-out **compression**, not a new raw-response validation estimate.

Scores are the best-prefix relative prediction L2 errors at budgets
2/4/8/16/32/64/128, and interpolated term counts needed for relative L2
errors 1%/3%/10%. The prefix envelope is explicitly a descriptor selected
on compression-validation errors. Counts not reached within 128 terms
are censored bounds, stored with flags; the numeric sentinel 129 does
not assert an actual count. Response scale cancels. Tiny numerical nulls
are excluded by relative feature/singularity thresholds, not by effect size.

Tests compare every selected prefix with direct training least squares,
and cover correlated columns, duplicate atoms, column rescaling, coordinate
transport and the user's two sparsity vectors. Rotation transport of both
training and validation conditional Gram systems matches direct rotated rows.

CR3 completed with **zero full-plane passes**. Best: poly3 logit,
10% prediction count, rank 6/37 and 4/6 planes. Conditional U prediction
energy was positive for every fit. Endpoint errors were numerical noise.
Other datasets were skipped; this is a scientific failure of this screen.

## N: joint U/X block separation

Represent the fixed projection in a product orthogonal basis: centered
isotropic Gaussian U, scales 0.5/1/2, times observed univariate X marginals.
Degree-ordered empirical X polynomials remove null higher degrees on finite
support. This is a declared product-reference descriptor, not a claim
that the observed X coordinates or U/X are independent.

For every pair (Ui,Xj), unfold the full coefficient tensor into that pair
versus all remaining variables. Squared singular values measure separation
energy. Report the same effective-support and relative-tail metrics on
each spectrum, averaged and maximized across all pairs. Also score the
double-centered block (remove the constant row and column). Total product
reference function norm must stay invariant under arbitrary U rotations.

`waveN_joint_spectra_resume_seed393` completed CR3 with **zero full-plane
passes**. Best passing-plane count: 4/6. A poly3 logit full-block mean
soft count at Gaussian scale 0.5 ranked 3/37. Other models were skipped.
The initial run had no score checkpoints when the execution host restarted;
it is explicitly marked INTERRUPTED_TOOL_HOST_RESTART, not a metric failure.

Full code verification after these additions: **74 tests passed**, including
complete WY-only pipelines for both methods and actual CUDA comparisons.

Related primary reading (the implementations here are custom diagnostics):

- [Tropp and Wright, computational methods for sparse approximation](https://tropp.caltech.edu/papers/TW10-Computational-Methods-preprint.pdf)
- [Gorodetsky et al., functional tensor train](https://arxiv.org/abs/1510.09088)
- [Adcock et al., compressed-sensing polynomial approximation](https://arxiv.org/abs/1703.06987)

Next distinguish pure X/U nuisance from UX coupling in the full tensor,
and inspect homogeneous U degrees without mixing their response amplitudes.
