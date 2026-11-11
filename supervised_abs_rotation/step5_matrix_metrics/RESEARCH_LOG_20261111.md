# Continuation on 2026-11-11 (America/New_York)

The authorized goal remains open: one common WY-only effective-sparsity
criterion for C1/C2/C3/CR3, followed by stability and blind rotation
optimization. CR3 is always screened first. A partial or single-estimator
pass is not declared universal success. All numerical work uses the real
Windows checkout and py11 environment; unrelated local edits are preserved.

## Completed unknown-link experiment

`waveD_adaptive_seed384` completed on CR3. It had zero adequate full-plane
candidates. The best passing-plane count was 4/6, baseline rank 4/37,
for poly3 probability component direction variation. The remaining models
were not launched. Earlier slow partial runs have explicit stopped runtime
statuses, rather than false COMPLETE or scientific-failure labels.

Cached interpolation matched 370 original anonymous scores to 6.94e-18.
The adaptive batch reused a fit only when its stored validation history
selected exactly the same iteration under patience=3. This was tested in
the complete numerical pipeline. Test error and baseline identity were not
used for stopping or checkpoint reuse.

## Fixed cubic lead rejected by C1

`waveJ3_cubic_projection_seed389` prospectively fixed poly3 logit,
scale 0.2, worst-axis effective L0.5 separation support, before testing
the remaining datasets. This explicitly tested a low-order projection
statistic, without relabeling Wave J's higher-order/kernel failures.

| Dataset | Baseline rank | Passing planes | Fit adequate |
| --- | ---: | ---: | --- |
| CR3 | 1/37 | 6/6 | Yes |
| C1 | 15/37 | 0/6 | Yes |

C2/C3 were not launched after the fixed rule failed C1. This is not a
universal criterion. Source function banks now have immutable file
manifests and explicitly distinguish function preparation from case
scoring. No dummy case-score results are created for preparation.

## Wave L: direct joint U/X projection support

Completed CR3 as `waveL_joint_projection_seed391`. It fits complete
isotropic polynomial projections of degrees 2/3/4 directly on all
2,500,000 original W/Y rows, rather than compressing X into one response
mode. CUDA is used only for double-precision matrix sufficient statistics;
no original decoder or neural weights are read.

| Degree | Logit validation NMSE | Probability validation NMSE |
| --- | ---: | ---: |
| 2 | 0.0781663 | 0.0865902 |
| 3 | 0.0173500 | 0.0372309 |
| 4 | 0.00892709 | 0.0165713 |

There were zero full-plane candidates. The closest was degree-4
probability UU-cross-term fractional support at 1%/3% relative L2 error:
rank 3/37, 5/6 planes. The source control coefficient norms stayed
invariant under exact generic U-coordinate coefficient transformations.
No failed plane was excluded.

## Stronger numerical scale controls

Support normalization now divides by maximum energy before summation,
avoiding a false numerical zero at very small but representable scales.
Context/point balancing has no absolute amplitude floor. Global field
balancing uses a single maximum-point Frobenius norm, which preserves
primary support scores and rotation controls while keeping normalized units.

Tests cover overall amplitudes from 1e-152 to 1e152. On 19,536 actual
ANOVA/spectral energy vectors, the stable implementation changed scores
by at most 1.215e-11, below 1e-10. Old frozen outputs and their original
code revision are retained. The audit stays in ignored local outputs.

## Next methods

Prioritize actual prediction-compression tails at fixed relative error,
profiling out X-only nuisance functions and respecting empirical feature
correlations. This can distinguish effective sparsity from arbitrary
coefficient splitting among correlated dictionary columns.

Also investigate full joint U/X block-separation spectra. Single-U spectra
cannot identify axes of a purely linear U response; mixed U/X partitions
can distinguish sparse bilinear coupling from rotated dense coupling.
All method definitions must remain identical across the four datasets.

## Verification

## Waves M/N completed

`waveM_prediction_seed392`: empirical-support forward prediction compression,
after X-only nuisance projection, failed CR3. Best poly3 logit 10% relative
error count ranked 6/37 and passed 4/6 planes. Counts above 128 terms are
explicitly censored; no exact count is claimed for them.

`waveN_joint_spectra_resume_seed393`: full and double-centered mixed U/X
block spectra failed CR3. Best count 4/6, with poly3 logit scale-0.5 mean
soft support ranking 3/37. Both methods skipped C1/C2/C3 after CR3 rejected
all candidates. Norm and endpoint controls were checked. See WAVE_M_N.md.

The original N process was interrupted by an execution-host restart before
score checkpoints; its protocol records interruption, not scientific failure.
The identical protocol was rerun in a fresh output directory to completion.

74 tests now pass, including full WY-only M/N pipelines, forward-selection
prefix equivalence to direct least squares, aliased feature invariance,
observed Gram transport, and finite-support empirical marginal bases.

## Previous verification checkpoint

## P/Q: diagnostic and local-support failures

The input-only kNN entropy diagnostic P failed CR3: best neighbor setting
10 ranked 3/37 and passed 4/6 planes. It uses no response information and
is kept separate from response-simplicity evidence.

`waveQ_local_support_seed396` completed all polynomial3/5, RBF1/2 and
both channels on all subjects. It scored support before averaging, instead
of only pooling energy. No full-plane candidate remained; best count 3/6.
Its observed-X gradient contrasts are not actual UX partial derivatives.

83 tests passed after O/P/Q and actual joint derivative additions. Wave R
is running genuine mixed UX derivatives using all-data joint projections.
Its first three datasets leave exactly one common poly4/logit candidate:
UX-column subject-median fractional support at 10% relative L2 tail.
C3 and subsequent stability checks were pending at that checkpoint.

## R: one common grid candidate completed

C3 subsequently passed 15/15 for that same poly4/logit/UX-column
subject-median fractional-support rule (10% relative L2 tail). CR3/C1/C2
each pass 6/6, baseline rank 1 in all four. The exact common intersection
contains one row. `compare_prepared_runs.py` validates all score hashes,
source manifests, source/scoring code settings and actual numeric backends.

Response validation NMSE for the selected joint projection is CR3 .00892709,
C1 .0000809827, C2 .00146661, C3 .00294670. All 5,000 subjects and all original
rows were fitted. The initial derivative descriptor uses eight observed
contexts per person. S now evaluates 32, paired subject bootstrap, 128
random multi-plane rotations and 16 blind rotation-optimization starts.
Numerical convergence failures are retained explicitly, not hidden.

89 tests passed after exact derivative lowering, covariance-score
equivalence, piecewise rotation gradients and paired-bootstrap checks.
The full objective remained open pending those scientific validations.

## S rejects R's small-context candidate

`waveS_ux_validation_seed398` completed CR3. With 32 contexts the fixed
lead ranked 6/37 and passed only 2/6. Subject-bootstrap simultaneous
minimum-margin 5th percentile was -0.0550581, and zero bootstrap replicates
kept all interior margins positive. The baseline score 3.64068044 exceeded
the random-rotation minimum 3.40093147, an explicit global-recovery
counterexample. BFGS attained 3.33153902 but all starts reported precision
loss; no convergence or global minimum is claimed. Later models were skipped.

Next compute exact per-person derivative covariance over **every original
trial**, streaming actual W rows. Do not select a favorable small stimulus
subset, omit failed planes, or call the four-grid lead a validated success.

## T: exact all-trial moments also fail CR3

`waveT_all_trials_seed399` completed all 2.5 million rows for each of
degrees 2/3/4, with per-person UU/UX/gradient moments. It found no full-plane
candidate; the best count was 5/6, rank at least 4/37. Other datasets were
skipped. Exact covariance transport matches brute-force local derivatives
and supports unequal trial counts. See WAVE_T.md.

## U: link-invariant all-trial UX geometry fails CR3

`waveU_link_ux_seed400` completed eight raw/profiled/tangent, point/column
normalization variants for degrees 2/3/4 and both channels. Every original
row contributed. No full-plane candidate survived; best count 5/6, rank
2/37 for some variants. Other datasets were skipped. Local chain-rule
invariance, rotation equivariance and extreme-scale controls passed, as did
the complete WY-only pipeline. See WAVE_U.md.

Next move actual UX derivative evaluation to a nonparametric U-kernel
times a generic X dictionary, with raw held-out-subject prediction audit.

## V: nonparametric U kernel with actual UX derivatives

`waveV_product_kernel_seed401` completed CR3 RBF1/2 on all subjects and
all raw rows. The product dictionary permits arbitrary cross-basis
coefficients. X interpolation ridge uses training U subjects only; a
separate training-only U fit predicts the original held-out raw Y.
Validation NMSE was .0161565/.0352302 for RBF1 and .0150766/.0344631
for RBF2 (logit/probability). No adequate full-plane candidate survived.
Other datasets were skipped. See WAVE_V.md. 97 tests passed.

## V2/W/X/Z continuation and user's UX emphasis

V2 improved raw CR3 logit validation NMSE to .00597715 for U-RBF2 while
retaining every original row; it still had no full-plane criterion.
W's full joint product-reference ANOVA retained seven common settings
through CR3/C1/C2, but C3 rejected all seven. X's observed-neighborhood
contrast design passed its full-rank guard at k=32/64 but found no adequate
full-plane CR3 criterion. See WAVE_V2_W_X_Z.md for complete definitions.

The user reiterated approximation followed by complexity measurement and
made UX sparsity the priority. Z directly implements that separation:
polynomial/RBF U-response functions first, held-out-X omission contrasts,
then normalized support of Ui-Xj first/total-effect energies. It preserves
X's observed joint design and does not read a task equation. CR3 found no
adequate full-plane candidate; some settings reached 5/6, rank 2/37.
No remaining dataset was launched after the empty intersection.

The initial separated-rank infrastructure was synthetic-only at that point.
It has since run as AB/AC; the subsequent actual pilots and their unresolved
approximation error are documented in WAVE_AA_AE.md.

## UX-focused approximation/complexity continuation

AA completes a degree-five full-W approximation on all 2.5 million CR3 rows
(logit validation NMSE .00439294). Its exact all-trial UX support fails the
strict six-plane rule. AD compresses only the UX polynomial terms after
profiling both pure-U and pure-X nuisance; it also has no full-plane pass.
AB/AC compress fitted polynomial/RBF UX functions with learned separable
products. Their rank-four, 20-epoch pilots have no grid survivor and retain
substantial prediction error (.351/.393 mean relative MSE), so remain
provisional rather than a conclusive method rejection.

AE now tests cross-fitted conditional Ui/Xj deletion on observed whole-block
empirical designs. It emphasizes per-X normalized UX effective support,
uses all 5000 subjects in out-of-fold scoring, and checks each X task's
function-reconstruction error before scoring. 123 tests passed before
launching the real CR3 batch. See WAVE_AA_AE.md and the retained run folders.

AE's feature-covariance cutoff failed its RBF reconstruction guard; the
fresh direct-SVD run passes reconstruction but has zero CR3 criteria
(best 2/6 planes). AF/AG exact-grid ALS improves separated approximation to
rank-16 mean relative MSE .0312556/.0789873 without a grid survivor. AI/AJ
are extending the preserved fits to ranks 32/64.

AH tests minimum U-variable subsets for each held-out-X contrast, rather
than one-variable deletion importance. All 5000 subjects are scored out
of fold. Its full-U approximation is adequate, but no full-plane CR3
criterion survives (best 3/6 planes). AH2/AF2/AG2 classify dominance scores
as controls without changing numerical-score hashes. Budget-one tails,
budget-curve area and 30% counts are excluded from primary sparse-support
claims in response to the user's explicit amplitude example.

AI/AJ finish rank 64 without a grid survivor (mean relative MSE
.00151381/.0203627). AL's free-X-profile polynomial term compression also
fails (best 3/6), despite its degree-seven RBF-function projection error
<=.000230. AM's globally integrable output correction reaches 5/6 planes,
baseline rank 2, but still fails the strict CR3 screen. AN's normal-rank
U-coordinate convention fails with adequate approximation (best 3/6).
AK all-point support, AO conditional X-balanced global correction and AP
bounded-rank input coordinates are still running. 152 tests pass. See
WAVE_AK_AL.md, WAVE_AM.md and WAVE_AN_AO_AP.md.

## Wave O: degree-resolved and pure-coupling structure

`waveO_resolved_seed394` completed CR3 with no full-plane candidate.
It explicitly removed U-only and X-only product-reference components
before mixed-block spectra, and separately scored homogeneous U orders
and equal-weighted X profiles. Settings remained prospective and shared.

Several degree-resolved supports reached 5/6 and rank 2/37; for example
poly3 probability, cubic U-axis-subset fractional support at 1%/3% tail.
The failing sixth plane was retained, so C1/C2/C3 were skipped.
Pure-coupling norm controls remained invariant. None of these exploratory
near-passes establishes the requested common criterion.

65 tests passed before the next statistics-bank extension, including
actual CUDA-vs-NumPy double-precision checks, exact joint polynomial
coordinate changes, the user's sparsity example, extreme rescaling,
immutable function manifests, and source-only complete pipelines.
