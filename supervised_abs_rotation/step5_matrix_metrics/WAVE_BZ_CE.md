# Conditional fitting, nearby minima and predictive complexity

Completed on the user's Windows computer, 2026-11-12 America/New_York.
Every main conditional fit and final score uses all 5000 subjects. Scores
use actual case W/Y matrices and explicitly derived artifacts. Reference
angles and baseline labels enter only after scores are frozen. No known
response equation, original decoder, true theta or original saved R is used.

## Main approximate candidate and capacity checks

The retained BU criterion uses p_nn and, for each observed X column,
singular energies of the conditional Shapley response-function shares.
Normalize the combined Ui-by-function-mode energies within that X column,
then sum p/(p+.03^2) and average across X columns. This measures relative
support, including functional profile complexity; total coupling amplitude
is not the objective. It uses the same definition and settings in all models.

All-coalition conditional predictions use generic polynomial/radial
regressions with an unpenalized affine span and GCV-selected ridge. These
are approximations learned from the observed matrices, not supplied forms.
CB removes higher polynomial powers and uses 256 radial centers. CC uses
complete degree-five polynomial features and 64 radial centers. BU/BU3
use complete degree-three features and 32/64 radial centers.

| Model family | CR3 | C1 | C2 | C3 |
|---|---|---|---|---|
| BU, degree 3 + radial 32 | 0 / 0% | 0 / 0% | 15 / -1.77090% | 15 / -.589545% |
| BU3, degree 3 + radial 64 | 0 / 0% | 0 / 0% | 15 / -1.78523% | 15 / -.618530% |
| CB, affine + radial 256 | 0 / 0% | 0 / 0% | 15 / -1.76960% | 15 / -.638247% |
| CC, degree 5 + radial 64 | 0 / 0% | 0 / 0% | 15 / -1.99139% | 15 / -.603577% |
| CD, GCV choice among the above | 0 / 0% | 0 / 0% | 15 / -1.77587% | 15 / -.622947% |
| CE, CD with unassigned residual | 0 / 0% | 0 / 0% | 15 / -1.58817% | 15 / -.559810% |

Cells give observed coarse-grid axis deviation in degrees / score change
relative to reference baseline. C2's raw best angle is 75 degrees, equivalent
to 15-degree axis deviation when axis permutations are allowed. The coarse
grid alone does not establish uniqueness, a narrow optimum, or arbitrary
random-start recovery. The same observed targets are reused by these
related checks; they are not independent replications.

CD selects the lowest normalized GCV candidate for each W-only input
subspace, never by baseline rank, angle or dataset-specific weights. Saved
output bases are aligned by orthogonal transport with a recorded numerical
reconstruction bound. New subspaces fit the same four candidates using all
subjects. CE fixes a full-input conditional fit across coordinate frames,
removes its residual's equal Shapley allocation, and appends that residual
as an explicit unassigned group. The groups still sum exactly to the
observed response. No residual is presumed noise or discarded.

In CR3 the CE p_nn unassigned energy is only .0005%-.0030% of the respective
X-profile energies. Thus a large unresolved full-response residual does not
explain the existing CR3 displacement. CE preserves the useful approximate
positions but does not solve global identification.

## Frozen-point paired stability

BU6 refines each earlier blind best and an independently random-start end
point with one exact-score Givens pass at smaller steps. The lower refined
points have maximum matched axis deviations 7.524/1.664/24.098/15.707
degrees in CR3/C1/C2/C3, with score changes -1.12978/-1.16976/-3.87061/
-2.76939%. The CR3 point comes from the random-only arm. Both arms remain
in the audit: distant coordinates can also produce nearby scores.

BU7 compares each frozen refined point against baseline on 12 paired 80%
subject draws at each of the 32/64-center capacities. Full-5000 comparisons
first replay the frozen baseline and point. All 96 paired comparisons favor
the displaced point. Median changes in percent, with 5th-95th percentiles:

| Model | 32 centers | 64 centers |
|---|---|---|
| CR3 | -1.1006 [-1.3598, -.8125] | -1.1387 [-1.3830, -.8583] |
| C1 | -1.2257 [-1.6319, -.6616] | -1.1992 [-1.5796, -.6153] |
| C2 | -3.7131 [-4.6602, -2.7121] | -3.7453 [-4.7028, -2.7543] |
| C3 | -2.8727 [-3.1425, -2.6385] | -2.8656 [-2.9796, -2.6374] |

W geometry, the observed-X bank, response bases and scoring gauges stay
fixed. Conditional regressions are refitted and U/response means are
recentered per draw. This checks conditional-fit stability, not independent
original-Y generalization or a full response-bank refit. BU5 used the same
draws on earlier nearby points; the two sets are correlated evidence.

## Conditional-emulator quality

`audit_conditional_emulator_quality.py --batch` replays all proper
coalitions of the anonymous observed-W frame for BU/BU3/CB/CC. It verifies
saved coefficients against the corresponding affine-plus-ridge hat matrix
and computes exact fixed-fit PRESS residuals. Features, response basis,
chosen capacity and ridge remain fixed; selection is not nested.

CR3 p_nn mean relative PRESS MSE for one/two/three input dimensions:

| Family | 1 input | 2 inputs | 3 inputs |
|---|---|---|---|
| BU | .731552 | .470691 | .226141 |
| BU3 | .731552 | .470721 | .225973 |
| CB | .731776 | .470866 | .226112 |
| CC | .731591 | .471638 | .226518 |

Each proper coalition deliberately omits inputs, so its error includes
unpredictable omitted-input variation. These numbers are not full-function
approximation errors and must not trigger an absolute .01 cutoff. Compare
families for the same input coalition. The closely agreeing capacities
support a robustness observation; they do not prove population conditional
means are accurate. Hashes bind the target basis, saved models and scores.

## BZ/CA: feature-span and PRESS projection alternatives

BZ projects generic Ui innovation/context features out of the other-U
feature span, orthonormalizes the residual span, then projects actual
observed X-response functions onto it. SVD support includes function-mode
complexity. Numerical residuals below a recorded relative rank tolerance
are not promoted to unit features. The same unresidualized-span control
is retained. Feature whitening is a projection onto a span, not output
whitening or a known task formula.

CA uses exact fixed-W-space PRESS residual covariance reduction from lower
to full features. Negative eigenvalue mass is audited; only positive
improvements enter support. Finite predictive improvements can be indefinite,
so these are not asserted to be PSD population variance increments.

The same p_nn mode-support 3% criterion reaches 45-degree coarse minima
in all four models, with BZ gains 2.32/3.76/3.65/2.44% and CA gains
2.26/3.99/3.61/2.67%. Energy-only 3% support reaches deviations
15/0/15/30 degrees; its CR3 gain is about 1.19% across a total grid
contrast of only 1.33%. The baseline is almost the worst point on that
low-contrast CR3 grid. Small absolute gaps alone are insufficient to infer
a useful direction. PRESS closely reproduces the projection behavior, so
training-fit optimism alone does not explain it. All results are retained.

## Continuous reporting and the next variance experiment

`record_rotation_optima.py` now records absolute degrees, permutation-aware
axis deviation, baseline rank on the supplied grid, relative score gap,
gap as a fraction of the observed grid contrast, and 1/3/5/15% near-minimum
score bands. These quantities have separate units. Rank percentages refer
to the provided grid including endpoint permutations, not a probability
over random rotations. Near-minimum bands omit duplicated 90-degree
endpoints. A band may contain distant angles even if the best point is 15
degrees. No angle, rank or MSE threshold automatically discards a result.

CF investigates a distinct issue: squaring function Shapley shares assigns
a pure k-way interaction only 1/k of its original summed energy. In an
oracle squared-loss predictive-power game this dilution is absent. CF
builds predictive skill from fixed-fit PRESS, allows each coalition to use
its best subcoalition, and exactly allocates skill by Shapley weights.
An explicit unassigned-risk row makes every active X column sum to one.
The score therefore measures relative UX allocation support. Envelope
corrections and selection limitations are audited rather than hidden.
This is a finite-estimator proposal, not already verified population Sobol
effects. The CR3-first run is separate from the completed BZ-CE evidence.

The predictive-power motivation follows Covert, Lundberg and Lee,
[*Understanding Global Feature Contributions With Additive Importance
Measures*](https://proceedings.neurips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html).
CF adds its own finite PRESS and monotone-envelope choices; those choices
should not be attributed to that paper or called a direct SAGE replication.

Run commands from the repository root in py11:

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_shapley_profile_support.py --batch --degree 1 --centers 256 --run-tag CB
python supervised_abs_rotation/step5_matrix_metrics/run_shapley_profile_support.py --batch --degree 5 --centers 64 --run-tag CC
python supervised_abs_rotation/step5_matrix_metrics/run_adaptive_shapley_profiles.py --batch
python supervised_abs_rotation/step5_matrix_metrics/run_adaptive_shapley_profiles.py --batch --split-residual
python supervised_abs_rotation/step5_matrix_metrics/run_shapley_profile_resampling.py --batch --refinement
python supervised_abs_rotation/step5_matrix_metrics/audit_conditional_emulator_quality.py --batch
python supervised_abs_rotation/step5_matrix_metrics/run_predictive_shapley_variance.py --model CR3
```

Completed directories are frozen and verified before reuse. Incomplete
directories are retained for diagnosis rather than silently overwritten.
