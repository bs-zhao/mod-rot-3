# Pair correction and data-only nearest-neighbor conditional covariance

Completed on the user's Windows computer, 2026-11-12 America/New_York.
CI is a CR3 pilot; CJ completes CR3/C1/C2/C3 in that order. All main fits,
conditional-variance estimates and final scores use all 5000 subjects.
Only actual W/Y-derived response functions and explicit frozen artifacts
are used. No new Y evaluations, hybrid inputs or task-specific formulas
are introduced. The same criteria are retained for all models, with
nearby optima and continuous gaps instead of rigid MSE/zero-angle vetoes.

## CI: remove within-bin self pairs

CH's empirical group-mean quadratic includes each observation paired with
itself. CI tests the off-diagonal alternative for every proper U coalition:

    C_S = sum_g (s_g s_g.T - sum_{i in g} y_i y_i.T) / (N*(n_g-1)),

where s_g=sum_{i in g} y_i, and groups with fewer than two observations
contribute zero with coverage recorded. Full-U covariance remains the
observed-U endpoint. This removes the within-bin quadratic mean-estimation
self term. Output compression and partitions still use the same sample;
there is no claim of an independently unbiased population variance.

Finite corrected operators need not be PSD or nested. Their exact Shapley
sum still reconstructs observed covariance. For support, CI retains the
absolute eigenvalues of signed projected components, including negative
modes, and records their mass. It does not silently discard negative
statistical contributions or describe absolute spectra as a PSD covariance
decomposition. Conservation remains a hard numerical identity.

The fixed logit mean-X 10% mode criterion has a 15-degree CR3 coarse minimum,
score change -.395714%, close to CH2's -.398129%. CH3's new distant frame
remains lower by 1.54965%, close to its previous 1.62123% advantage. Thus
this self-pair correction does not resolve the displacement. Projected
negative spectral mass is negligible in this real pilot, about 1.76e-15
of absolute mass; indefinite behavior remains supported and audited rather
than assumed absent in general. Other CI models are not run.

## CJ: nearest-neighbor conditional residual covariance

Broto, Bachoc and Depecker,
[*Variance reduction for estimation of Shapley effects and adaptation to
unknown input distribution*](https://arxiv.org/html/1812.09168v2#S6.SS1.SSS1),
give a data-only nearest-neighbor conditional-variance estimator in their
equation (18). Their knn version uses already observed input/output pairs;
their separate mix version requires new response evaluations. CJ adapts
the data-only version to matrix-valued observed response functions.

For each proper input subspace S, every subject is an anchor. Find its
three nearest observed subjects in U_S, including the anchor where it is
uniquely closest. Use one shared global U scale; do not standardize axes
separately or assume an input distribution. Ties use W-only randomized
ordering with a recorded seed. If Y_a contains the three existing response
vectors in anchor a's neighborhood, estimate

    R_S = mean_a sample_covariance(Y_a), with denominator K-1 and K=3.
    C_S = C_observed - R_S.

An equivalent weighted-Gram computation avoids storing every local
covariance matrix. R_S is PSD, and this is checked numerically. Finite C_S
and their Shapley increments may be indefinite, so the same explicit signed
operator/absolute-spectrum policy as CI is used. The full-U endpoint is
the exact observed-U covariance, with repeated-identical-U uncertainty
retained separately. All Shapley increments sum exactly to observed
covariance. No regressor or response-function form is fitted.

Euclidean conditioning depends on the input subspace, not its choice of
orthonormal basis. Accordingly CJ caches by a W-only subspace projector,
unlike the axis-partition cache needed for CH/CI. Neighbor radii and tie
fractions are recorded. Three neighbors and the scoring definition were
fixed in the CR3 pilot and kept unchanged for the other models. This is a
finite matrix extension and complexity objective; the paper's asymptotic
conditions and population-identification guarantees are not asserted for
these observed NN response banks.

## Identical CJ criterion across all four models

Use actual logit responses, per-X projected absolute covariance-mode weights,
10% relative-L2 soft support, and mean over X columns. Normalize the joint
Ui-by-mode weights within each X before counting; optimize support rather
than total effect amplitude. Full results for all other registered metrics
and channels remain in each model's CSV files.

| Model | Raw grid angle | Axis deviation | Score change from reference | Baseline rank |
|---|---|---|---|---|
| CR3 | 75 degrees | 15 degrees | -1.32189% | 8/37 |
| C1 | 75 degrees | 15 degrees | -3.46156% | 2/37 |
| C2 | 75 degrees | 15 degrees | -6.63115% | 5/37 |
| C3 | 75 degrees | 15 degrees | -1.67999% | 5/91 |

The largest score gap is 6.63%, and all observed coarse minima have
15-degree axis deviation up to signs/permutations. This is a useful new
approximate common candidate. It is not discarded for failing exact
baseline rank one. Rank percentages remain separate: CR3 is 21.62% of the
supplied grid, not within a hypothetical 15% rank cutoff. The baseline's
gap is 18.84/10.28/26.54/13.07% of each model's full grid contrast. Absolute
gaps, positions and relative contrast are all retained, without choosing a
different formula for different datasets.

Strict plane diagnostics are 2/6, 5/6, 5/6 and 12/15; there is no common
strict four-model metric. Maximum negative fractions of projected absolute
spectral mass are .146%, .607%, .615% and .410%. Small negative mass does
not prove accurate population sensitivity estimates, especially in weak
modes. The three-dimensional versus five-dimensional neighborhoods have
different finite-sample resolution, recorded by radius quantiles.

In CR3, two older frozen frames remain lower by 1.423/1.156%; BP2 is worse
by 1.309%. The newer CF2/CG2/CH3 frames are worse by .251/1.902/2.851%.
This changes the ranking of transferred alternatives but does not prove
that a fresh search cannot find a lower distant frame.

CJ2 therefore freezes this criterion and uses fresh random seed 509 for
an anonymous-W plus 32 random-SO proposal search, two 80-evaluation Powell
starts, and a Givens pass. It uses all 5000 subjects and verifies source
score replay before optimization. The completed CR3 search has score
change -8.03747%, mean matched axis deviation 35.856 degrees and maximum
52.245 degrees. Selected proposal indices remain in the blind report. This point is
retained alongside the common 15-degree coarse-grid positions; neither
small grid gaps nor successful transferred checks prove localization.
Other CJ2 models are not run under the CR3-first policy. No global optimum
is presumed.

All logit/p fields above are logit_nn/p_nn from the actual frozen NN, not
generating-equation outputs. The four-model Y provenance audit checks the
stored NN logit/probability/argmax against the stored two-class NN scores
on every original row. Analytic logit/probability can aid discovery only
and cannot count as final empirical success. Source eligibility is separate
from the rotation-quality conclusions above.

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_quantile_conditional_variance.py --model CR3 --run-tag CI --audit-latest --pairwise-correction
python supervised_abs_rotation/step5_matrix_metrics/run_neighbor_conditional_variance.py --batch
python supervised_abs_rotation/step5_matrix_metrics/run_neighbor_variance_blind.py --model CR3
python supervised_abs_rotation/step5_matrix_metrics/report_predictive_variance_evidence.py
python supervised_abs_rotation/step5_matrix_metrics/write_approximate_shortlist.py
```

The literature audit also checked the withdrawn 2020 rank-statistics
preprint against its [2026 replacement](https://arxiv.org/abs/2605.23760).
CJ does not rely on unverified multivariate TSP or rank-estimator claims.
Its implemented conditional estimator is the observed-data kNN variance
formula above. No TSP algorithm is claimed as implemented.
