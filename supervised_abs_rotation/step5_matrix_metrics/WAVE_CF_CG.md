# Predictive variance allocation and coherent polynomial covariance

Four-model grid and transferred-frame experiments completed on the user's
computer on 2026-11-12, America/New_York. Order: CR3, C1, C2, C3. All main
fits and scores use all 5000 subjects, actual W/Y-derived response profiles,
and the same criterion/selection policy in every model. Nearby positions,
small gaps and unsuccessful candidates are retained without a fixed MSE
or exact-baseline veto. Blind-search continuation is reported separately.

## Why another variance allocation?

BU allocates response functions by Shapley weights, then measures squared
profile energies and spectral complexity. For an independent pure k-way
interaction F, the function shares are F/k, so their summed squared energy
is Var(F)/k. This is mathematically valid for function explanation but can
underweight interactions relative to main components in a sparsity score.
CF/CG investigate variance allocation before support counting. Neither
assumes a known case-specific response form or independent U coordinates.

## CF: finite PRESS predictive-power game

Use the same observed X-response targets, output compression and generic
conditional-model library as CD: polynomial/radial 3/32, 3/64, affine/256,
and 5/64. GCV chooses the conditional model using all subjects, with no
reference angle or case-specific score weights. For each U coalition S and
each channel/X task, restore its fixed affine-plus-ridge smoother and form
exact PRESS residuals. With R(S) its mean squared PRESS error, define

    raw skill(S) = 1 - R(S)/R(empty).
    game(S) = max over T subset S of clip(raw skill(T), 0, 1).

The empty predictor is the leave-one-out mean. The monotone envelope lets
a coalition ignore extra inputs, and guarantees nonnegative exact Shapley
increments. It is part of this finite-estimator definition, not an assertion
that the fitted predictors are oracle conditional means. Raw skills and
envelope corrections are saved. The largest envelope increases across all
grid/transferred frames are 0/.002259/.001603/.002246 in CR3/C1/C2/C3.
Thus the CR3 outcome does not depend on an envelope correction.

Exactly allocate game(full) to the d U variables. Append 1-game(full) as an
unassigned-risk row, rather than dropping unexplained variance or calling
it noise. Every active X column then sums to one, with maximum recorded
conservation errors below 4.5e-16. Compute the same relative L2 effective
support and tail statistics on this UX allocation matrix. There is no
optimization of total effect amplitude. Zero-energy tasks remain zero.

The full-input predictor is fixed across coordinate frames. Model family,
ridge, W-feature maps and the response basis are fixed for PRESS; selection
is not nested. Best-subcoalition selection may still exploit finite-sample
CV noise. The result is a predictive-power diagnostic, not an independently
calibrated population Sobol effect or a direct SAGE replication.

## CF: retained four-model results

Cells below give permutation-aware coarse-grid axis deviation in degrees /
score improvement relative to reference baseline, in percent. Smaller
scores are preferred. A 75-degree raw minimum is a 15-degree axis deviation.

| Identical criterion | CR3 | C1 | C2 | C3 |
|---|---|---|---|---|
| logit, worst-X 3% soft support | 0 / 0% | 0 / 0% | 15 / -.748948% | 15 / -.224194% |
| p_nn, mean-X 3% soft support | 15 / -.361734% | 0 / 0% | 15 / -.070060% | 30 / -.279738% |
| p_nn, mean-X 1% fractional support | 15 / -.035384% | 45 / -.363689% | 0 / 0% | 30 / -.035491% |

The first row's strict plane diagnostics are 6/6, 6/6, 5/6 and 11/15.
It is a useful new approximate common candidate and is continued to blind
search unchanged. It was identified during the CR3 screen, before seeing
the other three models. The p_nn mean-X row was the prospectively preferred
view for the CF wave and remains in all reports. There is no identical
strict four-model combination among the full set.

For that unchanged logit criterion, baseline ranks are 1/37, 1/37, 4/37
and 7/91: one-based supplied-grid rank percentages 2.70%, 2.70%, 10.81%
and 7.69%. All are within the first 11% of this grid, separately from the
15-degree displacement statement. The fractions of other grid points
strictly below baseline are 0%, 0%, 8.33% and 6.67%. These supplied-grid
quantities include 90-degree permutation endpoints; they are not random-SO
probabilities or a replacement for the blind search.

Some allocation-support curves have small total contrast and wide near-minimum
bands. For example, the CF p_nn CR3 mean-X 3% gap is .3617% within a total
grid contrast .5584%; baseline is 64.8% of that contrast away from the grid
minimum. Reporting only its small absolute score gap would conceal this.
The ledger preserves both measurements, supplied-grid rank percentages,
and 1/3/5/15% near-minimum score bands. These are complementary diagnostics,
not new rejection thresholds.

## CG: coherent nested polynomial explained covariance

CF uses scalar prediction errors and separately fitted coalition models.
CG instead preserves an operator-valued function-complexity description
while using one coherent nested family. Form complete homogeneous polynomial
features with multinomial weights. Each block's feature inner product equals
the corresponding power of the input inner product. Every degree block uses its full-input empirical
norm scale, fixed for every coordinate frame and every input subset.

Full-input GCV selects degree from 2/3/5 and relative ridge from
1e-8/1e-6/1e-4/.01/.1, then freezes both degree and the resulting absolute
ridge tau for all coalitions. All four models select degree 5. This is a
finite generic approximation range; the boundary selection remains visible.
There is no task equation supplied to the fit and no separate subgroup
normalization or ridge retuning that would destroy nesting.

For centered feature matrix X_S, define

    H_S = X_S (X_S.T X_S + tau I)^(-1) X_S.T.
    C_S = Y.T H_S Y/N.

For nested input subspaces, the tensor-feature kernels satisfy K_S <= K_T
in PSD order. Centering preserves the order. With the same absolute tau,
H_S <= H_T, so all covariance increments C_T-C_S are PSD. Exact Shapley
weights allocate these increments to each Ui. Append

    C_unassigned = Y.T (I-H_full) Y/N.

The components sum to actual observed covariance Y.T Y/N. This regularized
unassigned covariance is not the squared fitted residual covariance, which
would contain (I-H_full)^2. Both prediction error and the unassigned trace
are reported separately. Per-X projected component eigenvalues give joint
Ui-by-function-mode support; their traces give the energy-only graph view.
Only negligible negative roundoff eigenvalues are clipped, with numerical
PSD/trace/conservation audits. No population conditional-variance or causal
identification claim follows from this feature-space construction.

## CG: retained four-model results and blind continuation

| Identical p_nn criterion | CR3 | C1 | C2 | C3 |
|---|---|---|---|---|
| joint U-by-mode 3% support | 15 / -1.37070% | 0 / 0% | 15 / -3.06148% | 45 / -1.47837% |
| U-allocation 3% support | 15 / -.314104% | 0 / 0% | 0 / 0% | 30 / -.101219% |

The joint-mode row has plane diagnostics 4/6, 6/6, 3/6 and 5/15. The
energy-only row has 0/6, 6/6, 6/6 and 12/15. No identical strict four-model
combination exists. CG therefore does not uniformly improve the earlier BU
approximate positions. Its coherent PSD covariance identity is a valid
numerical construction, not evidence that its objective must recover axes.

CG2 applies the exact all-subject p_nn joint-mode criterion to anonymous
W plus 32 random SO proposals, two 80-evaluation Powell starts and a Givens
pass. Its completed CR3 result has score change -1.85661%, mean matched
axis deviation 6.596 degrees, maximum 9.197 degrees. The remaining-model
searches are separate from the completed four-model grids. CF2 tests the
unchanged CF logit worst-X candidate, using W plus 16 random SO proposals,
one 40-evaluation Powell start and a Givens pass. Both retain signs and
axis permutations as equivalent and freeze scores/frames before angle
assessment. Search starts and the random-only arms are recorded separately;
the anonymous observed-W frame already equals reference axes in C3.

The finite searches do not prove global optimality. A search that finishes
above the known evaluation baseline has failed to find that baseline within
its budget; it is not, by itself, a counterexample to the objective.

## Reproduction and reports

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_predictive_shapley_variance.py --batch
python supervised_abs_rotation/step5_matrix_metrics/run_nested_polynomial_variance.py --batch
python supervised_abs_rotation/step5_matrix_metrics/run_predictive_variance_blind.py --model CR3
python supervised_abs_rotation/step5_matrix_metrics/run_nested_variance_blind.py --batch
python supervised_abs_rotation/step5_matrix_metrics/report_predictive_variance_evidence.py
python supervised_abs_rotation/step5_matrix_metrics/write_approximate_shortlist.py
```

The global `outputs/supervised_abs_rotation/predictive_variance_evidence`
folder contains full-plane plots, selected grid positions, transferred-frame
scores, numerical/quality checks and completed blind-search assessments.
Each model's full metrics and frozen artifacts remain under its own step5
directory. Existing completed runs are verified and reused; incomplete
directories are retained. Tests cover pure-interaction variance allocation,
permutation/sign/rotation equivalence, PSD monotonicity, conservation and
reload/replay of both blind scorers.

Primary predictive-power motivation: Covert, Lundberg and Lee,
[*Understanding Global Feature Contributions With Additive Importance
Measures*](https://proceedings.neurips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html).
Our fixed-fit PRESS, monotone envelope and covariance smoother are explicitly
separate implementation choices and are not attributed to that paper.
