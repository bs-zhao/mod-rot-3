# Sign-sensitive GP profiles and all-order conditional interaction profiles

Completed four-model experiments on the user's computer, 2026-11-12,
America/New_York. All main fits and final scores use all 5000 subjects.
Only actual case W/Y, row metadata and explicitly derived frozen artifacts
enter the scores. Nearby optima and continuous gaps remain in the ledger;
no exact-zero-angle or MSE=.01 veto is imposed.

## BW: positive-ray complexity of a fixed GP derivative function

The previously fitted isotropic BB GP is restored without refitting. Its
actual derivative response profiles g(U) are reconstructed from its own
saved parameters and actual matrix-derived targets. Before scoring, their
pointwise inner products must reproduce BB's frozen gradient covariance.
Across all four models the largest relative L2 replay error is 2.46e-11.
This verifies reconstruction of that fitted function, not the derivative
accuracy of the unknown original response.

For every candidate Ui direction and X-response function, append its norm:

    Psi(g) = (g, ||g||) / sqrt(2).

The lift preserves total squared profile energy exactly. Global axis sign
changes and orthogonal output-mode changes preserve its singular spectrum.
Rank one means all nonzero profiles lie on the same positive ray. A scalar
profile that reverses sign can have ordinary rank one but lifted rank two.
This is a precise sign-sensitive profile-complexity measure, not a count of
sign reversals, a general monotone decomposition, or a full roughness index.
Joint Ui-by-mode relative L2 support uses the same operator-mode rules as BS.
Ordinary unlifted spectra are retained as controls.

BW completes CR3/C1/C2/C3 with 0/20/0/0 strict primary combinations. The
largest SVD trace error across all tasks is 7.62e-14. One CR3 near-baseline
criterion continues identically in all four models:

Channel logit_nn,
`norm_lift_UX_operator_columns_worst_fractional_count_eps010`.

| Model | Best grid angle | Axis deviation | Score change | Strict planes (diagnostic) |
|---|---:|---:|---:|---:|
| CR3 | 15 | 15 | -.27193% | 5/6 |
| C1 | 0 | 0 | 0% | 6/6 |
| C2 | 45 | 45 | -16.14221% | 5/6 |
| C3 | 45 | 45 | -12.92742% | 9/15 |

The old BP2 frozen CR3 direction is also 15.176% lower on this criterion.
Thus sign sensitivity alone has not improved the common orientation result.
These are retained observations, not automatically discarded by a cutoff.

## BX: retain UiUk--X interactions instead of dividing all into individual Ui

For every proper U coalition S, reuse BU's frozen numerical conditional
response function m_S. Keep m_empty=0 and m_full=the observed centered
response profile. Compute all conditional function dividends

    D_S = sum_{T subset S} (-1)^(|S|-|T|) m_T.

All nonempty D_S sum to the actual observed response, up to its audited
numerical output compression. Their allocation back to each Ui as
sum_(S containing i) D_S/|S| equals BU's Shapley function share. Tests verify
both identities for four and six dimensions, and separately retain a sparse
pair interaction as its own hyperedge.

No input independence or orthogonality is assumed. The conditional
dividends are generally nonorthogonal: squared energies need not add to
the response variance. Therefore these are not population Sobol variance
components. Per-frame energy-sum/response-energy ratios, full-subset
residual energies and allocation checks are retained explicitly. Conditional
estimation artifacts can enter the higher-order dividends; exact algebraic
reconstruction does not by itself establish population interaction accuracy.

Within each X column, score U-subset-by-functional-mode support using the
same relative L2 tails. Energy-only U-subset--X support is also retained.
All nonempty subsets are included; no supplied interaction order is forced.

BX completes all four models with 0/16/3/0 strict primary combinations and
no strict common combination. All grid and old transferred-frame models are
replayed without a new conditional fit. Only the new BU2 frozen audit frame
requires fits: 14/14/14/62 proper coalitions in CR3/C1/C2/C3. The maximum
relative squared reconstruction error is 1.03e-32; maximum SVD trace error
is 1.91e-13. These are numerical checks, not statistical accuracy bounds.

The same p_nn operator-profile criterion in every model is
`conditional_dividend_UX_operator_columns_mean_soft_count_eps003`:

| Model | Best grid angle | Axis deviation | Score change | Strict planes (diagnostic) |
|---|---:|---:|---:|---:|
| CR3 | 15 | 15 | -1.50574% | 3/6 |
| C1 | 45 | 45 | -5.82944% | 1/6 |
| C2 | 45 | 45 | -.58759% | 4/6 |
| C3 | 45 | 45 | -4.25696% | 2/15 |

The three older CR3 counterframes and BU2's fresh frame are respectively
9.8743%, 16.7037%, 6.4090%, 1.1294% higher than baseline. This useful CR3
transfer behavior does not turn the other models' displaced grid optima
into nearby axis recovery.

The energy-only p_nn 1% fractional-support counterpart has gaps
-.15612%/-.86281%/-.29322%/-.24553% and grid angles 15/45/60/60.
All are recorded. Tiny value differences alone do not establish angular
localization; the full curve spans are in `four_model_optima.csv`.

## BU6: completed smaller-step continuation of the preferred BU criterion

See WAVE_BS_BU.md for the original definition and 96 paired refits. The
same p_nn 3% Shapley-profile criterion is unchanged. From the frozen BU2
best and independent-random-start frames, score exact +/- .02/.05/.1-radian
Givens steps in one pass. Every candidate uses all subjects and refits only
the necessary proper-coalition regressions under the frozen policy. Score
and frame hashes freeze before labels are read.

| Model | Lower of the two refined scores | Mean matched axis angle | Maximum matched axis angle |
|---|---:|---:|---:|
| CR3 | -1.12978% | 6.549 deg | 7.524 deg |
| C1 | -1.16976% | 1.526 deg | 1.664 deg |
| C2 | -3.87061% | 15.468 deg | 24.098 deg |
| C3 | -2.76939% | 10.723 deg | 15.707 deg |

CR3's lower refined point comes from the independent-random-start arm.
The random-only arms in C2/C3 remain farther away, at maximum offsets
41.821/56.731 degrees and score changes -.28022%/-.69273%. They are
preserved alongside the lower-score, closer-axis points. C3's observed-W
start was already equivalent to reference axes, as disclosed in the
initialization audit. No global uniqueness or arbitrary-start recovery
claim follows. The refined BU6 points have not received the earlier BU5
paired-refit audit; the 96 comparisons specifically concern BU2 frames.

## Next batch: empirical tree detail support

BY starts from CR3. Instead of minimum leaf counts at a fixed error level,
it scores relative L2 support of orthogonal empirical split contrasts in
a full multi-output CART representation of the observed response profiles.
The split contrast is sqrt(n_left*n_right/n_parent) times the difference
of child mean response functions. Root and leaf residual energy are kept
as an explicit unassigned block. Parseval reconstruction is audited.
This is finite empirical detail sparsity, not a claim of minimum rule
program length or a population smoothness estimator.

BY's CR3 pilot completes all 35 distinct frames, each with 5000 leaves.
The predeclared node-support 3% criterion has grid optimum 30 degrees and
score change -6.06491%; its full grid span is 14.85341%. The old BC3 frame
is 14.80708% lower, so the earlier continuous counterexample remains.
The axis-aggregated counterpart has a small -.61973% grid gap at 30 degrees,
but its entire grid span is only .74488% and no plane passes the strict
diagnostic. Its small gap is accompanied by weak orientation contrast.
All these values are retained. This is a CR3-only pilot, not a four-model
BY result. Earlier tree results and this coefficient-support variant remain
distinct experiments.

The first synthetic pipeline test exposed a NumPy integer in the JSON audit
metadata; integer conversion fixed it before the real-data pilot launched.
The complete suite now passes 250 tests, including actual coefficient
reconstruction and Parseval checks with unequal leaf sizes/nonzero residuals,
target-scale/output-basis invariance and the end-to-end matrix pipeline.

The shared matrix reader also now explicitly records its initial anonymous
W read in addition to subsequent coordinate reads. This fills an audit
listing omission and does not change scores. Historical audits remain
unchanged. Read logs are application-level records, not an OS sandbox.

## Commands and primary references

    python supervised_abs_rotation/step5_matrix_metrics/run_norm_lift_profiles.py --batch
    python supervised_abs_rotation/step5_matrix_metrics/run_mobius_function_profiles.py --batch
    python supervised_abs_rotation/step5_matrix_metrics/run_shapley_local_refinement.py --batch
    python supervised_abs_rotation/step5_matrix_metrics/run_tree_haar_profiles.py
    python supervised_abs_rotation/step5_matrix_metrics/write_approximate_shortlist.py
    python supervised_abs_rotation/step5_matrix_metrics/report_shapley_profile_evidence.py

- Sundararajan, Dhamdhere & Agarwal, *The Shapley Taylor Interaction Index*:
  https://proceedings.mlr.press/v119/sundararajan20a.html
- Tsai, Yeh & Ravikumar, *Faith-Shap: The Faithful Shapley Interaction Index*:
  https://jmlr.org/papers/volume24/22-0202/22-0202.pdf
- Elisha & Dekel, *Wavelet decompositions of Random Forests -- smoothness
  analysis, sparse approximation and applications*:
  https://www.jmlr.org/papers/volume17/15-203/15-203.pdf

These sources motivate explicit interactions and sparse function details.
BX is the full conditional Moebius transform, not a truncated Shapley-Taylor
or Faith-Shap implementation. BY uses empirical orthogonal binary split
contrasts of one CART tree; it is not the paper's random-forest algorithm.
