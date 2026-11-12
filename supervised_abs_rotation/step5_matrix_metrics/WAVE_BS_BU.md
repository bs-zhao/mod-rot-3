# Operator complexity and conditional Shapley function profiles

Experiments on the user's Windows computer, 2026-11-12, America/New_York.
All main fits use 5000 subjects. Every score derives from the actual case
W/Y matrices and row metadata. No generator, true theta, original decoder
weights or original saved rotation is read. Nearby optima are retained;
neither MSE=.01 nor exact baseline rank is an acceptance veto.

## BS/BT: retain functional modes instead of only edge energy

The conditional Ui witness and observed-X response functions are unchanged
from BC2/BR. For each Ui/Xj cross-moment matrix, BS/BT retain its squared
singular values rather than only its squared Frobenius norm. Ui-by-mode
entries share a relative L2 tail budget within each X column. Thus a weak
edge or a weak functional mode can enter the tail; a large edge is not
automatically judged dense. No edge-versus-complexity mixture is tuned.
The direct SVD trace is checked against the original cross-moment norm.
Ui-without-context remains an explicitly marked control.

BS uses raw responses; BT applies the previously frozen, polynomial-GP
learned global output map to the observed Y values. Both complete all four
models. Strict primary-combination counts in CR3/C1/C2/C3 are BS 1/31/1/1
and BT 1/30/0/0; neither has an identical strict combination across four.

One retained BT p_nn criterion is
`Ui_innovation_with_context_UX_operator_columns_worst_fractional_count_eps010`.
Its grid axis deviations are 0/0/15/15 degrees and C2/C3 improvements over
baseline are only .04493%/.87827%. However, three frozen CR3 directions
are approximately 15%--18% lower. This transfers the earlier continuous
counterexamples; the near-baseline grid pattern alone is insufficient.

## BU/BV: allocate complete observed UX response profiles

For each normalized observed X-response function F(U), estimate E[F|U_S]
for every proper U coalition S with the same generic complete polynomial
degree 3 plus 32-center radial regression. The affine span is unpenalized;
GCV chooses among the same five ridge values for all coalitions and models.
The full coalition is the observed centered F, rather than its prediction.
The empty coalition is zero. Exact finite-coalition Shapley weights give

    Phi_i = sum_{S not containing i} |S|! (d-|S|-1)! / d!
            * (estimated E[F|U_(S+i)] - estimated E[F|U_S]).

All interactions in the full observed profile are allocated: sum_i Phi_i=F
to numerical precision. Independence of the rotated inputs is not assumed.
These are estimated conditional **function shares**, whose squared spectra
are scored; they are not exact population Shapley variance effects.

Source output modes are numerical SVD compression of the response matrices,
not assumed response equations. A shared response basis permits all tasks
to use the same conditional fits. Per-task compression errors, allocation
efficiency, SVD energy traces, actual read paths, frozen scores and labels
are retained. Conditional model generalization has not been independently
verified. Proper-coalition prediction error also contains variation from
omitted U; it is not a direct approximation error for the full response.

BU uses raw Y profiles. BV uses the learned global output map: its CR3
pilot has zero strict primary combinations and has not been extended to
the other three models. Its pilot is retained rather than reported as a
completed four-model experiment.

## Retained same-definition four-model candidate

Channel p_nn, metric
`shapley_profile_UX_operator_columns_mean_soft_count_eps003`.
It counts Ui-by-functional-mode support with soft threshold eps=.03
relative to the total per-X L2 energy, then averages equally across X.
The same metric, channel, direction and capacity policy are used throughout.

| Model | BU best grid angle | Axis deviation up to signs/permutations | Score change from baseline | Strict planes (diagnostic) |
|---|---:|---:|---:|---:|
| CR3 | 0 | 0 | 0% | 6/6 |
| C1 | 0 | 0 | 0% | 6/6 |
| C2 | 75 | 15 | -1.77090% | 4/6 |
| C3 | 15 | 15 | -.58955% | 11/15 |

BU3 repeats every model with 64 radial centers. Positions and plane counts
are unchanged; C2/C3 gaps are -1.78523%/-.61853%. The general-capacity
runner refuses to reuse a completed run with different capacity arguments.

The three frozen CR3 counterexamples from BC3, BK2 and BP2 have BU primary
scores respectively +11.9397%, +15.2018%, +15.6697% above baseline.
Their BU3 counterparts are +11.5965%, +15.0120%, +15.4283%.
This is a useful transfer check and does not replace a new blind search.

## BU2: new blind searches, separate from the supplied case pool

Starting with the anonymous W frame and 32 fresh random SO frames,
the two best proposals receive 80-evaluation Powell searches and one
Givens pass with steps +/-.1 and +/-.25 radians. No labelled case frame
is an optimization start. Source conditional models are replayed on an
anonymous case before searching. New conditional models live in a new
folder; the source run is immutable. Scores and frames freeze before labels
are read for sign/permutation-matched axis assessment.

| Model | Score change | Mean matched axis angle | Maximum matched axis angle |
|---|---:|---:|---:|
| CR3 | -.27454% | 12.721 deg | 17.309 deg |
| C1 | -1.16976% | 1.526 deg | 1.664 deg |
| C2 | -3.85467% | 16.207 deg | 25.713 deg |
| C3 | -2.22716% | 10.435 deg | 13.915 deg |

All four fresh runs are complete. These are finite search results, not
proven global minima. Angle and relative score gap are recorded separately.
In particular, a near score does not certify near axes.

Post-freeze initialization assessment separates the observed-W proposal
from the 32 random SO proposals. The former is initially 30 degrees from
reference axes in CR3/C1/C2 and already equivalent to the reference axes
in C3. The independently random Powell starts, before the final Givens pass,
have score changes +.06867%/+3.64438%/-.03756%/+.37512% and maximum
matched axis offsets 8.426/15.388/46.357/58.167 degrees. Their individual
scores/angles are kept in `individual_start_positions.json`. This arm
distinction prevents a favorable supplied coordinate frame from being
reported as recovery from an independent random start. In C2/C3 the
random-start near scores still have large axis offsets.

## BU5: paired subject refits and capacity sensitivity

The fresh frame is frozen before sampling. Each sample refits every proper
coalition for the baseline and that frame, using the same 80% subject sample
for both. U and the observed functions are recentered. The observed-X bank,
output modes and task gauges remain frozen. These are conditional-fit
stability checks, not independent prediction validation or full-bank refits.
The independent all-5000, 32-center refit must replay the saved baseline and
fresh-frame scores before any sampled comparisons are accepted.

CR3: 12/12 samples favor the frozen fresh frame at both capacities.

| Centers | All-5000 score change | Sample 5% / median / 95% score changes |
|---|---:|---:|
| 32 | -.27454% | -.37597 / -.30972 / -.09126% |
| 64 | -.28779% | -.40241 / -.30449 / -.15194% |

Thus the small CR3 departure is reproducible in this audit and is retained
as approximate evidence. All four datasets finish 12 paired refits per
capacity: 96/96 comparisons favor the frozen fresh frames. C1/C2/C3's
32/64-center median gains are 1.2257%/1.1992%, 3.5465%/3.5860%, and
2.2976%/2.2355%. C3's all-subject 64-center gain is 2.2345%, close to
its 32-center gain of 2.2272%. No 15-degree or 15-percent rule is silently
introduced as a veto. These repeated subject draws overlap and are not
independent confirmations or confidence intervals for recovery.

BU6 subsequently refines the frozen best and independently random-start
frames with smaller exact Givens steps +/-.02/.05/.1 radians. CR3's frozen
best refines to -1.02974% with mean/max offsets 10.068/12.917 degrees.
The independent-random-start arm refines to a still lower score, -1.12978%,
and smaller mean/max offsets 6.549/7.524 degrees. The three remaining
models continue with the identical policy. All refined scores and frames
freeze before axis assessment. These new points have not yet received the
paired refit audit; BU5 refers to the earlier BU2 frozen points.

## Reproduction and local artifacts

    python supervised_abs_rotation/step5_matrix_metrics/run_operator_mode_support.py
    python supervised_abs_rotation/step5_matrix_metrics/run_operator_mode_support.py --polynomial-link
    python supervised_abs_rotation/step5_matrix_metrics/run_shapley_profile_support.py --batch
    python supervised_abs_rotation/step5_matrix_metrics/run_shapley_profile_support.py --batch --centers 64 --run-tag BU3
    python supervised_abs_rotation/step5_matrix_metrics/run_shapley_profile_blind.py --batch
    python supervised_abs_rotation/step5_matrix_metrics/run_shapley_profile_resampling.py --batch
    python supervised_abs_rotation/step5_matrix_metrics/report_shapley_profile_evidence.py
    python supervised_abs_rotation/step5_matrix_metrics/write_approximate_shortlist.py

The report writes complete plane curves and continuous grid/transfer/blind
tables under `outputs/supervised_abs_rotation/waveBU_continuous_profile_evidence`.
The retained-candidate shortlist also includes displaced minima from older
families. Bulky local matrices and derived conditional model caches are not
committed. The complete suite passes 244 tests; independent subspace-basis
refits, exact Shapley efficiency, response reconstruction, SVD trace,
permutation endpoints and reader replay are covered.

Primary methodological sources:

- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*:
  https://papers.nips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
- Owen & Prieur, *On Shapley Value for Measuring Importance of Dependent Inputs*:
  https://arxiv.org/abs/1610.02080
- Eckart & Young, *The Approximation of One Matrix by Another of Lower Rank*:
  https://doi.org/10.1007/BF02288367

These motivate allocation and matrix-mode complexity. They do not prove
identifiability of the axes in this experiment or turn our function-share
spectra into population variance Shapley effects.
