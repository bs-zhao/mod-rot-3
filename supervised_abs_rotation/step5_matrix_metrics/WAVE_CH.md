# Quantile conditional-variance UX support

CH is a completed CR3 pilot; CH2 completes CR3/C1/C2/C3 in that order on
the user's computer, 2026-11-12 America/New_York. All fits and scores use
all 5000 subjects. CH2 repeats the same algorithm and adds the newly frozen
CF2/CG2 stress frames. No response equation, decoder, true theta, original
saved R, or case-specific mixture weights enter the score.

## Empirical conditional means without a response-function formula

For every proposed Ui coordinate, build marginal empirical-quantile codes
using W alone. Use q=max(2,floor((N/20)^(1/(d-1)))): six bins for four-U
models, three bins for C3's six U variables. This is the same resolution
policy based on sample count and dimension in every model. Exact boundary
ties get a symmetric separate class, so changing axis signs preserves the
partition. All members of a tied value remain together.

For every proper U coalition S, group subjects by their selected quantile
codes and compute their actual observed response-function means. If P_S
is that group-mean projection, C_S=Y.T P_S Y/N. Nested coalitions refine
partitions, so covariance increments are PSD. No polynomial, GP or other
response-function family is fitted in this wave. Partitions depend on
individual axes, not merely their span; caching therefore uses sets of
axis projectors, not a subspace-projector hash.

The full coalition conditions on exact observed U tuples. Any unresolved
variation among repeated identical tuples is appended as an unassigned
covariance. In these real datasets U tuples are unique, so this endpoint
uses the actual observed response covariance. Proper coalitions are
coarsened and the full endpoint is exact. This finite-estimator asymmetry
is explicit; it is not a claim that continuous conditional expectations
have been recovered or that a lookup of unique U tuples generalizes.

Exact Shapley weights allocate the nested covariance increments to Ui.
Per-X projected eigenvalues describe Ui-by-function-mode variance support;
their traces give a graph-energy view. Each X column is normalized before
relative-L2 support scoring. The main operator score uses p/(p+.1^2), with
p the normalized variance eigenvalues, then averages over X columns.
Absolute effect amplitude is not optimized. The same support family obeys
the user's sparse-tail example.

Numerical checks enforce PSD nesting, sign/permutation equivalence, and
reconstruction of observed covariance. Group occupancy and fixed-partition
PRESS are retained. Singleton cells use a global leave-one-out mean only
for the PRESS diagnostic. Proper-coalition errors include omitted-input
variation; no absolute prediction-error cutoff is used. Full-U unique-cell
PRESS cannot establish out-of-sample prediction quality. In CR3 the proper
one/two/three-coordinate partitions have about 6/36/216 occupied groups;
the maximum singleton-subject fraction is .0004.

## Retained four-model results

These are exploratory same-formula comparisons, not independent replications
or a claim that one exact optimum has been established. Cells are coarse
axis deviation in degrees / score change relative to reference baseline.

| Same criterion | CR3 | C1 | C2 | C3 |
|---|---|---|---|---|
| logit, mean-X U-by-mode 10% soft support | 15 / -.398129% | 0 / 0% | 15 / -.725179% | 15 / -.364259% |
| logit, worst-X graph 3% soft support | 15 / -.018731% | 0 / 0% | 15 / -.054435% | 15 / -.008151% |
| p_nn, mean-X U-by-mode 3% support | 30 / -1.04912% | 30 / -2.73232% | 45 / -4.11375% | 45 / -1.60322% |

The mode-support row has baseline ranks 7/37, 1/37, 3/37 and 13/91.
Its largest one-based grid-rank percentage is 18.92%, separate from the
15-degree position and the <.73% score gap. The graph-only row ranks
2/37, 1/37, 3/37 and 3/91, within the first 8.11% of the supplied grids.
The graph curves have smaller contrast and do not, by themselves, localize
axes globally. No identical strict four-model combination exists. Failed
p_nn views and all angles/gaps remain recorded, without rigid vetoes.

The logit mode-support candidate was identified exploratorily from the
completed grid comparisons and then frozen for a new-seed blind test. It
ranks the three older distant CR3 frames worse by 1.512/1.740/.590%.
The CF2 CR3 frame with maximum axis deviation 59.6 degrees is worse by
1.711%; the CG2 distant C2/C3 frames are worse by 3.996/2.131%. These
transferred checks are encouraging, not a substitute for fresh optimization.
Nearby BU6/CG2 points can be slightly lower and remain in the ledger.

## Blind continuation and reproducibility

CH3 freezes the logit mean-X 10% variance-mode criterion and uses seed 501,
different from the preceding seed-478 searches. Proposals are anonymous
observed W plus 32 random SO matrices, followed by two 80-evaluation Powell
starts and one Givens pass. The exact score always uses all 5000 subjects.
Source-case replay must succeed before search. A bounded covariance cache
controls memory without changing the objective. Frames/scores are frozen
before reading reference labels; starts and random-only arms are reported
separately. Observed W is already equivalent to reference axes in C3.
The finite search does not prove global optimality.

CH3's completed CR3 search selects random proposals 14 and 19. The frozen
best has score change -1.62123%, mean matched axis deviation 29.622 degrees,
maximum 37.303 degrees. This new point was not used to define the CH2 score.
The favorable transferred checks therefore did not establish a narrow
global optimum. It is retained as continuous evidence rather than rejected
by a fixed threshold. Other CH3 models are not run, following CR3-first
prioritization. CI/CJ subsequently include this frozen point in their audits.

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_quantile_conditional_variance.py --model CR3
python supervised_abs_rotation/step5_matrix_metrics/run_quantile_conditional_variance.py --batch --run-tag CH2 --audit-latest
python supervised_abs_rotation/step5_matrix_metrics/run_quantile_variance_blind.py --model CR3
python supervised_abs_rotation/step5_matrix_metrics/report_predictive_variance_evidence.py
python supervised_abs_rotation/step5_matrix_metrics/write_approximate_shortlist.py
```

Outputs stay in each model's step5 directory. The global
`predictive_variance_evidence` folder holds full-plane figures, continuous
positions, transferred-frame scores, group/numerical audits and completed
blind assessments. Unit/integration tests cover boundary ties, exact
fixed-partition PRESS including singleton fallback, repeated U tuples,
PSD/conservation, and source-score replay by the blind scorer.
