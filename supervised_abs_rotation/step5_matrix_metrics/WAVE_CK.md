# Local affine correction for nearest-neighbor conditional covariance

CK completes CR3/C1/C2/C3 on the user's computer, 2026-11-12
America/New_York. All 5000 subjects are both available observations and
anchors in the conditional estimates. Responses are the case matrices'
actual frozen-NN logit and probability outputs, checked by the four-model
Y provenance audit. Analytic logit/probability cannot count as final success.

## Estimator fixed before the four-model continuation

CJ estimates conditional variance by sample covariance of the three nearest
response vectors. Finite neighborhoods also vary in the conditioning U
coordinates, so variation of the conditional mean can enter that estimate.
CK tests a local first-order numerical correction, rather than prescribing
a global response-function equation.

For every proper U subspace, find 32 nearest observed subjects around each
anchor. Locally regress the existing response-function vectors on an
intercept plus the conditioning coordinates. If A_a is this local design
and P_a its column-space projection, estimate conditional residual covariance
using the local residual degrees of freedom:

    R_S = mean_a Y_neighbors.T (I-P_a) Y_neighbors / (32-rank(A_a)).
    C_S = C_observed - R_S.

This is an own local-affine extension of CJ's data-only neighborhood
estimator, not an asserted exact population conditional-variance formula.
It assumes no supplied case-specific utility equation. The local regression
is a generic numerical approximation learned from the same actual W/Y.

A weighted-Gram implementation avoids forming each large local covariance
separately. Residual covariance is PSD up to checked roundoff. Finite
explained-covariance Shapley components can be signed; their sum still
reconstructs observed covariance. Absolute projected eigenvalues supply
the explicitly labeled complexity weights, including any negative modes.
This is not described as a PSD variance decomposition after taking absolute
values. Neighbor radii, effective residual degrees of freedom and signed
spectral mass are recorded.

Tie candidates are sorted by subject ID before seeded random priorities,
so seeded ties are reproducible under axis signs and alternative bases.
This numerical refinement is explicit; CJ's original tie convention is
retained as a supported default for replaying its frozen artifacts.
Neighbor count 32, local degree 1 and tie policy are identical in all four
models. There are no model-specific signs, response forms or score weights.

## Retained NN-probability criteria

Both below were identified in the CR3 screen and continued unchanged to
the other models. Each X field is normalized before support counting.
Fractional 10% L2 support counts how many joint Ui-by-mode weights are
needed to retain 99% of spectral mass, with fractional boundary weight.
The second score uses 3% soft support after giving each X equal mass.

| Same p_nn criterion | CR3 | C1 | C2 | C3 |
|---|---|---|---|---|
| Worst-X 10% fractional mode support | 0 / 0% | 0 / 0% | 15 / -6.30378% | 30 / -10.21211% |
| Equal-X 3% soft mode support | 0 / 0% | 0 / 0% | 15 / -5.42284% | 30 / -1.82958% |

Cells give coarse axis deviation in degrees / score change from reference.
The first criterion's strict plane diagnostics are 6/6, 6/6, 5/6 and 2/15,
with baseline ranks 1/37, 1/37, 2/37 and 45/91. The second has 6/6, 6/6,
5/6 and 10/15 planes, ranks 1/37, 1/37, 4/37 and 14/91. CK therefore
improves some CR3 transferred contrasts but does not uniformly improve
C3. Both common approximate results remain recorded; no exact-zero-angle
or fixed prediction-error veto is imposed.

For the worst-X criterion, CR3's three old distant frames score worse by
22.609/23.737/37.002%. The newer CF2 and CH3 frames are worse by
35.771/19.379%. Nearby BU6 and CG2 frames can score about .192%/.170%
lower. The latest CJ2 frame was created after the original CK pilot and is
not silently claimed as included in that pilot's transferred-frame set.
Full spectra and all tested channels/metrics are retained, including the
less favorable logit variants.

CK2 independently reloads the frozen NN-p worst-X criterion, verifies its
anonymous source-case score and starts a seed-521 CR3 blind search using
all 5000 subjects. Proposals are observed W plus 32 random SO frames,
two 80-evaluation Powell starts and one Givens pass. This search is
separate from the completed four-model grids and does not assume that
the reference is the unique optimum. The completed CR3 result has score
change -1.09277%, mean matched axis deviation 21.122 degrees and maximum
33.066 degrees. It is retained with the coarse-grid 6/6 result and the
transferred-frame contrasts; a grid win alone does not establish narrow
global localization. Other CK2 models are not run under CR3-first priority.

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_neighbor_conditional_variance.py --batch --run-tag CK --neighbors 32 --local-linear --stable-ties
python supervised_abs_rotation/step5_matrix_metrics/run_neighbor_variance_blind.py --source-tag CK --model CR3
python supervised_abs_rotation/step5_matrix_metrics/audit_response_provenance.py
python supervised_abs_rotation/step5_matrix_metrics/report_predictive_variance_evidence.py
```

Tests compare the efficient covariance against explicit local least-squares
residuals and check seeded ties, subspace geometry, CLI settings and frozen
source replay. Y-source eligibility is distinct from a claim of successful
global coordinate recovery.
