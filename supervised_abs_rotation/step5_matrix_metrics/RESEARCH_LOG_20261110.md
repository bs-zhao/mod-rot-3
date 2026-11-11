# Real Windows experiments, 2026-11-10 (America/New_York)

Checkout: D:\Projects\mod-rot-3; Anaconda environment: py11.
Initial Wave A scoring revision: db82999; scoring SHA256 is recorded in
every protocol. Only Step 5 code/docs are changed by this research session.
Unrelated pre-existing local changes are preserved.

## Wave A: completed four-model audit

All actual cases and all trial rows were validated. All 5,000 subjects
were included in final fitting and derivative evaluation in each model.
C1/C2/C3 have 2,003,448 rows each; CR3 has 2,500,000. Derivatives were
evaluated at 40,000 observed rows per model. The maximum orthogonal
registration residual was below 3.7e-8; invariant controls passed and
equivalent signed-permutation endpoints matched to numerical precision.
No forbidden files were opened through the matrix loader audit.

| Same metric: poly3 UX_entries_entropy | Logit baseline rank | Logit planes | Probability baseline rank | Probability planes |
| --- | ---: | ---: | ---: | ---: |
| C1 | 1/37 | 6/6 | 1/37 | 6/6 |
| C2 | 1/37 | 6/6 | 1/37 | 6/6 |
| C3 | 1/91 | 15/15 | 1/91 | 15/15 |
| CR3 | 13/37 | 2/6 | 10/37 | 1/6 |

The number of adequate estimator/channel/metric combinations satisfying
every endpoint test was C1=16, C2=3, C3=2, CR3=0. These are correlated
exploratory settings, not independent statistical discoveries.

CR3's best adequate setting by passing-plane count was poly3 logit
UU_edges_tail4_l2: rank 3/37, 4/6 planes. No sign was reversed and no
failed plane was removed. Its logit/probability validation NMSE was
0.017499/0.037298. The short/medium/long RBF settings with only 128 centers
all failed the fixed CR3 prediction adequacy screen, with logit NMSE
0.961704/0.695405/0.242424. Their derivative rankings cannot be used as
reliable evidence about the unknown function.

The final comparison is saved locally at:
`outputs/supervised_abs_rotation/step5_waveA_seed381_four_models.csv`.
It contains zero all-four and zero cross-family successes.
Each model also has `inspection/plane_curves.csv`,
`inspection/plane_diagnostics.csv` and representative figures.

## Wave B: CR3-first derivative-free projection screen

Protocol: `waveB_additive_seed382`, documented in WAVE_B.md.
All 2.5 million rows contributed to subject response profiles, followed by
all-subject additive/pairwise function fitting. Trial-held-out profile NMSE
was 0.0239445 for logit and 0.0542431 for probability.

| U basis | Worst pairwise test NMSE, logit / probability | Additive NMSE baseline rank, logit / probability | Additive passing planes, logit / probability |
| --- | --- | --- | --- |
| poly3 | 0.0012905 / 0.0022473 | 25 / 22 | 0 / 0 |
| poly4 | 0.0032031 / 0.0055312 | 26 / 23 | 0 / 0 |
| rbf3 | 0.0797118 / 0.0845673 | 19 / 16 | 0 / 0 |
| rbf4 | 0.0500951 / 0.0537834 | 19 / 17 | 0 / 1 |

Every family met the prediction adequacy screen, but none of the nine
metrics in either channel satisfied the full six-plane criterion. The
best plane count was poly4 logit component_tail1_l2: rank 6/37, 4/6.
Consequently C1/C2/C3 were not launched for Wave B.

This is a ranking failure despite good approximation, rather than merely
an inadequate function estimator. Empirical pair projections remove the
additive span but do not produce mutually orthogonal variance shares.
No result is described as an independent-input Sobol index.

## Wave C: conditional direction and finite-scale effects

Completed on CR3 as `waveC_finite_seed383`. Its numeric X bank was verified
identical across all 5,000 subjects, so all 500 recorded stimulus responses
per channel entered fitting directly. No X interpolation was needed. The
common support intersection retained all 5,000 subjects at both steps.

| U-to-response estimator | Logit / probability held-out NMSE |
| --- | --- |
| poly3 | 0.0178723 / 0.0186057 |
| poly5 | 0.00714793 / 0.00747493 |
| rbf1, 512 centers | 0.0101238 / 0.00988693 |
| rbf2, 512 centers | 0.00109705 / 0.00107670 |

There were zero adequate full-plane candidates. For the most accurate
rbf2 logit fit, fixed-X direction inconsistency ranked baseline 26/37,
conditional k=8 ranked 21/37, and conditional k=32 ranked 18/37. All
passed 0/6 planes. Both 0.2 and 0.5 finite steps gave those same ranks and
plane counts. The best adequate plane count over the entire wave was
rbf1 logit UU_edges_tail2_l2, rank 11/37 and 3/6 planes.

The response-gradient Gram eigenvalue shares were consistently nonzero
under every estimator. For poly5 logit they were approximately
0.0678, 0.2003, 0.2602 and 0.4717. This does not establish identifiability,
but does not support explaining the failure as a completely inactive U
direction. Prediction accuracy alone still does not certify derivatives.

## Further branches (running; no success claim)

Wave D, `waveD_unknown_link_seed384`, learns monotone response
transformations without a supplied link formula, then tests additive
compression in held-out original-Y units. See WAVE_D.md.

Wave E, `waveE_profiled_seed385`, profiles local chain-rule curvature
from the saved numerical fields. It explicitly distinguishes local
compression from a globally integrable response link. See WAVE_E.md.

Both start on CR3 and stop before C1/C2/C3 if no candidate survives.

## Research direction

Investigate conditional monotonicity and support-screened finite-scale
effects from generic U-to-response kernel fits. In CR3, a shared X bank can
be detected directly from the numeric matrix, allowing recorded responses
at fixed X to be analyzed without stimulus equations or X interpolation.
Keep ordinary first-gradient concentration and global smoothness controls
separate from conditional sign consistency. A CR3 candidate must survive
approximation and scale checks before other datasets are launched.

Methodological sources verified in this session:
[Durrande et al.](https://arxiv.org/abs/1106.3571),
[Duvenaud et al.](https://arxiv.org/abs/1112.4394),
[Chastaing et al.](https://arxiv.org/abs/1112.1788), and
[Apley & Zhu](https://arxiv.org/abs/1612.08468).
These motivate centering, low-dimensional composition and support-aware
effects; custom exploratory scores are not claimed to reproduce their
exact algorithms or theoretical guarantees.

## Verification

The original Wave A/B revision passed 22 numerical/pipeline tests in py11, covering finite-difference
derivatives, all-subject fitting, orthogonal controls, signed permutations,
strict endpoint screening, dependent-input additive projections, and
within-subject response-profile validation. Re-running the CR3-first
wrapper with `--all-models` verified and reused all four frozen outputs
and successfully regenerated the zero-candidate comparison.
The later Wave C/D/E revision adds mathematical output-link invariance,
finite support/context controls, monotone calibration and full-pipeline
checks; its test result is recorded with the corresponding commit.
