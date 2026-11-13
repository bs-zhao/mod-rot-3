# Independent random-rotation percentile of the fixed Shapley UX criterion

CL is a validation wave, not a new metric. It fixes BU's NN-p response and
the same mean-X 3% Ui-by-function-mode soft support in all four models.
This is the earlier approximate common criterion with coarse axis
deviations 0/0/15/15 degrees. All conditional fits and scores use all 5000
subjects. The actual Y source is frozen-neural-network probability, with
its provenance separately audited; no analytic logit/probability is used.

The earlier BU2 random proposal sets had 0/32 orientations at least as good
as reference in each model. That is useful exploratory information, but
many methods had subsequently inspected related frames. CL therefore
freezes the candidate and collects a separate random set after selection.
It does not combine the old 32 with the new draws or use an observed-W
initialization as a random sample.

## Frozen plan and independent draws

Before the first new score, save `plan_frozen_before_sampling.json` under
`outputs/supervised_abs_rotation/waveCL_independent_Haar_percentile_seed541`.
It fixes the metric, actual NN-p channel, N=5000, 64 random frames per
model, model order, seeds, confidence family and numerical tie rule. There
is no adaptive stopping or tuning after seeing these new scores.

Draw independent Gaussian matrices, apply QR with the positive-diagonal
correction and determinant correction, obtaining Haar SO(d) rotations.
Seeds are 541 + 1009 * fixed_model_index for CR3/C1/C2/C3. For each model,
first independently replay anonymous source case 000. Then score every one
of the 64 random frames using all 5000 subjects. Freeze score/frame hashes
before reading the reference label. Only then replay and assess reference.

## What the percentile means

For this fixed dataset and numerical scoring policy, define

    p = P_Haar(score(random frame) <= score(reference) + numerical_tolerance).

The reported estimate is k/64. Numerical ties count against a high reference
rank: a flat score gives k=64 and cannot masquerade as a good percentile.
Strictly better points, ties and the total favorable tail are all recorded.
The tolerance is a fixed 1e-7 * max(1,abs(reference score)); it is a numerical
comparison convention, not a prediction-error or angle acceptance cutoff.

Use a one-sided exact binomial upper bound, with alpha=.05/4 per model.
Bonferroni makes the four upper bounds simultaneous at confidence at least
95%, without requiring independence between datasets. For k<n,

    upper = Beta_quantile(1-.05/4; k+1, n-k), with n=64.

For k=n the upper bound is one. For k=0 the formula reduces to
1-(.05/4)^(1/64), about 6.618%. Thus the benchmark can quantify a reference
that ranks in the first 15% of random orientations while still allowing
some optimized alternatives to be slightly better. The 15% comparison is
recorded as an additional rank interpretation, not asserted to be the sole
meaning of the user's approximate-position example.

These probabilities are conditional on the observed data and fixed scorer.
They are not confidence intervals over a population of subjects, a claim
of unique global optimality, or a guarantee that an optimizer recovers the
reference axes from every starting point. Nearby and distant optimized
lower-score points remain in the separate frozen-search reports. An axis
offset in degrees, a relative score gap and a random-orientation percentile
are three different quantities and remain separate.

## Actual completed four-model result

| Model | Fresh random frames at least as good as reference | Empirical fraction | Simultaneous 95% upper bound |
|---|---|---|---|
| CR3 | 0/64 | 0% | 6.618% |
| C1 | 0/64 | 0% | 6.618% |
| C2 | 0/64 | 0% | 6.618% |
| C3 | 0/64 | 0% | 6.618% |

No numerical ties occur. This fixed NN-p criterion therefore places the
reference in the first approximately 6.62% of Haar orientations in all
four observed datasets, with simultaneous confidence at least 95%. This
supports its status as a useful approximate common UX-complexity criterion.
It does not erase the 15-degree coarse displacements or the lower optimized
points recorded earlier, and it does not establish monotonicity or a unique
global coordinate solution. The old 32 proposal scores were not pooled in
these estimates.

## Reproduction and artifacts

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_independent_random_percentile.py --batch
python supervised_abs_rotation/step5_matrix_metrics/report_random_percentile.py
```

The per-model folders hold random scores, Haar frames, hashes, source replay
checks, conditional-fit audits and the assessment. The global
`four_model_assessment.json` is updated only after a model's full fixed-size
sample completes; `report.md` is written for the requested completed batch.
Existing complete runs are verified before reuse; incomplete outputs are
retained, and changed plans/code are not silently resumed.

Tests check the exact zero-count formula, binomial-tail inversion, proper
orthogonal rotations, and the flat-score tie safeguard. These tests do not
stand in for the real four-model run.
