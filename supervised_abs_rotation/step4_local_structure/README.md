# Step 4 — Blind **local response fingerprints** on C1 and CR3

Standalone experiment: `supervised_abs_rotation/step4_local_structure/`.

This next wave directly addresses the Step-3 finding that global linear stimulus
fingerprint sparsity can favor the Step-1 supervised coordinates in C1 but
fails systematically for CR3. **It does not start from an analytic generating
equation.** All mathematical estimators use only the numeric W/Y matrices
already saved in Step 2.

No true theta, C1/CR3 formulas, decoder checkpoint/weights or supervised R*
are read while calculating a loss. Cases are first assigned anonymous IDs
and scores frozen before labels, angles and baseline are revealed.

## 1. Why local gradients instead of one global B?

Step 3 studied a single linear response fingerprint

\[
F_s\approx b+U_s^\top B_{\rm global},
\]

where F is a subject-by-stimulus or subject-by-response-feature matrix
constructed entirely from W and Y.

In an intrinsically nonlinear response function, the same latent axis may
have different effects in different regions of U. Global B can average
opposite effects away. This experiment instead estimates a field of local
slopes using ONLY numerical data:

\[
B(u_1),B(u_2),\dots,B(u_A).
\]

Each B(u) has shape (d latent axes, p response features). These are
**regression approximations to local derivatives**, not exact Hessians or
decoder-autograd derivatives.

### Data-driven response profiles F

Reuse Step-3 `profiles.py` on the common W/Y table, unchanged:

- **Shared stimulus bank** (typically CR3): F[s,j] = Y for subject s,
  actual stimulus j. Full original bank used for every gradient estimate.
- **Nonshared X** (typically C1): subject-specific ridge coefficients
  in generic constant/linear/quadratic X basis. No true C1 equation is
  assumed or loaded.
- Analyze both `full` and `contrast` response views, and each of
  `logit_nn`, `p_nn`, `choice_nn`, `choice_observed` separately.
- Each F column is normalized using only its TRAIN subjects' data,
  with the same normalization for all rotations.

### Local estimation protocol

1. Seeded **70% TRAIN subject**, 30% TEST subject partition.
2. Select 128 TEST subjects as local evaluation anchors.
3. For every anchor, find the nearest `k=96` or `k=256`
   **TRAIN** subjects in 4D U space; the anchor is never a fitting neighbor.
4. Give neighbors isotropic Gaussian radial weights and fit an
   intercept+4 slope local weighted ridge regression.
   Ridge strength scales with the local covariance trace, so
   the estimator is equivariant under orthogonal transformations.
5. The resulting local slope B(u) is evaluated on common normalized F.
   Report TEST-anchor profile reconstruction MSE and condition numbers.

The two neighborhood sizes are **predeclared robustness settings**, not
selected by baseline rank. Alternate sizes are exposed by CLI, to test
sensitivity. There is no neural-network retraining.

## 2. Exactly the same local estimator under every rotation

For orthogonal transformations U_case = H U_reference:

- Euclidean neighbor distances and radial weights are unchanged.
- The local ridge estimator is rotation-equivariant:
  \(B_{\rm case}(u)=H B_{\rm reference}(u)\).
- Therefore local slopes are computed once in an **anonymous** reference
  case chosen by the seeded blind shuffle.
- For every other case, H is estimated via orthogonal Procrustes from
  the two **numeric subject U arrays extracted from W**, without
  reading Step-1 R, per-case R or rotation angles.
- The exact original Y is used unchanged across all cases.
- Registration errors are checked; inconsistent geometry or X row
  alignment raises an error.

This makes comparisons faster and avoids re-estimation noise across angles,
without supplying the correct orientation.

## 3. Predeclared metrics

We test complementary families, all sign- and axis-permutation-invariant.

| Family | Metrics | What it asks |
|---|---|---|
| Pointwise local | `local_point_entropy`, `local_point_tail1`, `local_point_tail2`, `local_point_l1`, `local_point_hoyer` | At each subject/stimulus, how many U axes carry effect? |
| Across-anchor RMS | `local_rms_entropy`, `local_rms_tail1`, `local_rms_hoyer` | Does the same U axis consistently carry the effect across subjects? |
| Across-anchor mean absolute | `local_meanabs_entropy`, `local_meanabs_tail1`, `local_meanabs_hoyer` | Is there an axis-supported pattern even when local slopes flip sign? |
| Joint gradient structure | `local_gram_offdiag_fraction`, `local_group_l1_over_l2` | Are axes separated across all local slopes? |
| Nonlinear gradient variation | `nonlinear_local_*` | Are changes in B(u) over subjects simpler in one orientation? |
| Global comparator | `global_local_*` | How do the same metrics behave for one global B? |
| Invariant controls | `control_*`, `global_control_*` | Sanity check: norms/spectra and variation fraction must NOT depend on rotation |

**Important limitations:**

- Joint gradient-gram offdiagonal minimization can favor an eigenbasis
  unrelated to the true theory parameters. It is an empirical *candidate
  loss*, not a structural theorem about latent interpretation.
- Nonlinear-gradient variation is computed after subtracting the mean
  local B(u), *not* by directly estimating the full model Hessian.
- Metric correlation is substantial; do not interpret counts of successful
  related metrics as independent confirmations.
- The per-subject response-feature approximation for C1 can distort
  the actual dependency structure; this must be examined separately.

## 4. Blind evaluation and strict diagnostics

After ALL case losses are frozen in `blind_scores.csv`:

1. Save `blind_key.csv` mapping anonymous IDs to original Step-2 cases.
2. Reveal baseline (0° = original Step-1 supervised R*) and six two-axis
   sweeps (15°, 30°, 45°, 60°, 75°, 90° relative to R*).
3. Report case ranking for each metric and every Y channel, view and k.
4. Report the number of planes where both endpoints are preferred,
   90°/0° discrepancy, and median interior-vs-endpoint contrast.
5. Draw separate six-plane curves and local-gradient RMS heatmaps.

Zero and 90 degrees are signed permutations of the same coordinate system;
any proposed metric should treat them as equivalent up to floating-point
error. Flat curves are NOT successful recovery.

The subject-level holdout is only an analysis/estimation diagnostic:
the original S5 encoder inferred Z from **all** observed choices,
including those same subjects' trials. This is NOT clean new-subject or
held-out-trial predictive generalization.

## 5. Commands, Windows CMD

From repository root:

```cmd
conda activate py11
cd /d D:\Projects\mod-rot-3
git pull origin main

python -m pytest -q supervised_abs_rotation/step4_local_structure/tests

python supervised_abs_rotation/step4_local_structure/run.py --model C1
python supervised_abs_rotation/step4_local_structure/run.py --model CR3

python supervised_abs_rotation/step4_local_structure/compare.py
```

Both models use the exact Step-2 `step2_pairwise_sweep/` folder
corresponding to their existing Step-1 beta, context fractions, repeat and
objective. This script does not run Step 1, Step 2, or Step 3 for you.

To run an inexpensive first-pass logit screen into a separate destination:

```cmd
python supervised_abs_rotation/step4_local_structure/run.py --model CR3 --channels logit_nn --neighborhoods 96,256 --anchors 96 --no-plots --output outputs/CR3_step4_logit_screen
```

If an output exists, use `--overwrite` only when you intend to replace
that particular output folder.

## 6. Saved results

Default per-model location:

```text
outputs/supervised_abs_rotation/MODEL/CONDITION/repeat01/abs_mse/
    step4_local_structure/
        summary.json
        blind_scores.csv              # scores without case/angle names
        blind_key.csv                 # identity reveal AFTER score freeze
        revealed_scores.csv
        metric_summary.csv            # key result: baseline rank; 0/90 checks; 6 planes
        plane_curves.csv              # complete angle-by-score table
        rankings.csv                  # all 37 cases, every metric
        local_fit_quality.csv         # diagnostic approximation error and condition numbers
        geometry_checks.csv           # orthogonal registration residuals
        subject_split.csv             # fixed TRAIN/TEST and anchor allocation
        plots/
            curve_logit_nn_contrast_k96_local_point_entropy.png
            curve_logit_nn_contrast_k256_local_rms_entropy.png
            ...
        diagnostic_gradient_maps/
            baseline_k256.npz
            baseline_local_gradient_rms.png
            baseline_nonlinear_gradient_rms.png
            u1_u2_045_k256.npz
            ...
```

Only numerical CSV summaries and comparatively small diagnostics live
in Step 4. Multi-million-row W files and common Y are **not duplicated**.
The output can therefore be zipped and shared without sending original
trial W/Y arrays.

## Post-hoc C1 vs CR3 summary

After both independent analyses have finished, execute
`python supervised_abs_rotation/step4_local_structure/compare.py`.
It locates both default output directories and writes:

- `outputs/supervised_abs_rotation/step4_C1_CR3_comparison/cross_model_all_metrics.csv`
  with both models' baseline ranks and successful-plane counts;
- `cross_model_shortlist.csv`, ordered by the explicitly declared
  *joint* success checks (NOT used to choose/modify the original scores);
- `local_vs_global.csv`, reporting exactly how each local metric changes
  baseline rank and the number of successful planes relative to the
  corresponding global-B metric within each model;
- `comparison.json`, with joint strict/relaxed counts and key limitations.

Set `--c1`, `--cr3` or `--output` to override the defaults.
Do not substitute Step-3 files for Step-4 metric-summary files.

## 7. Interpreting the next comparison

Compare `global_local_point_entropy`,
`local_point_entropy`, `local_rms_entropy`, and
`nonlinear_local_point_entropy` on the **same Y view**:

- If C1 remains near 6/6 success while CR3 improves from 0–1/6 to
  5–6/6, the data support the hypothesis that the CR3 axis structure
  is chiefly **local or nonlinearly varying**, rather than global.
- If CR3 continues to favor intermediate rotations, the local
  gradient-sparsity hypothesis has failed this test too; do not
  rename or tune a nearly identical score to force success.
- If 0° and 90° differ materially, inspect `geometry_checks.csv`
  and local estimator invariance before interpreting anything.

All scientific conclusions must wait for the **actual C1 and CR3 output
matrices**. Passing unit tests is only evidence that the code executes
and obeys its designed mathematical invariants.
