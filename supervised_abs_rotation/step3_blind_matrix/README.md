# Step 3 — Blind matrix-only discovery: does any WY simplicity prefer baseline?

**Standalone scope:** `supervised_abs_rotation/step3_blind_matrix/`.
Start with C1 and CR3; nothing is regenerated, retrained or decoded. The only
scientific observations available to the scoring algorithms are **numeric
trial matrices W and one shared Y** from Step 2.

**Crucial:** This step does NOT begin with the true generative equation,
does NOT read true parameters θ, and does NOT use the original best R to
calculate loss values. Case identities are assigned randomized blind IDs
`B001,B002,...`; the angle and `baseline` names are unmasked **only after**
all loss values have been calculated and saved to `blind_scores.csv`.

## Question and scope

All W matrices encode the same Y-relevant information: the Z-part differs by
an invertible orthogonal transformation, while X and Y are identical. A
coordinate-free criterion such as rank, singular values, the Bayes-optimal
prediction error, or a Frobenius norm of a linearly equivariant B cannot
prefer baseline by construction.

We ask if there is an **axis-dependent** description of the common
input–output relation which is simplest in the baseline coordinate system.
Every scored metric is fixed before unblinding. Successful baseline
recovery is not assumed and failure is documented.

### 1. Build a response-profile matrix F from W and Y

One row per subject; output columns summarize stimulus-response behavior.
No simulator equations enter the procedure.

- If the actual X matrix contains an identical stimulus bank for every
  subject (detected **from numeric X values**, not the C1/CR3 model name),
  then F[s, j] is simply that subject's recorded Y at stimulus index j.
  For typical CR3 this gives **5000 × 500** per response channel.
- Otherwise, each subject's trial-wise Y is regressed on the **same**
  generic feature basis: intercept, normalized X, squared normalized X and
  all pairwise X products. The coefficients become their response profile
  F[s, k]. The ridge-regularized per-subject regressions serve only as
  a common numeric summary. This is appropriate for C1's independently
  sampled stimulus vectors, without assuming the C1 generative equation.
- Two views of F: `full` (including intercept/global response level)
  and `contrast` (subtract per-subject stimulus mean for repeated banks;
  remove the intercept coefficient for variable designs). Contrasts
  examine *how stimuli matter*, rather than only response biases.

The same subject profile F is reused unchanged for ALL candidate
rotations, since Y and X are common across cases.

**Target channels (all reported separately):**

`logit_nn`, `p_nn`, `choice_nn`, `choice_observed`.

The NN logit and probability are the most informative continuous
diagnostics; discrete and observed choices test robustness to a noisier or
coarsened output. The choices are NOT transformed into invented logits.

### 2. Linear association matrix B: stimulus-response fingerprints

For each candidate, take the subject-level rotated U from its W rows and
fit an identical isotropic ridge regression:

\[
F=\mathbf1\mu_F^\top +(U-\bar U)B+\epsilon
\]

U is N × d, F is N × p, and B is d × p. Every F column is
normalized with moments from the same TRAIN subjects. The linear
fit uses the same absolute ridge coefficient in every case, preserving
the exact orthogonal-coordinate equivariance of B up to floating-point
noise.

**Seven candidate scalar losses on B:**

- `B_column_l1_loss`: mean L1 of each unit-L2 response column.
- `B_column_hoyer_loss`: dimension-normalized Hoyer dense-axis loss.
- `B_column_entropy_loss`: mean normalized entropy of energy across U axes.
- `B_column_top1_tail_loss`: energy outside each response column's
  strongest latent axis.
- `B_global_l1_over_fro_loss`: full coefficient L1 normalized by Frobenius.
- `B_varimax_negative_loss`: negative variance of normalized squared
  coefficients across response features (Varimax-type axis concentration).
- `B_row_overlap_loss`: mean absolute cosine overlap of B rows across
  response features.

**Two explicit negative controls:** `control_B_fro_norm` and
`control_B_singular_fro`, which should be constant across orthogonal
rotations. They are never treated as evidence for baseline.

### 3. Nonlinear decomposition with fixed subject splits

Fit a *generic complete degree-3 polynomial basis* in U, using no X
formula. The target remains the shared subject-level F and the
train/test split is fixed once for all rotations. Train and evaluate
on disjoint **subjects**.

- `function_add_quad_mse`: intercept, U_i, U_i² only.
- `function_full_quad_mse`: same, plus U_i U_j terms.
- `function_quad_interaction_gap`: additive MSE − complete quadratic MSE.
- `function_add_cubic_mse`: intercept, U_i, U_i², U_i³ only.
- `function_full_cubic_mse`: full cubic polynomial including cross terms.
- `function_cubic_interaction_gap`: additive MSE − complete cubic MSE.

If one set of axes makes the function closer to additive, the corresponding
gap should be smaller. **Complete** quadratic and cubic polynomial spans
are invariant to an invertible linear coordinate transformation in the
unregularized limit, giving informative capacity controls.

**Compression / description-length proxies:** fit the complete cubic model
on TRAIN, rank each polynomial basis function by the L2 norm of its
multi-output coefficient vector, keep the top fixed budgets of
`4,8,12,20` nonconstant terms, refit on TRAIN and measure held-out
subject MSE. `function_budget_auc_loss` averages these four errors
(lower is better). `function_full_cubic_group_l1_loss` measures the
sparsity of the full cubic group coefficients.

These are **predictive-compression proxies, not formal minimum
description length**: chosen basis, degree, feature scaling and
regularization all matter. In shared-bank mode, the nonlinear stage
uses up to **64 evenly spaced stimulus columns** (configurable), to
keep the cubic fits practical; the linear B stage still uses the full
available stimulus-response profile.

### 4. Freeze scores first, then reveal identity and angle

The chronological order in `run.py` is:

1. Read Step-2 W case paths, randomize, assign blind IDs.
2. Compute **every** (channel × view × metric × case) numeric loss using
   only W, Y, subject IDs and the fixed TRAIN/TEST split.
3. Freeze the complete numeric table in `blind_scores.csv`.
4. **Only now** create `blind_key.csv` and `revealed_scores.csv`.
5. Rank all candidates, compare 0° and 90° endpoints to 15°–75°
   intermediate angles across all six pairs, and draw angle curves.

0° is the **Step-1 supervised best R\*** and 90° is signed axis
permutation. An axis/sign-symmetric simplicity criterion should treat
these endpoints as equivalent. Flat curves are NOT counted as a success.
`metric_summary.csv` explicitly reports baseline rank, the lowest-loss
case, number of planes with endpoint-favored structure, median
middle-angle minus endpoint complexity, and 0°–90° discrepancy.

There is **no post-hoc metric tuning** to force baseline success; scores
from other C1/CR3 models may disagree. Treat this first sweep as
hypothesis generation, with an independent future replication required
to validate a selected criterion.

## Windows CMD

From `D:\Projects\mod-rot-3`:

```cmd
conda activate py11
git pull origin main

python -m pytest -q supervised_abs_rotation/step3_blind_matrix/tests
python supervised_abs_rotation/step3_blind_matrix/run.py --model C1
python supervised_abs_rotation/step3_blind_matrix/run.py --model CR3
```

Step 1 `stage1_data` and Step 2 `step2_pairwise_sweep` must already
have completed for this model and configuration. The script never calls
earlier stages automatically.

To do a faster preliminary logit-only trial with a separate output,
without overwriting the full experiment:

```cmd
python supervised_abs_rotation/step3_blind_matrix/run.py --model CR3 --channels logit_nn --output outputs/supervised_abs_rotation/CR3_step3_logit_screen
```

After code changes rerun with `--overwrite` only for the intended output
directory. Other supported arguments: `--step2-dir`, `--output`,
`--beta`, `--latent-dim`, `--repeat`, `--objective`,
`--seed`, `--train-fraction`, `--fingerprint-ridge`,
`--nonlinear-ridge`, `--profile-ridge`,
`--max-profile-outputs`, `--no-plots`, `--overwrite`.

## Output paths and files

```text
outputs/supervised_abs_rotation/<MODEL>/<CONDITION>/repeat01/abs_mse/
  step3_blind_matrix/
    summary.json             # provenance, settings, counts and limits
    blind_scores.csv         # ALL scores with anonymous IDs; no angle/name
    blind_key.csv            # separate case identity reveal
    revealed_scores.csv      # same frozen scores + case name
    rankings.csv             # baseline ranking for EVERY score/channel/view
    metric_summary.csv       # baseline rank, plane successes, 90° checks
    plane_curves.csv         # full 0/15/30/45/60/75/90 curves per U pair
    subject_splits.csv       # reproducible fixed TRAIN/TEST subject IDs
    plots/
      curve_logit_nn_B_column_entropy_loss.png
      curve_logit_nn_B_column_top1_tail_loss.png
      curve_logit_nn_function_quad_interaction_gap.png
      ...
    diagnostic_B/
      B_logit_nn_baseline.npy
      B_logit_nn_baseline.png
      B_logit_nn_u1_u2_045.npy
      B_logit_nn_u1_u2_045.png
      ...
```

The `diagnostic_B` matrix outputs and plots are generated **after**
the blind scoring is frozen, solely for explanation. No analytical
generator, simulator parameter, or decoder weight is loaded.

## Interpretation and limitations

- **No true T, generative expression, or neural model internals are used.**
  Only the supplied WY sample and metadata needed to align subject and
  trial indices.
- The first-stage best R\* was **supervised** using T. Step 3 is blind
  about that provenance, not an unsupervised proof that any metric must
  prefer R\*.
- **Same information ≠ same axis simplicity.** All cases are bijective
  rotations of the same latent data; complete-flexibility prediction,
  U-covariance eigenvalues, singular values, etc. do not prove recovery.
- **Subject holdout is not independent trial holdout:** the S5 posterior
  used all choice labels to infer each subject's Z. Report predictive
  error here as geometric/profiling diagnostics, not prospective generalization.
- C1 non-shared stimulus profiles are **approximations**: a poor per-person
  X regression can mask real interactions. CR3 shared-bank raw profiles
  are direct observed output matrices.
- A large number of correlated metrics are screened. The analysis is
  exploratory; apply multiple-testing-aware replication before a final claim.
