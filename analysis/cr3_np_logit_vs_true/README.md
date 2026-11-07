# CR3 Neural Process decoder versus true logits and probabilities

This independent, read-only analysis compares ALL 2,500,000 (subject, trial)
predictions of a frozen CR3 Neural Process checkpoint to the exact simulator.

Default frozen checkpoint:

    outputs/choice/CR3/s3_train/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/s3_train__checkpoint.pt

Default CR3 data:

    outputs/choice/CR3/s1_generate/s1_generate__dataset.npz

The original S1 file contains `parameters`, `x`, `y`, and
`predictive_true` (the true p). Contrary to what might be expected, the
committed S1 generator does **not** save a separate pre-sigmoid true logit.
This analysis reconstructs true logit via the **exact generator source**
`np/reference/s0_cr2_design_recovery.py:logits`, and raises an error
unless sigmoid(reconstructed logit) reproduces the saved true p.

## Run

From the repository root in the existing py11 environment:

```cmd
conda activate py11
git pull origin main
python -m unittest discover -s analysis/cr3_np_logit_vs_true -p "test_*.py" -v
python analysis/cr3_np_logit_vs_true/run.py
```

Optional (reduce inference memory, restrict plots):

```cmd
python analysis/cr3_np_logit_vs_true/run.py --batch-size 8 --max-plot-points 100000
```

To analyze another checkpoint or repeat, specify `--checkpoint` explicitly,
and optionally `--condition` or `--repeat`. The implementation
validates the S1 dataset hash stored in the checkpoint, binary model
configuration, latent dimension 4, and beta 5e-5.

## Exact computational protocol

For each subject independently:

1. Gather all 500 (X, observed binary choice) pairs. These are
   **encoder observations**. Use the checkpoint's ORIGINAL 7-channel
   encoder mean/std to normalize the encoder input.
2. Set the context mask to all 500 observed trials (all True) and run
   `model.encode_context(...)`. Store both posterior mean
   `mu` and log variance.
3. Use **mu**, NOT a randomly drawn posterior latent, as the subject's z.
4. Feed the subject's mu and **all 500 original, unnormalized** X trials
   to `model.decode(X,mu,mask)`.
5. The binary network outputs two logits, `(l0,l1)`.
   Its Bernoulli logit (choice 1 versus 0) is `l1-l0`. The neural
   choice probability is `softmax([l0,l1])[1]`.
6. Compare neural and true logit **and separately** neural and true p.

Do not compare the decoder's raw second class logit `l1` alone against
the simulator's scalar logit: that would be incorrect because adding the
same constant to both class logits changes neither softmax probability
nor the Bernoulli log-odds.

**Important interpretation:** all 500 target choices are included in
their subject's encoder context. Consequently this experiment measures
all-context **reconstruction**, not prediction on held-out trials.
The original subject train/validation/test splits are still reported
separately, but even test subjects provide all their 500 observed choices
for z inference. Do not use these metrics as a held-out-trial test score.

## Metrics

For both true-vs-neural Logit and true-vs-neural p, calculate:

- pooled Pearson r, RMSE, MAE, prediction R², mean bias;
- regression slope and intercept (neural value on true value);
- within-subject centered correlation across trials;
- within-trial centered correlation across subjects;
- per-subject correlation, RMSE and other metrics (5,000 CSV rows);
- per-trial correlation, RMSE and other metrics (500 CSV rows);
- correlations of subject mean scores and of trial mean scores;
- separate reports for train/validation/test subject splits.

The **within-trial** correlation is particularly important to study
whether the decoder preserves variation between subjects on the SAME
stimulus, rather than relying on strong trial-mean effects.

Additional probability diagnostics: calibration against true p and
against observed binary choices, Brier score, binary cross-entropy,
choice accuracy, and comparisons with the simulator oracle.

## Outputs

All outputs are placed in a NEW folder, without changing the checkpoint:

    outputs/analysis/cr3_np_logit_vs_true/
      beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/

- `all_subject_trial_predictions.npz`: full (5000,500) arrays
  `true_logit`, `pred_logit`, `true_p`, `pred_p`, and `binary_choice`;
  also all-trial z/posterior logvar, true theta, fixed X, split IDs and
  source SHA256 checksums. NO point sampling in this NPZ.
- `summary.json`: all and split-specific logit/probability metrics,
  calibration/Brier/CE/accuracy, provenance and inference description.
- `logit__per_subject.csv` and `probability__per_subject.csv`:
  one row per person, with true parameters, posterior mean z, and metrics
  across that subject's 500 trials.
- `logit__per_trial.csv` and `probability__per_trial.csv`:
  one row per fixed trial with X, metrics across all subjects.
- `logit__binned_by_true.csv` and
  `probability__binned_by_prediction.csv`: response
  curves / probability calibration.
- Both response channels have all-point scatter/hexbin and residual
  plots, per-subject/per-trial correlation histograms and RMSE
  histograms, plus subject-means and trial-means comparison figures.
- `logit_and_probability__calibration.png`: paired logit response
  curve and p calibration against both oracle p and actual choices.
- `logit_and_probability__splits.png`: within-trial correlations
  across the original train/validation/test subject splits.

Plots may subsample points for rendering only, using a fixed seed;
**every numeric metric and every saved prediction uses ALL observations**.

No new model training, no mutation of S1/S3/S5 outputs, and no coupling to
other experimental analyses.
