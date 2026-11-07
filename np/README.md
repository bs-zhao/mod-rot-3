# NP research: independently executed steps

Each experiment is its **own runnable Python file**, with actual stage logic in that file. No `np/run.py`, no `--stage` argument and no `all` orchestration.

| Order | Python entrypoint | What it does | Required previous output |
|---|---|---|---|
| 1 | `np/s1_generate.py` | Exact C1/C2/C3/CR3 simulator, theta/X sampling and binary choice Y | None; CR3 will prepare original design bank |
| 2 | `np/s2_precheck.py` | Original validity, identification and CR3 FOSD checks | S1 dataset |
| 3 | `np/s3_train.py` | Train original NP; choose beta, lambda, latent dimension, optimizer hyperparameters | S2 qualification |
| 4 | `np/s4_evaluate.py` | Saved NP validation/test accuracy, CE, KL, learning curves; if S5 exists, also posterior-Z histograms, per-axis variance and KL | S3 checkpoint; S5 optional for Z diagnostics |
| 5 | `np/s5_extract_z.py` | Encode every subject using **ALL observed trials**, save frozen posterior-mean Z | S3 checkpoint and S1 dataset |
| 6 | `np/s6_pca.py` | TRAIN-only PCA, FastICA and FactorAnalysis; threshold-selected PC/Z pair plots plus TEST PC/IC/FA-to-theta heatmaps and all-pair scatter | S5 Z |
| 7 | `np/s7_regression.py` | Full-Z and top-k-PC Ridge regression to simulator theta; validation-selected alpha; held-out test | S5 Z and S6 PCA |

Run commands **one at a time** from repository root, e.g. for CR3 beta=0 L8:

```cmd
python np/s1_generate.py --model CR3 --subjects 5000
python np/s2_precheck.py --model CR3
python np/s3_train.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --device auto
python np/s4_evaluate.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
python np/s5_extract_z.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
python np/s6_pca.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
python np/s7_regression.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
```

Training-dependent steps must use matching `--beta`, `--lambda-sparse`, `--latent-dim`, `--context-min`, `--context-max`, and `--repeat` to locate the same checkpoint and Z.

**Code organization**: `np/src/modlearn_rot/` contains frozen upstream simulator/NP/training mathematics. `np/src/modrot3/pipeline.py` contains only shared paths/data adaptation/provenance functions; `np/src/modrot3/analysis.py` contains only shared numerical analysis utilities. Each step's actual implementation lives in its corresponding `np/sN_*.py` file.

All data, checkpoints, figures, and parameter recovery CSV/JSON files are under **root-level `outputs/choice/<MODEL>/`**, excluded by `.gitignore`. In particular the rotation platform consumes the unchanged `s4_evaluate__latent.npz` file from S5. Neither S6 nor S7 alters that file. True theta is never used for NP training, Z extraction, or fitting rotations.
### Inspecting posterior Z distributions in S4

S4 keeps its original checkpoint metrics and plots, and **optionally** adds
all-subject latent diagnostics when matching S5 frozen Z is already available.
The original S4-before-S5 order still works: S4 prints a clear skip notice
when S5 does not exist. After S5, simply rerun S4 (no retraining):

```cmd
python np/s5_extract_z.py --model CR3 --beta 0.0005 --latent-dim 8
python np/s4_evaluate.py --model CR3 --beta 0.0005 --latent-dim 8
```

New files are next to the existing S4 results under
`outputs/choice/<MODEL>/s3_evaluate/<CONDITION>/repeat01/`:

* `s3_evaluate__z_dimension_histograms.png`: original-coordinate posterior-mean
  histograms for every Z axis; all subjects plus TRAIN/VAL/TEST outlines.
* `s3_evaluate__z_dimension_stats.csv` and
  `s3_evaluate__z_diagnostics.json`: mean, standard deviation, between-subject
  variance (population ddof=0), min/max, 1/5/25/50/75/95/99 percentiles,
  mean absolute value, mean/median posterior variance, exact per-axis Gaussian
  KL, and aggregate posterior variance.
* `s3_evaluate__z_variance_and_kl.png`: between-subject posterior-mean
  variance, mean conditional posterior variance and axis-wise KL.
* `s3_evaluate__z_correlation_heatmap.png`: labeled Pearson-r heatmaps of
  original posterior-mean Z dimensions for ALL subjects and TRAIN subjects.
* `s3_evaluate__z_correlations_{all,train,validation,test}.csv`: all four
  splits' exact numeric Z-by-Z Pearson correlation matrices, with labeled rows
  and columns. The corresponding numeric matrices are also in
  `s3_evaluate__z_diagnostics.json`.
* `s3_evaluate__z_pca_spectrum.png`: **TRAIN-only** posterior-mean PCA absolute
  eigenvalues, variance ratios and participation-ratio effective dimension.

The `z` column is the S5 **posterior mean**, not a draw from the posterior.
A small variance of posterior means alone does **not** prove posterior
collapse. The per-axis diagonal-Gaussian KL is calculated directly as
`0.5 * mean(mu**2 + exp(logvar) - 1 - logvar)`. The plots use the S5 all-
observed-trials inference regime, whereas the training KL was measured on
random context masks. S5 must agree with S3's checkpoint hash, dataset hash,
latent dimension and subject splits; no theta is accessed.


### S6: PC vs original-Z pairwise dependence diagnostics

S6 accepts two **independent thresholds** (both select dimensions using
TRAIN subjects only; comparisons use **strictly greater than**):

* `--pc-var-threshold 0.05`: select PCA components whose **individual**
  TRAIN explained-variance ratio is > 0.05, e.g. PC1–PC4.
* `--z-var-threshold 0.01`: select **original z coordinates** whose
  between-subject TRAIN variance (ddof=1) is > 0.01; unlike the PC
  threshold, this is an **absolute variance** in raw z units.

Example:

```cmd
python np/s6_pca.py --model CR3 --beta 0.0005 --latent-dim 8 --pc-var-threshold 0.05 --z-var-threshold 0.01
```

New files under `outputs/choice/<MODEL>/s12_z_diagnostics/<CONDITION>/repeat01/`:

* `s12__selected_pc_correlations.csv` and
  `s12__selected_z_correlations.csv`: labeled **ALL-subject** signed
  Pearson r matrices for retained original axes. Analogous
  `_train.csv`, `_validation.csv`, and `_test.csv` versions included.
* `s12__selected_pc_correlation_heatmap.png` and
  `s12__selected_z_correlation_heatmap.png`: three side-by-side panels
  for ALL, TRAIN, TEST, each annotated with signed r.
* `s12__selected_pc_pairwise_scatter.png` and
  `s12__selected_z_pairwise_scatter.png`: scatterplot matrices
  (lower triangle ALL gray plus TEST orange, diagonal histograms,
  upper triangle signed r).
* `s12__z_train_variances.csv`: original Z-axis variance and selection.
* `s12__selected_dimensions.json`: selected indices/thresholds and
  split-specific correlation matrices.

The existing `s12__pca.npz`, `s12__pca_variance.csv`,
`s12__pca_summary.json`, theta-vs-Z/PC correlations and original plots
remain intact. PCA TRAIN components have near-zero Pearson correlations
**by construction**; inspect ALL/TEST matrices and scatterplots, and
do not equate uncorrelatedness with statistical independence. When a
threshold selects zero axes, S6 records that status rather than failing.


### S6: TEST correlations and scatterplots from PCA, ICA and 4-factor FA

All three representations are fitted **only using TRAIN subjects' frozen
posterior-mean Z**, without access to simulator theta:

1. **PC1..PCL**: all components of the existing TRAIN PCA, whether or not
   they pass `--pc-var-threshold`. (Thresholded PC plots above remain intact.)
2. **IC1..ICL**: sklearn FastICA of TRAIN-standardized original Z,
   `whiten='unit-variance'`, deterministic seed 31, all latent dimensions.
3. **FA1..FA4**: sklearn FactorAnalysis, four factors, `rotation='varimax'`,
   on the same TRAIN-standardized Z. If latent dimension is smaller than 4,
   the factor count is reduced to the available dimension.

The fitted transforms are then applied to ALL subjects. Each representation
is compared to ALL true parameters on **held-out TEST subjects only**, never
used for transform fitting, axis selection, sign flips or optimization.

New files under the same `s12_z_diagnostics/<condition>/repeatXX/` folder:

* `s12__{pc,ic,fa}_theta_correlations.csv`: signed Pearson-r matrix with
  true parameters in rows and every corresponding component in columns.
* `s12__{pc,ic,fa}_theta_heatmap.png`: annotated signed-r heatmap,
  fixed red/blue range [-1, 1].
* `s12__{pc,ic,fa}_theta_scatter.png`: full theta-by-component scatter
  grid on TEST subjects, annotated with signed `r` and `r²`.
* `s12__{pc,ic,fa}_theta_pairwise.csv`: one row per parameter/component
  combination, signed `r`, squared correlation, and test sample count.
* `s12__ica.npz`, `s12__factor_analysis.npz`: fitted transform outputs,
  TRAIN feature normalization and provenance; original `s12__pca.npz`
  remains unchanged for S7.
* `s12__component_theta_summary.json`: results, configuration and
  convergence warnings.

For the already trained L4 run with beta=5e-5:

```cmd
python np/s6_pca.py --model CR3 --beta 0.00005 --latent-dim 4 --pc-var-threshold 0.05 --z-var-threshold 0.01 --fa-factors 4
```

Important interpretation: ICA components and FA factors are identifiable
only up to certain permutations, signs and potentially other ambiguities;
they are not forced to match the generating theta. Four factors extracted
from **four** original Z variables are generally an underidentified factor
model. The correlations are descriptive comparisons, not evidence of
uniquely recovered causal latent factors.
