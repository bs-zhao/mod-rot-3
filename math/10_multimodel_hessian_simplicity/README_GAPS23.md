# Gap 2 + Gap 3: simultaneous numeric-WY Hessian estimation, after Gap 1

This extension to [math/10](README.md) implements two **matched** branches after the analytic Gap 1:

- **Gap 2**: generate numeric Y from the true formula at the TRAIN-mapped \(\widehat T=A Z+b\), infer Hessians from the numeric (W,Y) table only.
- **Gap 3**: use the **original S5 all-trial inferred Z** and **frozen S3 NP decoder** to produce numeric Y at precisely the same W, then infer Hessians with precisely the same estimator. Never differentiate the neural network.

**Use one command (Windows CMD, from repository root):**

~~~cmd
conda activate py11
git pull origin main
python math/10_multimodel_hessian_simplicity/run_pipeline.py --models C1 C2 C3 CR3
~~~

This first checks that saved analytic Gap 1 results belong to the current S1 datasets (exact SHA256 hashes). If valid, it **reuses** them; otherwise it runs Gap 1. It then executes the Gap 2 / Gap 3 paired numeric experiment. The separate run_all.cmd now invokes this whole pipeline. Use --rerun-gap1 to force re-running the analytic screen or --only-gap1 to stop after it.

Because the uploaded Gap1 output already exists on your Windows computer, you can explicitly run just the next two gaps:

~~~cmd
python math/10_multimodel_hessian_simplicity/run_pipeline.py --models C1 C2 C3 CR3 --only-gaps23
~~~

That command requires completed Gap1 and current matching S1 hashes; if the files are absent or stale, it fails with a precise error instead of claiming to have validated a different data source.

## 1. Actual model identity and NP configuration

The four actual generators are C1, C2, C3, CR3. The repository does **not** implement a distinct CR1. Gap2/3 deliberately reject the spelling CR1 to prevent an accidental false CR1 claim.

Default pre-trained NP conditions:

| Model | True T and Z dimension | β | X width |
|---|---:|---:|---:|
| C1 | 4 | 0.0005 | 3 |
| C2 | 4 | 0.0005 | 4 |
| C3 | **6** | 0.0005 | 4 |
| CR3 | 4 | 0.00005 | 6 |

All four use lambda_sparse=0, context fractions 0.25–0.8, repeat01. The three major configuration arguments are exposed: --zdim, --pca {0,1}, --beta-c and --beta-cr3. The run requires zdim equal to T dimension; no lossy dimensionality reduction is silently substituted.

The S5 file is the one created with **all observed trials** by np/s5_extract_z.py, even for TEST subjects. This experiment assesses functional structure in that S5 representation, **not held-out trial predictive performance**.

Required files per model:

~~~text
outputs/choice/MODEL/s1_generate/s1_generate__dataset.npz
outputs/choice/MODEL/s3_train/CONDITION/repeat01/s3_train__checkpoint.pt
outputs/choice/MODEL/s4_evaluate/CONDITION/repeat01/s4_evaluate__latent.npz
~~~

The code checks file presence, matching SHA256, complete TRAIN/VALIDATION/TEST partitions, full-trial S5 provenance, and the original theta dimensions. It will **not** generate pseudo-Z, silently retrain NP, or substitute fake NN predictions when a checkpoint is missing.

## 2. Three-way experimental design

TRAIN-only affine map to the true theory parameter coordinate:

\[
\widehat T_s=A_{\mathrm{train}}\,V_s+b_{\mathrm{train}},\quad
V_s=Z_s\ \mathrm{or\ full}\ d\text{-dimensional PCA}(Z_s).
\]

PCA, when enabled, retains **all** d components fitted using TRAIN Z and is invertible. NN decoding always uses original unmodified S5 Z, never projected T_hat.

Define a standardized common input coordinate:

\[
W_{st}=\left[\frac{\widehat T_s-\mu_{T,\mathrm{TRAIN}}}{s_T},
                  \frac{X_{st}-\mu_X}{s_X}\right].
\]

The **same physical TRAIN/VALIDATION/TEST subject IDs and selected X trials** are used for all three references:

| Name | How Y is obtained | How H is obtained |
|---|---|---|
| analytic_T_hat | Original generator formula at T_hat | Analytic ZZ/ZX, evaluated on matched TEST W |
| gap2_formula_Y | Original generator formula emits only logit and probability | Numeric-WY fitted regression, then differentiate the fitted regression |
| gap3_nn_Y | Frozen NP decoder at original Z and raw X emits two-class logit difference and probability | **Exactly the same** numeric-WY fitted regression procedure |

The numerical labels have four response columns:

\[
Y_{\mathrm{fit}}=
[\,\eta_{\mathrm{formula}}, p_{\mathrm{formula}},
  \eta_{\mathrm{NN}},p_{\mathrm{NN}}\,].
\]

**One multi-output regression fit** estimates all four sources together at each estimator setting, so formula Y and decoder Y have the same feature representation, regularization and subject split. NN weights are only used for forward Y; no autograd or PyTorch Hessian is used.

For C1/C2/C3 the original randomized X trials are sampled from each person's S1 observations. For CR3 the same selected X indices from the official 500-item bank are used for **every** subject. Rows where the analytic CR3 X-derivatives are undefined (zero amounts or non-interior probabilities) are excluded from the **joint valid support**; invalid T_hat values are also excluded. Metadata records the conditional evaluation domain.

## 3. Two complementary estimators and selection

- **Polynomial sieve**: total-degree 2 or 3 in the standardized W (all ZZ/ZX mixed terms allowed). Ridge fit on TRAIN numerical Y only; polynomial derivatives are computed analytically from fitted coefficients.
- **Isotropic Gaussian RBF ridge**: Nystrom landmarks from TRAIN W, Gaussian features, ridge, analytic second derivatives of the fitted RBF surface.

The degree, ridge, bandwidth grids are selected by **VALIDATION numeric Y RMSE** pooled over the four columns after per-channel response-scale normalization. The **same winning hyperparameters apply to Gap 2 and Gap 3**. The true analytic Hessian, Gap1 angle minima, TEST Y, and true theory axis are NOT used for estimator selection.

The common surrogate fitted at reference W is held fixed when varying R. The **local Hessians are transformed** by the proper chain rule at each rotation. This deliberately avoids angle-wise refitting artefacts: it studies the simplicity of the same estimated function in different bases, rather than unstable angle-by-angle regression noise.

Third/fourth derivative and oracle-only function-value contrast metrics are **not currently estimated by the Gap2/3 branches**. These are listed in the missing/unsupported files, not assigned fabricated values. Gap1's complete metric atlas remains unchanged; Gap2/3 use a **frozen shortlist** selected from Gap1 before inspecting decoder results. The selective implementation has unit tests matching the original metrics output exactly. --metric-mode all computes all compatible second-order metrics (slower); default --metric-mode shortlist is faster.

### Lightweight run

~~~cmd
python math/10_multimodel_hessian_simplicity/run_pipeline.py --models C1 C2 C3 CR3 --only-gaps23 --train-subjects 40 --validation-subjects 12 --test-subjects 12 --trials-per-subject 12 --angles 0,45,90 --top-per-model 10 --top-global 10 --plots 0
~~~

### Larger experiment

~~~cmd
python math/10_multimodel_hessian_simplicity/run_pipeline.py --models C1 C2 C3 CR3 --only-gaps23 --train-subjects 300 --validation-subjects 60 --test-subjects 80 --trials-per-subject 40 --angles 0:90:5 --top-per-model 60 --top-global 100
~~~

### Alternate PCA/β settings

~~~cmd
python math/10_multimodel_hessian_simplicity/run_pipeline.py --models C1 C2 C3 CR3 --only-gaps23 --pca 1 --repeat 1 --beta-c 0.0005 --beta-cr3 0.00005
~~~

A different β/latent dimension/repeat requires the **matching** S3 checkpoint and S5 latent, not only a CLI change.

## 4. Three layers of comparisons

1. **Gap1 analytic results**, at the original true T and prior/design sample measure Q (may not equal the later T_hat TEST empirical measure).
2. **Gap2 inference error**: fitted formula-Y Hessian vs exact formula Hessian evaluated at the *same T_hat TEST W*.
3. **Gap3 added NN output effect**: fitted decoder-Y Hessian vs fitted formula-Y Hessian using the same model, regularization, coordinate, TEST points and rotation; also reports **NN vs oracle total difference**.

Accuracy is reported separately for ZZ, ZZ diagonal, ZZ offdiagonal, ZX: MAE, RMSE, relative Frobenius error, cosine and RMS scale. Rotation-curve pattern similarity is reported as Pearson r, normalized shape RMSE, minimum-angle shift, flatness and true-axis endpoint/near-endpoint minima, separately for each of the three comparisons. A signed mean or vector norm alone is not used as proof of successful structure recovery.

**Important limitation:** The Z-to-T affine mapping uses true T on TRAIN subjects. It is an intentionally supervised *benchmark coordinate*, not an unsupervised solution to rotating Z. The NN gap combines decoder fidelity and Z-to-T mapping mismatch. No NN-Y accuracy result implies automatically accurate Hessians.

## 2026-11-09 output audit and reranking correction

The uploaded C2.zip actually contains all four models, C1/C2/C3/CR3. The original cross_model_threeway_ranking.csv could rank an indicator with 100% near-axis success from C1 alone above indicators tested in all four models: the example zx_mean_abs_column_tail2sq_k1_loss has 24 comparisons all from C1 and none from C2/C3/CR3. This is **not** universal success.

The ranking has therefore been corrected. For the four models, complete coverage requires (6+6+15+6) pairwise planes x 2 methods x 2 Y channels = **132 records per comparison**, and the new fields include models_covered, expected_conditions, full_coverage, worst-model near-axis fraction, both analytic-and-estimated near-axis fraction, and worst-model curve correlation. Rerank previously saved outputs without refitting or decoding:

~~~cmd
git pull origin main
python math/10_multimodel_hessian_simplicity/rerank_existing.py
~~~

This writes cross_model_threeway_ranking_coverage_aware.csv; it preserves the old ranking. In the uploaded results, 71 candidate loss metrics have full four-model/method/channel/plane coverage. None is yet a verified four-model universal recovery solution.

Also, C1/C2/C3 have analytically zero Logit ZZ Hessian. Relative error divided by a zero oracle is **undefined**, not an enormous numerical percentage. New field_errors.csv outputs use reference_zero=1, relative_frobenius blank, and the absolute fitted RMSE to quantify false curvature.

The leading scientific bottleneck is CR3: polynomial sieve gives Gap2 logit ZZ cosine ~0.933 but ZX cosine ~0.540; Gap2 probability ZX cosine ~0.253. High decoder-vs-formula output correlation alone cannot fix these derivative errors. A useful next experiment is numeric logit regression followed by the known sigmoid chain rule to obtain probability Hessian, compared to direct probability regression. That proposed estimator enhancement is **not implemented in this version**.

---

## 5. Outputs

All new files are written separately from the Gap1 root under:

~~~text
outputs/math/10_multimodel_hessian_simplicity/gaps23/
  manifest.json
  all_models_landscape_comparisons.csv
  cross_model_threeway_ranking.csv
  C1/beta0p0005_lambda0_ctx0p25to0p8_L4_pca0/
    metadata.json
    gap1_frozen_metric_shortlist.csv
    hyperparameter_validation.csv
    field_errors.csv
    threeway_curves.csv
    threeway_landscape_comparisons.csv
    sieve__test_hessians.npz
    rbf__test_hessians.npz
    sieve__missing_oracle_metrics.csv
    rbf__missing_oracle_metrics.csv
    sieve/plots/*.png
    rbf/plots/*.png
  C2/...
  C3/...
  CR3/...
~~~

### Testing

~~~cmd
python -m unittest discover -s math/10_multimodel_hessian_simplicity -p "test_*.py" -v
~~~

Automated tests cover exact analytic mixed derivatives, correct shared W/Hessian transformations, both numeric-WY estimators, exact shortlist-score equivalence and end-to-end Gap2/3 control flow using FAKE forward Y. The fake test does **not** establish real NP accuracy; that requires running the commands above on your own checkpoint/data.

Related research: [readmes/history](../../readmes/history/README.md) and [rotation_research](../../readmes/rotation_research/02_hessian_simplicity_experiment.md).
