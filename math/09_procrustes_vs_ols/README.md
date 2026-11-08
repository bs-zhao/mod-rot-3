# 09 · Oracle affine alignment versus orthogonal Procrustes (C1 / C2 / C3 / CR3)

**Independent supervised geometry control.** This is NOT an unsupervised rotation method and **does not use X or Y to fit axes**. It tests whether true T can be recovered from frozen NP Z by a genuine orthogonal transform, or whether general linear mixing is needed. True T is used **only for this oracle experiment**.

## Inputs

The NP step S5 already saved `z`, `theta_true` and `train_ids`, `validation_ids`, `test_ids` in:

`outputs/choice/<MODEL>/s4_evaluate/<CONDITION>/repeatXX/s4_evaluate__latent.npz`

The loader checks S3 checkpoint SHA, S1 dataset SHA, context mode, and subject splits. No training, decoding or re-extraction.

| Model | Z and T dimensions | beta | lambda_sparse | context |
| --- | --- | --- | --- | --- |
| C1 | 4 | 0.0005 | 0 | 0.25–0.80 |
| C2 | 4 | 0.0005 | 0 | 0.25–0.80 |
| C3 | **6** | 0.0005 | 0 | 0.25–0.80 |
| CR3 | 4 | **0.00005** | 0 | 0.25–0.80 |

## Math and leakage control

All transforms/normalizations are fitted on **TRAIN** subjects only:

\[
Z_s=(Z-\mu_{Z,\mathrm{train}})/\sigma_{Z,\mathrm{train}},\quad
T_s=(T-\mu_{T,\mathrm{train}})/\sigma_{T,\mathrm{train}}.
\]

A standardized fitted map is `U_std = Z_std @ B`; in original coordinates it is an affine map `T_hat = A_raw @ Z + b_raw` (column-vector convention). Standardization implies TRAIN regression intercept zero; it does not imply the original intercept is zero.

Six supervised oracle methods are evaluated on identical TRAIN/VALIDATION/TEST splits:

1. `ols_affine` — unconstrained least squares for B.
2. `procrustes_od` — orthogonal Procrustes, Q.T Q = I; reflections/sign flips permitted.
3. `procrustes_so` — proper rotation, additionally det(Q) = +1.
4. `procrustes_od_scaled` — orthogonal Procrustes plus separately fitted scaling per U axis (changes R², not |correlation|).
5. `whiten_procrustes_od` — whiten Z by TRAIN covariance and then apply O(d) Procrustes (not a pure rotation of original Z).
6. `whiten_procrustes_od_scaled` — whiten, rotate, calibrate each axis separately.

No transformation or label matching is fitted on TEST. Reports cover all U-to-T correlations (diagonal and off diagonal), U-to-U correlations, per-parameter R² and standardized RMSE, conditioning, orthogonality, fitted maps, and held-out scatter plots.

## Windows CMD

Run independently from the repo root:

```cmd
conda activate py11
git pull origin main
python -m unittest discover -s math/09_procrustes_vs_ols -p "test_*.py" -v
python math/09_procrustes_vs_ols/run.py --model C1 --beta 0.0005 --latent-dim 4
python math/09_procrustes_vs_ols/run.py --model C2 --beta 0.0005 --latent-dim 4
python math/09_procrustes_vs_ols/run.py --model C3 --beta 0.0005 --latent-dim 6
python math/09_procrustes_vs_ols/run.py --model CR3 --beta 0.00005 --latent-dim 4
python math/09_procrustes_vs_ols/compare_four.py
```

The single-model script now selects beta by model when `--beta` is omitted: **0.0005 for C1/C2/C3, 0.00005 for CR3**. The summary script uses independent `--beta` (C models) and `--cr3-beta` (CR3), with the same defaults. If all four result folders already exist, simply run `python math/09_procrustes_vs_ols/compare_four.py`; there is no need to rerun the four analyses.

Additional flags: `--lambda-sparse`, `--context-min`, `--context-max`, `--repeat`, `--output`, `--no-plots`, `--max-scatter`. Z dimensions auto-select 4/4/6/4 when omitted.

## Saved results

Each model: `outputs/math/09_procrustes_vs_ols/<MODEL>/<CONDITION>/repeatXX/`

- `summary.json`: matrices, intercepts, training statistics, split metrics, and TEST deltas relative to OLS.
- `method_split_summary.csv`: signed/absolute diagonal correlations, off-diagonal summary, R², RMSE for TRAIN/VAL/TEST.
- `per_parameter.csv`: each true parameter and each method's TEST r, |r|, R², RMSE/MAE.
- `corr__<method>__<split>.csv` and `uu_corr__<method>__<split>.csv`: full correlation matrices.
- `test_corr__<method>.png`, `test_scatter__<method>.png`, `test_method_comparison.png`.
- `mappings_and_U.npz`: standardized and original-scale matrices and intercepts, all U, train/val/test IDs.

`compare_four.py` **only summarizes already saved results** into `four_models_test_comparison.csv` and `.png`. It never fits or reruns experiments.

Interpretation: A poor SO(d) fit in the presence of a good O(d) fit often means the optimal alignment contains an odd reflection/sign flip; latent-axis signs are generally arbitrary. OLS minimizes TRAIN squared error, not necessarily TEST r or R². Near-standard-normal marginal Z does not imply identity *joint* covariance; whitening checks this. This supervised oracle control does **not** demonstrate that a no-T simplicity objective can discover the alignment. Real four-model results require the user's locally saved frozen Z data.

## Stage 2: Optimize orthogonal axes AND four/six signed scales jointly (without rerunning Stage 1)

The confirmed CR3 condition is **beta=0.00005, L4** (not beta=0.0001).
On held-out TEST (n=750), the **direct OLS Oracle** reached mean diagonal
|r| = **0.8741**, max off-diagonal |r| = **0.1545**, mean R² = **0.7639**.
For comparison: ordinary O(d) Procrustes had mean R²=0.7418,
O(d)+axis scaling reached 0.7609, and whitened O(d)+scaling reached 0.7619.
These are user-reported real runs, not synthetic test numbers.

**Important:** the direct OLS Oracle has NOT been deleted or replaced:
it is exactly the saved Stage-1 `ols_affine__B` matrix (original units:
`ols_affine__A_raw_Z_to_T` and `ols_affine__b_raw_Z_to_T`).
The optimization below uses original TRAIN rows *only* and never touches
these saved matrices.

We now optimize:

\[
\min_{Q^\top Q=I,\;s\in\mathbb R^d}
\|Z_{\rm std}Q\operatorname{diag}(s)-T_{\rm std}\|^2_{\rm TRAIN}.
\]

This is **orthogonal + independently optimized scales**, NOT unrestricted OLS.
Each candidate Q is optimized via an antisymmetric exponential update;
the signed optimal scales are solved analytically for each Q.
Multiple starts (plain Procrustes, polar(OLS), seeded perturbations) reduce
local-minimum risk. The OLS baseline remains the first method in reports.

Four models, each as a separate command:

```cmd
git pull origin main
python -m unittest discover -s math/09_procrustes_vs_ols -p "test_*.py" -v
python math/09_procrustes_vs_ols/refine_existing.py --model C1
python math/09_procrustes_vs_ols/refine_existing.py --model C2
python math/09_procrustes_vs_ols/refine_existing.py --model C3
python math/09_procrustes_vs_ols/refine_existing.py --model CR3
python math/09_procrustes_vs_ols/compare_four.py
python math/09_procrustes_vs_ols/compare_refinements.py
```

The first four commands **only read** already saved Stage-1 results
(including CR3 beta5em05...L4). `compare_four.py` continues to summarize
the **unchanged Stage-1 six-method comparison**, while `compare_refinements.py` summarizes the new optimized results in
`four_models_stage2_comparison.csv`, `four_models_stage2_per_parameter.csv`,
and `four_models_stage2_comparison.png`. Each Stage-2 analysis
produces its own new results under:

`outputs/math/09_procrustes_vs_ols/<MODEL>/<CONDITION>/repeat01/stage2_joint_orthogonal_scaled/`

Output: `summary.json`, `method_comparison.csv`,
`per_parameter.csv`, `wrong_parameter_correlations.csv`,
full TEST U–T and U–U correlation matrices (CSV), fitted `Q`/scales/B
(NPZ), method comparison and per-parameter figures (PNG).

No Stage-2 results are claimed until these commands are run on the local
real saved frozen Z files. The fitted orientation and scaling do NOT
recover T without supervision; this is a geometry control only.

### Optional, different experiment: CR3 L8

`recheck_cr3_l8.py` is an **optional separate L8-to-T4 audit** and is
NOT the source of the confirmed successful **L4 beta=0.00005** Oracle.
Do not confuse L4 and L8 NP checkpoints or corresponding beta values.
