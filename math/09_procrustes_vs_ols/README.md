# 09 · Oracle affine alignment versus orthogonal Procrustes (C1 / C2 / C3 / CR3)

**Independent supervised geometry control.** This is NOT an unsupervised rotation method and **does not use X or Y to fit axes**. It tests whether true T can be recovered from frozen NP Z by a genuine orthogonal transform, or whether general linear mixing is needed. True T is used **only for this oracle experiment**.

## Inputs

The NP step S5 already saved \`z\`, \`theta_true\` and \`train_ids\`, \`validation_ids\`, \`test_ids\` in:

\`outputs/choice/<MODEL>/s4_evaluate/<CONDITION>/repeatXX/s4_evaluate__latent.npz\`

The loader checks S3 checkpoint SHA, S1 dataset SHA, context mode, and subject splits. No training, decoding or re-extraction.

| Model | Z and T dimensions | beta | lambda_sparse | context |
| --- | --- | --- | --- | --- |
| C1 | 4 | 0.0005 | 0 | 0.25–0.80 |
| C2 | 4 | 0.0005 | 0 | 0.25–0.80 |
| C3 | **6** | 0.0005 | 0 | 0.25–0.80 |
| CR3 | 4 | 0.0005 | 0 | 0.25–0.80 |

## Math and leakage control

All transforms/normalizations are fitted on **TRAIN** subjects only:

\[
Z_s=(Z-\mu_{Z,\mathrm{train}})/\sigma_{Z,\mathrm{train}},\quad
T_s=(T-\mu_{T,\mathrm{train}})/\sigma_{T,\mathrm{train}}.
\]

A standardized fitted map is \`U_std = Z_std @ B\`; in original coordinates it is an affine map \`T_hat = A_raw @ Z + b_raw\` (column-vector convention). Standardization implies TRAIN regression intercept zero; it does not imply the original intercept is zero.

Six supervised oracle methods are evaluated on identical TRAIN/VALIDATION/TEST splits:

1. \`ols_affine\` — unconstrained least squares for B.
2. \`procrustes_od\` — orthogonal Procrustes, Q.T Q = I; reflections/sign flips permitted.
3. \`procrustes_so\` — proper rotation, additionally det(Q) = +1.
4. \`procrustes_od_scaled\` — orthogonal Procrustes plus separately fitted scaling per U axis (changes R², not |correlation|).
5. \`whiten_procrustes_od\` — whiten Z by TRAIN covariance and then apply O(d) Procrustes (not a pure rotation of original Z).
6. \`whiten_procrustes_od_scaled\` — whiten, rotate, calibrate each axis separately.

No transformation or label matching is fitted on TEST. Reports cover all U-to-T correlations (diagonal and off diagonal), U-to-U correlations, per-parameter R² and standardized RMSE, conditioning, orthogonality, fitted maps, and held-out scatter plots.

## Windows CMD

Run independently from the repo root:

\`\`\`cmd
conda activate py11
git pull origin main
python -m unittest discover -s math/09_procrustes_vs_ols -p "test_*.py" -v
python math/09_procrustes_vs_ols/run.py --model C1 --beta 0.0005 --latent-dim 4
python math/09_procrustes_vs_ols/run.py --model C2 --beta 0.0005 --latent-dim 4
python math/09_procrustes_vs_ols/run.py --model C3 --beta 0.0005 --latent-dim 6
python math/09_procrustes_vs_ols/run.py --model CR3 --beta 0.0005 --latent-dim 4
python math/09_procrustes_vs_ols/compare_four.py
\`\`\`

Additional flags: \`--lambda-sparse\`, \`--context-min\`, \`--context-max\`, \`--repeat\`, \`--output\`, \`--no-plots\`, \`--max-scatter\`. Z dimensions auto-select 4/4/6/4 when omitted.

## Saved results

Each model: \`outputs/math/09_procrustes_vs_ols/<MODEL>/<CONDITION>/repeatXX/\`

- \`summary.json\`: matrices, intercepts, training statistics, split metrics, and TEST deltas relative to OLS.
- \`method_split_summary.csv\`: signed/absolute diagonal correlations, off-diagonal summary, R², RMSE for TRAIN/VAL/TEST.
- \`per_parameter.csv\`: each true parameter and each method's TEST r, |r|, R², RMSE/MAE.
- \`corr__<method>__<split>.csv\` and \`uu_corr__<method>__<split>.csv\`: full correlation matrices.
- \`test_corr__<method>.png\`, \`test_scatter__<method>.png\`, \`test_method_comparison.png\`.
- \`mappings_and_U.npz\`: standardized and original-scale matrices and intercepts, all U, train/val/test IDs.

\`compare_four.py\` **only summarizes already saved results** into \`four_models_test_comparison.csv\` and \`.png\`. It never fits or reruns experiments.

Interpretation: OLS minimizes TRAIN squared error, not necessarily TEST r or R². Near-standard-normal marginal Z does not imply identity *joint* covariance; whitening checks this. This supervised oracle control does **not** demonstrate that a no-T simplicity objective can discover the alignment. Real four-model results require the user's locally saved frozen Z data.