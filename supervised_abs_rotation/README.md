# Step 1 — Supervised full-sample rotation + diagonal scaling

An **independent experiment folder**. This code **does not train NP, re-extract Z, or modify the existing `math/09_procrustes_vs_ols` experiment**. It loads the already frozen **S5 posterior-mean** Z and true simulator parameters T for **C1, C2, C3, CR3**.

**Per latest experiment design: all subjects are used to optimize R and D.** Original S5 TRAIN/VALIDATION/TEST indices are loaded solely to validate the source data, but all subjects are pooled for every fitted quantity. **This is an in-sample supervised Oracle, not an independent evaluation nor an unsupervised axis-discovery algorithm.**

## Configuration from project overview

| Model | Z/T dimensions | NP beta | lambda_sparse | Context fraction |
|---|---:|---:|---:|---|
| C1 | 4 | 0.0005 | 0 | 0.25–0.80 |
| C2 | 4 | 0.0005 | 0 | 0.25–0.80 |
| C3 | **6** | 0.0005 | 0 | 0.25–0.80 |
| CR3 | 4 | **0.00005** | 0 | 0.25–0.80 |

The corresponding frozen latents are at:

`outputs/choice/<MODEL>/s4_evaluate/<CONDITION>/repeat01/s4_evaluate__latent.npz`

S5 stores `z`, `theta_true`, the original split IDs and S1/S3 provenance hashes. This script verifies all of them and requires `context_mode=all_observed_trials`. A missing S5 file is an error, not an invitation to retrain silently.

## Definition, optimization and identification

Let Z and T be N×d matrices (rows = subjects). Define statistics using **all N subjects**:

\[
Z_c=Z-\mathbf1\bar Z^\top,\qquad
T_{\rm std}=(T-\mathbf1\bar T^\top)\,\mathrm{diag}(\sigma_T)^{-1}.
\]

Z is **centered, not standardized or whitened**: only genuinely orthogonal rotations of the original Z coordinate system are admissible. For column-vector notation:

\[
R^\top R=I,\quad D=\mathrm{diag}(s_1,\ldots,s_d),\quad
U_s=R(Z_s-\bar Z),\quad \hat T_{{\rm std},s}=D U_s.
\]

**Default objective — literal absolute-value recovery** (`--objective abs_mse`):

\[
\min_{R^\top R=I,\,D}\;\frac1{Nd}
\sum_{s=1}^N\sum_{j=1}^d
\left(\left|[DR(Z_s-\bar Z)]_j\right|-|T_{{\rm std},sj}|\right)^2.
\]

For fixed R let v=Zc Rᵀ. The optimal scale **magnitude** for each output axis has the closed form

\[
|s_j|=\frac{\sum_s |v_{sj}|\,|T_{{\rm std},sj}|}
              {\sum_s v_{sj}^{\,2}}.
\]

This allows a search **over R alone**: a multistart manifold parameterization `R=R0 exp(K)`, where `K` is skew-symmetric (6 angles for 4D; 15 angles for 6D). Starting points include an orthogonal Procrustes solution, polar(OLS), identity and seeded random orthogonal matrices; each is optimized via SciPy L-BFGS-B. Best means **lowest observed full-sample loss among starts**, not a mathematically certified global minimum.

**Crucial sign ambiguity:** the magnitude objective cannot identify the signs of individual D values (or flipped rows of R). After fitting, sign is chosen using `sign(sum_s v_sj Tstd_sj)` as a diagnostic calibration; this **does not change the magnitude objective**. Reported signed prediction correlations depend on that convention. A high `|prediction|`–`|Tstd|` correlation is not sufficient to conclude that the correct *signed* parameter was recovered.

**Alternative objective — signed parameter recovery** (`--objective signed_mse`):

\[
\min_{R^\top R=I,\,D}\;\frac1{Nd}\|Z_c R^\top D-T_{\rm std}\|_F^2,\qquad
s_j=\frac{\sum_s v_{sj}T_{{\rm std},sj}}{\sum_s v_{sj}^2}.
\]

This is a separate comparison; it does **not** minimize the absolute-value loss. Unlike whitening or unconstrained OLS, the fitted rows of `DR` remain orthogonal (up to scaling). For centered Z and unit-variance centered Tstd, minimizing this profiled signed MSE is equivalent to maximizing the **sum of squared signed Pearson correlations** between matching U/T axes (and hence their squared absolute correlations); it is **not** the same objective as fitting the absolute magnitudes.

## Windows CMD, from repository root

```cmd
conda activate py11
cd /d D:\Projects\mod-rot-3
git pull --ff-only origin main
python -m unittest discover -s supervised_abs_rotation/tests -p "test_*.py" -v
python supervised_abs_rotation/run.py --model C1
python supervised_abs_rotation/run.py --model C2
python supervised_abs_rotation/run.py --model C3
python supervised_abs_rotation/run.py --model CR3
```

For signed-value recovery, repeat a single model with e.g.:

```cmd
python supervised_abs_rotation/run.py --model CR3 --objective signed_mse
```

Each invocation handles **only one model**, not a batch of other rotation methods. The defaults automatically select the intended beta and dimension for the named model. Optional flags: `--beta`, `--latent-dim`, `--lambda-sparse`, `--context-min`, `--context-max`, `--repeat`, `--restarts`, `--maxiter`, `--seed`, `--no-plots`, `--output`, `--overwrite`.

If an S5 file is missing, extract it **using the matching already-trained S3 checkpoint** (no retraining needed), for example:

```cmd
python np/s5_extract_z.py --model C3 --beta 0.0005 --latent-dim 6
python np/s5_extract_z.py --model CR3 --beta 0.00005 --latent-dim 4
```

## Additional diagnostic: whiten U after fitting R

After optimizing R and D, define **unscaled** rotated U from the already fitted rotation: `U = (Z - mean_all(Z)) @ R.T`. Using all subjects' **U only**, calculate `Cov(U)` with `ddof=0` and apply symmetric **ZCA whitening**:

```text
W = eigvecs @ diag(1 / sqrt(eigenvalues)) @ eigvecs.T
U_whitened = (U - mean_all(U)) @ W
```

This whitening is **post hoc only**: no true T enters W; it does not refit R, re-optimize D, or change existing predictions. The additional signed-Pearson heatmap `corr__U_whitened_Tstd.png` (with numeric `corr__U_whitened_Tstd.csv`) displays **rows = whitened U axes, columns = true standardized parameters**, with the same full-sample data. It lets you compare `corr__U_Tstd.png` (before whitening) against `corr__U_whitened_Tstd.png` (after whitening).

Other outputs: `U_zca_whitening_matrix.csv`, `U_whitened_covariance.csv`, `corr__U_whitened_U_whitened.csv`, and arrays `U_whitened`, `U_zca_whitening_matrix`, eigenvalues and covariance matrices in `oracle_full_sample.npz`. `summary.json` stores the whitening rank and covariance diagnostics.

**Important interpretation:** whitening is generally a nonorthogonal mixing of the rotated axes, so `U_whitened` does **not** satisfy the original pure rotation plus post-axis scaling constraint in general. Consequently, an improvement in U_white–T correlations is evidence about the effect of decorrelation/whitening, **not** evidence that the originally fitted orthogonal R became better. If the empirical U covariance is rank deficient under the stated eigenvalue threshold, the script uses truncated pseudoinverse whitening; the resulting covariance is a projection rather than the identity and this is explicitly flagged in the JSON.

When re-running a previously completed model/objective, append `--overwrite` so the new whitening outputs can be added:

```cmd
python supervised_abs_rotation/run.py --model C1 --overwrite
```

## Files written per model and objective

`outputs/supervised_abs_rotation/<MODEL>/<CONDITION>/repeat01/<OBJECTIVE>/`

- `summary.json`: full R, D, DR, objective, each start's optimizer trace, sample count, normalizers, parameter-wise and aggregate diagnostics, entire correlation matrices, provenance SHA256.
- `R.csv`, `D.csv`, `DR.csv`, `A_rawZ_to_Tstd.csv`: fitted matrices; `R.png`, `D.png`, `DR.png`: annotated coefficient heatmaps. `b_rawZ_to_Tstd.csv` saves the intercept. The full raw-Z affine map is `prediction = (Z @ A.T) + b`, where `A=DR` and `b=-DR @ mean_all(Z)`.
- `per_parameter.csv`: each true-parameter axis' fitted D, signed and magnitude Pearson r, signed and magnitude R², MSE and worst incorrect-parameter signed correlation.
- `corr__*.csv`: all dimensions × all dimensions Pearson matrices for raw Z–Z, Tstd–Tstd, raw Z–Tstd, unscaled U–U, U–Tstd, calibrated prediction–Tstd, absolute prediction–absolute Tstd, and calibrated prediction internal correlations (signed/magnitude).
- Matching `corr__*.png` annotated correlation heatmaps.
- `oracle_full_sample.npz`: all-subject arrays `Z_centered`, `T_std`, `U_unscaled`, `prediction_Tstd`, `abs_prediction`, `abs_Tstd`, R, D, DR, original T and normalization statistics, for subsequent experiments.

A correlation matrix for magnitudes is `corr(|prediction_i|,|Tstd_j|)`, **not** `|corr(prediction_i,Tstd_j)|`; both versions are saved separately to avoid mixing these distinct questions.

Outputs cannot be silently replaced; use `--overwrite` to redo an existing configuration deliberately. No result values are asserted in this README before execution on the actual locally frozen Z data.
