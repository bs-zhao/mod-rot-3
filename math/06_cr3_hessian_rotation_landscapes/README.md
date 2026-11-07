# 06 · Hessian rotation landscapes around the recovered A-star

## Research question

Does the recovered linear map \(A^*\), which maps learned posterior means
\(Z\) to the four true CR3 parameters \(T=(\alpha,\lambda,\gamma,\tau)\),
have a distinctive Hessian pattern/simplicity signature? Is that
signature preserved when the Hessian comes from **numeric Y only** rather
than from the original CR3 generating formula?

**This is a NEW independent experiment folder.** It does not modify
experiments 01–05, the frozen NP, any rotation package, or any
previously saved Y.

## Geometry and exact 3×3 methods

First re-estimate using the **same original step-04 map-fit IDs**:

\[
T_n \approx A^*Z_n+b,\quad A^*\in\mathbb R^{4\times4}.
\]

The implementation verifies the refitted A*,b match the step-04
saved mapping. Intercept b is never silently discarded.

For each of the **6 pairs** of four parameter dimensions:

- \(\alpha\)--\(\lambda\) [01]
- \(\alpha\)--\(\gamma\) [02]
- \(\alpha\)--\(\tau\) [03]
- \(\lambda\)--\(\gamma\) [12]
- \(\lambda\)--\(\tau\) [13]
- \(\gamma\)--\(\tau\) [23]

sweep angles from **0° to 90° inclusive**, 5° apart by default
(19 positions per plane). For each pair build a proper 4×4 rotation
\(R_{ij}(\theta)\) and compute:

\[
A_{ij}(\theta)=R_{ij}(\theta)A^*,\quad
U_{ij}(\theta)=A_{ij}(\theta)Z+b.
\]

The intercept b is kept fixed. Equivalently,
\(U=b+R_{ij}(\theta)(A^*Z)\). The 0° candidate is \(U=A^*Z+b\).

**Crucial:** We do **not** regenerate data when rotating A.
The original **same 5000 Z subjects × same 500 X stimuli × six saved Y**
are held fixed at every angle. The underlying CR3 function is also held
fixed. The data-only regression is refit in the new U coordinates each
time.

All nine methods expose one consistent programming interface:

\`run_nine(A,Z,X,Y,oracle_z,estimator,b)\`

and return nine full 10×10 mean Hessians for that A, indexed by
(source, Y type). The extra arguments carry the original saved analytic
reference and the frozen regression configuration.

| Hessian source | Choice | Logit | Probability |
| --- | --- | --- | --- |
| **Exact analytic CR3** | analytic choice *mean* | analytic logit | analytic probability |
| **Fit to original CR3-generated Y** | original Bernoulli choices | original logit | original p |
| **Fit to frozen NN-generated Y** | NN hard choices | NN logit | NN p |

For Choice, an analytic Hessian of **observed 0/1 samples** does NOT
exist. Therefore analytic "Choice" means

\[
H\{\mathbb E[\mathrm{Choice}\mid Z,X]\}=H\{p(Z,X)\},
\]

and is necessarily identical to the analytic Probability Hessian.
The two data-only Choice estimates are computed from their respective
saved numeric binary Y; they are not identical to Probability estimates.

The NN is never retrained or differentiated. The NN-generated Y
are the **identical saved step-04 outputs** at every angle.

### Exact analytic Hessian at every rotated A

Step 04 already saved original analytic 10×10 mean Hessian \(H_{ZX}\)
at the original Z,X. Under \(U=A Z+b\), the exact change of
coordinates is

\[
J_A=\operatorname{diag}(A^{-1},I_6),\quad
H_{UX}^{(\mathrm{analytic})}=J_A^\top H_{ZX}^{(\mathrm{analytic})}J_A.
\]

For \(A=R A^*\) this also gives, letting \(H_*\) denote the analytic
Hessian in the baseline \(A^*Z+b\) coordinates:

\[
H_{UU}(\theta)=R H_{TT}(0)R^\top,\qquad
H_{UX}(\theta)=R H_{TX}(0),\qquad
H_{XX}(\theta)=H_{XX}(0).
\]

**Thus the analytic Frobenius norms of full H, ZZ/UU, ZX/UX and XX
must be rotation invariant.** The program asserts these invariants
at EVERY tested angle.

But coordinate-specific off-diagonal energy, L1, element signs,
and sparsity can change with rotation. These are the actual
orientation-sensitive experimental metrics.

The stored analytic means are *conditional means* when some CR3
amount derivatives are undefined at X=0. The exact same fraction-defined
mask is retained; no undefined derivatives are invented.

## Estimating six Hessians from numeric Y

The numeric methods use the optimized step-05 **factorized full
10D tensor-polynomial estimator**:

\[
\hat f(U,X)=\phi_U(U)^\top B\phi_X(X).
\]

Default: monomial total degree 4 on U, degree 4 on X, and independent
two-sided ridge \(\lambda_U=\lambda_X=10^{-6}\); optional
Chebyshev basis, degrees and regularizations via CLI.

The X basis and its SVD as well as the large \`Y @ U_X\` projection
are cached **once across all 114 angle/plane configurations**.
The U basis, U SVD, coefficient matrices, and Hessian moments are
recomputed for each candidate A. All six Y are fitted simultaneously.

This is critical: using a rotated algebraic Hessian from the original
fit *instead* of refitting would automatically force coordinate
equivariance and hide an estimator's rotation-dependent bias.

No Hessian ground truth is used to select the polynomial degrees or
ridge. Defaults were suggested by step 05 on Y prediction, NOT tuned
to this experiment's rotation curves. A limited held-out Y RMSE
is recorded at every angle as a function-fit diagnostic.

## Hessian metrics and output

For each angle and each of 9 methods, save a full 10×10 Hessian.

Score every **ZZ, ZX, XX, FULL, ZZ diagonal, ZZ non-diagonal,
XX diagonal, XX non-diagonal** block.

1. **Intrinsic orientation / simplicity**
   - Frobenius norm, absolute L1, normalized L1
   - Largest |H| / Frobenius
   - Off-diagonal Frobenius-energy ratio for symmetric blocks
   - Row concentration and row entropy for the 4×6 ZX block
   - Effective rank (rotation-invariant sanity control)
2. **Agreement with analytic CR3**, at the **same angle and same
   U-coordinate system**:
   - Cosine, Pearson, Spearman rank of absolute element strengths
   - Relative Frobenius error, MAE
   - Important-element sign agreement and error concentration
3. **Estimator equivariance / refitting bias**:
   - Transport each angle's fitted H back to the baseline
     A* coordinate system, and compare with its **own 0° fitted H**.
   - An exact analytic change of coordinates has zero such drift.
     Drift for a fitted Y measures rotation sensitivity of the
     chosen numerical estimator *on the same data*.

Because all these measures use **mean Hessians**, this experiment
does not prove that pointwise Hessians of the frozen NN or of each
individual subject have been recovered.

### Files

\`\`\`text
outputs/math/06_cr3_hessian_rotation_landscapes/
  beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/
    monomial_u4_x4_ridgeu1e-06_ridgex1e-06/
      summary.json
      all_curves.csv
      best_angles.csv
      heldout_y_rmse.csv
      curves__01_alpha_lambda.csv
      curves__02_alpha_gamma.csv
      ... 4 other plane CSVs
      hessians__01_alpha_lambda.npz
      ... 5 other NPZs, each (n_angles, 9, 10, 10)
      dashboard__01_alpha_lambda__normalized_l1.png
      dashboard__01_alpha_lambda__cosine_to_analytic.png
      ... other angle/metric dashboards
      PLOT_INDEX.md
\`\`\`

Dashboard layout: **3 rows = Choice/Logit/Probability,
3 columns = ZZ/ZX/XX**. Each panel plots THREE lines
(analytic, fit-to-analytic-Y, fit-to-NN-Y) vs degrees.
Each dashboard therefore shows all NINE requested methods.

\`best_angles.csv\` summarizes where each metric is maximal/minimal
within the scanned interval, and whether 0° attains the extremum.
This is a descriptive search: there is no theorem that the true
original CR3 parameter axes minimize off-diagonal curvature. In fact,
CR3 parameters interact nonlinearly, so an off-axis angle may give a
simpler mean Hessian than A* without invalidating the recovery.

## New: rich sparsity losses and three-way error diagnosis

The ordinary matrix \`L1\`, row entropy and row concentration indicators
are NOT enough to detect tiny "extra" X effects in a row with one,
two, or several dominant X effects. We therefore export **multiscale
orientation-dependent simplicity losses** with all of these properties:

- Treat every XZ row separately as a 6-component vector of the
  X effects for ONE latent parameter. Also export mean and worst-row
  (max) loss; an average alone may hide a badly mixed dimension.
- Test **multiple fixed top-k tails** \`k=1..4\` (L1, L2, squared L2)
  and next-largest-effect residuals. Do not assume k=1.
- **Adaptive / soft support**: capped L1, capped squared L2, smooth
  soft-L0, log-sum, fractional penalty, and
  \`min_k[lambda*k + sum_{j>k}s_j^2]\`. Four relative scales:
  0.0005, 0.001, 0.005, 0.02 of each row's largest absolute effect.
- Hoyer loss, \`L_p\` quasi-norms (\`p=0.25,0.5,0.75\`), Shannon/Rényi
  entropies and Gini impurity.
- The original unscaled physical derivatives AND X-column-standardized
  derivatives, because gain/loss and probability columns of X have
  different units. This multiplies \`H_ZX[:,j]\` by \`sd(X_j)\`,
  separately for the original X bank. The same units apply at every
  angle and for all three Hessian sources.
- Diagonal and off-diagonal blocks have their own sparsity losses, with
  duplicate elements of symmetric Hessians counted once.

All new keys ending in \`_loss\` mean **MINIMIZE**. Hard support sizes
are \`_diagnostic\` values and are NOT optimized directly.

These losses are descriptive until validated. High curve correlation is
not sufficient: **all three** sources (exact analytic, numeric analytic-Y
regression, numeric NN-Y regression) must have highly correlated
angle curves and a true 0°/90° MINIMUM. Flat curves are rejected.
The automatic screening files report thresholds, qualifying pairs and
six-plane consistency.

### Distinguish three possible discrepancies

You highlighted an essential diagnostic: if the true analytic Hessian
has a clean endpoint minimum, but the fitted Hessian from **analytic
generated Y** does not, this is evidence of the data-only
Hessian-estimation/inference pipeline failing its oracle-Y control.
It is premature to blame the NN.

For EACH angle, Y and block we save the exact vector identity:

\[
\underbrace{\widehat H_{NN-Y}-H_{\mathrm{analytic}}}_{\mathrm{total}}
=
\underbrace{\widehat H_{\mathrm{formula-Y}}-H_{\mathrm{analytic}}}_{\mathrm{estimator/sampling}}
+
\underbrace{\widehat H_{NN-Y}-\widehat H_{\mathrm{formula-Y}}}_{\mathrm{extra\ fitted-Y\ gap}}.
\]

The filename is \`three_way_hessian_gaps.csv\`. It includes the
relative and absolute norms of the three residuals, cosine alignment,
and the squared-error interaction/cancellation term.

A separate scalar-curve screen (\`sparsity_screen_all.csv\`) reports
ALL THREE pairwise correlations:
\`analytic_vs_formula\`, \`analytic_vs_NN\`, and \`formula_vs_NN\`,
plus minima and a diagnostic label. The label is **not** a unique
causal explanation of the network's own unknown Hessian.

For Choice, formula-Y is sampled Bernoulli while NN-Y is hard
threshold choice, so differences also involve sampling/response-type
mismatch. Interpret these separately from Logit/Probability.

### Rescore saved step-06 matrices immediately (no refitting)

If step 06 was already completed, DO NOT rerun 1,026 regressions
to obtain the new metrics. Use the saved \`hessians__*.npz\` files:

\`\`\`cmd
python math/06_cr3_hessian_rotation_landscapes/rescore_existing.py --input "outputs\math\06_cr3_hessian_rotation_landscapes\beta5em05_lambda0_ctx0p25to0p8_L4\repeat01\monomial_u4_x4_ridgeu1e-06_ridgex1e-06"
\`\`\`

Alternatively, from the previous ZIP:

\`\`\`cmd
python math/06_cr3_hessian_rotation_landscapes/rescore_existing.py --input "D:\Downloads\monomial_u4_x4_ridgeu1e-06_ridgex1e-06.zip"
\`\`\`

Specify \`--step04-source "outputs\math\cr3_zx_hessian_threeway\beta5em05_lambda0_ctx0p25to0p8_L4\repeat01"\`
if the original step-04 directory is not auto-discovered.
Without it, unit-standardized X losses are omitted but RAW
Hessian sparsity scores and analytic/formula/NN contrasts still work.

The rescore creates a **separate** \`rich_sparsity_rescored\` directory,
never overwriting the original step-06 output. It writes:

- \`all_curves.csv\` with every new metric for all 9 Hessians/angle
- \`sparsity_screen_all.csv\`, \`sparsity_screen_groups.csv\`,
  \`sparsity_screen_ranking.csv\` for strict three-way endpoint minima
- \`three_way_hessian_gaps.csv\` for estimator-versus-NN attribution
- \`metric_catalog.csv\` and \`best_angles.csv\`
- \`xz_sparsity__...png\` figures per plane/metric (unless \`--no-plots\`)

A normal fresh \`run.py\` now produces these new metrics automatically,
and includes X-standardization by reading the original fixed X bank.

## Windows CMD commands

First pull and verify isolated unit tests:

\`\`\`cmd
conda activate py11
git pull origin main
python -m unittest discover -s math/06_cr3_hessian_rotation_landscapes -p "test_*.py" -v
\`\`\`

**Recommended cheap smoke experiment: one plane, 0–90° in 10° steps:**

\`\`\`cmd
python math/06_cr3_hessian_rotation_landscapes/run.py --planes 01 --angles 0:90:10
\`\`\`

**Full requested experiment: all 6 planes, 0–90° in 5° steps:**

\`\`\`cmd
python math/06_cr3_hessian_rotation_landscapes/run.py
\`\`\`

This computes 6 × 19 × 9 = **1026 mean Hessians** (342 analytic
reference Hessians and 684 data-only Hessian estimates).

**Higher angular resolution on one plane:**

\`\`\`cmd
python math/06_cr3_hessian_rotation_landscapes/run.py --planes 02 --angles 0:90:1
\`\`\`

**Chebyshev alternative:**

\`\`\`cmd
python math/06_cr3_hessian_rotation_landscapes/run.py --basis chebyshev --angles 0:90:10
\`\`\`

**Already have step-04 ZIP but not the extracted output directory?**
The script accepts an original step-04 ZIP directly, including
\`repeat01(2).zip\` or \`beta5em05_lambda0_ctx0p25to0p8_L4.zip\`:

\`\`\`cmd
python math/06_cr3_hessian_rotation_landscapes/run.py --source "D:\path\to\repeat01(2).zip" --planes 01 --angles 0:90:10
\`\`\`

**Other useful flags:** \`--ridge-u\`, \`--ridge-x\`,
\`--degree-u\`, \`--degree-x\`, \`--basis\`,
\`--validation-limit 0\` (skip Y RMSE for faster sweeps),
\`--no-plots\`, \`--no-save-matrices\`,
\`--output path\`, and \`--plot-metrics\`.

## Critical interpretation

An arbitrary invertible transform of Z cannot be identified as a
unique "true" rotation by prediction alone: under consistent
coordinate substitution, f and its predictions are unchanged.

Some analytic structure metrics (Hessian L1, diagonal dominance,
off-diagonal ratios) genuinely vary with R; others (Frobenius norm,
singular values/effective rank of appropriate blocks) are invariant
under the orthogonal transformation. A metric with a 0° minimum
may support alignment with the original parameters *on these data*,
but does not by itself establish causal interpretability or global
parameter recovery.

This experiment deliberately separates (i) coordinate-dependent
simplicity, (ii) Hessian recovery error, and (iii) estimator
rotation-equivariance so that each curve has a valid interpretation.
