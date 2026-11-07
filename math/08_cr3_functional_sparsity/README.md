# 08 — CR3 functional sparsity: analytic Hessian vs function-value interactions

**Completely independent experiment** under `math/`. It does not import another `math/XX` experiment, train a neural network, modify NP checkpoints, or require the learned latent Z. It asks an **oracle question** first: when the true CR3 function is perfectly known, do particular definitions of simplicity prefer the original parameter axes?

## Main question

Let `T=(alpha, lambda, gamma, tau)`, `X=(gain_A, loss_A, prob_A, gain_B, loss_B, prob_B)`, and:

- `eta(T,X) = tau * (V_A - V_B)` with CR3's exact CPT-like value function;
- `p(T,X) = sigmoid(eta(T,X))`.

We separately analyze **logit** and **probability**. Observed binary choice is not a smooth function; its conditional mean is `p`. Choice estimation and learned-Z/NN-based variants are outside this **oracle-only** first experiment.

For a fair rotation, standardize each real parameter using an independently drawn prior reference `v_k=(T_k-mean_k)/sd_k`, then rotate `U=R_ij(theta) v`. At 0°, `U` corresponds to the four true parameter axes, up to centering/scaling. The fixed underlying `T,X` pairs and their Y values **never change with angle**. All six 2D parameter planes are tested, with angle 0 to 90 degrees by default.

## Implemented metrics

### A. Exact analytic Hessian

The included `oracle.py` reproduces CR3's two independent probability-weighting calls (one for `p`, one for `1-p`). It derives exact `TT` and `TX` Hessians of logit and transforms probability curvature by the sigmoid chain rule. It uses the original CR3 four-parameter prior.

- `H_UX = R * diag(sd_T) * H_TX * diag(sd_X)`;
- `H_UU = R * diag(sd_T) * H_TT * diag(sd_T) * R^T`;
- `M_ZX[i,j] = sqrt(mean_{T,X}(H_UX[i,j]^2))`;
- `M_ZZ[i,j] = sqrt(mean_{T,X}(H_UU[i,j]^2))`.

Thus M_ZX is a **4×6 functional-connectivity strength matrix**, with no positive/negative cancellation. Track the diagonal ZZ self-curvature and the off-diagonal ZZ interaction separately.

To prefer a small **number of active Z–X links**, rather than low L1/L2 magnitude, use a saturating soft-support penalty:

`softL0(M)=sum_ij M_ij^2/(M_ij^2+epsilon^2)`, with `epsilon=relative_threshold*sqrt(mean_ij M_ij^2)`.

The score is relative, not a calibrated hypothesis test. `hard_support` counts entries above the same epsilon. The analytic ZX Frobenius squared energy must remain constant under exact orthogonal rotation; this invariant is asserted for every angle.

### B. Derivative-free controlled function-value effects

At identical `T,X` anchors, construct a 3×3 grid in one rotated `U_i` and one original `X_j`. Re-evaluate **the original CR3 function** at these counterfactual inputs, then double-center the nine function values across rows and columns. The RMS residual is the local **2-way factorial nonadditivity gap**: an X effect depending on U gives a nonzero value. This is computed separately for all 24 Z–X pairs, logit and probability.

Likewise, evaluate `f(U_i-h),f(U_i),f(U_i+h)` with other inputs held fixed. RMS of `f(-h)-2f(0)+f(+h)` is a **self-nonlinearity function-value contrast**.

Important distinctions:

- The 3×3 local double-centering is **not a Sobol/functional-ANOVA population variance index**, even though it uses the same two-factor additivity algebra.
- These oracle counterfactual function evaluations are possible only because the generator is known. If we later have **only a fixed WY table**, generating unobserved corners requires a surrogate or a dedicated local estimator; this step does **not** already solve that inference problem.
- Finite contrasts depend on the chosen step `--step`; compare multiple steps before drawing conclusions.
- Both branches may detect high-order interaction participation, but they need not rank pair strengths identically.

### C. Scores and strict six-plane audit

Save, for each plane, angle, and Y type:

- Hessian ZX 4×6 RMS matrix; ZX relative soft-L0, hard support, total Frobenius invariant.
- Hessian ZZ 4×4 RMS matrix; self diagonal, off-diagonal interaction and soft-support diagnostics.
- Function-value 3×3 pair-gap RMS 4×6 matrix, its soft support, and own 3-point contrast 4-vector.
- Pearson correlation across the 24 corresponding ZX entries (descriptive).
- Optional combined score `ZX softL0 + --self-weight * ZZ diagonal softL0` (exploratory weight, not tuned to truth).
- Minimum angle, values at 0°/90°, endpoint-is-min flag, and flat-curve flag.

**A true-axis minimum is NOT assumed.** With a genuine nonlinear generator like CR3, some planes may favor interior rotations. Even an endpoint minimum is not a theorem of theoretical-parameter recovery. The 90° endpoint corresponds to a signed/permuted version of the original parameter axes.

## CR3-specific evaluation domain

The official CR3 **fixed 500-row X bank** is loaded from `outputs/choice/CR3/s0_design/CR3_4P_fixed500.npz`, and prior T values are sampled independently. For analytic TX and symmetric finite-value contrasts, only X rows with strictly positive amounts and probabilities away from 0 and 1 are eligible. Zero-amount trials have undefined amount derivatives for alpha in (0,1); excluding them changes the estimand. The program records eligible counts, selected original X indices, parameter and X scales, bank identity, seed and every normalization choice in `summary.json`.

It uses Cartesian prior-subject × eligible-X pairs; these are NOT raw repeated observations of a learned NP Z. Classical independent-input Sobol indices are deliberately **not** reported: six X features come from a fixed, designed, dependent bank, and rotating non-Gaussian T coordinates destroys their independence. A general dependent-input Sobol or appropriately defined functional projection would be a separate experiment.

## Run on Windows CMD (repository root)

Install existing requirements once if needed: `pip install -e ".[dev]"`.

```cmd
conda activate py11
cd /d D:\Projects\mod-rot-3
git pull --ff-only
python -m unittest discover -s math/08_cr3_functional_sparsity -p "test_*.py" -v
python math/08_cr3_functional_sparsity/run.py --planes 01 --angles 0,45,90 --subjects 12 --max-x 8
python math/08_cr3_functional_sparsity/run.py --planes all --angles 0:90:5 --subjects 48 --max-x 30 --output outputs/math/08_cr3_functional_sparsity_full
```

The default output directory is `outputs/math/08_cr3_functional_sparsity/`. The smoke test and the full run use separate output directories. Existing nonempty output directories cannot be overwritten unless `--overwrite` is explicitly given.

If the official CR3 bank is not present, generate it using the existing CR3 data pipeline first. For **code-only debugging**, you may run:

```cmd
python math/08_cr3_functional_sparsity/run.py --synthetic-bank --subjects 12 --max-x 8 --planes 01 --angles 0,45,90 --output outputs/math/08_debug_synthetic
```

This synthetic X bank is **not** the CR3 experimental design and must not be used for scientific conclusions.

## Output files

```text
outputs/math/08_cr3_functional_sparsity_full/
  summary.json
  all_curves.csv
  best_angles.csv
  curves__01_alpha_lambda.csv
  ... (six plane-specific CSV files)
  matrices__01_alpha_lambda.npz
  ... (six plane-specific NPZ files)
  plots/
    curves__01_alpha_lambda__logit.png
    curves__01_alpha_lambda__probability.png
    heatmaps__01_alpha_lambda__logit.png
    heatmaps__01_alpha_lambda__probability.png
    ... (six planes)
```

NPZ datasets: `angles`, `hessian_zz_rms` (angles,2,4,4), `hessian_zx_rms` (angles,2,4,6), `value_self_contrast_rms` (angles,2,4), and `value_pair_anova_gap_rms` (angles,2,4,6). Channel order: logit, probability.

## Validation

`test_oracle.py` checks exact analytic ZZ/ZX elements against central finite differences for both Y channels, the true logit's lambda–gain **structural zero**, appearance of new connections under rotation, orthogonal ZX energy invariance, function-value effects, and soft-L0's preference for a weak fourth link versus four substantive links.

The experiment compares mathematical targets and exact function values **before** doing any WY-only estimator or learned-decoder replication. This separation is intentional: failed oracle-axis minima must not be blamed on regression or NN approximation errors.
