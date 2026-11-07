# 05 · CR3 ZX Hessian optimization and algorithm expansion

This independent experiment builds directly on **04** (original encoder Z,
fixed X, six numerically generated Y, and analytic CR3 Hessian transformed
into the identical (Z,X) coordinates). It never retrains or differentiates
a neural network. The previous four experiments are unchanged.

## What is changed?

### Inputs are held EXACTLY fixed

Read these files from the completed step-04 run:

- `all_Y_and_mapping.npz`: all 5,000 original encoder Z,
  all 500 fixed X, six Y (`oracle_choice`, `oracle_logit`,
  `oracle_probability`, `nn_choice`, `nn_logit`, `nn_probability`),
  and affine A,b.
- `all_matrices.npz`: the **actual analytic** ZZ, ZX, XX reference
  and saved old polynomial estimates, including fraction-defined masks.
- `summary.json`: original data/model provenance.

All 6 Y and the analytic reference remain **identical to step 04**.
This is an algorithm and numerical conditioning experiment, not a new
simulation run or a different trained network.

### Extended full 10×10 (Z,X) estimators

The original step-04 method is
`f(Z,X)=phi_Z(Z)^T B phi_X(X)` with a degree-4 monomial
basis in Z, degree-3 in X, and two-sided ridge
`lambda_Z=lambda_X=0.01`.

This experiment adds:

1. **Independent Z/X ridge sweeps:** default grid
   `lambda_Z,lambda_X in {1e-2,1e-4,1e-6}` with independent
   choices for each group. The full grid adds `1e-8`.
2. **Different degrees:** quick uses Z4 and X3/X4; full uses
   Z3/Z4/Z5 and X2/X3/X4.
3. **Two polynomial bases:** ordinary monomials (faithful to 04)
   and Chebyshev polynomials, differentiated analytically in the original
   Z/X physical coordinates.
4. **One SVD per basis/degree design:** all six Y are fitted together
   with cached SVDs and second-derivative moment vectors, so ridge sweeps
   do not repeat expensive decompositions.
5. **Strict baseline check:** `--verify-step04-baseline` recomputes
   old monomial Z4/X3 and ridge .01/.01 from the **exact original 04
   train subjects** and aborts if any of the six saved 10×10 mean
   Hessians differ by more than 1e-8.

Chebyshev and monomial polynomial bases span the same total-degree
polynomial function space at fixed degrees, but ridge regularization is
basis-dependent. This provides a direct controlled test of basis and
regularization effects.

### Two additional ZZ-only estimators

- **Nyström Gaussian kernel ridge regression (KRR)** with subject-wise
  training/validation and analytic kernel second derivatives.
- **Penalized cubic tensor B-splines** with knots in the **actual
  Z training ranges**, not the original T prior bounds. Analytic
  spline second derivatives and a second-difference roughness penalty.

The last two fit the mean of the 500 Y values per subject:
`g(Z)=mean_{500 fixed X} f(Z,X)` and therefore estimate
**ONLY 4×4 ZZ**, never ZX or XX. Their ZZ mean Hessians are compared
against the step-04 analytical ZZ mean reference.

No data-only estimator can be called an oracle Hessian; they are all
independent regressions on numeric (Z,X,Y).

## Model selection without answer leakage

The original step-04 polynomial used 80% of subjects to train and 20%
to test. The 80% training IDs are preserved *exactly*, and the untouched
20% is split into 10% validation and 10% final test (seeded).

**Models are selected only by held-out Y RMSE on validation subjects.**
Neither the true CR3 analytic Hessian nor a Hessian pattern metric is
used to choose any degree, ridge, basis, kernel bandwidth or smoothing
penalty. All oracle comparison scores are POST-HOC diagnostics.

The NN Y has the same all-context latent-inference caveat as step 04:
the original Z was inferred from each person's observed choices, and
so this is not held-out-trial NP prediction.

## Evaluation: numerical accuracy AND pattern

Every full-matrix candidate is scored on:

- `ZZ`, `ZZ_diag`, `ZZ_offdiag`,
  `ZX`, `XX`, `XX_diag`,
  `XX_offdiag`, and `FULL`.
- Frobenius cosine, relative Frobenius error, MAE/RMSE, norm ratio,
  Pearson correlation, Spearman correlation of absolute-element ranks,
  sign accuracy of substantively large true elements, and
  contribution of the largest one/two error elements.
- Off-diagonal elements in symmetric blocks are counted **once**,
  not twice; a small cosine alone does not mean no substructure.
- The fraction-defined mask uses the original analytic CR3 rules for
  fractional powers at zero monetary amounts.

Patterns are evaluated on the original average 10×10 Hessian, not
a neural-network Hessian.

### Important nuance

For NN outputs, difference from original analytic CR3 can represent
**both** a change of generating function AND regression error.
The NN hard choice is a threshold, not a smooth choice probability;
its comparison to the smooth CR3 probability Hessian remains an
**external comparator**, not a true NN hard-class Hessian.

Choice Y from the two sources also differs (analytic Bernoulli vs NN hard
argmax). Do not equate their noise levels.

## Commands (Windows CMD, run from repo root)

Install dependencies already in the existing `py11` environment
(NumPy, SciPy, PyTorch for earlier stage 04; this step uses NumPy/SciPy
and Matplotlib only).

```cmd
conda activate py11
git pull origin main
python -m unittest discover -s math/05_cr3_zx_hessian_optimization -p "test_*.py" -v
```

**Fast first pass: reproduce original method exactly, only adjust ridge
and increase X degree:**

```cmd
python math/05_cr3_zx_hessian_optimization/run.py --methods tensor --preset quick --verify-step04-baseline --no-plots
```

**All four algorithm families (quick preset):**

```cmd
python math/05_cr3_zx_hessian_optimization/run.py --methods tensor krr spline --preset quick
```

**Broader polynomial degree/ridge sweep:**

```cmd
python math/05_cr3_zx_hessian_optimization/run.py --methods tensor --preset full --no-plots
```

**Targeted low-regularization test:**

```cmd
python math/05_cr3_zx_hessian_optimization/run.py --methods tensor --basis monomial --degree-z 4 --degree-x 3 4 --ridge-z 0.000001 --ridge-x 0.000001 --output outputs\math\05_cr3_zx_hessian_optimization\focused_low_ridge
```

**ZZ-only nonparametric comparison:**

```cmd
python math/05_cr3_zx_hessian_optimization/run.py --methods krr spline --preset quick --no-plots
```

When the step-04 output was saved elsewhere, pass its repeat directory:

```cmd
python math/05_cr3_zx_hessian_optimization/run.py --source-dir "D:\path\to\step04\repeat01" --methods tensor
```

The default source location remains the **old, unnumbered output path**
so existing results are found automatically:

```text
outputs/math/cr3_zx_hessian_threeway/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/
```

## Outputs

Default (quick):

```text
outputs/math/05_cr3_zx_hessian_optimization/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/quick/
```

- `tensor_grid_pattern_metrics.csv`: **all** tensor hyperparameter
  candidates × six Y × eight Hessian blocks, including numerical and
  structural metrics, for retrospective diagnosis.
- `summary.json`: source SHA256 hashes, train/val/test sizes,
  selected per-channel hyperparameters, validation and test Y RMSE,
  selected block-wise Hessian recovery scores, KRR/spline ZZ results.
- `selected_hessians.npz`: chosen full 10×10 Hessians,
  polynomial coefficients, chosen 4×4 ZZ KRR/spline Hessians,
  original analytical and old baseline Hessians, and split IDs.
- `{basis}__{channel}__{ZZ|ZX|XX}.png`: side-by-side comparison
  of analytic CR3, step-04 baseline, and new selected estimate
  (unless `--no-plots`).

### Research questions answered by this design

1. Does relaxing excessive ridge on the same Z,X,Y recover analytic
   probability ZZ without changing the data?
2. Does increasing X degree fix the pA/pB diagonal artefact and the
   spurious pA-pB cross derivative in XX?
3. At the same degrees, do monomial vs Chebyshev bases change
   inferred curvature because of the regularization geometry?
4. If a flexible ZZ-specific estimator (B-spline/KRR) fails on NN Y
   but succeeds on oracle Y, is the mismatch persistent across
   independent estimator families?
5. Do high cosine and sign/rank agreement survive validation-based
   hyperparameter choice without consulting the Hessian truth?

No mechanism is declared proven merely from a single hyperparameter
configuration or cosine similarity.
