# CR3 Hessian from SIX T-X-Y response channels (NO NN Hessian)

A new independent math experiment that **reuses the exact prior 10D
T-X-Y polynomial sieve and analytic CR3 comparator** from
`math/cr3_hessian_expectation/` without changing it.

## Objective

Fixed dataset: CR3 with 5,000 subjects, 500 common 6D stimulus vectors
X, and four true model parameters T=(alpha,lambda,gamma,tau).

1. Read **S5 all-trial posterior means Z** saved using all 500 observed
   (X, actual binary Y) pairs per subject for the checkpoint
   `outputs/choice/CR3/s3_train/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/`.
   Do not retrain, re-infer Z from fewer trials, or rotate Z.
2. Estimate train-subject-only ordinary least squares **affine** map
   T ~= Z B + b in row-vector notation (equivalent to T_col ~= A Z_col+b,
   A=B.T). Verify B is nonsingular, report singular values, condition
   number, per-parameter train/validation/test R2, and mapped-back Z
   differences. If singular/unstable, fail instead of silently
   pseudoinverting.
3. Apply inverse to TRUE T, obtaining Zmap=(T-b)@inv(B), and pass
   (Zmap, original X) through **the frozen NN DECODER ONLY**. Its
   two-class output is mapped to binary log-odds l1-l0, p=softmax(logits)[1],
   hard choiceprediction=(p>=0.5).
4. Assemble six numeric Y arrays, each shape (N,500):

   | Original CR3 | NN decoder evaluated at mapped Z=A^-1(T-b) |
   |---|---|
   | original_choice (observed 0/1 Bernoulli) | nn_choiceprediction (hard 0/1 prediction) |
   | original_logit (recomputed exact S1 logit) | nn_logit (l1-l0) |
   | original_probability (S1 predictive_true) | nn_probability (softmax) |

5. Fit **the same** `TensorPolynomial(T,X,Y)` six times, using
   four TRUE original T parameters as inputs, not original S5 Z and not
   decoder-mapped Z. Report entire 10x10 average Hessian (TT, TX, XX),
   conditional mean/strict mean, RMS/SD and fraction-defined, plus
   holdout response prediction metrics. All channels have exactly the
   same train/test partition and T,X basis.
6. Compare all six estimated Hessians with the exact **original CR3**
   analytic Hessian and compare NN-derived versus original-derived
   fitted Hessian, both on subject-sample expectations and optional
   prior quadrature. Save all numbers and heatmaps.

**Absolutely no decoder Jacobian, neural-network Hessian, or
automatic differentiation is used anywhere in this experiment.**
The NN is used ONLY to supply numerical Y.

## Math and interpretation

The affine map is estimated by TRAIN-only least squares. Its inverse is
an *extrapolative counterfactual decoder coordinate mapping* and need
not coincide exactly with encoder-inferred Z. Inspect R2, the matrix
condition number and Z differences before interpreting NN outputs.

The analytic oracle for original binary choice is the Hessian of its
smooth conditional choice probability. For NN-generated logit and p,
comparison to original CR3 analytic Hessian is a measure of **surrogate
mismatch plus data-only estimation error**, NOT a claim to have measured
the NN function's own true Hessian. The NN hard-class output has a
discontinuity on the decision boundary, so the ordinary classical
Hessian is zero off boundaries and undefined at boundaries. The
comparison to original smooth CR3 p Hessian is only a common external
reference, not a true Hessian recovery score for the hard classifier.

As in the original 10D analysis, continuous X derivatives cannot be
identified from a finite 500-stimulus bank without structural
assumptions. Original CR3 derivatives with respect to zero amounts may
not exist. Thus `conditional_mean` for TX/XX is not the
original unconditional mean; always inspect `fraction_defined`
and `strict_mean`.

## Run from repository root (Windows CMD / py11)

```cmd
conda activate py11
git pull origin main
python -m unittest discover -s math/cr3_hessian_six_y -p "test_*.py" -v
python math/cr3_hessian_six_y/run.py
```

If the S5 posterior mean Z file is not available, run once:

```cmd
python np/s5_extract_z.py --model CR3 --beta 0.00005 --lambda-sparse 0 --latent-dim 4 --context-min 0.25 --context-max 0.8 --repeat 1
```

Then rerun the analysis.

A faster sample-only check (skip 10^4-node prior quadrature):

```cmd
python math/cr3_hessian_six_y/run.py --sample-only --no-plots
```

Optional:

- `--no-intercept`: fit strictly T=Z B with no b.
- `--degree-t 4 --degree-x 3 --ridge-t .01 --ridge-x .01`:
  match the previously published TensorPolynomial defaults.
- `--batch-size 32`: decoder and oracle chunk sizes.
- `--quad-nodes 10`: prior Gauss-Legendre nodes per parameter
  dimension, 10^4 grid points by default.
- `--latent PATH --checkpoint PATH --data PATH`: inspect
  alternative saved checkpoints and associated S5 posterior Z (the
  checkpoint L4/CR3 and S1 SHA256 must agree).

## Outputs

Default directory:

```text
outputs/math/cr3_hessian_six_y/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/
```

- `summary.json`: affine map, invertibility, R2 per parameter
  per split, source checksums, holdout predictions for six channels,
  TT/TX/XX/full oracle discrepancies, and notes on interpretation.
- `six_response_channels.npz`: actual T and X, all 6
  (N,500) response arrays, raw S5 posterior means, counterfactual mapped
  Z, learned A,b.
- `all_matrices.npz`: oracle original CR3 matrices,
  estimated matrices for all 6 channels, including mean/RMS/SD/valid
  fraction, and pairwise NN-minus-original fitted differences.
- `sample__<channel>__estimated_mean.csv` and
  corresponding prior CSV files.
- `<channel>__TT/TX/XX__comparison.png`: original oracle
  vs estimated from T,X,Y vs difference, one per response channel.

This experiment never overwrites existing NP checkpoints, S5 posterior
Z, the earlier Hessian estimator results, or any rotation outputs.
