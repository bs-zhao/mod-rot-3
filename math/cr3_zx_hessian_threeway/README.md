# CR3: THREE-WAY Hessian comparison in original encoder Z and fixed X

This independent \`math/\` experiment implements **exactly three** parallel paths
for estimating the **10×10 Hessian in the SAME (Z,X) coordinate system**.
It does not train a network or compute the network's Hessian.

## Inputs

- 5,000 × 500 CR3 S1 dataset (true T and 500 shared, fixed X stimuli).
- S3 frozen checkpoint at
  \`outputs/choice/CR3/s3_train/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/\`.
- S5 frozen original posterior-mean Z, inferred from **all 500 actual
  observed (X, binary choice) trials per subject**, at
  \`outputs/choice/CR3/s4_evaluate/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/s4_evaluate__latent.npz\`.

We use the **original S5 Z**, not inverse-mapped Z created from true T.
The script verifies SHA256, split IDs, and full-context provenance.

## Step 1: linear map Z -> T

Fit least-squares affine regression (column-vector notation):

\[
  \widehat T_i = A Z_i + b .
\]

By default fit A,b on **all** subjects, as in the experiment specification.
Use \`--mapping-fit train\` to fit only on original S3 train subjects.
An intercept is included by default because T need not be centered and Z may
be centered. Use \`--no-intercept\` for strictly \`AZ=T\` if desired.

Report A, b, the singular values/condition number, and per-parameter R²/RMSE
for all/train/validation/test. Never silently clip \`T_hat\` to a prior bound:
doing so would change the function and invalidate the simple Hessian chain rule.

## Step 2: generate six Y arrays, WITHOUT neural differentiation

At exactly the same observed original \`(Z_i,X_t)\` pairs:

| Response | Analytic CR3 branch | Frozen network branch |
|---|---|---|
| choice (0/1) | sample Bernoulli from \`p_CR3(AZ+b,X)\` (fixed seed) | network predicted hard class (\`p_NN>=0.5\`) |
| scalar logit | exact \`eta_CR3(AZ+b,X)\` | neural \`class1_logit - class0_logit\` |
| probability | exact \`sigmoid(eta_CR3)\` | neural \`softmax(class_logits)[...,1]\` |

Both branches have the same Z and X, and both yield Y matrices with shape
\`(5000,500)\`. S1's original binary choices and true p correspond to *true*
T and **must not be substituted** for new Y at \`T_hat=AZ+b\`.

Optional \`--nn-choice-mode sampled\` uses independent Bernoulli draws from NN
probabilities; these share the same uniform random variates as the oracle
draws, which makes a fairer stochastic-choice comparison. The default hard
choice is compatible with the previous \`choiceprediction\` definition.

## Step 3: data-driven matrix Hessians (two paths)

For the three analytic-Y channels and three NN-Y channels, fit the
**identical** full \`TensorPolynomial(Z,X,Y)\` estimator, using the same
4D Z features, 6D X features, regularization and subject-level holdout
splits. This is the exact polynomial-sieve matrix method of the original
\`math/cr3_hessian_expectation\` experiment, now applied to original Z,X.

Estimates include mean 10×10 Hessian, local-curvature RMS and standard
deviation moments, and a defined-domain mask for zero-amount X derivatives.

The source code is copied into this experiment's own \`data_only.py\` to keep
math folders independent.

## Step 4: analytic answer (third path)

For EACH real observed Z, evaluate the original CR3 **analytic** Hessian
at \`T_hat=A Z+b\`, together with the fixed stimuli X.

**Coordinate conversion is essential.** The simulator computes Hessian
in (T,X). Our two polynomial fits compute Hessian in (Z,X). Since
\`T=A Z+b\`, the correct reference is:

\[
J=\begin{pmatrix}A&0\\0&I_6\end{pmatrix}, \qquad
H_{ZX}^{\mathrm{analytic}}=J^\top H_{TX}^{\mathrm{analytic}}J.
\]

Specifically:

- \`H_ZZ = A.T @ H_TT @ A\`,
- \`H_ZX = A.T @ H_TX\`,
- \`H_XX = H_XX\`.

The implementation transforms **each local analytic Hessian before**
computing the averaged mean, RMS, SD or undefined-domain fraction.
This is necessary: RMS/SD do not transform merely by applying A to
their already-aggregated matrices.

The implementation transforms blocks individually so NaNs from zero
amount derivatives do not pollute a valid ZZ block.

**Absolutely NO decoder gradient, Hessian, autograd, torch.func, or
derivatives of any NN module are computed.** The network is *only* a
frozen map from Z/X to numerical Y.

## Run on Windows CMD (py11)

From the repository root:

\`\`\`cmd
conda activate py11
git pull origin main
python -m unittest discover -s math/cr3_zx_hessian_threeway -p "test_*.py" -v
python math/cr3_zx_hessian_threeway/run.py
\`\`\`

If the all-trial S5 latent file is absent:

\`\`\`cmd
python np/s5_extract_z.py --model CR3 --beta 0.00005 --lambda-sparse 0 --latent-dim 4 --context-min 0.25 --context-max 0.8 --repeat 1
\`\`\`

Rerun this experiment after extracting Z. The checkpoint, Z and dataset
must match each other by hashes; this script will not overwrite them.

Other useful runs:

\`\`\`cmd
python math/cr3_zx_hessian_threeway/run.py --mapping-fit train
python math/cr3_zx_hessian_threeway/run.py --nn-choice-mode sampled
python math/cr3_zx_hessian_threeway/run.py --no-intercept
python math/cr3_zx_hessian_threeway/run.py --no-plots --batch-size 8
\`\`\`

Outputs go to

\`outputs/math/cr3_zx_hessian_threeway/beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/\`

- \`summary.json\`: affine map A,b, singular values, fit metrics and
  six Y response fit statistics; mean Hessian comparisons for ZZ/ZX/XX/FULL.
- \`all_Y_and_mapping.npz\`: original Z, actual T, mapped T_hat, fixed X,
  A,b and six complete Y arrays.
- \`all_matrices.npz\`: original CR3 analytic Hessian in ZX coordinates,
  data-only six Hessian means/RMS/SD/validity masks and NN-minus-analytic
  response-based fitted differences.
- \`analytic_logit__mean.csv\`, \`analytic_probability__mean.csv\`,
  and \`fitted_<six_channel>__mean.csv\`: full 10×10 numeric matrices.
- \`threeway_choice_TT.png\`, \`threeway_logit_ZX.png\` etc.: five-panel
  plots of exact answer, sieve-from-oracle-Y, sieve-from-NN-Y, and both errors.

## Correct interpretation

The analytic reference is CR3 at \`T_hat=A Z+b\`, **not** at the original
true T. This deliberately isolates the alternative functions of
original Z and X after a fixed affine mapping.

If the sieve from analytic-Y already has substantial error, a mismatch
from NN-Y does **not** establish bad decoder curvature: the sieve itself
has approximation error. If the sieve on oracle Y is accurate but the NN-Y
estimate is not, the mismatch can arise from the decoder's different
function of Z/X, as well as the sieve's sensitivity to that function.

The hard-thresholded choice output is not smooth: its Hessian is zero
almost everywhere / undefined at a threshold. Its comparison with
the CR3 smooth probability Hessian is an **external diagnostic only**,
not an estimate of the hard classifier's Hessian.

Finally, continuous X derivatives remain model-dependent because only
500 discrete stimuli are observed. The original CR3 has potentially
undefined derivatives at zero gain/loss; read \`fraction_defined\` and
\`strict_mean\` for ZZ/ZX/XX comparisons.
