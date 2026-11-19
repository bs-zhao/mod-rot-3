# Gaussian-Integrated Functional Rotation (GIFR)

**Independent, role-free rotation suite** for frozen CR3 (four-parameter,
4D latent), CR4/CR5 (three-parameter, 3D latent) Neural Process decoders.

The crucial change from empirical-sample rotation is its **integration
measure**. No participant-specific \`z_s\` is used to evaluate a decoder
derivative. The complete subject population contributes ONLY a single
mean vector and full covariance matrix, from which we define a Gaussian
reference measure. An arbitrary number of deterministic Gaussian nodes
then drives the frozen decoder's Hessian analysis. Source Z and true
parameter matrices are accessed later ONLY to evaluate the frozen R
against an all-subject supervised Oracle R.

This is a research **hypothesis**, not guaranteed parameter recovery.
The initial default objective deliberately retains our former generic
global Hessian RMS-L1 sparsity loss; previous empirical results showed
this loss can favor an energy-concentrating direction over psychologically
meaningful axes. New integration need not change that. No output-gain,
parameter role, analytical CPT formula, or axis-specific heuristics
enter any unsupervised optimization.

## 1. Mathematics and coordinate conventions

Let the frozen NP decoder return two logits, so define
\`f(z,x)=logit_1(z,x)-logit_0(z,x)\`. Fit the full empirical native-Z
mean and covariance using **all subjects**:

    mu = mean_s(z_s)
    Sigma = cov_s(z_s)   # full d by d covariance, ddof=1
    Sigma = B B^T
    q = B^{-1}(z-mu), q ~ N(0,I_d) [REFERENCE ASSUMPTION]

For CR3, d=4; for CR4/CR5, d=3. The eigenbasis, ordering and signs
of B are deterministic. No native-Z residuals are silently discarded:
the source latent dimension must be EXACTLY d.

A rotation acts as:

    u = R q,    R^T R = I, det R = +1
    f_R(u,x) = f(mu + B R^T u, x)

The decoder is frozen and predictions are unchanged. The NN's
q-Hessian is

    H(q,x) = B^T (d²f/dz²) B

and its rotated Hessian is \`R H(q,x) R^T\`, with q+X integration
evaluated on the *original same* q Gaussian and X measure for every R.
Optional mixed derivative in standardized Decoder X coordinates is
\`J(q,x)=d²f/dq dXstd\`, rotating as \`R J\`.

The first version estimates small tensor integrals:

    C_abcd = E_q,X[H_ab H_cd]       # d^4
    D_abj  = E_q,X[J_aj J_bj]      # d x d x 6 (optional)

After these are computed once, no more Decoder calls are needed to
evaluate \`L(R)\` or its autograd rotation gradient.

## 2. Gaussian Q integration options

By default:

    --set z_method=sobol
    --set z_cutoff=2.0

This samples QMC Sobol points distributed as the **coordinatewise
truncated standard normal** q_j~N(0,1) conditioned on
\`-2<=q_j<=2\`. It is NOT a uniformly distributed [-2,2]^d box.
Inverse Gaussian CDF gives correctly weighted points; every Sobol
point represents an equal mass of the target truncated-normal law.

Alternative full Gaussian: \`--set z_cutoff=null\`.
Gaussian point counts must be powers of two, default 512 FIT points
and 256 VALID points. Independent scrambled Sobol seeds are used to
assess quadrature repeatability, **NOT** split subjects.

For low-dimensional deterministic quadrature use either:

    --set z_method=hermite --set z_cutoff=null --set z_grid_order=7
    --set z_method=grid --set z_cutoff=2 --set z_grid_order=7

Hermite integrates the **full** normal distribution (order^d nodes).
Grid integrates a truncated Gaussian using per-axis Gaussian density
and trapezoidal quadrature weights; never a uniform average.
VALID uses z_grid_validation_order=9 by default (larger grid).
In every mode the integration weights are explicitly normalized.

A full Gaussian is only an *approximation* to the distribution of NP
posterior means. Marginal normality is not sufficient to establish
joint Gaussianity or decoder faithfulness far outside observed Z.
We do NOT equate posterior-mean q with samples of the NP prior.

## 3. Stimulus X integration

**\`--x-mode bank\`** (default): CR3, CR4 and CR5 each have an actual
shared immutable 500x6 physical-X bank. Sum equally over **all 500**,
with no density fitting, interpolation, rejection, or X derivatives
unless an explicit joint/UX ablation is selected.

CR3/CR5 retain original labels: gain-only (60), gamma-neutral (60),
loss-only (40), general (340). \`--grouping balanced\` computes exactly
the SAME scalar objective separately per group and averages four
scores equally. Default pooled follows the original 500-item mixture.

**\`--x-mode generative\`**: for CR4, read and SHA-verify the real S0
design JSON and use its six independent uniform-coordinate generating
law (four monetary attributes + two probabilities). FIT and VALID
each use independent Sobol X samples. No reconstruction from an
empirical 500-item histogram. The original CR4 bank is NOT modified.

For a future model with an explicitly known independent-uniform X
generator, a custom JSON can be used with \`--x-spec-file\`:

    {
      "family": "independent_uniform",
      "low":  [5, 5, 0.05, 5, 5, 0.05],
      "high": [200, 200, 0.95, 200, 200, 0.95]
    }

This intentionally does not claim to reproduce CR3's complex
FOSD-conditional generation or mixture of special conditions.
Non-factorized stimulus generators require explicit new samplers.

X remains physical for the integration law. The checkpoint's
**persisted** per-coordinate mean and SD convert it into Decoder-X
coordinates prior to differentiation. A missing/mismatched scaler
raises an error; CR3 requires the NEW \`_xstd\` decoder.

## 4. Simplicity objective and optimization

Default is **\`--derivatives uu_only\`**. Fixed X values are only
conditions of f; no X differential is taken. Thus CR3's amount-zero
and p=0/1 X derivative singularities are not relevant to this
initial UU-only diagnostic.

The role-free empirical Hessian RMS-L1 is retained as a *baseline*,
but all moments use the new Gaussian-X measure:

    L_UU(R) =
      sum_ij sqrt(E[(R H R^T)_ij^2] + eps * E||H||_F^2 / d^2)
      / (d * sqrt(E||H||_F^2))

Optional \`ux_only\` and \`joint\`:

    L_UX(R) =
      sum_ij sqrt(E[(R J)_ij^2] + eps * E||J||_F^2/(6d))
      / (sqrt(6d) sqrt(E||J||_F^2))

    L_joint = (w_UU L_UU + w_UX L_UX)/(w_UU+w_UX)

We do NOT minimize the total Frobenius energy, which is invariant to
orthogonal R. In the default loss the symmetric off-diagonal UU
entries both contribute.

The same formula and full SO(d) search apply to CR3/4/5, with 6
Givens angles for SO(4) and 3 for SO(3). Each run includes 16 seeded
L-BFGS-B initializations, full FIT/VALID Gaussian integrals and all
pairwise 0..90° diagnostic curves. The optimized R* is chosen by
**FIT Gaussian simplicity only**; VALID integrals are diagnostics.
No true CPT parameter is ever accessed during fitting.

Crucially, the Gaussian Z nodes used to fit R are **never observed
people's Z values**. Only the complete-cohort \`mu\` and \`Sigma\`
are estimated from people.

## 5. Full-population Oracle comparison (after freeze)

Only AFTER R* and its fit_complete.json have been saved:

1. Read all subjects' real Z and true theta.
2. Calculate SO(d) Procrustes Oracle (standardized theta).
3. Calculate SO(d) correlation Oracle, maximizing mean of parameter-
   specific squared Pearson correlation over ALL subjects.
4. Evaluate **I, blind R*, Procrustes Oracle and correlation Oracle**
   on the **identical precomputed FIT/VALID Gaussian Hessian losses**.
5. Report \`L(R_Oracle)-L(R*)\`, UU/UX/group contributions and in-sample
   full-subject correlation matrices/r².

The Oracle does NOT select the blind metric, the unsupervised solution,
its initialization, or its optimization settings. This suite uses NO
participant TEST split for rotation. Full-sample correlations should
never be reported as generalization results.

## 6. Windows CMD — direct commands

From repository root, with existing S1/S3/S5 artifacts and \`py11\`:

    git pull origin main
    python -m pytest rotations/gaussian_integrated_rotation/tests -q

Path and Gaussian plan (no Decoder calls):

    python rotations/gaussian_integrated_rotation/run.py --model CR3 --dry-run

Cheap real frozen-decoder smoke (still Gaussian, 24 fixed X, 16 Q):

    python rotations/gaussian_integrated_rotation/run.py --model CR3 --tag smoke --set z_fit_points=16 --set z_validation_points=16 --set x_bank_limit=24 --set restarts=3 --set maxiter=80

Full **fixed 500-trial bank**, Gaussian Q integral and UU-only:

    python rotations/gaussian_integrated_rotation/run.py --model CR3 --tag main
    python rotations/gaussian_integrated_rotation/run.py --model CR4 --tag main
    python rotations/gaussian_integrated_rotation/run.py --model CR5 --tag main

CR4 continuous ORIGINAL generator (instead of empirical 500-bank):

    python rotations/gaussian_integrated_rotation/run.py --model CR4 --x-mode generative --tag generative

Alternative full Gaussian and joint derivative:

    python rotations/gaussian_integrated_rotation/run.py --model CR3 --set z_cutoff=null --tag fullnormal
    python rotations/gaussian_integrated_rotation/run.py --model CR5 --derivatives joint --tag mixed
    python rotations/gaussian_integrated_rotation/run.py --model CR3 --grouping balanced --tag balanced

Default 512q × 500X is much more work than a 16q × 24X smoke test,
but contains NO derivative evaluation at subject-specific Z. The
large NP network is evaluated in configurable \`z_batch=4\` and
\`x_batch=32\` chunks; the moment tensors remain tiny. Reduce the
Q/X quadrature resolution **explicitly** for a speed experiment
instead of reducing the number of subjects.

## 7. Outputs

Separate from all previous rotation packages, under:

    outputs/choice/<MODEL>/rotations/gaussian_integrated_rotation/
      <condition>/repeat01/<tag>_<derivatives>_<xmode>_<fingerprint>/

- \`protocol.json\`: source SHA256, covariance, quadrature laws and settings
- \`quadrature_fit.npz\` / \`quadrature_validation.npz\`: reusable q/X nodes and weights
- \`moments_fit.npz\` / \`moments_validation.npz\`: integrated UU/UX moment tensors
- \`rotation.npz\`: R, B, affine mean, covariance and diagnostic all-subject U
- \`loss_diagnostics.json\`: every blind restart, FIT/VALID/identity losses
- \`plane_landscapes.csv\`, \`plane_u*_u*.png\`: pairwise loss curves
- \`invariance_audit.json\`: orthogonality, prediction invariance
- \`fit_complete.json\`: blind R frozen before theta read
- \`oracle_comparison.json\`, \`oracle_rotations.npz\`,
  \`oracle_loss_comparison.csv\`, \`oracle_loss_comparison.png\`

NOTE: under the default FIXED BANK, FIT vs VALID differ only in their
independent Gaussian quadrature nodes, not in X or people. Under
GENERATIVE X, both q and X integration nodes differ.

The main next experiment is whether switching from empirical-subject
integration to this explicitly specified Gaussian measure changes the
gap \`L(R_Oracle)-L(R*)\` and parameter alignment. Do not interpret a
lower L1 sparsity loss alone as evidence of recovery.
