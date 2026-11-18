# Role-free Hessian Sparsity Rotation — CR3, CR4, CR5

An **independent rotation package**, deliberately distinct from
\`cr4_functional_axes\`. No output-gain assumption, no tau/lambda/gamma/alpha
role or sequential greedy axis discovery, no true CPT formula. Exactly the
same structural loss and optimizer apply to all three frozen NPs.

## Mathematical objective

Let q be d TRAIN-PCA-whitened latent dimensions (d=4 CR3, d=3 CR4/CR5).
Take \`U=Rq\` with \`R in SO(d)\`. For each selected subject and original X,
differentiate the frozen Decoder logit and obtain

    H = d² logit / dq dq^T
    J = d² logit / dq dX_standardized^T

Then \`H_R = R H R^T\` and \`J_R=R J\`. Critically the full Frobenius
energies are rotation-invariant and CANNOT select R. Instead minimize
*elementwise RMS-L1 sparsity*, with moments over selected real subjects
and original 500-bank stimuli:

    L_UU(R) = sum_{ij} sqrt(E[(H_R)_ij²]+eps*E||H||_F²/d²) /
              (d sqrt(E||H||_F²))

    L_UX(R) = sum_{ij} sqrt(E[(J_R)_ij²]+eps*E||J||_F²/(6d)) /
              (sqrt(6d) sqrt(E||J||_F²))

    L_joint = (w_UU L_UU + w_UX L_UX)/(w_UU + w_UX)

Both terms use the exact same formula for CR3/4/5. The zero/weak-value
smoothing eps defaults to 1e-8. Symmetric UU off-diagonal elements count
twice. No special parameter role is encoded.

Three predeclared modes: \`joint\` (default, w_UU=w_UX=1),
\`uu_only\` (no X differentiation at all), and \`ux_only\`.
For CR3/CR5 \`--grouping balanced\` applies *the same scalar loss per
original condition*, averaging the four values equally. The default
\`pooled\` instead uses the natural mixture of original 500 trials. CR4
has only one random-X condition. Neither condition label nor theta
enters a role-specific rule.

## Optimization and validation

- Optimize ALL d(d-1)/2 Givens angles jointly over SO(d), not a sequence
  of axes or a two-dimensional-only search.
- FIT = up to 48 NP TRAIN people × 96 original X trials;
  VALID = up to 24 disjoint NP VALID people × 64 disjoint original X trials.
- Precompute second-moment tensors once; fast exact torch-autograd gradient
  of the scalar rotation loss; 16 deterministic multistart L-BFGS-B runs.
- Choose the winning restart by **FIT alone**. VALID is only diagnostic.
  Save all restart scores and validation-vs-PCA comparisons.
- Scan all pairs of U coordinates 0...90° after optimizing and save the
  FIT/VALID curves. Endpoints 0/90 are mathematically equivalent by
  signed-permutation symmetry; not a parameter recovery guarantee.
- Persist R, U and source/provenance hashes, audit zero change in frozen
  predictions, then and only then load true theta. Use TRAIN theta to
  choose signed/permutation matching; report held-out TEST r/r².

**CR3:** requires the newly trained \`_xstd\` S3 checkpoint, not old raw-X.
For p=0/1 CR3/CR5, smooth NN X-derivatives are finite but do NOT equal
the potentially singular analytic CPT boundary derivatives. Always compare
joint with \`uu_only\`. Original X, trial types, and physical units are never
modified. Derivatives wrt X use the saved *standardized decoder* coordinates.

## Windows CMD commands

Test the new package:

    git pull origin main
    python -m pytest rotations/hessian_sparsity/tests -q

Inspect paths first:

    python rotations/hessian_sparsity/run.py --model CR3 --dry-run

Fast CR3 smoke (the full one is costlier):

    python rotations/hessian_sparsity/run.py --model CR3 --mode joint --tag smoke --set fit_subjects=8 --set valid_subjects=8 --set fit_x=20 --set valid_x=16 --set restarts=3 --set maxiter=80

Full runs (same metric, model-specific latent dimension):

    python rotations/hessian_sparsity/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L4_xstd --mode joint
    python rotations/hessian_sparsity/run.py --model CR4 --condition beta0p0005_lambda0_ctx0p25to0p8_L3 --mode joint
    python rotations/hessian_sparsity/run.py --model CR5 --condition beta0p0005_lambda0_ctx0p25to0p8_L3 --mode joint

Main Hessian-only ablation, avoiding problematic boundary X gradients:

    python rotations/hessian_sparsity/run.py --model CR3 --mode uu_only --tag uu_main
    python rotations/hessian_sparsity/run.py --model CR4 --mode uu_only --tag uu_main
    python rotations/hessian_sparsity/run.py --model CR5 --mode uu_only --tag uu_main

Original condition balanced sensitivity (CR3, CR5):

    python rotations/hessian_sparsity/run.py --model CR3 --mode joint --grouping balanced --tag balanced
    python rotations/hessian_sparsity/run.py --model CR5 --mode joint --grouping balanced --tag balanced

Requirements: CR3/4/5 S1, S3 and S5 completed for the SAME condition.
S5 has all-observed-trials posterior mean Z. Default source folders
are the existing S3/S5 paths; the new package NEVER trains a decoder.

## Results and interpretation

Isolated paths:

    outputs/choice/<MODEL>/rotations/hessian_sparsity/<condition>/repeat01/
        <tag>_<mode>_<grouping>_<fingerprint>/

Artifacts: \`protocol.json\`, \`rotation.npz\`, \`fit_fields.npz\`,
\`validation_fields.npz\`, \`loss_diagnostics.json\` (ALL restarts),
\`plane_landscapes.csv\` and \`plane_u*_u*.png\`, \`plane_diagnostics.json\`,
\`prediction_invariance.json\`, \`fit_complete.json\`, \`evaluation.json\`,
and candidate / identity heatmaps.

This is an UNSUPERVISED ROTATION HYPOTHESIS, not guaranteed to recover
CPT axes. A result must improve FIT/VALID losses, show separate held-out
TEST recovery, and reproduce under independent NP seeds to count as
strong evidence. Do not select hyperparameters from TEST theta. Compare
\`pooled\` with \`balanced\` to understand stimulus-mixture effects.
