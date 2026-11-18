# Role-free Hessian Sparsity Rotation — CR3, CR4, CR5

An **independent rotation package**, deliberately distinct from
\`cr4_functional_axes\`. No output-gain assumption, no tau/lambda/gamma/alpha
role or sequential greedy axis discovery, no true CPT formula. Exactly the
same structural loss and optimizer apply to all three frozen NPs.

## Mathematical objective

Let q be d ALL-SUBJECT-PCA-whitened latent dimensions (d=4 CR3, d=3 CR4/CR5).
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

## Optimization and full-subject descriptive evaluation

- **ALL subjects, with NO rotation-stage train/validation/test participant split.**
  The original NP training splits stay intact for NN training and source
  integrity, but have NO role in rotation, PCA, supervised Oracle or scoring.
- PCA/whitening is fit on the full subject Z matrix.
- FIT-X and VALID-X contain disjoint *original stimulus trial IDs*, and
  both contain **the identical complete set of all subjects**.
  Thus VALID-X is a stimulus sensitivity diagnostic, NOT held-out subjects.
- Optimize ALL d(d-1)/2 Givens angles simultaneously in SO(d).
- Precompute second-moment tensors once; differentiate this scalar loss via
  autograd; seeded multistart L-BFGS-B. R* selected using only FIT-X.
- Save FIT-X/VALID-X loss, optimizer restarts and all pairwise rotation
  landscapes. Persist R* BEFORE reading any generating theta.
- Report full-population Pearson correlations, matched ALL-subject r²,
  and TRAIN/VALID/TEST-free alignment to actual theta after R* frozen.
  These are **in-sample descriptions**, not generalization claims.
- Full cohort (typically 5,000) creates significantly more NN Hessian
  computation than prior 48-person subsetting. Tune `--set fit_x=...`
  and `--set valid_x=...` if stimulus budget is too costly; these
  choices do NOT drop any subjects. Batch size is configurable.


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

Full-cohort low-stimulus-cost CR3 run (still ALL subjects):

    python rotations/hessian_sparsity/run.py --model CR3 --mode joint --tag all_low_X --set fit_x=20 --set valid_x=16 --set restarts=3 --set maxiter=80

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
CPT axes. A result can improve FIT-X/VALID-X losses while decreasing
full-population parameter alignment. With no rotation TEST set,
correlations are descriptive in-sample. Replication under independent
NP seeds is required to assess stability. Do not select hyperparameters from TEST theta. Compare
\`pooled\` with \`balanced\` to understand stimulus-mixture effects.


## Post-freeze Oracle R vs blind R* loss comparison — ALL subjects

New full-subject runs automatically produce this comparison AFTER
the blind R* has been frozen to disk. It uses ALL subjects for both
unsupervised Hessian R* and supervised Oracle R. No TEST subjects
or TRAIN-only parameter matching exist in this protocol.

For each new completed full-subject run, rerun this cheap post-fit
analysis WITHOUT retraining NP, extracting derivatives or refitting R*:

    python rotations/hessian_sparsity/oracle_compare.py --model CR3
    python rotations/hessian_sparsity/oracle_compare.py --model CR4
    python rotations/hessian_sparsity/oracle_compare.py --model CR5

The resolver selects completed ALL-SUBJECT run folders only. Use
\`--run-dir\` with an exact output path if multiple matches exist.
Legacy runs fitted to 48 subjects are intentionally rejected: a
previous subsampled R* must NOT be mislabeled as full-cohort R*.

Oracle definitions use the **entire Z,theta population**:
1. \`R_oracle_procrustes\`: full-cohort standardized-theta SO(d)
   orthogonal least-squares fit.
2. \`R_oracle\`: full-cohort approximate SO(d) optimization maximizing
   mean named-axis Pearson r² (20 seeded starts by default).

Evaluate R=I, blind R*, Procrustes Oracle and correlation Oracle on the
**same original FIT-X and VALID-X second-derivative moments**.
Save \`L(R_oracle)-L(R*)\` in both X samples, including UU/UX separately.
Output files: \`oracle_comparison.json\`, \`oracle_rotations.npz\`,
\`oracle_loss_comparison.csv\`, \`oracle_loss_comparison.png\`,
\`oracle_axis_overlap.png\`. Original files are never overwritten.

The Oracle uses theta supervision, but only as a *post-freeze reference*.
Neither Oracle's supervised criterion nor the true theta enter the
blind optimization. All reported parameter correlation statistics
are in-sample over the 5,000 subjects. There is **no TEST r²**.

Run only the post-fit report by invoking \`oracle_compare.py\`. The
original \`run.py\` also automatically adds it on newly completed runs
unless \`--no-evaluate\` was requested. This full-cohort protocol creates
NEW folders; old 48-person runs are preserved but cannot be reclassified
as all-subject experiments.

This analysis is **diagnostic and supervised**. The blind R* rotation
remains a genuinely unsupervised simplicity optimizer; the Oracle
parameter mapping does NOT enter Hessian loss optimization. A higher
Oracle Hessian loss would indicate a conflict between the current
simple-function definition and parameter alignment, but does not
imply NN Hessian numerical error by itself.
