# CR4 functional-axis rotation (independent suite)

This is one isolated CR4 rotation package, not a new pipeline. It reads
frozen S3 decoder + S5 full-trial Z and the original 500-trial X bank from
S1. It does not retrain NP or import another rotation suite. The frozen
CR4 checkpoint MUST have the decoder-X standardization metadata created
by the CR4 branch of S3; this suite applies exactly that transform.

## Hypothesis and loss

CR4 fixes alpha=1; the true 3P generator has lambda, gamma, tau. In ideal
parameter coordinates, the tau direction has the **output gain** property,
while the lambda direction has zero pure lambda-lambda curvature.

This package tests these two functional properties without knowing their
parameter names during fitting. On TRAIN subjects and a common original
X sample, let eta(s,x) be frozen decoder logit and G(s,x) its gradient
along the three TRAIN PCA-whitened latent directions q:

  1. Regress G(s,x) over eta(s,x) separately for each participant; form
     residual A=E[(G-c_s eta)(G-c_s eta)^T] and B=E[GG^T].
     Identify the minimum generalized Rayleigh direction v solving
     A v=lambda (B+ridge I)v. Normalize by directional derivative energy,
     not raw residual energy, so weak directions are not wrongly preferred.
  2. In the orthogonal plane, scan/refine the SO(2) direction minimizing
     E[(v^T H_qq v)^2]/E[||H_qq||_F^2].
     This is an axial curvature, not a blanket ban on mixed UU interaction.
  3. The three ROWS of R are [low curvature, remaining, output gain].
     They define U(row)=q(row) R^T, R in SO(3).

All quantities come from autodifferentiating the frozen NN, not a fitted
polynomial surrogate. No sim theta, known CPT formula, or true axis labels
enter objective selection.

FIT uses up to 96 NP TRAIN people x 128 original X anchors; VALID uses
up to 48 NP VALID people x 64 DISJOINT X anchors. The gain and curvature
conditions have predeclared thresholds for BOTH splits, and a candidate
is flagged exploratory if either gate fails. Full decoder predictive
distribution equivalence, orthogonality, residual preservation and
source hashes are audited before theta access.

## Additional UX diagnostic

Compute actual mixed U-X Hessian for 24 TRAIN subjects x 48 original
stimuli, with X in the **saved decoder's standardized coordinates**.
Rotate only the U index. Record six X-column energy shares before/after.
Column energies must be orthogonally invariant, so R cannot repair a
decoder that improperly allocates its sensitivity across X variables.
This UX comparison is DIAGNOSTIC ONLY; it is not used in the loss and
does not select R. It directly checks the earlier CR3 failure pattern.

## Installation and commands (Windows CMD)

Prerequisites: CR4 pipeline S1, S2, S3 and S5 completed using the SAME
condition and repeat (S4 optional).

    git pull origin main
    python -m pytest rotations/cr4_functional_axes/test_suite.py -q

Check that source exists and inspect anchors without running:

    python rotations/cr4_functional_axes/run.py --model CR4 --condition beta0p0005_lambda0_ctx0p25to0p8_L3 --dry-run

Run a small smoke using an isolated output fingerprint:

    python rotations/cr4_functional_axes/run.py --model CR4 --condition beta0p0005_lambda0_ctx0p25to0p8_L3 --tag smoke --set fit_subjects=20 --set valid_subjects=12 --set fit_x=24 --set valid_x=16 --set ux_subjects=8 --set ux_x=12

Run the declared main protocol:

    python rotations/cr4_functional_axes/run.py --model CR4 --condition beta0p0005_lambda0_ctx0p25to0p8_L3 --repeat 1 --tag main

The condition must match an ACTUAL trained CR4 S3/S5 checkpoint.
For an L8 checkpoint, pass its L8 condition. This suite fixes the
rotation subspace at THREE TRAIN PCs, preserves unselected native Z
residuals exactly, and applies R only to those 3 q coordinates.

## Outputs

Under outputs/choice/CR4/rotations/cr4_functional_axes/<condition>/
repeat01/main_<fingerprint> :

- protocol.json: config and source/code SHA256, FIT/VALID ID lists
- rotation.npz: R, A=RW, U, q, native residuals, split identifiers
- fit_fields.npz / validation_fields.npz: frozen eta, G and H
- loss_diagnostics.json: gain eigenvalues/eigengap, directional energy,
  FIT/VALID relative gain residual, axial curvature separation and gates
- curvature_landscape.csv/png: entire in-plane scan
- ux_audit.npz and ux_diagnostics.json: exact mixed UX and X-column audit
- invariance_audit.json: native-Z and probability roundtrips
- fit_complete.json: R persisted BEFORE any theta diagnostic
- evaluation.json: TRAIN matching -> TEST r^2 for candidate and PCA identity
- candidate_correlations.png / identity_pca_correlations.png

The method's output is a functional candidate, not guaranteed causal or
parameter recovery. In a highly nonlinear NP representation, a globally
constant direction implementing tau scaling or lambda affinity may not
exist. Gate failures are scientifically meaningful and retained. Do not
select variants based on TEST theta. If successful, compare against
independent repeat-2 NP after defining the same precommitted protocol.


## Controlled CR5 transfer, same rotation algorithm

The exact SAME function-supported gain-axis / low-axial-curvature algorithm,
constants, no-oracle-fit contract and UX audit are applicable to CR5.
CR5 changes only the stimulus design to the EXACT original CR3 500-row
bank, including gain-only, loss-only, gamma-neutral and general trials,
while retaining CR4 alpha=1, three parameters and standardized decoder X.
No condition labels enter the rotation objectives, so this is a clean
out-of-sample design comparison; native labels remain in the CR5 bank
for **after-fit** diagnostic stratification.

After completing CR5 S3 and S5:

    python rotations/cr4_functional_axes/run.py --model CR5 --condition beta0p0005_lambda0_ctx0p25to0p8_L3 --repeat 1 --tag main

Results are isolated in `outputs/choice/CR5/rotations/cr4_functional_axes/`.
Never reuse the CR4 checkpoint, Z or stimulus scaler on CR5. Strict
CR5 source, checkpoint and S1/S5 hashes are checked before optimization.
