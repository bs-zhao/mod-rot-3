# Effect-Fingerprint Rank: complete independent rotation suite

This is a **standalone rotation experiment**, not an additional global metric
and not a sub-method in functional_composition. Every input loader, surrogate
model, function metric, orthogonal optimizer, diagnostic and test is inside
this directory.

It investigates whether **a latent axis has one or a few consistent stimulus
response-effect profiles**. The fitted ZXY response-profile polynomial is
the same modeling family used in functional_composition v2; hence comparing
initial results holds the surrogate family constant.

## Commands (Windows CMD, from repository root)

Run one CR3 rotation without changing NP:

    python rotations/effect_fingerprint/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4

Vary effective-rank definition or stimulus-channel metric:

    python rotations/effect_fingerprint/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --set rank_metric="participation" --set channel_mode="unitvar"

Test across independent FIT sample sizes, surrogate refits, optimizer seeds
and different rank/channel definitions (opt-in; many expensive fits):

    python rotations/effect_fingerprint/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --set stress_enabled=true --set stress_subject_counts=[320,640,1280] --set stress_seeds=[31,47,73]

Repeat with beta0p0001_lambda0_ctx0p25to0p8_L8 or other trained conditions
by changing ONLY --condition. Do not use true theta for the rotation search.

## Objective and limits

Given proxy gradients G(s,i,l)=d(response-profile-channel_l)/du_i,
rotate the coordinate index with R and form per-axis channel covariance
C_i = G_i^T G_i / N. The objective is sum_i(effective_rank(C_i)-1),
where effective rank is Shannon entropy rank, participation ratio or an
energy-tail variant. The channel metric ('raw', 'unitvar', 'whiten') is
computed ONCE from frozen FIT gradients, before searching R.

An optional weak-axis penalty is disabled by default. A tiny-gradient
axis may otherwise appear artificially simple; the diagnostic report
records each axis gradient energy.

A model with two distinct nonlinear stimulus effects can favor original
axes; a linear model can be completely nonidentifiable, and a radial model
can have no preferred orientation. Tests explicitly cover these cases.

## Outputs

The experiment saves R/A/U and TEST r² alongside
fingerprint__diagnostics.json, fingerprint__plane_curves.csv/png,
and optional fingerprint__stress.csv/json and
fingerprint__stress_rotations.npz (FIT-only resampled rotations).

The TEST theta audit follows R/U persistence and does not enter optimization.
In-sample FIT surrogate gradients are NOT exact CR3 derivatives; S5 latent Z
was computed using all trials per subject, so this is not unseen-trial generalization.
This is a research experiment, not a universally identifiable parameter estimator.


## Compare two independent beta runs (does not refit or search R)

After running this same suite separately for two NP conditions:

    python rotations/effect_fingerprint/compare.py --model CR3 --conditions beta0p0001_lambda0_ctx0p25to0p8_L8 beta0p0005_lambda0_ctx0p25to0p8_L8

This script first aligns axes by permutation and sign on NP TRAIN subjects,
then reports held-out TEST-subject axis correlation. It never searches another
rotation and it never uses TEST theta to define the alignment. The saved TEST
theta r² in the completed runs is copied for descriptive comparison only.

If you have multiple completed runs for one condition, pass --runs with
the exact run folders rather than silently choosing whichever finished last.
