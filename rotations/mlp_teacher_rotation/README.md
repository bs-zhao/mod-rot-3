# mlp_teacher_rotation — independent frozen-decoder rotation suite

**No other rotation suite is imported or launched.** Everything lives under
`rotations/mlp_teacher_rotation/`: its own CLI, experiment loader, teacher,
surrogate, 19 candidate objective definitions, optimizer, output writer, and
tests. No NP training code or existing suite is modified.

## What replaces Y

Given the frozen S3 Neural Process decoder `D_psi([X,Z])` and S5 full-trial
posterior-mean `Z`, define

`Y_MLP = softmax(D_psi([X,Z]))[...,1]`.

The categorical logits are decoded from **RAW X and Z** (the encoder alone
normalizes its input). Neither true simulator parameters nor the observed
hard Y are used to fit the new rotation: observed Y is used only to report
the decoder's descriptive FIT CE and accuracy. Checkpoint and latent hashes
must match, and all candidate rotations use NP TRAIN subjects only.

The package has **two complementary paths**, rather than pretending the
surrogate is the original decoder:

1. **MLP-soft response-profile path.** Project `Y_MLP` onto the same
   empirical polynomial X features and fit the package-local cubic-in-q
   response-profile function. Its operator08, fingerprint11, jointdiag13,
   pairwise-separation07, hypergraph09, and sparsity16 functionals have the
   same definitions as in the source functional-composition suite.
2. **Direct frozen-MLP derivative path.** Evaluate the trained decoder on
   identical, FIT-only X anchors for every FIT anchor subject. Use
   `torch.func.jvp` to compute exact `dP(Y=1|Z,X)/dq_i` and
   `d(logit1-logit0)/dq_i`. No refitting or stimulus-profile approximation
   occurs here. A PCA-whitened perturbation is mapped back into the raw
   8D latent Z, preserving the unselected PCA residual.

In both paths, every tested candidate `R` transforms fixed gradients
as `G_R=R G`. **No neural network is rerun at each rotation angle.**

## Independent candidate objectives

The package implements 19 scalar objective names:

| Family | Names | Interpretation |
|---|---|---|
| Calibrated operator/module support | `operator08`, `hypergraph09`, `polysparse16` | Sparse functional modules from the MLP-soft profile |
| Structure / separation | `fingerprint11`, `jointdiag13`, `separation07` | Low effective response-profile rank, contextual joint-diagonalization, pair separation |
| Direct effect rank | `teacher_fp_probability`, `teacher_fp_logit`, `teacher_fp_centered` | Per-axis entropy effective rank over common X anchors; centered variant isolates subject-dependent changes |
| Direct channel joint diagonalization | `teacher_jd_probability`, `teacher_jd_logit`, `teacher_jd_centered`, `teacher_global_jd` | Off-diagonal energy of multiple gradient second moments |
| Direct axiswise X-channel sparsity | `teacher_channel_sparse_probability`, `teacher_channel_sparse_logit` | Low L1/L2 ratio across latent axes per subject and stimulus |
| Predeclared combinations | `adaptive_profile`, `adaptive_teacher_probability`, `adaptive_teacher_logit`, `hybrid_profile_teacher` | Fixed identity-derived weighting; **exploratory**, not theoretically unique |

The two adaptive objectives use `J(R)+(F(I)/J(I))*F(R)` with a
small numerical floor; hybrid averages the identity-normalized profile
and probability-teacher adaptive objectives. **The default primary is
`hybrid_profile_teacher`, chosen before TEST inspection, not the best
TEST candidate.** All configured candidates are optimized separately,
and none is selected by its TEST θ recovery. Default config optimizes
16 candidates; the remaining three are selectable via `--set candidates=...`.

The 20 legacy proxy diagnostics (including expensive optional
prequential MDL) are also evaluated for Identity vs primary. Metric19
(discrete conditional synergy) is **not applicable to soft teacher Y**
and is recorded as null rather than misreported. Metric20 is null unless
`--set full_mdl=true`.

## Run

From repository root, using an existing S3 checkpoint and S5 frozen Z for
the requested model/condition (S1 dataset also required):

```cmd
cd /d D:\Projects\mod-rot-3
python rotations/mlp_teacher_rotation/run.py --model C1 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
python rotations/mlp_teacher_rotation/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
```

C2 uses 4 PCs, C3 uses 6 PCs by default. To run a quick smoke benchmark
without changing the default configuration file:

```cmd
python rotations/mlp_teacher_rotation/run.py --model C1 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --set candidates=["hybrid_profile_teacher","adaptive_profile","fingerprint11","jointdiag13"] --set starts=1 --set maxiter=20 --set sweep_steps=9
```

The full default run optimizes many objectives and may be CPU-intensive.
Use `--set teacher_device=cpu` or `--set teacher_device=cuda` to select the
decoder device. Model weights are never updated. For full metric20
calculation, use `--set full_mdl=true` (substantially more expensive).

## Outputs

`outputs/choice/<MODEL>/rotations/mlp_teacher_rotation/<CONDITION>/repeat01/run_<fingerprint>/`

* `rotation__matrix.npz`, `rotation__U.npz` — primary rotation
  (with the same affine PCA map and saved all-subject U contract).
* `rotation__method_arrays.npz` — fitted soft Y, direct probability/logit
  gradient tensors, shared FIT X anchors, PCA tangent vectors and polynomial
  coefficient tensors.
* `rotation__fit.json` — checkpoint identity, teacher FIT CE and accuracy,
  response-profile fidelity, all candidate optimizations and structural scores.
* `candidate__<name>/rotation__matrix.npz`, `rotation__U.npz`,
  `rotation__test_evaluation.json`, `rotation__test_corr2.png` — independent
  per-candidate transforms; TEST labels are read **after** saving all R/U.
* `teacher__candidate_comparison.csv/png` — descriptive TEST results for
  every candidate alongside identity.
* `teacher__all_metric_scores.csv`, `teacher__20_diagnostics.json`,
  `teacher__plane_curves.csv`, `teacher__summary.json`.

## Scientific interpretation / checks

* Direct-MLP gradient is the derivative of the *learned decoder*, not
  necessarily the true simulator gradient; predictive accuracy does not
  imply functional derivative recovery.
* Soft-Y projection is a **second approximation**; the separate direct
  derivative candidates audit whether it helps or hurts.
* C1/C2/C3 sample different X per subject: shared FIT-only stimulus
  anchors prevent differences in X support from masquerading as a difference
  in latent effect fingerprints.
* S5 Z already contains each participant's observed responses; this is
  descriptive functional rotation, not a novel-trial forecasting test.
* Previous C1/CR3 TEST outcomes motivated the objective catalogue;
  reporting performance on those same datasets is **exploratory** and
  must be verified using fresh NP trainings and independent generated data.
* Neither a small structural loss nor high in-sample/TEST parameter
  correlations establishes general latent identifiability.
