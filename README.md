# mod-rot-3

**Clean, reproducible binary-choice Neural Process training and independent post-hoc latent rotations.**

The project deliberately has **two separate pipelines**:

1. `baseline/`: exact C1/C2/C3/CR3 simulation, original model-specific prechecks, shared binary-choice Neural Process training, checkpoint predictive evaluation, full-observed-trial latent Z, PCA and true-parameter linear Ridge evaluation.
2. `rotations/`: experiment with objective functions on **frozen** Z, X, Y. Nothing in this module retrains NP or modifies the generators. The included identity, quartic and kurtosis procedures are baseline controls, not a claim of recovering ground-truth parameters. Add future UXY objectives here.

All source data/checkpoints/plots/rotation matrices are written under **`outputs/` (gitignored)**. No generated datasets or models are committed.

## Source structure

```text
mod-rot-3/
  baseline/run.py                       # generate/precheck/train/extract/analyze/all
  rotations/run.py                      # post-hoc, independent rotation methods
  src/modrot3/
    pipeline.py                         # model-specific I/O / exact NP orchestration
    analysis.py                         # TRAIN PCA; VAL-selected Ridge; TEST scores
    rotation.py                         # frozen-ZXY rotation interface and baseline methods
    sanity.py                           # small selftest
  src/modlearn_rot/
    generators/                        # EXACT original C1/C2/C3 simulators
    diagnostics/precheck/              # EXACT original full C1/C2/C3 prechecks
    nn/                                 # EXACT original Set-Transformer NP
    training/                           # EXACT original discrete-choice NP trainer
    utils/                              # original output paths and random seed utilities
  reference/
    s0_cr2_design_recovery.py           # EXACT original CR2-4P design/prior for CR3
    cr3/s0_s1_cr3.py                    # EXACT original 500-trial CR3 generator/FOSD check
    cr3/s2_oracle_cr3.py                # optional CR3 oracle diagnostic
    functional_role_identifiability/   # reference oracle utilities
  tests/
  docs/SOURCE_PROVENANCE.md
  outputs/                               # ignored
```

Only source-root paths in the reference CR3 scripts were changed for the new tree. Mathematical formulas, true-parameter distributions, random stimulus/subject sampling, CR3 500-trial bank composition and the original NP architecture/training losses remain the same as upstream `modlearn-rot-2`.

## Setup (Windows CMD)

```cmd
cd /d D:\Projects
git clone https://github.com/bs-zhao/mod-rot-3.git
cd mod-rot-3
conda activate py11
pip install -e ".[dev]"
python baseline/run.py selftest
```

**C1, C2, C3** are the original independent-stimulus binary choice models with 4, 4 and 6 true factors. **CR3** uses the exact original four-parameter CPT simulator and 500-trial fixed stimulus bank with no strictly first-order dominant pair. Creating CR3 data for the first time may involve rebuilding the original CR2-4P fixed bank and therefore take substantially longer than C1/C2/C3. To guarantee identity with a historical run, reuse and hash-check its original bank.

## Five independent steps

**1. Generate an unchanged C model:**

```cmd
python baseline/run.py generate --model C1 --data-seed 20261006
python baseline/run.py generate --model C2 --data-seed 20261006
python baseline/run.py generate --model C3 --data-seed 20261006
python baseline/run.py generate --model CR3 --design-seed 20261023 --data-seed 20261028 --subjects 5000
```

**2. Original precheck:**

```cmd
python baseline/run.py precheck --model C1
python baseline/run.py precheck --model C2
python baseline/run.py precheck --model C3
python baseline/run.py precheck --model CR3
```

The full upstream `PrecheckSuite` (identifiability, oracle recovery, relevance, response balance, predictive ceiling, etc.) is retained **unchanged** for C1/C2/C3. For CR3, the original fixed-bank/FOSD/structural S2 checks are retained. An optional computationally expensive CR3 oracle MLE check is available:

```cmd
python baseline/run.py precheck --model CR3 --oracle-precheck --oracle-subjects 100 --oracle-starts 4
```

**3. NP training with explicit hyperparameters:**

C3 (same intended condition as prior experiments):

```cmd
python baseline/run.py train --model C3 --beta 0.00625 --lambda-sparse 0 --latent-dim 16 --context-min 0.25 --context-max 0.80 --batch-size 64 --learning-rate 0.0005 --weight-decay 0 --max-epochs 1000 --patience 50 --repeat 1 --train-seed 31 --device auto
```

CR3 (native 8D, exactly beta=0):

```cmd
python baseline/run.py train --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --train-seed 31 --device auto
```

**4. Separately report the original NP validation/TEST prediction accuracy and learning curves (no retraining):**

```cmd
python baseline/run.py evaluate-np --model C3 --beta 0.00625 --lambda-sparse 0 --latent-dim 16 --context-min 0.25 --context-max 0.80 --repeat 1
python baseline/run.py evaluate-np --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.80 --repeat 1
```

**5. Extract every subject's posterior mean Z using ALL observed trials:**

```cmd
python baseline/run.py extract --model C3 --beta 0.00625 --lambda-sparse 0 --latent-dim 16 --context-min 0.25 --context-max 0.80 --repeat 1 --device auto
python baseline/run.py extract --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.80 --repeat 1 --device auto
```

**6. PCA + held-out linear parameter recovery:**

```cmd
python baseline/run.py analyze --model C3 --beta 0.00625 --lambda-sparse 0 --latent-dim 16 --context-min 0.25 --context-max 0.80 --repeat 1
python baseline/run.py analyze --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.80 --repeat 1
```

PCA is fitted on TRAIN subjects only. Ridge alpha is selected on VALIDATION labels only, then refitted on TRAIN+VALIDATION and evaluated on TEST. The report includes all-Z parameter regression, PC1..PCk regression curves, PCA variance explained, raw-Z/PC correlation heatmaps, ridge scatter plots and CSV/JSON summaries. NP training **never reads theta**. The extracted Z uses every observed trial regardless of the randomized subset mechanism used during original NP training.

For a completely fresh dataset and condition, replace `train` with `all` to generate, precheck, train, extract and analyze sequentially. Existing verified data and checkpoints are preserved; no silent overwrite.

## Separate rotation pipeline

```cmd
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --repeat 1 --representation pca --rotation-dim 4 --method identity
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --repeat 1 --representation pca --rotation-dim 4 --method kurtosis --starts 4 --maxiter 200
```

The default 4D CR3 rotation uses the first four TRAIN-fitted whitened PCs of the frozen native 8D Z. For a full 8D orthogonal rotation use `--representation full --rotation-dim 8`. No model is retrained and no theta is read by the rotation optimizer. The rotation writes `R`, the original Z-to-U affine matrix `A` and offset, each subject's U, a JSON loss record, and a separate postfit TEST true-parameter correlation heatmap.

`src/modrot3/rotation.py` contains the **only** rotation method registry. Existing methods are intentionally simple distribution-only controls (`identity`, `quartic`, `kurtosis`); new interaction/Hessian/Functional-MDL/UXY objectives belong in this module or standalone files imported by it. No old UXR1–5 accumulation or circular imports are copied.

## Outputs

```text
outputs/choice/C3/
  s1_generate/s1_generate__dataset.npz
  s2_precheck/s2_precheck__report.json
  s3_train/beta0p00625_lambda0_ctx0p25to0p8_L16/repeat01/
    s3_train__checkpoint.pt
    s3_train__history.csv
    s3_train__summary.json
  s3_evaluate/<same-condition>/repeat01/
    s3_evaluate__summary.json
    s3_evaluate__learning_curves.png
  s4_evaluate/<same-condition>/repeat01/
    s4_evaluate__latent.npz
    s4_evaluate__summary.json
    s4_evaluate__linear_predictions.npz
  s12_z_diagnostics/<same-condition>/repeat01/
    s12__diagnostics.json
    s12__pca.npz
    s12__pca_ridge_curve.csv
    s12__full_z_ridge.csv
    s12__pca_and_regression.png
    s12__theta_vs_z.png
    s12__theta_vs_pca.png
    s12__ridge_scatter.png
  rotations/<method>/<condition>/repeat01/
    rotation__matrix.npz
    rotation__U.npz
    rotation__fit.json
    rotation__test_evaluation.json
    rotation__test_corr2.png
```

## Reproducibility and safety

- Inputs and frozen CR3 stimulus banks have separate SHA-256 provenance.
- Changing a previously used training condition's learning rate, batch size, patience or training budget is **not** silently treated as the same checkpoint: use a new `--repeat` or archive previous outputs.
- Modifying a source dataset after training invalidates the checkpoint's provenance and blocks Z extraction.
- Existing Z is never silently replaced by Z extracted from a different checkpoint.
- True theta is present in saved datasets and post-training evaluation but never in unsupervised rotation loss or NP training.
- CR3 β=0 and β=0.0005 occupy distinct folders and cannot be mixed.
- All `outputs/` files are Git-ignored; commit only reproducible source and tests.

See [SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md) for exact upstream details and unavoidable environment-version caveats.