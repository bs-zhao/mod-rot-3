# Neural Process: one self-contained research pipeline

Everything related to binary-choice generation, original precheck diagnostics,
Neural Process construction, training, checkpoint evaluation, full-trial Z extraction,
PCA and linear parameter recovery is located **within this `np/` directory**.

```text
np/
  run.py                    # one CLI for all six stages
  src/modlearn_rot/         # original model generators, prechecks, NP and trainer
  src/modrot3/              # orchestration, evaluation, all-trial Z, PCA and Ridge
  reference/                # unchanged CR3 CPT formula, prior, stimulus bank, oracle
  tests/                    # NP/data/precheck/unit-training tests
  SOURCE_PROVENANCE.md
```

The original C1/C2/C3 simulator and discrete-choice NP source code were moved,
not rewritten. CR3 uses the original 500-trial stimulus bank, CPT parameter
prior and sampling procedure; only the repository-relative import/root paths
were adjusted.

Run from repository root:

```cmd
python np/run.py selftest
python np/run.py generate --model CR3
python np/run.py precheck --model CR3
python np/run.py train --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --batch-size 128 --context-min 0.25 --context-max 0.80 --learning-rate 0.0005 --max-epochs 500 --patience 50
python np/run.py evaluate-np --model CR3 --beta 0 --latent-dim 8
python np/run.py extract --model CR3 --beta 0 --latent-dim 8
python np/run.py analyze --model CR3 --beta 0 --latent-dim 8
```

Or use `python np/run.py all ...` with the same training hyperparameters.
All NP outputs go to **repository-root `outputs/choice/<MODEL>/`**, excluded
from Git. S4 encodes ALL observed trials, saving one Z per subject. PCA is fitted
to TRAIN subjects; Ridge selection uses VALIDATION, and TEST reports are held out.
The separate `rotations/` module only consumes this frozen ZXY dataset.
