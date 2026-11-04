# mod-rot-3

The repository has **two physically separate research modules**:

```text
mod-rot-3/
  np/
    run.py                     # generate/precheck/train/evaluate-np/extract/analyze
    src/modlearn_rot/          # original generators, NP, trainer, prechecks
    src/modrot3/              # pipeline, PCA/Ridge, reporting
    reference/                 # original CR3 CPT/prior/fixed bank/oracle
    tests/                     # tests only for NP
    README.md
  rotations/
    run.py                     # shared CLI for any number of methods
    api.py                     # ZXY/R contracts
    engine.py                  # discovery and reproducible output manager
    data.py                    # X/Y alignment and postfit test evaluation
    identity/                  # EACH rotation method gets a complete folder
      __init__.py
      config.json
      README.md
      test_method.py
    kurtosis/
    quartic/
    xy_effect/
    ...                        # arbitrary independent methods and submodules
    template_method.py
    plan_example.json
    README.md
  outputs/                     # all generated artifacts, ignored by Git
  pyproject.toml
```

There is no root `src/`, `reference/`, `baseline/`, or `tests/`
folder. New rotation methods never require changes to `np/`.

## Installation

```cmd
cd /d D:\Projects\mod-rot-3
pip install -e ".[dev]"
pytest -q
```

## NP workflow

```cmd
python np/run.py selftest
python np/run.py all --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --device auto
```

The `np/` folder contains all four binary-choice generators, original
prechecks and Neural Process source, checkpoint evaluation, full-trial Z extraction,
PCA and held-out linear regression. See [NP guide](np/README.md).

## Independent rotation methods

```cmd
python rotations/run.py --list-methods
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --method xy_effect
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --methods all --continue-on-error
```

Every method lives exclusively inside its own `rotations/<method>/` folder,
including implementation, optimization, configuration, tests and local docs.
See [Rotation guide](rotations/README.md). True theta is never provided to
the unsupervised rotation method and appears only in postfit TEST evaluations.

All data, checkpoints, figures, and rotation results are saved to the root
`outputs/`, excluded through `.gitignore`.
