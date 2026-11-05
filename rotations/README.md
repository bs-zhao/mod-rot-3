# Rotation experiment suites (fully independent)

**One subfolder is one complete rotation suite**, not one metric. A suite may
contain several interacting metrics, loss terms, optimizers, model fitting
stages and plots, all belonging to one experiment. The user runs that suite's
own `run.py`. Suites never import or run one another.

```text
rotations/
  identity/
    run.py
    __init__.py       # this suite's fit/objective
    experiment.py     # this suite's ZXY loading, fitting and result saving
    api.py            # this suite's local rotation optimizer/contract
    data.py           # local subject alignment and TEST evaluation
    config.json
    test_method.py
    README.md
  kurtosis/
    ...
  quartic/
    ...
  xy_effect/
    ...
  future_composite_suite/
    run.py
    __init__.py
    metrics.py
    optimization.py
    diagnostics.py
    experiment.py
    config.json
    README.md
```

No global `rotations/run.py`, no registry, no batch launcher, and no
`--methods all` command. There is no shared rotation API or engine outside
these package folders. Each suite keeps its own copy of I/O and evaluation
helpers so it can evolve without modifying other suites.

Only S5 frozen full-trial latent Z, S1 observed X and binary Y enter a fit.
Parameters theta are accessed after R and U are persisted for TEST diagnostics.

## Run exactly one selected suite

```cmd
python rotations/identity/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
python rotations/kurtosis/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --starts 8 --maxiter 300
python rotations/xy_effect/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --set ridge=0.001
```

Use `--representation full --rotation-dim 8` to retain all 8 NP dimensions.
Every suite writes under ignored `outputs/choice/<MODEL>/rotations/<SUITE>/`
with a fingerprint for the data, source code and hyperparameters.
