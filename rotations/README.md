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

## Functional composition research suite

[`functional_composition/`](functional_composition/README.md) is **one self-contained rotation experiment** containing 20 operationalized functional-complexity diagnostics and a joint loss (operator-module codelength + effect-fingerprint rank + contextual gradient structure). Run **only** its own entrypoint:

```cmd
python rotations/functional_composition/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
```

Its own folder contains `surrogate.py`, `metrics.py`, `mdl.py`, `optimizer.py`, `diagnostics.py`, `experiment.py`, `runner.py`, `run.py`, `config.json`, and tests. It neither imports nor runs other rotation suites. Not all 20 indicators are exact population statistics; the suite documents which are estimators, proxies, and negative controls.


## Effect fingerprint rank suite

[`effect_fingerprint/`](effect_fingerprint/README.md) is an independent full experiment to validate metric 11 under multiple rank definitions, channel metrics and FIT-only resampling. It has its own run.py, surrogate, loss, optimizer, diagnostics and tests. No other rotation suite is executed.

## Stimulus-contrast covariance investigation

[`stimulus_contrast/`](stimulus_contrast/README.md) contains a standalone frozen-decoder rotation and controlled diagnostics. Exact counterexamples reject it as a general recovery criterion; the source and fixed configuration are retained for reproducibility. See [`research/RESEARCH_LOG_2026-11-06.md`](../research/RESEARCH_LOG_2026-11-06.md) for evidence, including stimulus-basis dependence and surrogate fidelity.
