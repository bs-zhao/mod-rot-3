# One folder = one complete rotation method

```text
rotations/
  run.py              # common batch CLI
  api.py              # frozen-ZXY contract and optional generic optimizer
  engine.py           # dynamic plugin discovery + saving R, A, U, results
  data.py             # subject alignment, postfit parameter evaluation
  identity/
    __init__.py       # independently implemented rotation
    config.json       # this method's parameters
    README.md         # documentation
    test_method.py    # this method's tests
  kurtosis/           # same self-contained structure
  quartic/            # same self-contained structure
  xy_effect/          # same self-contained structure
  ...                 # add as many independent algorithms as needed
  template_method.py  # example (not registered)
  plan_example.json
  test_platform.py
  test_methods.py
```

There is **no intermediate `methods/` directory** and no central method
registry. Add `rotations/new_algorithm/__init__.py`, local config, README,
tests, helper estimators, optimization algorithms or model files **all in that
single directory**. The shared runner discovers it automatically.

Each method supplies `NAME='new_algorithm'`, `VERSION`, `DEFAULTS` and
either `objective(data,R,config)` (generic multistart SO(K) optimization) or
`fit(data,config)` (fully customized optimization). `data.q` is whitened
frozen Z for FIT subjects; `data.x`, `data.y` and `data.mask` contain
subject-aligned X and binary Y. The method NEVER receives theta.

You may import `RotationResult` or `optimize_objective` from `rotations.api`,
or implement an entirely independent estimator and optimizer. Local relative
imports of arbitrary auxiliary modules work inside the method package.
`config.json` (optional) overlays local `DEFAULTS`; CLI `--set` and
experiment-plan overrides take precedence. The loader hashes every Python file
inside a method; changes create a new run fingerprint.

```cmd
python rotations/run.py --list-methods
python rotations/run.py --describe-method xy_effect
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --method xy_effect
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --method xy_effect --set xy_effect.ridge=0.001
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --methods all --continue-on-error
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --plan rotations/plan_example.json
pytest -q rotations
```

The runner saves `R`, the original-Z-to-U affine transformation `A`, U for
all subjects, a config/source/code fingerprint, and postfit TEST theta recovery
with matched and off-target correlations in ignored root
`outputs/choice/<MODEL>/rotations/<METHOD>/...`.
Untrusted Python plugins are NOT a sandbox.
