# mod-rot-3

The two independent parts of the project are:

- [np/](np/README.md): **seven manually executed stages** for C1/C2/C3/CR3
generation, precheck, NP training, prediction evaluation, full-trial Z,
PCA and parameter regression.
- [rotations/](rotations/README.md): **one folder per complete rotation
suite**, with its own run.py, all of its metrics and training/optimization
steps, configuration, tests, post-fit evaluation and output handling.

There is **no** command to execute all rotation suites. Each suite runs by
calling its own `run.py` directly. No central method registry, shared rotation
engine or batch launcher exists.

```cmd
cd /d D:\Projects\mod-rot-3
git pull --ff-only
pip install -e ".[dev]"
python rotations/xy_effect/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
```

All data, models, rotations and plots are saved to the root ignored
`outputs/` directory. Rotation optimization sees only frozen ZXY; theta
is used strictly afterward for held-out TEST diagnostics.
