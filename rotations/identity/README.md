# identity: independent rotation suite

This folder contains the complete **identity** experiment. It imports no other
rotation suite and has its own execution entrypoint, objective(s), local
optimizer/API, input/output handling, hyperparameters and tests.

Run from the repository root:

```cmd
python rotations/identity/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
```

Change only this suite's `config.json`, or pass `--set KEY=VALUE`.
For iterative optimizers you may use `--starts 8 --maxiter 300`.

The suite fits only frozen NP TRAIN ZXY and saves R/A/U; known theta is
accessed only afterward for held-out TEST recovery diagnostics. Results are
written under ignored `outputs/choice/MODEL/rotations/identity/`.

A suite may contain any number of complementary metrics, optimization stages,
surrogate models or other helper files, but they form **one** objective/rotation
experiment controlled by this folder's `run.py`.
