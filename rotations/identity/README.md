# identity

Identity reference: no additional orthogonal transformation of PCA-whitened Z.

This **entire algorithm** lives in this folder: `__init__.py` defines the method and its objective/optimizer; `config.json` holds local hyperparameters; `test_method.py` tests the method. Add any estimators, plots, alternative optimizers, or mathematical documentation here as separate files.

Run:

```cmd
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --method identity
```

The shared runner fits only on frozen ZXY; theta is used only afterward for TEST diagnostics.
