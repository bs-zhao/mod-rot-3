# Rotation module (post-hoc; independent of baseline)

Use `python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --method kurtosis` after running baseline S4 extraction.

All methods have the same signature:

```python
def objective(q_fit, original_X, original_Y, trial_mask_or_counts, R):
    u_fit = q_fit @ R.T
    return scalar_complexity
```

`R` is constrained to SO(k), and `U = (Z - mean_train) W.T R.T`. The equivalent affine transform in original Z coordinates is saved as `A = R W` and `offset = -A mean_train`.

Included baseline controls:
- identity — no extra orthogonal rotation of PCA-whitened q
- quartic — maximize fourth-moment concentration
- kurtosis — maximize squared excess-kurtosis concentration

These are not claims of parameter recovery; future UXY interaction/functional modules should be added here, never inside `baseline/`. The optimizer cannot inspect theta. Only after saving R does it run Hungarian-matched heldout TEST parameter recovery diagnostics.