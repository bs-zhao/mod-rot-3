# Recoverability audit (supervised diagnostic only)

Independent directory; does not import or change any existing rotation suite.
Reads a completed `mlp_teacher_rotation` output directory, using its identity
coordinates and saved TEST truth. Requires NumPy and SciPy; no checkpoint,
PyTorch, package installation, or training is needed.

This answers a prerequisite question: **does this fixed latent subspace still
contain the parameter information that a better functional rotation could find?**

It reports:

* Exact empirical unrestricted linear upper bounds (multiple correlation r²).
* A multistart supervised orthogonal reference, explicitly not a proven optimum.
* Repeated two-half fitting/evaluation of the supervised references. Directions
  and parameter assignments are fixed on the fitting half; the evaluation half
  does not reassign them.
* The Gram matrix of normalized linear parameter directions, in the original
  identity coordinate geometry (no extra axis standardization).

These are supervised diagnostics, never proposed unsupervised candidates. They
use already inspected NP TEST labels and do not constitute fresh validation.
Half splits overlap across repeats and must not be treated as independent NP
trainings. A high reference score establishes room for improvement, not that
decoder structure alone identifies the parameter directions.

```cmd
conda activate py11
python rotations/recoverability_audit/run.py --run "outputs/choice/CR3/rotations/mlp_teacher_rotation/CONDITION/repeat01/run_HASH" --out outputs/recoverability_audit/CR3
```

Substitute the actual completed run directory. Outputs remain under `outputs/`:
`audit.json`, `half_split_diagnostics.csv`, `REPORT.md`, and a clearly labeled
`SUPERVISED_DIAGNOSTIC_ONLY.npz`. Do not deploy that transform as a rotation.
