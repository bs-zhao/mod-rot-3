# Independently extensible ZXY rotation laboratory

This directory is deliberately independent of `baseline/`. It never retrains a
Neural Process or changes a simulation model. The only optimizer inputs are
**FIT subject** frozen PCA-whitened `q`, aligned observed `X`, binary `Y`, and
`mask`. The known simulator parameters `theta` are not included in the plugin
API; they are read only **after the fitted R and U are written to disk** for
separate held-out TEST recovery heatmaps.

## One algorithm = one file

```text
rotations/
  run.py                   # universal runner (never edit to register methods)
  methods/
    identity.py             # independent no-rotation baseline
    quartic.py              # fourth moment objective
    kurtosis.py             # ICA-like kurtosis objective
    xy_effect.py            # example real X/Y-sensitive objective
    _template.py            # copy to a new filename (not registered)
  plans/
    example.json            # run many independent methods/configurations
src/modrot3/
  rotation_api.py           # stable theta-free RotationInput and optimizer
  rotation_engine.py        # automatic loader, provenance and batch runner
  rotation_data.py          # subject alignment and TEST-only evaluation
outputs/choice/MODEL/rotations/   # ignored by git
```

No central `OBJECTIVES` map is maintained. Plugins are discovered dynamically by
scanning `rotations/methods/*.py` (plus optional `--plugin-dir`). Filenames
starting with `_` are not registered. Only install plugins you trust: Python
modules can execute arbitrary local code and **are not a security sandbox**.

### A simple method (only objective)

Create `rotations/methods/my_nonlinear.py`:

```python
NAME = 'my_nonlinear'
VERSION = '1.0'
DESCRIPTION = 'One independent definition of interaction complexity.'
DEFAULTS = {'starts': 4, 'maxiter': 200, 'seed': 31,
            'optimizer': 'Powell', 'alpha': 0.1}

def objective(data, R, config):
    u = data.rotated(R)        # q @ R.T, shape N x K
    x, y, mask = data.x, data.y, data.mask
    # Implement real finite scalar complexity from U, X, Y, mask.
    return float(config['alpha'] * (u ** 4).mean())
```

The generic multistart optimizer automatically searches **SO(K)** with the
exponential of a skew-symmetric matrix, validates orthogonality and saves
standard artifacts. All allowed config keys must be declared in `DEFAULTS`.

### A complex method (completely custom fit)

If a method needs pretrained surrogates, symbolic regression, differentiable
optimization, multiple stages, SciPy alternatives, or its own cache, supply
instead:

```python
from modrot3.rotation_api import RotationResult

def fit(data, config):
    # Only data.q, data.x, data.y, data.mask are available from the engine.
    R = ...     # your independent optimization, any dependencies you choose
    return RotationResult(R=R,
                          info={'fit_loss': 0.23, 'custom': 'diagnostics'},
                          extra_arrays={'some_operator': ...})
```

`fit` overrides `objective` when both exist. Returned `R` **must** be KxK,
finite, orthogonal and `det(R)=+1`. You can call `optimize_objective()` from
`modrot3.rotation_api` inside your method, but you do not have to. Independent
methods may import different third-party libraries; a missing dependency in
one plugin does not prevent unrelated methods from running.

### Explore and run

```cmd
python rotations/run.py --list-methods
python rotations/run.py --describe-method xy_effect
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --method identity
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --method kurtosis --starts 8 --maxiter 300
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --methods kurtosis quartic xy_effect
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --methods all --continue-on-error
```

For a method-specific hyperparameter without touching the runner:

```cmd
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --method xy_effect --set xy_effect.ridge=0.001 --set xy_effect.starts=8
```

For many parameter combinations, use a JSON array with `{method,config}`:

```cmd
python rotations/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --plan rotations/plans/example.json --continue-on-error
```

You can add an entire external method directory via `--plugin-dir D:\rotations_extra`.
A JSON config file may also contain `{"defaults":{...}, "methods":{"name":{...}}}`.
`--dry-run` lists experiments without loading ZXY. `--no-evaluate` runs the
unsupervised fit without reading theta. `--summary` rebuilds aggregated CSV
from already completed TEST evaluations without fitting anything.

### Results and provenance

Each independent run goes to:

`outputs/choice/<MODEL>/rotations/<METHOD>/<CONDITION>/repeat01/run_<sha16>/`

- `rotation__matrix.npz`: orthogonal R, original-Z affine map A, offset, PCA
  mean/eigenvectors, FIT subject IDs, and source fingerprints.
- `rotation__U.npz`: *all subjects'* rotated U and immutable split IDs.
- `rotation__fit.json`: method name, version, source-file SHA256, complete
  hyperparameter dictionary, source-data hashes, optimizer diagnostics.
- `rotation__method_arrays.npz`: optional method-specific operators, tensors etc.
- `rotation__test_evaluation.json`: separate postfit matched r² and leakage.
- `rotation__test_corr2.png`: TEST theta-vs-U correlation matrix (parameter
  rows, latent-axis columns, Hungarian matched cells outlined).
- `rotation__summary.csv` under the model's `rotations/` folder summarizes
  all existing experiments including multiple versions and hyperparameters.

The run ID fingerprints method source, method version, parameters, model,
representation, FIT IDs, seed, Z and dataset SHA256. Different method settings
cannot silently overwrite one another. By default exact duplicate jobs are
**skipped**; `--overwrite` explicitly replaces only the same fingerprint.

### Important research boundaries

- No theta in optimization, hyperparameter selection, loss evaluation or any
  plugin input. Never select a method using TEST theta and claim it was
  selected without supervision.
- PCA whitening is fit using TRAIN subjects. All method scores use the *same*
  FIT-subject inputs and obey original varying trial lengths (mask padding).
- The X/Y-sensitive example is only a demonstration; no promise of recovering
  the true parameters. Function simple does not necessarily mean theta-aligned.
- All methods are SO(k) rotations of an already chosen PCA-whitened subspace.
  The nonorthogonal full mapping from raw Z is exported as A plus offset.
- Different method code can have different computational costs and dependencies;
  plugin runs should be compared on identical source hashes and data splits.