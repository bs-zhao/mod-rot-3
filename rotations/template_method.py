"""Copy to `my_method.py`; underscore-prefixed files are NOT registered.

You can define `objective(data, R, config)` (simple method; common optimizer)
OR `fit(data, config) -> RotationResult` (fully custom optimization).
`data` contains ONLY FIT q, X, Y, mask; it deliberately has no theta.
"""

NAME = 'my_method'
VERSION = '1.0'
DESCRIPTION = 'What the loss measures and why it could prefer simpler action.'
DEFAULTS = {'starts': 4, 'maxiter': 200, 'seed': 31, 'optimizer': 'Powell', 'strength': 1.0}


def objective(data, R, config):
    u = data.rotated(R)  # data.q @ R.T
    # Implement your scalar complexity. It must return a finite float.
    return float(config['strength'] * (u * u).mean())