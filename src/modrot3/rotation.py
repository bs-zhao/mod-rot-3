"""Backward-compatible aliases for the original mod-rot-3 rotation entrypoint.

The active experiment platform is `modrot3.rotation_engine` with independently
loadable files in `rotations/methods/`. Do not add new methods to this module.
"""
from __future__ import annotations

from .rotation_api import RotationInput, skew_matrix, validate_result, optimize_objective
from .rotation_data import aligned_observations, corr, evaluate_after_freeze
from .rotation_engine import main, discover, BUILTINS, _invoke


def optimize(q, x, y, mask, method, starts=4, maxiter=200, seed=31):
    """Compatibility wrapper for older code expecting (R, dict)."""
    import numpy as np
    if x is None or y is None or mask is None:
        if method != 'identity' and method not in ('kurtosis', 'quartic'):
            raise ValueError('ZXY method requires X/Y/mask')
        x = np.zeros((len(q), 1, 1), dtype=float)
        y = np.zeros((len(q), 1), dtype=int)
        mask = np.ones_like(y, dtype=bool)
    plugins, failed = discover([BUILTINS])
    if method not in plugins:
        raise KeyError(f'Unknown method {method!r}: {failed.get(method, "not installed")}')
    plugin = plugins[method]
    cfg = dict(plugin.defaults)
    for key, value in [('starts', starts), ('maxiter', maxiter), ('seed', seed)]:
        if key in cfg:
            cfg[key] = value
    result = _invoke(plugin, RotationInput(np.asarray(q), np.asarray(x),
                                           np.asarray(y), np.asarray(mask)), cfg)
    return result.R, result.info


if __name__ == '__main__':
    main()