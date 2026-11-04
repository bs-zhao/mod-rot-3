"""Example true Z/X/Y objective: sparse axis-level stimulus-response effects.

Experimental: demonstrates preprocessing inside a standalone plugin. There is
no claim of identifying true psychological parameters.
"""
import numpy as np
from rotations.api import optimize_objective

NAME = 'xy_effect'
VERSION = '1.0'
DESCRIPTION = 'Group-sparsify latent axes of empirical X-dependent response moments.'
DEFAULTS = {'starts': 4, 'maxiter': 200, 'seed': 31, 'optimizer': 'Powell',
            'ridge': 1e-4}


def fit(data, config):
    valid = data.mask
    count = valid.sum(axis=1)
    x = np.asarray(data.x, dtype=float)
    y = np.asarray(data.y, dtype=float)
    if x.shape[-1] == 0:
        raise ValueError('xy_effect requires at least one stimulus feature')
    avg = x[valid].mean(axis=0)
    std = np.maximum(x[valid].std(axis=0), 1e-6)
    xs = (x - avg) / std
    # Moment per subject and stimulus feature, conditional on their own choices.
    features = ((2 * y - 1)[:, :, None] * xs * valid[:, :, None]).sum(axis=1) / count[:, None]
    features -= features.mean(axis=0)
    q = data.q - data.q.mean(axis=0)
    operator = (q.T @ features) / data.n_subjects
    eps = float(config['ridge'])
    if eps <= 0:
        raise ValueError('ridge must be positive')

    def loss(_, R, cfg):
        transformed = R @ operator
        return float(np.sum(np.sqrt(np.sum(transformed ** 2, axis=1) + eps ** 2)))

    result = optimize_objective(data, loss, config)
    result.info['moment_operator_shape'] = list(operator.shape)
    result.extra_arrays['stimulus_response_operator'] = operator
    return result