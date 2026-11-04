"""Independent distribution-only ICA-like kurtosis rotation baseline."""
import numpy as np

NAME = 'kurtosis'
VERSION = '1.0'
DESCRIPTION = 'Maximize sum of squared excess kurtoses of latent axes.'
DEFAULTS = {'starts': 4, 'maxiter': 200, 'seed': 31, 'optimizer': 'Powell'}


def objective(data, R, config):
    u = data.rotated(R)
    u = u - u.mean(0)
    var = np.maximum(np.mean(u**2, axis=0), 1e-10)
    k4 = np.mean(u**4, axis=0) / (var**2) - 3.0
    return -float(np.sum(k4**2))