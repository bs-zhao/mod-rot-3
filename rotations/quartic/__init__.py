"""Independent fourth-moment concentration rotation baseline."""
import numpy as np

NAME = 'quartic'
VERSION = '1.0'
DESCRIPTION = 'Maximize raw fourth moment summed across whitened latent axes.'
DEFAULTS = {'starts': 4, 'maxiter': 200, 'seed': 31, 'optimizer': 'Powell'}


def objective(data, R, config):
    return -float(np.mean(data.rotated(R)**4, axis=0).sum())