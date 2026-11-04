"""Identity (PCA-whitened coordinate) baseline; no optimization."""
import numpy as np
from rotations.api import RotationResult

NAME = 'identity'
VERSION = '1.0'
DESCRIPTION = 'No additional rotation; exact PCA-whitened baseline.'
DEFAULTS = {}


def fit(data, config):
    return RotationResult(R=np.eye(data.k), info={'fit_loss': 0.0, 'optimizer': 'none'})