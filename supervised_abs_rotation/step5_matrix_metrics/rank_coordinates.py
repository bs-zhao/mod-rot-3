"""W-only monotone marginal coordinate conventions; no independence model."""
import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri


def transform(z, method='identity'):
    if method == 'identity':
        return z
    ranks = np.column_stack([rankdata(z[:, i], method='average') for i in range(z.shape[1])])
    quantiles = (ranks-.5)/len(z)
    if method == 'normal_rank':
        return ndtri(quantiles)
    if method == 'uniform_rank':
        return (2*quantiles-1)*np.sqrt(3.)
    raise ValueError('Unknown marginal coordinate convention')
