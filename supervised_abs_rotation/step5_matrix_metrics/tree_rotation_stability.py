"""Blind rotations and paired subject refits of a frozen tree criterion."""
import numpy as np
from scipy.linalg import expm
from sklearn.tree import DecisionTreeRegressor

import tree_function_complexity as trees


def coupling_modes(responses, operators, channel=0, tolerance=1e-6):
    f = responses[:, channel]
    f = f-f.mean(0, keepdims=True)-f.mean(1, keepdims=True)+f.mean()
    targets, residuals = [], []
    for operator in operators:
        values = f @ operator.T
        norm = np.sqrt(np.mean(np.sum(values*values, axis=1)))
        if norm <= 0.:
            raise ValueError('A conditional X witness has zero coupling energy')
        values /= norm
        left, singular, right = np.linalg.svd(values, full_matrices=False)
        rank = min(int(np.searchsorted(np.cumsum(singular**2)/np.sum(singular**2), 1-tolerance))+1, len(singular))
        target = left[:, :rank]*singular[:rank]
        targets.append(target); residuals.append(values-target @ right[:rank])
    return targets, residuals


def sample_functions(targets, residuals, subject_ids):
    sampled, discarded = [], []
    for target, residual in zip(targets, residuals):
        y, r = target[subject_ids].copy(), residual[subject_ids].copy()
        # Refit the pure-X mean under exactly this empirical subject draw.
        y -= y.mean(0); r -= r.mean(0)
        sampled.append(y); discarded.append(np.sum(r*r, axis=1))
    return sampled, discarded


def training_count(u, target, discarded, relative_l2=.1, seed=421):
    estimator = DecisionTreeRegressor(random_state=seed).fit(u, target)
    tree = estimator.tree_
    # These are the exact empirical training sums of squared output errors.
    risk = np.maximum(tree.impurity, 0.)*tree.weighted_n_node_samples*target.shape[1]
    risk += estimator.decision_path(u).T @ discarded
    denominator = float(np.sum(target*target)+discarded.sum())
    leaves, errors, _ = trees.expansion_path(estimator, [risk/denominator])
    return trees.minimum_leaves(leaves, errors[0], relative_l2)


def canonical_rotation(z, q):
    import cp_regression as cp
    classes, _ = cp.canonical_classes(z, [q])
    return classes[0]


def score(z, q, targets, discarded, relative_l2=.1):
    q = canonical_rotation(z, q)
    u = (z @ q).astype(np.float32)
    counts = [training_count(u, target, tail, relative_l2) for target, tail in zip(targets, discarded)]
    return float(max(counts)), counts


def random_rotation(d, rng):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    q *= np.sign(np.diag(r))[None]
    if np.linalg.det(q) < 0.:
        q[:, 0] *= -1.
    return q


def increment(parameters, d):
    a = np.zeros((d, d)); upper = np.triu_indices(d, 1)
    a[upper] = parameters; a -= a.T
    return expm(a)
