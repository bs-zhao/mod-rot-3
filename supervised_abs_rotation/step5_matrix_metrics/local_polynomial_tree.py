"""Piecewise complete-polynomial response functions on W-only U coordinates.

CART proposes axis-aligned partitions. A complete polynomial is fitted at
every node by SVD least squares. The compression path expands nodes by
training polynomial error reduction, never query error or case labels.
"""
import heapq

import numpy as np
from scipy.linalg import lstsq
from sklearn.tree import DecisionTreeRegressor

import run as wave


def local_fit(train_u, train_y, query_u, powers, weights):
    mean = train_u.mean(0)
    scale = max(float(np.sqrt(np.var(train_u, axis=0).mean())), 1e-12)
    design = wave.sur.monomials((train_u-mean)/scale, powers, weights)
    query = wave.sur.monomials((query_u-mean)/scale, powers, weights)
    coef, _, rank, singular = lstsq(design, train_y, cond=1e-10, lapack_driver='gelsd')
    return design @ coef, query @ coef, dict(rank=int(rank), features=len(powers),
        relative_smallest_singular=float(singular[-1]/singular[0]) if len(singular) else 0.)


def distortion_path(estimator, risks):
    tree = estimator.tree_; left, right = tree.children_left, tree.children_right
    values = np.asarray(risks)
    internal = np.flatnonzero(left >= 0)
    gains = np.zeros(tree.node_count)
    gains[internal] = values[0, internal]-values[0, left[internal]]-values[0, right[internal]]
    if gains.min() < -1e-8:
        raise ArithmeticError('Nested polynomial partitions increased training error')
    available = []
    if left[0] >= 0:
        heapq.heappush(available, (-float(gains[0]), 0))
    current = values[:, 0].copy(); history = [current.copy()]; splits = []
    while available:
        _, node = heapq.heappop(available)
        a, b = int(left[node]), int(right[node])
        current += values[:, a]+values[:, b]-values[:, node]
        splits.append(node); history.append(np.maximum(current, 0.).copy())
        for child in (a, b):
            if left[child] >= 0:
                heapq.heappush(available, (-float(gains[child]), child))
    return np.arange(1, len(history)+1), np.asarray(history).T, splits


def fit_path(u, target, train, query, discarded=None, seed=424, degree=2, leaf_ratio=2):
    u, target = np.asarray(u), np.asarray(target)
    discarded = np.zeros(len(u)) if discarded is None else np.asarray(discarded)
    powers = wave.sur.powers_for(u.shape[1], degree); weights = wave.sur.weights_for(powers)
    minimum_leaf = max(len(powers)*leaf_ratio, 2)
    estimator = DecisionTreeRegressor(min_samples_leaf=minimum_leaf, random_state=seed).fit(u[train], target[train])
    train_path = estimator.decision_path(u[train]).tocsc()
    query_path = estimator.decision_path(u[query]).tocsc()
    risks = np.zeros((2, estimator.tree_.node_count)); minimum_rank = len(powers)
    smallest = 1.; minimum_rows = len(train)
    normalizers = [float(np.sum(target[rows]**2)+discarded[rows].sum()) for rows in (train, query)]
    if min(normalizers) <= 0.:
        raise ValueError('A response-function task has zero energy')
    for node in range(estimator.tree_.node_count):
        tr = train_path.indices[train_path.indptr[node]:train_path.indptr[node+1]]
        qr = query_path.indices[query_path.indptr[node]:query_path.indptr[node+1]]
        train_rows, query_rows = train[tr], query[qr]
        a, b, audit = local_fit(u[train_rows], target[train_rows], u[query_rows], powers, weights)
        minimum_rank = min(minimum_rank, audit['rank']); minimum_rows = min(minimum_rows, len(tr))
        smallest = min(smallest, audit['relative_smallest_singular'])
        for k, rows, prediction in ((0, train_rows, a), (1, query_rows, b)):
            risks[k, node] = (np.sum((prediction-target[rows])**2)+discarded[rows].sum())/normalizers[k]
    leaves, errors, splits = distortion_path(estimator, risks)
    return leaves, errors, dict(degree=degree, polynomial_features=len(powers), minimum_leaf_rows=minimum_rows,
        leaf_ratio=leaf_ratio, minimum_node_design_rank=minimum_rank, minimum_relative_singular=smallest,
        full_leaves=estimator.get_n_leaves(), full_depth=estimator.get_depth(), train_subjects=len(train),
        scored_subjects=len(query), minimum_query_relative_mse=float(errors[1].min()),
        full_query_relative_mse=float(errors[1, -1]), full_training_relative_mse=float(errors[0, -1]),
        root_training_relative_mse=float(errors[0, 0]), root_query_relative_mse=float(errors[1, 0]),
        split_axis=estimator.tree_.feature[splits].tolist(), split_node_ids=splits)
