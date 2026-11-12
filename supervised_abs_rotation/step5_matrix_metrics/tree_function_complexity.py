"""Axis-aligned response-function compression at fixed relative distortion.

The tree is a generic piecewise-constant approximant, not a behavioral
equation. Counts describe rule complexity, not a causal UX graph.
"""
import heapq

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def node_risks(estimator, u, target, discarded=None):
    """Exact query squared error if each reached node were made a leaf."""
    target = np.asarray(target, dtype=float)
    discarded = np.zeros(len(target)) if discarded is None else np.asarray(discarded)
    path = estimator.decision_path(u)
    means = estimator.tree_.value[:, :, 0]
    sums = path.T @ target
    squares = path.T @ (np.sum(target*target, 1)+discarded)
    counts = np.asarray(path.sum(0)).ravel()
    error = squares-2*np.sum(means*sums, 1)+counts*np.sum(means*means, 1)
    denominator = float(np.sum(target*target)+discarded.sum())
    if denominator <= 0.:
        raise ValueError('Tree complexity requires nonzero response-function energy')
    if float(error.min()) < -1e-10*denominator:
        raise ArithmeticError('Node prediction risk lost numerical precision')
    return np.maximum(error, 0.)/denominator


def expansion_path(estimator, risks):
    """Best available training-impurity split; query risk never selects splits.

    This is one nested greedy path, not a globally optimal tree search.
    A full grown CART tree makes every available branch explicit. Its
    splits are then expanded in best-first training-risk order.
    """
    tree = estimator.tree_; left, right = tree.children_left, tree.children_right
    weight = tree.weighted_n_node_samples
    raw_risk = np.maximum(tree.impurity, 0.)*weight*tree.n_outputs
    gain = np.zeros(tree.node_count)
    internal = np.flatnonzero(left >= 0)
    gain[internal] = np.maximum(raw_risk[internal]-raw_risk[left[internal]]-raw_risk[right[internal]], 0.)
    frontier = []
    if left[0] >= 0:
        heapq.heappush(frontier, (-float(gain[0]), 0))
    values = np.asarray(risks, dtype=float)
    current = values[:, 0].copy(); history = [current.copy()]; split_nodes = []
    while frontier:
        _, node = heapq.heappop(frontier)
        a, b = int(left[node]), int(right[node])
        current += values[:, a]+values[:, b]-values[:, node]
        history.append(np.maximum(current, 0.).copy()); split_nodes.append(node)
        for child in (a, b):
            if left[child] >= 0:
                heapq.heappush(frontier, (-float(gain[child]), child))
    return np.arange(1, len(history)+1), np.asarray(history).T, np.asarray(split_nodes)


def fit_path(u, target, train, query, discarded=None, seed=421):
    estimator = DecisionTreeRegressor(random_state=seed).fit(u[train], target[train])
    discarded = np.zeros(len(u)) if discarded is None else np.asarray(discarded)
    train_risk = node_risks(estimator, u[train], target[train], discarded[train])
    query_risk = node_risks(estimator, u[query], target[query], discarded[query])
    leaves, errors, splits = expansion_path(estimator, [train_risk, query_risk])
    direct = estimator.predict(u[query]); direct = direct.reshape(len(query), -1)
    expected = (np.sum((direct-target[query])**2)+discarded[query].sum())/(
        np.sum(target[query]**2)+discarded[query].sum())
    if abs(float(errors[1, -1])-float(expected)) > 1e-10:
        raise ArithmeticError('Pruning-path risk differs from explicit full-tree prediction')
    return leaves, errors, dict(full_leaves=estimator.get_n_leaves(), full_depth=estimator.get_depth(),
        train_subjects=len(train), scored_subjects=len(query), minimum_query_relative_mse=float(errors[1].min()),
        full_query_relative_mse=float(expected), full_training_relative_mse=float(errors[0, -1]),
        split_axis=estimator.tree_.feature[splits].tolist(), split_node_ids=splits.tolist())


def minimum_leaves(leaves, errors, relative_l2):
    qualifying = np.flatnonzero(np.asarray(errors) <= relative_l2**2)
    return int(leaves[qualifying[0]]) if len(qualifying) else int(leaves[-1])+1


def scores(counts):
    """counts[threshold,X], with each X target normalized before any fit."""
    out = {}
    for k, tag in enumerate(('010', '030', '100')):
        for pooling, value in (('Xmean', np.mean(counts[k])), ('Xworst', np.max(counts[k]))):
            out[f'UX_function_tree_leaves_relative_L2_eps{tag}_{pooling}'] = float(value)
    return out
