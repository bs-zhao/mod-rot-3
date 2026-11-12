"""W-only conditional Ui feature innovations, with optional other-U context."""
import numpy as np
from scipy.linalg import eigh, null_space
from scipy.stats import rankdata

import conditional_ux as conditional


def nystrom(x, centers, seed):
    x = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    sample = x[rng.choice(len(x), min(512, len(x)), replace=False)]
    distances = np.maximum(np.sum(sample*sample, 1)[:, None]+np.sum(sample*sample, 1)[None]-2*sample @ sample.T, 0.)
    positive = distances[np.triu_indices(len(sample), 1)]
    positive = positive[positive > 1e-12]
    length2 = float(np.median(positive)) if len(positive) else 1.
    ids = rng.choice(len(x), min(centers, len(x)), replace=False)
    distance = np.maximum(np.sum(x*x, 1)[:, None]+np.sum(x[ids]**2, 1)[None]-2*x @ x[ids].T, 0.)
    kernel = np.exp(-distance/(2*length2)); cc = kernel[ids]
    values, axes = eigh((cc+cc.T)/2)
    good = values > max(float(values.max())*1e-10, 1e-12)
    features = kernel @ (axes[:, good]/np.sqrt(values[good]))
    a, b = rng.integers(len(x), size=(2, 2048))
    truth = np.exp(-np.sum((x[a]-x[b])**2, axis=1)/(2*length2))
    prediction = np.sum(features[a]*features[b], axis=1)
    return features, dict(centers=len(ids), rank=int(good.sum()), lengthscale=float(np.sqrt(length2)),
        sampled_kernel_relative_squared_error=float(np.sum((truth-prediction)**2)/max(np.sum(truth*truth), 1e-30)))


def features(z, direction, seed=430, centers=16, folds=5):
    coordinate = z @ direction
    p = (rankdata(coordinate, method='average')-.5)/len(z)
    marginal, ma = nystrom(p[:, None], centers, seed)
    marginal -= marginal.mean(0)
    rest = z @ null_space(np.asarray(direction)[None])
    context, ca = nystrom(rest, centers, seed+1)
    context -= context.mean(0)
    context /= max(float(np.mean(np.sum(context*context, axis=1)))**.5, 1e-30)
    design, da = conditional.complement_design(z, direction, degree=3, centers=64,
        lengthscale=ca['lengthscale'], seed=seed+2)
    partition = np.array_split(np.random.default_rng(seed+3).permutation(len(z)), folds)
    residual = marginal.copy(); history = []
    ridges = (1e-8, 1e-6, 1e-4, .01, .1)
    for k, query in enumerate(partition):
        train = np.setdiff1d(np.arange(len(z)), query)
        order = np.random.default_rng(seed+4+k).permutation(train)
        cut = int(.75*len(train)); inner, valid = order[:cut], order[cut:]
        predictions = conditional.regressions(design, marginal, inner, valid, da['linear_count'], ridges)
        normalizer = max(float(np.mean((marginal[valid]-marginal[inner].mean(0))**2)), 1e-30)
        errors = [float(np.mean((v-marginal[valid])**2))/normalizer for v in predictions]
        chosen = int(np.argmin(errors))
        predicted = conditional.regressions(design, marginal, train, query, da['linear_count'], (ridges[chosen],))[0]
        residual[query] -= predicted
        history.append(dict(fold=k, ridge=ridges[chosen], W_only_validation_relative_feature_mse=errors))
    residual -= residual.mean(0)
    context = np.column_stack((np.ones(len(z)), context))
    joined = (residual[:, :, None]*context[:, None]).reshape(len(z), -1)
    result = {}
    for name, value in (('Ui_innovation', residual), ('Ui_innovation_with_context', joined)):
        scale = float(np.mean(np.sum(value*value, axis=1)))**.5
        result[name] = value/max(scale, 1e-30)
    audit = dict(all_subjects=len(z), W_only_tuning=True, response_values_used=False,
        scalar_embedding='RBF of marginal midrank, with median-distance bandwidth',
        marginal_kernel=ma, context_kernel=ca, conditional_design=da, crossfit_history=history,
        residual_feature_fraction=float(np.sum(residual**2)/max(np.sum(marginal**2), 1e-30)),
        context_interpretation='constant plus centered joint-other-U radial features')
    return result, audit


def moments(feature, target):
    """Squared conditional feature covariance and its diagonal-removed version.

    Crossfitted/empirically centered features are dependent across subjects,
    so the diagonal removal is not claimed to be exactly unbiased here.
    """
    n = len(feature); cross = feature.T @ target
    raw = float(np.sum(cross*cross))/(n*n)
    diagonal = float(np.sum(np.sum(feature*feature, axis=1)*np.sum(target*target, axis=1)))
    return raw, (float(np.sum(cross*cross))-diagonal)/(n*(n-1))
