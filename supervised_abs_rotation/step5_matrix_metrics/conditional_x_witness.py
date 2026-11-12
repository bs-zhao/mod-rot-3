"""W-only conditional X feature witnesses on the observed joint X design.

This is a finite-sample structural diagnostic, not a calibrated KCI test.
For column j the witness is (phi(Xj)-E[phi(Xj)|X-j]) tensor psi(X-j).
Both embeddings and the conditional mean are numerical RBF approximations.
Only observed X enters bandwidth and regularization choices.
"""
import numpy as np
from scipy.linalg import eigh


def center_gram(gram):
    return gram-gram.mean(0, keepdims=True)-gram.mean(1, keepdims=True)+gram.mean()


def rbf_gram(x):
    x = np.asarray(x, dtype=float)
    distance = np.maximum(np.sum(x*x, 1)[:, None]+np.sum(x*x, 1)[None]-2*x @ x.T, 0.)
    positive = distance[np.triu_indices(len(x), 1)]
    positive = positive[positive > 1e-12]
    length2 = float(np.median(positive)) if len(positive) else 1.
    return np.exp(-distance/(2*length2)), length2


def features(gram, tolerance=1e-12):
    values, vectors = eigh((gram+gram.T)/2)
    peak = max(float(values.max()), np.finfo(float).tiny)
    if float(values.min()) < -1e-8*peak:
        raise ArithmeticError('Conditional witness kernel is not positive semidefinite')
    good = values > tolerance*peak
    return vectors[:, good]*np.sqrt(values[good]), dict(rank=int(good.sum()),
        relative_negative_eigenvalue=max(0., -float(values.min())/peak),
        relative_discarded_trace=float(np.maximum(values[~good], 0.).sum()/max(values.sum(), 1e-30)))


def regression_maps(kernel, train, query, ridges):
    """Dual kernel ridge with an exactly unpenalized intercept."""
    tt = kernel[np.ix_(train, train)]
    qt = kernel[np.ix_(query, train)]
    centered = center_gram(tt)
    query_centered = qt-qt.mean(1, keepdims=True)-tt.mean(0, keepdims=True)+tt.mean()
    values, vectors = eigh((centered+centered.T)/2)
    values = np.maximum(values, 0.)
    scale = max(float(values.mean()), 1e-12)
    left = query_centered @ vectors
    right = vectors.T-vectors.sum(0)[:, None]/len(train)
    return [np.ones((len(query), len(train)))/len(train)+
            (left/(values[None]+ridge*scale)) @ right for ridge in ridges]


def operators(x, seed=420, folds=5, ridges=(1e-4, .001, .01, .1, 1.)):
    """Return factors B_j with B_j.T @ B_j equal to a unit-trace witness Gram.

    F @ B_j.T is a finite observed-design conditional feature covariance.
    It retains context interactions through the tensor product rather than
    testing covariance with Xj alone, which could miss Xj*Xk effects.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < max(folds*2, 10) or not np.isfinite(x).all():
        raise ValueError('Conditional X witnesses require a finite two-dimensional design')
    scale = x.std(0); scale[scale == 0.] = 1.
    z = (x-x.mean(0))/scale
    n, dx = z.shape
    partition = np.array_split(np.random.default_rng(seed).permutation(n), folds)
    result, audits = [], []
    for j in range(dx):
        marginal, marginal_length2 = rbf_gram(z[:, j:j+1])
        phi, marginal_audit = features(center_gram(marginal))
        context, context_length2 = rbf_gram(np.delete(z, j, axis=1))
        residual = phi.copy(); history = []; constant_error = 0.
        for fold, test in enumerate(partition):
            train = np.setdiff1d(np.arange(n), test)
            order = np.random.default_rng(seed+1+fold).permutation(train)
            split = int(.75*len(order)); inner, valid = order[:split], order[split:]
            maps = regression_maps(context, inner, valid, ridges)
            denominator = max(float(np.mean((phi[valid]-phi[inner].mean(0))**2)), 1e-30)
            errors = [float(np.mean((mapping @ phi[inner]-phi[valid])**2))/denominator for mapping in maps]
            chosen = int(np.argmin(errors))
            mapping = regression_maps(context, train, test, (ridges[chosen],))[0]
            residual[test] -= mapping @ phi[train]
            constant_error = max(constant_error, float(np.max(np.abs(mapping.sum(1)-1.))))
            history.append(dict(fold=fold, ridge=ridges[chosen], validation_relative_feature_mse=errors,
                                heldout_X_nodes=len(test)))
        if constant_error > 1e-7:
            raise ArithmeticError('Conditional mean regression lost its unpenalized intercept')
        # Population conditional centering implies ordinary centering too;
        # enforce that finite-sample identity before the context product.
        residual -= residual.mean(0)
        gram = center_gram((residual @ residual.T)*(1.+context))
        witness, audit = features(gram)
        trace = float(np.sum(witness*witness))
        if trace > 0.:
            witness /= np.sqrt(trace)
        result.append(witness.T)
        audits.append(dict(X_column=j, conditional_crossfit_folds=folds, history=history,
            marginal_lengthscale_squared=marginal_length2, context_lengthscale_squared=context_length2,
            marginal_feature_audit=marginal_audit, witness_feature_audit=audit,
            conditional_feature_residual_fraction=float(np.sum(residual**2)/max(np.sum(phi**2), 1e-30)),
            unnormalized_witness_trace=trace, constant_annihilation_error=constant_error,
            final_witness_centering_error=float(np.max(np.abs(witness.sum(0)))) if witness.size else 0.))
    return result, dict(seed=seed, W_only_tuning=True, response_values_used=False,
        standardized_X_columns=True, joint_observed_X_nodes=n, ridges=ridges,
        context_embedding='constant plus joint-other-X RBF', columns=audits,
        interpretation='finite empirical conditional feature covariance; not a calibrated independence test')
