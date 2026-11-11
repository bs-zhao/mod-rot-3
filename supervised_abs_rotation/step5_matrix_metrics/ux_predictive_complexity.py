"""UX sparsity: U-ANOVA of cross-fitted leave-X-column-out response contrasts.

X uses its recorded joint design, not independent marginal resampling.
U references are explicitly local isotropic Gaussians within each anchor.
"""
import numpy as np
from scipy.linalg import solve

import run as wave
import kernel_anova as ka
import tail_support as ts


def regression_map(design_train, design_query, linear_count, ridge):
    """Unpenalized affine X span plus regularized nonlinear features."""
    l, n = design_train[:, :linear_count], design_train[:, linear_count:]
    lq, nq = design_query[:, :linear_count], design_query[:, linear_count:]
    inverse = np.linalg.pinv(l, rcond=1e-10)
    correction = inverse @ n
    residual = n-l @ correction
    gram = residual.T @ residual
    scale = max(float(np.trace(gram))/max(len(gram), 1), 1e-12)
    beta = solve(gram+ridge*scale*np.eye(len(gram)), residual.T, assume_a='pos')
    return lq @ inverse+(nq-lq @ correction) @ beta


def omission_operators(x, scale, response_bank, train_subjects, seed=405, folds=5, degrees=(2, 4), centers=32):
    """X-only maps; tuning Y uses training U subjects, never case labels."""
    z = np.asarray(x, dtype=float)/scale
    m, dx = z.shape; rng = np.random.default_rng(seed)
    partition = np.array_split(rng.permutation(m), folds)
    target = response_bank[train_subjects].transpose(2, 0, 1).reshape(m, -1)
    channels, subjects = response_bank.shape[1], len(train_subjects)
    result, audit = {}, {}
    for degree in degrees:
        operators = []
        for omit in range(dx):
            rest = np.delete(z, omit, axis=1)
            powers = wave.sur.powers_for(dx-1, degree); weights = wave.sur.weights_for(powers)
            poly = wave.sur.monomials(rest, powers, weights)
            residual_operator = np.eye(m); history = []
            for fold, test in enumerate(partition):
                train = np.setdiff1d(np.arange(m), test)
                local_rng = np.random.default_rng(seed+degree*100+fold)
                center_ids = local_rng.choice(train, min(centers, len(train)), replace=False)
                center = rest[center_ids]
                kernel = np.exp(-np.sum((rest[:, None]-center[None])**2, axis=-1)/8.)
                kc = kernel[center_ids]
                eigenvalues, vectors = np.linalg.eigh((kc+kc.T)/2)
                good = eigenvalues > max(float(eigenvalues.max())*1e-10, 1e-12)
                transform = vectors[:, good]/np.sqrt(eigenvalues[good])
                design = np.column_stack((poly, kernel @ transform))
                order = local_rng.permutation(train)
                inner_train, inner_valid = order[:int(.75*len(train))], order[int(.75*len(train)):]
                variance = np.maximum(target[inner_train].reshape(len(inner_train), subjects, channels).var(axis=(0, 1)), 1e-12)
                trials = []
                for ridge in (1e-8, 1e-6, 1e-4, .01):
                    h = regression_map(design[inner_train], design[inner_valid], dx, ridge)
                    error = ((h @ target[inner_train]-target[inner_valid]).reshape(len(inner_valid), subjects, channels)**2).mean(axis=(0, 1))/variance
                    trials.append((float(error.mean()), ridge, error))
                _, ridge, error = min(trials, key=lambda v: v[0])
                mapping = regression_map(design[train], design[test], dx, ridge)
                residual_operator[np.ix_(test, train)] -= mapping
                history.append(dict(fold=fold, ridge=ridge, validation_nmse=error.tolist(), heldout_X_nodes=len(test)))
            constant_error = float(np.max(np.abs(residual_operator @ np.ones(m))))
            affine_error = float(np.max(np.abs(residual_operator @ rest)))
            if max(constant_error, affine_error) > 1e-7:
                raise ArithmeticError('Omission map did not preserve the unpenalized affine span')
            operators.append(residual_operator)
            audit[f'd{degree}_X{omit}'] = dict(history=history, constant_annihilation_error=constant_error,
                other_X_affine_annihilation_error=affine_error)
        result[f'omit_d{degree}'] = operators
    return result, audit


def source_coefficients(bank, name):
    bank.model(name)
    with np.load(bank.folder/f'model_{name}.npz', allow_pickle=False) as saved:
        scaler = object.__new__(wave.sur.Scaler); scaler.mean, scaler.scale = saved['mean'].copy(), saved['scale'].copy()
        coef, center_points = saved['coef'].copy(), saved['centers'].copy()
    nstim = bank.bank.shape[2]; channels = len(bank.data.channels)
    if name.startswith('poly'):
        powers = wave.sur.powers_for(bank.data.d, int(name[4:])); weights = wave.sur.weights_for(powers)
        coefficients = coef*weights[:, None]
        meta = dict(kind='poly', powers=powers)
    else:
        model = wave.sur.RBF(scaler, center_points, float(name[3:])); model.set_coef(coef)
        coefficients = np.concatenate((coef[:1], model.beta), axis=0)
        meta = dict(kind='rbf', centers=model.centers, lengthscale=model.lengthscale)
    return coefficients.reshape(len(coefficients), channels, nstim), scaler(bank.u), meta


def u_axis_matrices(meta, q, anchor, sigma):
    d = len(q); mean, seconds = [], []
    if meta['kind'] == 'rbf':
        outer, _, raw_second = ka.moments(meta['centers'] @ q, anchor @ q, meta['lengthscale'], sigma)
        for a, s in zip(outer, raw_second):
            mu = np.sqrt(np.maximum(np.diag(a), 0.))
            ma = np.ones((len(mu)+1, len(mu)+1)); so = ma.copy()
            ma[0, 1:] = ma[1:, 0] = mu; ma[1:, 1:] = a
            so[0, 1:] = so[1:, 0] = mu; so[1:, 1:] = s
            mean.append(ma); seconds.append(so)
    else:
        powers = meta['powers']; rotated_anchor = anchor @ q
        # Coefficients are separately transported into these U coordinates.
        for j in range(d):
            exponents = powers[:, j]; moments = [1., float(rotated_anchor[j])]
            for k in range(2, int(2*exponents.max())+1):
                moments.append(rotated_anchor[j]*moments[-1]+(k-1)*sigma**2*moments[-2])
            moments = np.asarray(moments)
            mu = moments[exponents]
            mean.append(mu[:, None]*mu[None, :]); seconds.append(moments[exponents[:, None]+exponents[None, :]])
    first, total = [], []
    for i in range(d):
        centered = seconds[i]-mean[i]
        a, b = centered.copy(), centered.copy()
        for j in range(d):
            if j != i:
                a *= mean[j]; b *= seconds[j]
        first.append(a); total.append(b)
    full, constant = np.ones_like(mean[0]), np.ones_like(mean[0])
    for a, b in zip(mean, seconds):
        full *= b; constant *= a
    return first, total, full-constant


def polynomial_transport(meta, q, seed=405):
    powers = meta['powers']; weights = wave.sur.weights_for(powers)
    probes = np.random.default_rng(seed).normal(size=(4*len(powers), len(q)))
    weighted = ka.rotation_representation(powers, weights, q, probes)
    return weights[:, None]*weighted/weights[None, :]


def coefficient_grams(coefficients, operators):
    """Positive-semidefinite U coefficient Grams after observed-X omission."""
    result = {}
    for name, matrices in operators.items():
        table = []
        for residual in matrices:
            channels = []
            for c in range(coefficients.shape[1]):
                values = coefficients[:, c] @ residual.T
                channels.append((values @ values.T)/residual.shape[0])
            table.append(channels)
        result[name] = np.asarray(table).transpose(1, 0, 2, 3)
    return result


def energies(meta, grams, anchors, sigma, q):
    return energies_from_tables(moment_tables(meta, anchors, sigma, q), grams)


def moment_tables(meta, anchors, sigma, q):
    transform = polynomial_transport(meta, q) if meta['kind'] == 'poly' else None
    first, total, controls = [], [], []
    for anchor in anchors:
        f, t, v = u_axis_matrices(meta, q, anchor, sigma)
        if transform is not None:
            f = [transform.T @ a @ transform for a in f]
            t = [transform.T @ a @ transform for a in t]
            v = transform.T @ v @ transform
        first.append(f); total.append(t); controls.append(v)
    return dict(U_first=np.asarray(first), U_total=np.asarray(total), variance=np.asarray(controls))


def energies_from_tables(tables, grams):
    first = np.einsum('niab,xab->nix', tables['U_first'], grams, optimize=True)
    total = np.einsum('niab,xab->nix', tables['U_total'], grams, optimize=True)
    controls = np.einsum('nab,xab->nx', tables['variance'], grams, optimize=True)
    arrays = dict(U_first=np.asarray(first), U_total=np.asarray(total))
    scale = max(float(np.max(np.abs(controls))), np.finfo(float).tiny)
    negative = max(max(0., -float(a.min()))/scale for a in arrays.values())
    return {key: np.maximum(value, 0.) for key, value in arrays.items()}, dict(
        negative_relative_energy=negative, total_U_variance_by_anchor_X=np.asarray(controls).tolist())


def sparsity(energies):
    """energies[anchor,U,X], pooled or equal-anchor normalization."""
    out = {}
    for pooling in ('pooled', 'equal_anchor'):
        e = np.maximum(energies, 0.)
        if pooling == 'equal_anchor':
            total = e.sum((1, 2), keepdims=True)
            e = np.divide(e, total, out=np.zeros_like(e), where=total > 0)
        matrix = e.mean(0)
        for metric, value in ts.scores(matrix).items():
            out[pooling+'_UX_entries_'+metric] = value
        columns = [ts.scores(matrix[:, j]) for j in range(matrix.shape[1])]
        for metric in columns[0]:
            out[pooling+'_UX_columns_mean_'+metric] = float(np.mean([v[metric] for v in columns]))
            out[pooling+'_UX_columns_worst_'+metric] = float(np.max([v[metric] for v in columns]))
    return out
