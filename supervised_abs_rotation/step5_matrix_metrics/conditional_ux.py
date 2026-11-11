"""Cross-fitted empirical U/X omission of a fitted coupling function.

The whole observed U and X blocks form a product reference. Coordinates
inside each block can be dependent. This is a conditional-prediction
diagnostic, not the independent-input Sobol decomposition.
"""
import numpy as np
from scipy.linalg import eigh, null_space

import run as wave
import tail_support as ts


def source_features(z, meta):
    if meta['kind'] == 'poly':
        return wave.sur.monomials(z, meta['powers'], np.ones(len(meta['powers'])))
    ds = np.maximum(np.sum(z*z, 1)[:, None]+np.sum(meta['centers']**2, 1)[None]
                    -2*z @ meta['centers'].T, 0.)
    return np.column_stack((np.ones(len(z)), np.exp(-ds/(2*meta['lengthscale']**2))))


def balanced_latents(phi, coefficients, operators, tolerance=1e-7):
    """Compress output functions with an explicit error bound for EVERY X task.

    Each channel/omitted-X task has equal weight before output PCA. Latents
    have unit empirical covariance; this changes neither their function
    span nor the final task energies (up to recorded reconstruction error).
    """
    phi = phi-phi.mean(0)
    coefficients = coefficients-coefficients.mean(2, keepdims=True)
    sigma = phi.T @ phi/len(phi)
    values, vectors = eigh((sigma+sigma.T)/2)
    good = values > max(values.max()*1e-12, 1e-15)
    root = (vectors[:, good]*np.sqrt(values[good])) @ vectors[:, good].T
    inverse = (vectors[:, good]/np.sqrt(values[good])) @ vectors[:, good].T
    task_grams, normalizers, average = {}, {}, np.zeros_like(sigma)
    for mode, maps in operators.items():
        grams = []
        for c in range(coefficients.shape[1]):
            rows = []
            for residual in maps:
                b = coefficients[:, c] @ residual.T
                gram = b @ b.T/residual.shape[0]
                norm = max(float(np.sum(sigma*gram)), 1e-30)
                rows.append(gram); normalizers[mode, c, len(rows)-1] = norm
                average += gram/norm
            grams.append(rows)
        task_grams[mode] = np.asarray(grams)
    average /= len(normalizers)
    covariance = root @ average @ root
    importance, axes = eigh((covariance+covariance.T)/2)
    order = np.argsort(importance)[::-1]
    importance, axes = np.maximum(importance[order], 0.), axes[:, order]
    candidates = axes.T @ root
    fractions = []
    for mode, grams in task_grams.items():
        for c, row in enumerate(grams):
            for j, gram in enumerate(row):
                diagonal = np.einsum('ka,ab,kb->k', candidates, gram, candidates, optimize=True)
                fractions.append(np.cumsum(np.maximum(diagonal, 0.))/normalizers[mode, c, j])
    fractions = np.asarray(fractions)
    worst = np.min(fractions, axis=0)
    rank = min(int(np.searchsorted(np.maximum.accumulate(worst), 1-tolerance))+1, len(axes))
    projection = inverse @ axes[:, :rank]
    latents = phi @ projection
    transport = axes[:, :rank].T @ root
    grams = {mode: np.einsum('ka,cxab,lb->cxkl', transport, g, transport, optimize=True)
             for mode, g in task_grams.items()}
    weights = importance[:rank]; weights /= weights.sum()
    audit = dict(source_features=phi.shape[1], latent_rank=rank,
        relative_squared_reconstruction_tolerance=tolerance,
        maximum_task_relative_squared_reconstruction_error=float(max(0., 1-worst[rank-1])),
        minimum_task_retained_energy_fraction=float(worst[rank-1]),
        empirical_latent_covariance_error=float(np.max(np.abs(latents.T @ latents/len(latents)-np.eye(rank)))),
        all_U_subjects=len(phi), balance='equal channel and omitted-X task, before PCA',
        latent_loss_weights=weights.tolist())
    if audit['maximum_task_relative_squared_reconstruction_error'] > tolerance*1.1+1e-10:
        raise ArithmeticError('Source-function compression failed its per-X error budget')
    return latents, weights, grams, audit


def direction_classes(maps, tolerance=1e-6):
    directions, memberships = [], []
    for q in maps:
        indices = []
        for raw in q.T:
            v = raw.copy()/np.linalg.norm(raw)
            v *= np.sign(v[np.flatnonzero(np.abs(v) > 1e-10)[0]])
            match = next((i for i, old in enumerate(directions) if np.linalg.norm(v-old) < tolerance), None)
            if match is None:
                match = len(directions); directions.append(v)
            indices.append(match)
        memberships.append(indices)
    return np.asarray(directions), np.asarray(memberships)


def complement_design(z, direction, degree, centers, lengthscale, seed):
    # Any orthonormal complement is equivalent: complete homogeneous
    # polynomial blocks and radial kernels retain their isotropic geometry.
    rest = z @ null_space(np.asarray(direction)[None])
    powers = wave.sur.powers_for(rest.shape[1], degree)
    poly = wave.sur.monomials(rest, powers, wave.sur.weights_for(powers))
    scales = []
    for order in range(2, degree+1):
        mask = powers.sum(1) == order
        scale = max(float(np.mean(np.sum(poly[:, mask]**2, axis=1)))**.5, 1e-12)
        poly[:, mask] /= scale; scales.append(scale)
    ids = np.random.default_rng(seed).choice(len(z), min(centers, len(z)), replace=False)
    ds = np.maximum(np.sum(rest*rest, 1)[:, None]+np.sum(rest[ids]**2, 1)[None]
                    -2*rest @ rest[ids].T, 0.)
    kernel = np.exp(-ds/(2*lengthscale**2))
    eigenvalues, vectors = eigh((kernel[ids]+kernel[ids].T)/2)
    good = eigenvalues > max(eigenvalues.max()*1e-10, 1e-12)
    features = kernel @ (vectors[:, good]/np.sqrt(eigenvalues[good]))
    kernel_scale = max(float(np.mean(np.sum(features**2, axis=1)))**.5, 1e-12)
    return np.column_stack((poly, features/kernel_scale)), dict(
        nonlinear_degree_block_scales=scales, kernel_block_scale=kernel_scale,
        kernel_features=int(good.sum()), input_dimension=rest.shape[1], linear_count=rest.shape[1]+1)


def regressions(design, target, train, query, linear_count, ridges):
    """Partial out the affine span exactly; share only target-independent algebra."""
    affine, nonlinear = design[:, :linear_count], design[:, linear_count:]
    inverse = np.linalg.pinv(affine[train], rcond=1e-10)
    correction = inverse @ nonlinear[train]
    r = nonlinear[train]-affine[train] @ correction
    rq = nonlinear[query]-affine[query] @ correction
    affine_coef = inverse @ target[train]
    y = target[train]-affine[train] @ affine_coef
    gram = r.T @ r
    values, vectors = eigh((gram+gram.T)/2)
    values = np.maximum(values, 0.)
    scale = max(float(np.trace(gram))/max(len(gram), 1), 1e-12)
    rhs = vectors.T @ (r.T @ y)
    base = affine[query] @ affine_coef
    left = rq @ vectors
    return [base+left @ (rhs/(values[:, None]+ridge*scale)) for ridge in ridges]


def omission_covariance(z, direction, latents, weights, degree=5, centers=128,
                        lengthscale=2., seed=407, folds=5):
    design, audit = complement_design(z, direction, degree, centers, lengthscale, seed)
    partition = np.array_split(np.random.default_rng(seed+1).permutation(len(z)), folds)
    residual = np.empty_like(latents); history = []
    ridges = (1e-8, 1e-6, 1e-4, .01, .1)
    for k, query in enumerate(partition):
        train = np.setdiff1d(np.arange(len(z)), query)
        order = np.random.default_rng(seed+2+k).permutation(train)
        split = int(.75*len(order)); inner, validation = order[:split], order[split:]
        predictions = regressions(design, latents, inner, validation, audit['linear_count'], ridges)
        errors = [float(np.mean((p-latents[validation])**2, 0) @ weights) for p in predictions]
        index = int(np.argmin(errors)); ridge = ridges[index]
        predicted = regressions(design, latents, train, query, audit['linear_count'], (ridge,))[0]
        residual[query] = latents[query]-predicted
        history.append(dict(fold=k, scored_subjects=len(query), ridge=ridge,
                            inner_validation_weighted_mse=errors, selected_index=index))
    audit.update(history=history, all_subjects_scored=len(z), U_crossfit_folds=folds,
                 weighted_heldout_residual_energy=float(np.mean(residual**2, 0) @ weights))
    return residual.T @ residual/len(z), audit


def energies(covariances, membership, grams):
    by_axis = np.asarray(covariances)[membership]
    return np.maximum(np.einsum('iab,cxab->cix', by_axis, grams, optimize=True), 0.)


def sparsity(energy):
    """Support of the UX matrix, with primary column-wise scale invariance."""
    out = {}
    for label, matrix in (('raw', energy), ('equal_X', np.divide(energy, energy.sum(0, keepdims=True),
                         out=np.zeros_like(energy), where=energy.sum(0, keepdims=True) > 0))):
        for metric, value in ts.scores(matrix).items():
            # Global top-2 is unsuitable for graphs with several X columns.
            if metric != 'tail2_l2':
                out[label+'_UX_entries_'+metric] = value
    for label, slices in (('columns', energy.T), ('rows', energy)):
        scores = [ts.scores(row) for row in slices]
        for metric in scores[0]:
            out['UX_'+label+'_mean_'+metric] = float(np.mean([s[metric] for s in scores]))
            out['UX_'+label+'_worst_'+metric] = float(np.max([s[metric] for s in scores]))
    return out
