"""Empirical response profiles and nested additive regression, using W/Y only.

These are predictive projection diagnostics, NOT independent-input Sobol
indices. The empirical U distribution may be dependent in every rotation.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.linalg import solve

import surrogates as sur


def estimate(g, cross, ridge):
    penalty = np.eye(len(g)); penalty[0, 0] = 0
    scale = max(np.trace(g[1:, 1:]) / max(1, len(g)-1), 1e-12)
    return solve(g + ridge * scale * penalty, cross, assume_a="pos")


def x_profiles(data, *, centers=64, seed=382, ridges=(1e-6, 1e-3, 1e-1), probe_count=0):
    """Fit identical generic X bases per subject, with within-subject validation.

    All subjects and all their rows contribute to final profiles. A common
    empirical X Gram matrix defines the profile distance. This approximates
    integration against the pooled X measure, not an assertion of U/X
    independence or a guarantee of common support.
    """
    rng = np.random.default_rng(seed)
    # The X-only representation is invariant to every U rotation.
    x = data.w[:, data.d:]
    train_rows = np.flatnonzero(np.load(data.folder / "trial_index.npy") % 5 != 0)
    scaler = sur.Scaler(x, 1, row_mask=np.load(data.folder / "trial_index.npy") % 5 != 0)
    ids = rng.choice(train_rows, min(centers, len(train_rows)), replace=False)
    kernel = sur.RBF(scaler, x[ids], 2.)
    poly = sur.Polynomial(scaler, 2)

    def features(v):
        return np.column_stack([poly.features(v), kernel.features(v)[:, 1:]])

    p = features(x[:1]).shape[1]
    coefficients = np.empty((len(ridges), data.n, p, len(data.channels)))
    total_gram = np.zeros((p, p))
    validation_sse = np.zeros((len(ridges), len(data.channels)))
    train_sum = np.zeros(len(data.channels)); train_sq = train_sum.copy()
    nv = nt = 0
    for s, (a, b) in enumerate(zip(data.offsets[:-1], data.offsets[1:])):
        phi, y = features(x[a:b]), data.y[a:b]
        keep = np.arange(b-a) % 5 != 0
        pt, pv, yt, yv = phi[keep], phi[~keep], y[keep], y[~keep]
        ga, ca = phi.T @ phi, phi.T @ y
        gt, ct = pt.T @ pt, pt.T @ yt
        total_gram += ga
        for k, ridge in enumerate(ridges):
            beta = estimate(gt, ct, ridge)
            validation_sse[k] += ((pv @ beta-yv)**2).sum(0)
            coefficients[k, s] = estimate(ga, ca, ridge)
        train_sum += yt.sum(0); train_sq += (yt*yt).sum(0)
        nt += len(yt); nv += len(yv)
        if (s+1) % 1000 == 0:
            print(f"[WaveB] X profiles: {s+1}/{data.n} subjects", flush=True)
    variance = np.maximum(train_sq/nt-(train_sum/nt)**2, 1e-12)
    nmse = validation_sse/nv/variance
    best = int(np.argmin(nmse.mean(1)))
    g = total_gram/data.m
    values, vectors = np.linalg.eigh(g)
    transform = vectors * np.sqrt(np.maximum(values, 0.))[None, :]
    # ||beta @ transform||^2 = beta @ G_X @ beta.T.
    profiles = np.einsum("spc,pq->scq", coefficients[best], transform)
    quality = dict(feature_count=p, ridge=ridges[best],
                          validation_nmse=nmse[best].tolist(),
                          ridge_trials=nmse.tolist(), all_subjects=data.n,
                          all_rows=data.m, profile_measure="pooled empirical X",
                          sobol_independence_assumed=False)
    if probe_count:
        probe_ids = rng.choice(train_rows, min(probe_count, len(train_rows)), replace=False)
        values = np.einsum("spc,kp->sck", coefficients[best], features(x[probe_ids]))
        return values, np.asarray(x[probe_ids]), quality
    return profiles, quality


def axis_bases(u, train, family):
    """Centered empirical bases; sign/permutation equivariant spans.

    Pair components use the entire same one-axis basis. No eigenvector
    truncation chooses a preferred polynomial direction within a group.
    """
    bases = []
    for i in range(u.shape[1]):
        z = u[:, i]
        if family.startswith("poly"):
            degree = int(family[4:])
            raw = np.column_stack([z**k for k in range(1, degree+1)])
        elif family.startswith("rbf"):
            count = int(family[3:])
            knots = np.quantile(z[train], np.linspace(.05, .95, count))
            raw = np.exp(-.5*(z[:, None]-knots[None, :])**2)
        else:
            raise ValueError(family)
        raw -= raw[train].mean(0)
        cov = raw[train].T @ raw[train]/len(train)
        values, vectors = np.linalg.eigh(cov)
        good = values > max(values.max()*1e-8, 1e-12)
        bases.append(raw @ (vectors[:, good]/np.sqrt(values[good])))
    return bases


def designs(bases):
    n, d = len(bases[0]), len(bases)
    additive = np.column_stack([np.ones(n), *bases])
    pairs = list(combinations(range(d), 2))
    components = [np.einsum("ni,nj->nij", bases[i], bases[j]).reshape(n, -1)
                  for i, j in pairs]
    return additive, components, pairs


def fit_predict(design, target, train, valid, test, ridges=(1e-6, 1e-3, 1e-1)):
    gt, ct = design[train].T @ design[train], design[train].T @ target[train]
    trials = []
    for ridge in ridges:
        beta = estimate(gt, ct, ridge)
        err = float(np.mean(np.sum((design[valid] @ beta-target[valid])**2, axis=1)))
        trials.append((err, ridge))
    _, chosen = min(trials)
    fit_rows = np.r_[train, valid]
    coef = estimate(design[fit_rows].T @ design[fit_rows],
                    design[fit_rows].T @ target[fit_rows], chosen)
    pred = design[test] @ coef
    error = float(np.mean(np.sum((pred-target[test])**2, axis=1)))
    all_coef = estimate(design.T @ design, design.T @ target, chosen)
    return error, chosen, all_coef


def projection_scores(u, target, split, family):
    """Generalized additive and pairwise predictive compression.

    A pair gain is measured after projecting that component out of the
    entire additive design on all subjects. Correlated pair components
    remain correlated: their gains are not additive variance shares.
    """
    train, valid, test = split
    target = target-target[train].mean(0)
    variance = max(float(np.mean(np.sum(target[train]**2, axis=1))), 1e-12)
    target = target/np.sqrt(variance)
    bases = axis_bases(u, train, family)
    add, pairs, labels = designs(bases)
    add_error, add_ridge, add_coef = fit_predict(add, target, train, valid, test)
    full = np.column_stack([add, *pairs])
    full_error, full_ridge, full_coef = fit_predict(full, target, train, valid, test)
    resid = target-add @ add_coef
    pair_energy = []
    for component in pairs:
        # Joint empirical projection removes additive correlations. No
        # product-measure orthogonality or Sobol interpretation is claimed.
        regressed = np.linalg.lstsq(add, component, rcond=1e-9)[0]
        orth = component-add @ regressed
        beta = solve(orth.T @ orth + 1e-5*np.eye(orth.shape[1]), orth.T @ resid,
                     assume_a="pos")
        pair_energy.append(float(np.mean(np.sum((orth @ beta)**2, axis=1))))
    energy = np.asarray(pair_energy)
    total = max(float(energy.sum()), 1e-24)
    ordered = np.sort(energy)[::-1]
    scores = dict(additive_test_nmse=add_error,
                  pairwise_test_nmse=full_error,
                  additive_pairwise_gap=max(0., add_error-full_error),
                  pair_projection_tail1_l2=float(np.sqrt(ordered[1:].sum()/total)),
                  pair_projection_tail2_l2=float(np.sqrt(ordered[2:].sum()/total)),
                  pair_projection_total_energy=float(energy.sum()))
    # Complexity is an all-subject component-energy tail, separate from
    # held-out prediction. Correlated components can share energy.
    group_energy = []
    at = 1
    for component in [*bases, *pairs]:
        coef = full_coef[at:at+component.shape[1]]
        group_energy.append(float(np.mean(np.sum((component @ coef)**2, axis=1))))
        at += component.shape[1]
    e = np.sort(np.asarray(group_energy))[::-1]
    for label, k in (("1", 1), ("2", 2), ("dimension", u.shape[1])):
        scores[f"component_tail{label}_l2"] = float(np.sqrt(e[k:].sum()/max(e.sum(), 1e-24)))
    return scores, dict(additive_features=add.shape[1], pairwise_features=full.shape[1],
                        additive_ridge=add_ridge, pairwise_ridge=full_ridge,
                        pair_labels=labels, pair_projection_energy=energy.tolist())
