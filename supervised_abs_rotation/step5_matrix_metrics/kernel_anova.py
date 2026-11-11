"""Exact RBF ANOVA variances under local isotropic Gaussian references.

Each anchor defines an explicit product measure, not the dependent empirical
U law. Components are orthogonal only within that declared reference.
"""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np
from scipy.linalg import lstsq


def moments(centers, anchor, lengthscale, sigma):
    l2, s2 = lengthscale**2, sigma**2
    mean_outer, centered, second = [], [], []
    for i in range(centers.shape[1]):
        c = centers[:, i]
        mu = np.sqrt(l2/(l2+s2))*np.exp(-(anchor[i]-c)**2/(2*(l2+s2)))
        m = mu[:, None]*mu[None, :]
        pair = np.sqrt(l2/(l2+2*s2))*np.exp(
            -(c[:, None]-c[None, :])**2/(4*l2) -
            (anchor[i]-(c[:, None]+c[None, :])/2)**2/(l2+2*s2))
        mean_outer.append(m); centered.append(pair-m); second.append(pair)
    return mean_outer, centered, second


def variances(centers, gram, anchor, lengthscale, sigma):
    mean_outer, centered, second = moments(centers, anchor, lengthscale, sigma)
    d = centers.shape[1]
    subsets, values = [], []
    for order in range(1, d+1):
        for subset in combinations(range(d), order):
            kernel = np.ones_like(gram)
            for i in range(d):
                kernel *= centered[i] if i in subset else mean_outer[i]
            subsets.append(subset)
            values.append(float(np.sum(gram*kernel)))
    full, constant = np.ones_like(gram), np.ones_like(gram)
    for i in range(d):
        full *= second[i]; constant *= mean_outer[i]
    total = float(np.sum(gram*(full-constant)))
    values = np.asarray(values)
    error = abs(float(values.sum())-total)/max(abs(total), 1e-24)
    negative = max(0., float(-values.min()))/max(abs(total), 1e-24)
    return subsets, np.maximum(values, 0.), total, dict(
        sum_relative_error=error, negative_relative_energy=negative)


def sparsity(energy, prefix):
    energy = np.asarray(energy)
    total = float(energy.sum())
    if total <= 1e-24:
        return {prefix+"_"+k: 0. for k in ("tail1_l2", "tail2_l2", "tail4_l2", "entropy")}
    p = np.sort(energy/total)[::-1]
    out = {f"{prefix}_tail{k}_l2": float(np.sqrt(p[k:].sum())) for k in (1, 2, 4)}
    out[prefix+"_entropy"] = float(-np.sum(p*np.log(np.maximum(p, 1e-300)))/max(np.log(len(p)), 1.))
    return out


def scores(centers, gram, anchors, lengthscale, sigma):
    all_values, totals, audits = [], [], []
    for anchor in anchors:
        subsets, values, total, audit = variances(centers, gram, anchor, lengthscale, sigma)
        all_values.append(values); totals.append(total); audits.append(audit)
    energy = np.mean(all_values, axis=0)
    order = np.array([len(s) for s in subsets])
    interactions = order >= 2
    out = sparsity(energy, "all_components")
    out.update(sparsity(energy[interactions], "interaction_components"))
    out.update(sparsity(energy[order == 2], "pair_components"))
    out["interaction_fraction"] = float(energy[interactions].sum()/max(energy.sum(), 1e-24))
    out["higher_order_fraction"] = float(energy[order >= 3].sum()/max(energy.sum(), 1e-24))
    out["control_total_local_variance"] = float(np.mean(totals))
    audit = dict(sum_relative_error=max(a["sum_relative_error"] for a in audits),
                 negative_relative_energy=max(a["negative_relative_energy"] for a in audits),
                 subsets=[list(s) for s in subsets], component_energy=energy.tolist())
    return out, audit


def rotation_representation(powers, weights, q, probes):
    """Exact homogeneous polynomial change of coordinates, checked by blocks."""
    import surrogates as sur
    left = sur.monomials(probes @ q, powers, weights)
    right = sur.monomials(probes, powers, weights)
    transform = np.zeros((len(powers), len(powers)))
    degrees = powers.sum(1)
    for degree in np.unique(degrees):
        ids = np.flatnonzero(degrees == degree)
        block = lstsq(left[:, ids], right[:, ids], lapack_driver="gelsy")[0]
        if np.max(np.abs(block.T @ block-np.eye(len(ids)))) > 1e-7:
            raise ArithmeticError("Polynomial rotation representation lost orthogonality")
        transform[np.ix_(ids, ids)] = block
    return transform


def hermite_moments(powers, anchor, sigma):
    """E[U^alpha He_beta(Z)/sqrt(beta!)] for U=anchor+sigma*Z."""
    degree = int(powers.sum(1).max())
    matrix = np.ones((len(powers), len(powers)))
    for i, a in enumerate(anchor):
        one = np.zeros((degree+1, degree+1))
        for p in range(degree+1):
            for k in range(p+1):
                for r in range((p-k)//2+1):
                    one[p, k] += (math.factorial(p) * a**(p-k-2*r) * sigma**(k+2*r) /
                        (math.factorial(p-k-2*r) * 2**r * math.factorial(r) * math.sqrt(math.factorial(k))))
        matrix *= one[powers[:, i, None], powers[None, :, i]]
    return matrix


def polynomial_scores(powers, weights, coef_gram, anchors, sigma):
    subsets = [tuple(np.flatnonzero(p)) for p in powers]
    labels = sorted(set(subsets)-{()}, key=lambda s: (len(s), s))
    group = {s: [i for i, subset in enumerate(subsets) if subset == s] for s in labels}
    gram = coef_gram*weights[:, None]*weights[None, :]
    energies = []
    negative = 0.
    for anchor in anchors:
        matrix = hermite_moments(powers, anchor, sigma)
        e = np.sum(matrix*(gram @ matrix), axis=0)
        total = float(e[1:].sum())
        negative = max(negative, max(0., float(-e[1:].min()))/max(abs(total), 1e-24))
        energies.append([float(e[ids].sum()) for ids in group.values()])
    energy = np.maximum(np.mean(energies, axis=0), 0.)
    order = np.array([len(s) for s in labels])
    out = sparsity(energy, "all_components")
    out.update(sparsity(energy[order >= 2], "interaction_components"))
    out.update(sparsity(energy[order == 2], "pair_components"))
    out["interaction_fraction"] = float(energy[order >= 2].sum()/max(energy.sum(), 1e-24))
    out["higher_order_fraction"] = float(energy[order >= 3].sum()/max(energy.sum(), 1e-24))
    out["control_total_local_variance"] = float(energy.sum())
    return out, dict(sum_relative_error=0., negative_relative_energy=negative,
                    subsets=[list(map(int, s)) for s in labels], component_energy=energy.tolist())
