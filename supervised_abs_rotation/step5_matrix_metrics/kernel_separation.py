"""Single-axis functional separation spectra; scale-free spectral support."""
from __future__ import annotations

import numpy as np

import kernel_anova as ka
import tail_support as ts


def clean_spectrum(matrix):
    values = np.linalg.eigvalsh((matrix+matrix.T)/2)
    trace = max(float(np.trace(matrix)), 1e-100)
    negative = max(0., float(-values.min()))/trace
    values[values < 1e-8*trace] = 0.
    return np.maximum(values, 0.), negative


def polynomial_spectra(powers, hermite_gram):
    degree = int(powers.sum(1).max())
    spectra, negatives = [], []
    for axis in range(powers.shape[1]):
        groups = {}
        for index, p in enumerate(powers):
            if p[axis] > 0:
                groups.setdefault(tuple(np.delete(p, axis)), []).append((int(p[axis])-1, index))
        matrix = np.zeros((degree, degree))
        for group in groups.values():
            rows, indices = map(list, zip(*group))
            matrix[np.ix_(rows, rows)] += hermite_gram[np.ix_(indices, indices)]
        values, negative = clean_spectrum(matrix)
        spectra.append(values); negatives.append(negative)
    return spectra, dict(negative_relative_energy=max(negatives), quadrature_relative_error=0.)


def rbf_spectra(centers, coef_gram, anchor, lengthscale, sigma, quadrature=12):
    means, _, seconds = ka.moments(centers, anchor, lengthscale, sigma)
    nodes, weights = np.polynomial.hermite_e.hermegauss(quadrature)
    weights = weights/np.sqrt(2*np.pi)
    spectra, negatives, errors = [], [], []
    for axis in range(centers.shape[1]):
        z = anchor[axis]+sigma*nodes
        phi = np.exp(-(centers[:, axis, None]-z[None, :])**2/(2*lengthscale**2))
        mean = np.sqrt(lengthscale**2/(lengthscale**2+sigma**2))*np.exp(
            -(anchor[axis]-centers[:, axis])**2/(2*(lengthscale**2+sigma**2)))
        left = (phi-mean[:, None])*np.sqrt(weights)[None, :]
        right = coef_gram.copy()
        for j in range(centers.shape[1]):
            if j != axis:
                right *= seconds[j]
        matrix = left.T @ right @ left
        total = float(np.sum((seconds[axis]-means[axis])*right))
        error = abs(float(np.trace(matrix))-total)/max(abs(total), 1e-100)
        values, negative = clean_spectrum(matrix)
        spectra.append(values); negatives.append(negative); errors.append(error)
    return spectra, dict(negative_relative_energy=max(negatives), quadrature_relative_error=max(errors))


def summarize(spectra_by_anchor):
    keys = list(ts.scores(spectra_by_anchor[0][0]))
    table = np.asarray([[[ts.scores(e)[key] for key in keys] for e in spectra]
                        for spectra in spectra_by_anchor])
    by_axis = table.mean(0)
    out = {}
    for k, key in enumerate(keys):
        out["axis_mean_"+key] = float(by_axis[:, k].mean())
        out["axis_worst_"+key] = float(by_axis[:, k].max())
    return out


def polynomial_scores(powers, weights, gram, anchors, sigma):
    weighted = gram*weights[:, None]*weights[None, :]
    spectra, audits = [], []
    for anchor in anchors:
        matrix = ka.hermite_moments(powers, anchor, sigma)
        values, audit = polynomial_spectra(powers, matrix.T @ weighted @ matrix)
        spectra.append(values); audits.append(audit)
    return summarize(spectra), dict(
        negative_relative_energy=max(a["negative_relative_energy"] for a in audits),
        quadrature_relative_error=0., spectra=[[e.tolist() for e in s] for s in spectra])


def rbf_scores(centers, gram, anchors, lengthscale, sigma):
    spectra, audits = [], []
    for anchor in anchors:
        values, audit = rbf_spectra(centers, gram, anchor, lengthscale, sigma)
        spectra.append(values); audits.append(audit)
    return summarize(spectra), dict(
        negative_relative_energy=max(a["negative_relative_energy"] for a in audits),
        quadrature_relative_error=max(a["quadrature_relative_error"] for a in audits),
        spectra=[[e.tolist() for e in s] for s in spectra])
