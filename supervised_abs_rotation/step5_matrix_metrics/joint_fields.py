"""Exact standardized-coordinate derivatives of a joint polynomial projection."""
import numpy as np

import surrogates as sur


def derivative_coefficients(powers, weights, coef):
    degree, dimension = int(powers.sum(1).max()), powers.shape[1]
    gp, hp = sur.powers_for(dimension, degree-1), sur.powers_for(dimension, max(0, degree-2))
    gw, hw = sur.weights_for(gp), sur.weights_for(hp)
    gl = {tuple(p): i for i, p in enumerate(gp)}; hl = {tuple(p): i for i, p in enumerate(hp)}
    gradient = np.zeros((len(gp), dimension, coef.shape[1]))
    hessian = np.zeros((len(hp), dimension, dimension, coef.shape[1]))
    for a, p in enumerate(powers):
        for i in np.flatnonzero(p):
            lower = p.copy(); lower[i] -= 1; b = gl[tuple(lower)]
            gradient[b, i] += p[i]*weights[a]/gw[b]*coef[a]
            for j in np.flatnonzero(lower):
                twice = lower.copy(); twice[j] -= 1; b = hl[tuple(twice)]
                hessian[b, i, j] += p[i]*lower[j]*weights[a]/hw[b]*coef[a]
    return gp, gw, gradient, hp, hw, hessian


def evaluate(z, powers, weights, coef, chunk=4096):
    gp, gw, gc, hp, hw, hc = derivative_coefficients(powers, weights, coef)
    gradient, hessian = [], []
    for a in range(0, len(z), chunk):
        b = min(a+chunk, len(z)); x = z[a:b]
        g = sur.monomials(x, gp, gw) @ gc.reshape(len(gp), -1)
        h = sur.monomials(x, hp, hw) @ hc.reshape(len(hp), -1)
        gradient.append(g.reshape(len(x), *gc.shape[1:]).transpose(0, 2, 1))
        hessian.append(h.reshape(len(x), *hc.shape[1:]).transpose(0, 3, 1, 2))
    return np.concatenate(gradient), np.concatenate(hessian)


def observed_rows(data, per_subject=8):
    ids = []
    for a, b in zip(data.offsets[:-1], data.offsets[1:]):
        if b-a < per_subject:
            raise ValueError('Equal-context protocol requires at least eight recorded rows per subject')
        ids.extend(a+np.linspace(0, b-a-1, per_subject, dtype=int))
    return np.asarray(ids, dtype=int)
