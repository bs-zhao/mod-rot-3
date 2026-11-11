"""Exact subject-level UX support and its piecewise analytic rotation gradient."""
import numpy as np
from scipy.linalg import expm, expm_frechet


class UXObjective:
    def __init__(self, hux, eps=.1, aggregation='median', kind='fractional'):
        # hux[subject, observed_context, U_axis, X_axis].
        covariance = np.einsum('nsix,nsjx->nxij', hux, hux, optimize=True)/hux.shape[1]
        trace = np.trace(covariance, axis1=-2, axis2=-1)
        self.covariance = np.divide(covariance, trace[..., None, None],
            out=np.zeros_like(covariance), where=trace[..., None, None] > 0)
        self.active = trace > 0
        self.eps, self.aggregation, self.kind = eps, aggregation, kind
        self.d = hux.shape[2]

    def values_gradient(self, q, with_gradient=True):
        cq = np.einsum('nxij,ja->nxia', self.covariance, q, optimize=True)
        raw = np.maximum(np.einsum('ia,nxia->nxa', q, cq, optimize=True), 0.)
        total = raw.sum(-1, keepdims=True)
        p = np.divide(raw, total, out=np.zeros_like(raw), where=total > 0)
        if self.kind == 'soft':
            values = np.sum(p/(p+self.eps**2), axis=-1)
            derivative = self.eps**2/(p+self.eps**2)**2
        elif self.kind == 'fractional':
            permutation = np.argsort(-p, axis=-1)
            sorted_p = np.take_along_axis(p, permutation, axis=-1)
            cumulative = np.cumsum(sorted_p, axis=-1); target = 1-self.eps**2
            index = np.minimum(np.sum(cumulative < target, axis=-1), self.d-1)
            previous = np.take_along_axis(cumulative, np.maximum(index-1, 0)[..., None], axis=-1)[..., 0]
            previous = np.where(index > 0, previous, 0.)
            pivot = np.take_along_axis(sorted_p, index[..., None], axis=-1)[..., 0]
            fraction = (target-previous)/np.maximum(pivot, 1e-300)
            values = index+fraction
            sorted_derivative = np.where(np.arange(self.d) < index[..., None],
                -1/np.maximum(pivot[..., None], 1e-300), 0.)
            np.put_along_axis(sorted_derivative, index[..., None],
                             (-fraction/np.maximum(pivot, 1e-300))[..., None], axis=-1)
            derivative = np.empty_like(p)
            np.put_along_axis(derivative, permutation, sorted_derivative, axis=-1)
        else:
            raise ValueError('Supported rules: fractional or soft')
        values = np.where(self.active, values, 0.)
        if not with_gradient:
            return values
        weights = np.zeros_like(values)
        n, nx = values.shape
        if self.aggregation == 'mean':
            weights[:] = 1/(n*nx)
            loss = float(values.mean())
        elif self.aggregation == 'median':
            order = np.argsort(values, axis=0)
            middle = (n-1)//2, n//2
            for k in middle:
                weights[order[k], np.arange(nx)] += .5/nx
            loss = float(np.median(values, axis=0).mean())
        else:
            raise ValueError('Supported aggregations: mean or median')
        derivative *= weights[..., None]
        derivative = np.divide(derivative-np.sum(derivative*p, axis=-1, keepdims=True), total,
            out=np.zeros_like(derivative), where=total > 0)
        derivative *= self.active[..., None]
        gradient = 2*np.einsum('nxa,nxia->ia', derivative, cq, optimize=True)
        return loss, gradient


def skew_parameters(parameters, d):
    ii, jj = np.triu_indices(d, 1); out = np.zeros((d, d))
    out[ii, jj] = parameters; out[jj, ii] = -parameters
    return out


def parameter_objective(parameters, initial, objective):
    a = skew_parameters(parameters, objective.d)
    q = initial @ expm(a)
    loss, gradient = objective.values_gradient(q)
    result = []
    for i, j in zip(*np.triu_indices(objective.d, 1)):
        e = np.zeros_like(a); e[i, j] = 1.; e[j, i] = -1.
        direction = initial @ expm_frechet(a, e, compute_expm=False)
        result.append(float(np.sum(gradient*direction)))
    return loss, np.asarray(result)
