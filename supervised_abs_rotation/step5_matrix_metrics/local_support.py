"""Compute normalized effective support before, rather than after, averaging."""
import numpy as np


def row_scores(energy):
    e = np.maximum(np.asarray(energy, dtype=float), 0.)
    if e.ndim < 2 or not np.isfinite(e).all():
        raise ValueError('Finite energy vectors required')
    peak = e.max(axis=-1, keepdims=True)
    p = np.divide(e, peak, out=np.zeros_like(e), where=peak > 0)
    total = p.sum(axis=-1, keepdims=True)
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    p = np.sort(p, axis=-1)[..., ::-1]
    active = total[..., 0] > 0
    out = dict(effective_support_p0p5=np.sum(p**.25, axis=-1)**(4/3),
               effective_support_p1=np.sum(np.sqrt(p), axis=-1)**2,
               tail2_l2=np.sqrt(p[..., 2:].sum(axis=-1)))
    cumulative = np.cumsum(p, axis=-1)
    for label, eps in (('001', .01), ('003', .03), ('010', .1)):
        out['soft_count_eps'+label] = np.sum(p/(p+eps**2), axis=-1)
        target = 1-eps**2
        index = np.minimum(np.sum(cumulative < target, axis=-1), p.shape[-1]-1)
        previous = np.take_along_axis(cumulative, np.maximum(index-1, 0)[..., None], axis=-1)[..., 0]
        previous = np.where(index > 0, previous, 0.)
        value = np.take_along_axis(p, index[..., None], axis=-1)[..., 0]
        count = index+(target-previous)/np.maximum(value, 1e-300)
        out['fractional_count_eps'+label] = np.where(active, count, 0.)
    return out


def pooled_scores(energy):
    """energy[subject,context,entry], with equal subject/context sampling."""
    out = {}
    for pooling, e in (('point', energy), ('subject', np.mean(energy, axis=1)),
                       ('context', np.mean(energy, axis=0))):
        for metric, values in row_scores(e).items():
            out[pooling+'_mean_'+metric] = float(np.mean(values))
            out[pooling+'_median_'+metric] = float(np.median(values))
    return out


def evaluate(g, h, delta, q):
    d = len(q); ii, jj = np.triu_indices(d, 1)
    gr = g @ q
    hr = q.T @ h @ q
    diagonal = np.diagonal(hr, axis1=-2, axis2=-1)**2
    edges = hr[..., ii, jj]**2
    contrasts = delta @ q
    groups = dict(UU_edges=edges, UU_all=np.concatenate((diagonal, 2*edges), axis=-1),
                  U_curvature=diagonal, U_gradient=gr**2, UX_observed_contrast=contrasts**2)
    out = {}
    for group, energy in groups.items():
        for metric, value in pooled_scores(energy).items():
            out[group+'_'+metric] = value
    out['control_UU_fro2'] = float(np.mean(np.sum(hr**2, axis=(-2, -1))))
    out['control_U_gradient_fro2'] = float(np.mean(np.sum(gr**2, axis=-1)))
    out['control_UX_contrast_fro2'] = float(np.mean(np.sum(contrasts**2, axis=-1)))
    return out
