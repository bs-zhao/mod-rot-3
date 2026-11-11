"""Rotation-equivariant local UX geometry, with no output-link formula."""
import numpy as np

import all_trial_moments as am
import fast_projection as fp
import joint_fields as jf
import local_support as ls

PROFILES = tuple(f'{scope}_{balance}' for scope in ('raw', 'scalar_UX', 'scalar_UW', 'tangent_X')
                 for balance in ('point', 'column'))


def unit_vector(vector):
    peak = np.max(np.abs(vector), axis=-1, keepdims=True)
    v = np.divide(vector, peak, out=np.zeros_like(vector), where=peak > 0)
    norm = np.sqrt(np.sum(v*v, axis=-1, keepdims=True))
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


def balanced(matrix, mode):
    peak = np.max(np.abs(matrix), axis=(-2, -1), keepdims=True)
    v = np.divide(matrix, peak, out=np.zeros_like(matrix), where=peak > 0)
    axes = (-2, -1) if mode == 'point' else (-2,)
    norm = np.sqrt(np.sum(v*v, axis=axes, keepdims=True))
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


def remove_roundoff_residual(residual, original):
    peak = np.max(np.abs(original), axis=(-2, -1), keepdims=True)
    r = np.divide(residual, peak, out=np.zeros_like(residual), where=peak > 0)
    h = np.divide(original, peak, out=np.zeros_like(original), where=peak > 0)
    tiny = np.sum(r*r, axis=(-2, -1)) <= 1e-24*np.sum(h*h, axis=(-2, -1))
    return np.where(tiny[..., None, None], 0., residual)


def fields(g, h_u, d):
    """g[...,W], h_u[...,U,W]; gradient directions only, no fitted link."""
    hx = h_u[..., d:]
    u, x, w = unit_vector(g[..., :d]), unit_vector(g[..., d:]), unit_vector(g)
    bx = u[..., :, None]*x[..., None, :]
    bw = u[..., :, None]*w[..., None, :]
    scalar_x = hx-np.sum(hx*bx, axis=(-2, -1), keepdims=True)*bx
    residual_w = h_u-np.sum(h_u*bw, axis=(-2, -1), keepdims=True)*bw
    tangent = hx-np.sum(hx*x[..., None, :], axis=-1, keepdims=True)*x[..., None, :]
    source = dict(raw=hx, scalar_UX=remove_roundoff_residual(scalar_x, hx),
                  scalar_UW=remove_roundoff_residual(residual_w[..., d:], hx),
                  tangent_X=remove_roundoff_residual(tangent, hx))
    return {f'{scope}_{mode}': balanced(matrix, mode) for scope, matrix in source.items()
            for mode in ('point', 'column')}


def moments(data, saved, subjects_per_chunk=32, progress=None):
    gp, _, gc, hp, _, hc = jf.derivative_coefficients(saved['powers'], saved['weights'], saved['coef'])
    scaler = am.FrozenScaler(saved['mean'], saved['scale'])
    gradient_model = fp.Polynomial(scaler, int(gp.sum(1).max()))
    hessian_model = fp.Polynomial(scaler, int(hp.sum(1).max()))
    dimension, d, dx, channels = saved['powers'].shape[1], data.d, data.dx, saved['coef'].shape[1]
    hc = hc[:, :d]
    out = {profile: np.empty((data.n, channels, dx, d, d)) for profile in PROFILES}
    zeros = {profile: np.zeros(channels, dtype=int) for profile in PROFILES}
    rows_used = 0
    for start in range(0, data.n, subjects_per_chunk):
        stop = min(start+subjects_per_chunk, data.n)
        a, b = data.offsets[start], data.offsets[stop]
        g = (gradient_model.features(data.w[a:b]) @ gc.reshape(len(gp), -1)).reshape(b-a, dimension, channels).transpose(0, 2, 1)
        h = (hessian_model.features(data.w[a:b]) @ hc.reshape(len(hp), -1)).reshape(b-a, d, dimension, channels).transpose(0, 3, 1, 2)
        profiles = fields(g, h, d)
        for profile, values in profiles.items():
            zeros[profile] += np.sum(np.sum(values*values, axis=(-2, -1)) == 0., axis=0)
            for subject in range(start, stop):
                lo, hi = data.offsets[subject:subject+2]-a
                field = values[lo:hi]
                out[profile][subject] = np.einsum('scix,scjx->cxij', field, field, optimize=True)/(hi-lo)
        rows_used += int(b-a)
        if progress and (stop % 1024 < subjects_per_chunk or stop == data.n):
            progress(stop, rows_used)
    if rows_used != data.m:
        raise ValueError('Not all original trial rows were processed')
    return out, dict(rows_used=rows_used, zero_field_rows={k: v.tolist() for k, v in zeros.items()},
                     numerical_residual_relative_norm_threshold=1e-12)


def scores(covariance, q):
    energy = np.maximum(np.einsum('ia,nxij,ja->nax', q, covariance, q, optimize=True), 0.)
    out = {}
    for scope, values in (('UX_entries', energy.reshape(len(energy), -1)),
                          ('UX_columns', energy.transpose(0, 2, 1))):
        for metric, values in ls.row_scores(values).items():
            out[scope+'_subject_mean_'+metric] = float(np.mean(values))
            out[scope+'_subject_median_'+metric] = float(np.mean(np.median(values, axis=0)))
    out['control_UX_fro2'] = float(energy.sum((1, 2)).mean())
    return out
