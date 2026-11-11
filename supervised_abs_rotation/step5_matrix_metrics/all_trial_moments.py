"""Exact per-subject derivative second moments over every recorded W row."""
import numpy as np

import fast_projection as fp
import joint_fields as jf
import local_support as ls


class FrozenScaler:
    def __init__(self, mean, scale):
        self.mean, self.scale = mean, scale

    def __call__(self, w):
        return (np.asarray(w, dtype=float)-self.mean)/self.scale


def symmetric_basis(d):
    basis = []
    for i in range(d):
        b = np.zeros((d, d)); b[i, i] = 1.; basis.append(b)
    for i, j in zip(*np.triu_indices(d, 1)):
        b = np.zeros((d, d)); b[i, j] = b[j, i] = 1/np.sqrt(2); basis.append(b)
    return np.asarray(basis)


def moments(data, saved, subjects_per_chunk=32, progress=None):
    gp, gw, gc, hp, hw, hc = jf.derivative_coefficients(saved['powers'], saved['weights'], saved['coef'])
    scaler = FrozenScaler(saved['mean'], saved['scale'])
    gradient_model = fp.Polynomial(scaler, int(gp.sum(1).max()))
    hessian_model = fp.Polynomial(scaler, int(hp.sum(1).max()))
    if not np.array_equal(gradient_model.powers, gp) or not np.array_equal(hessian_model.powers, hp):
        raise ValueError('Derivative feature ordering changed')
    d, dx, channels = data.d, data.dx, saved['coef'].shape[1]
    basis = symmetric_basis(d); nk = len(basis)
    symcoef = np.einsum('pijc,kij->pkc', hc[:, :d, :d], basis, optimize=True)
    uxcoef = hc[:, :d, d:]
    gradcoef = gc[:, :d]
    uu = np.empty((data.n, channels, nk, nk)); ux = np.empty((data.n, channels, dx, d, d))
    gradient = np.empty((data.n, channels, d, d))
    rows_used = 0
    for start in range(0, data.n, subjects_per_chunk):
        stop = min(start+subjects_per_chunk, data.n)
        a, b = data.offsets[start], data.offsets[stop]
        hphi = hessian_model.features(data.w[a:b])
        hs = (hphi @ symcoef.reshape(len(hp), -1)).reshape(b-a, nk, channels).transpose(0, 2, 1)
        hx = (hphi @ uxcoef.reshape(len(hp), -1)).reshape(b-a, d, dx, channels).transpose(0, 3, 2, 1)
        gphi = gradient_model.features(data.w[a:b])
        g = (gphi @ gradcoef.reshape(len(gp), -1)).reshape(b-a, d, channels).transpose(0, 2, 1)
        for subject in range(start, stop):
            lo, hi = data.offsets[subject:subject+2]-a
            m = hi-lo
            uu[subject] = np.einsum('sci,scj->cij', hs[lo:hi], hs[lo:hi], optimize=True)/m
            ux[subject] = np.einsum('scxi,scxj->cxij', hx[lo:hi], hx[lo:hi], optimize=True)/m
            gradient[subject] = np.einsum('sci,scj->cij', g[lo:hi], g[lo:hi], optimize=True)/m
        rows_used += int(b-a)
        if progress and (stop % 1024 < subjects_per_chunk or stop == data.n):
            progress(stop, rows_used)
    if rows_used != data.m:
        raise ValueError('Every original row must contribute exactly once')
    return dict(UU=uu, UX=ux, gradient=gradient, trial_counts=np.diff(data.offsets),
                rows_used=np.asarray(rows_used))


def rotation_energies(uu, ux, gradient, q):
    basis = symmetric_basis(len(q))
    rotated_basis = np.einsum('ia,rab,jb->rij', q, basis, q, optimize=True)
    transform = np.einsum('rij,kij->rk', rotated_basis, basis, optimize=True)
    eu = np.maximum(np.einsum('ra,nab,rb->nr', transform, uu, transform, optimize=True), 0.)
    ex = np.maximum(np.einsum('ia,nxij,ja->nax', q, ux, q, optimize=True), 0.)
    eg = np.maximum(np.einsum('ia,nij,ja->na', q, gradient, q, optimize=True), 0.)
    return eu, ex, eg


def scores(uu, ux, gradient, q):
    d = len(q); eu, ex, eg = rotation_energies(uu, ux, gradient, q)
    groups = dict(UU_all=eu, UU_edges=eu[:, d:]/2, U_curvature=eu[:, :d],
                  UX_entries=ex.reshape(len(ex), -1), U_gradient=eg)
    out, tables = {}, {}
    for group, energy in groups.items():
        tables[group] = ls.row_scores(energy)
        for metric, values in tables[group].items():
            out[group+'_subject_mean_'+metric] = float(values.mean())
            out[group+'_subject_median_'+metric] = float(np.median(values))
    columns = ls.row_scores(ex.transpose(0, 2, 1))
    for metric, values in columns.items():
        out['UX_columns_subject_mean_'+metric] = float(values.mean())
        out['UX_columns_subject_median_'+metric] = float(np.median(values, axis=0).mean())
    for metric in tables['UU_edges']:
        total = tables['UU_edges'][metric]+tables['UX_entries'][metric]
        out['UU_UX_subject_total_mean_'+metric] = float(total.mean())
        out['UU_UX_subject_total_median_'+metric] = float(np.median(total))
    out['control_UU_fro2'] = float(eu.sum(1).mean())
    out['control_UX_fro2'] = float(ex.sum((1, 2)).mean())
    out['control_U_gradient_fro2'] = float(eg.sum(1).mean())
    return out
