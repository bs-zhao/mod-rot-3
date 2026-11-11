"""Generic U-kernel/polynomial times X-dictionary WY function representation."""
import json

import numpy as np
from scipy.linalg import solve

import run as wave
import fast_projection as fp
import joint_fields as jf
import link_ux as lu

PROFILES = ('raw_global',)+lu.PROFILES


class XDictionary:
    def __init__(self, data, seed=383, centers=64, degree=2):
        x = data.w[:, data.d:]
        trial = np.load(data.folder/'trial_index.npy', allow_pickle=False)
        keep = trial % 5 != 0
        self.scaler = wave.sur.Scaler(x, 1, row_mask=keep)
        self.center_ids = np.random.default_rng(seed).choice(np.flatnonzero(keep), min(centers, int(keep.sum())), replace=False)
        self.kernel = wave.sur.RBF(self.scaler, x[self.center_ids], 2.)
        self.poly = fp.Polynomial(self.scaler, degree)
        self.powers, self.weights = self.poly.powers, self.poly.weights
        p = len(self.powers)
        gp, gw, gc, _, _, _ = jf.derivative_coefficients(self.powers, self.weights, np.eye(p))
        self.gp, self.gw, self.gc = gp, gw, gc
        self.first = self.kernel.centers[:, :, None]*self.kernel.transform[:, None, :]

    def evaluate(self, x, gradient=False):
        z = self.scaler(x); k = self.kernel.kernel(z); radial = k @ self.kernel.transform
        phi = np.column_stack((self.poly.features(x), radial))
        if not gradient:
            return phi
        gp = wave.sur.monomials(z, self.gp, self.gw) @ self.gc.reshape(len(self.gp), -1)
        gp = gp.reshape(len(x), z.shape[1], len(self.powers)).transpose(0, 2, 1)
        first = (k @ self.first.reshape(len(self.kernel.centers), -1)).reshape(len(x), z.shape[1], -1)
        gr = (first-z[:, :, None]*radial[:, None, :])/self.kernel.lengthscale**2
        return phi, np.concatenate((gp, gr.transpose(0, 2, 1)), axis=1)


def ridge_fit(gram, cross, ridge):
    p = len(gram); penalty = np.eye(p); penalty[0, 0] = 0.
    scale = max(float(np.trace(gram[1:, 1:]))/max(p-1, 1), 1e-12)
    return solve(gram+ridge*scale*penalty, cross, assume_a='sym')


def x_projection(bank, dictionary, x, seed=401):
    """Choose X interpolation ridge using training U subjects only."""
    phi = dictionary.evaluate(x); m = len(x)
    order = np.random.default_rng(seed).permutation(m)
    train_x, valid_x = order[:int(.75*m)], order[int(.75*m):]
    target = bank.bank[bank.train].transpose(2, 0, 1).reshape(m, -1)
    channels = len(bank.data.channels); n = len(bank.train)
    variance = np.maximum(target[train_x].reshape(len(train_x), n, channels).var(axis=(0, 1)), 1e-12)
    pt, pv = phi[train_x], phi[valid_x]
    gt, cross = pt.T @ pt, pt.T @ target[train_x]
    trials = []
    for ridge in (1e-8, 1e-6, 1e-4, .01):
        coef = ridge_fit(gt, cross, ridge)
        error = ((pv @ coef-target[valid_x]).reshape(len(valid_x), n, channels)**2).mean(axis=(0, 1))/variance
        trials.append((float(error.mean()), ridge, error))
    _, ridge, error = min(trials, key=lambda r: r[0])
    mapping = ridge_fit(phi.T @ phi, phi.T, ridge).T
    return mapping, dict(ridge=ridge, validation_nmse=error.tolist(), feature_count=phi.shape[1],
        heldout_x_probes=len(valid_x), training_u_subjects_used_to_select_ridge=len(bank.train),
        validation_u_subjects_used_to_select_ridge=0, interpretation='X interpolation check on source response bank')


def prepare_function(bank, name, mapping):
    # Reconstruct and verify the original U basis through the existing loader.
    bank.model(name)
    with np.load(bank.folder/f'model_{name}.npz', allow_pickle=False) as saved:
        scaler = object.__new__(wave.sur.Scaler)
        scaler.mean, scaler.scale = saved['mean'].copy(), saved['scale'].copy()
        original = saved['coef'].copy(); centers = saved['centers'].copy()
    model = (wave.sur.Polynomial(scaler, int(name[4:])) if name.startswith('poly') else
             wave.sur.RBF(scaler, centers, float(name[3:])))
    channels, probes = bank.bank.shape[1:]
    coef = np.einsum('fcm,mk->fck', original.reshape(len(original), channels, probes), mapping, optimize=True)
    model.set_coef(coef.reshape(len(coef), -1))
    phi = model.features(bank.u)
    full = (phi @ model.coef).reshape(bank.data.n, channels, -1)
    g, h = model.derivatives(bank.u)
    p, d = mapping.shape[1], bank.data.d
    g, h = g.reshape(bank.data.n, channels, p, d), h.reshape(bank.data.n, channels, p, d, d)
    # Reproduce a training-only U fit to audit actual held-out original Y.
    fit = json.loads((bank.folder/f'fit_{name}.json').read_text())
    pt = phi[bank.train]
    raw = bank.bank[bank.train].reshape(len(bank.train), -1)
    train_coef = ridge_fit(pt.T @ pt, pt.T @ raw, fit['ridge'])
    train_coef = np.einsum('fcm,mk->fck', train_coef.reshape(len(train_coef), channels, probes), mapping, optimize=True)
    validation_predictions = (phi @ train_coef.reshape(len(train_coef), -1)).reshape(bank.data.n, channels, p)
    return full, g, h, validation_predictions


def moments(bank, dictionary, name, mapping, progress=None):
    data = bank.data; channels, d, dx = len(data.channels), data.d, data.dx
    f, gu, hu, heldout = prepare_function(bank, name, mapping)
    counts = np.diff(data.offsets)
    shared = bool(np.all(counts == counts[0]))
    common_x = np.asarray(data.w[:counts[0], d:])
    if shared:
        for a, b in zip(data.offsets[:-1], data.offsets[1:]):
            if not np.array_equal(data.w[a:b, d:], common_x):
                shared = False; break
    cache = dictionary.evaluate(common_x, gradient=True) if shared else None
    covariance = {profile: np.empty((data.n, channels, dx, d, d)) for profile in PROFILES}
    train_set = set(map(int, bank.train)); sse_all = np.zeros(channels); sse_valid = sse_all.copy()
    train_sum, train_sq = sse_all.copy(), sse_all.copy(); nt = nv = rows_used = 0
    for subject, (a, b) in enumerate(zip(data.offsets[:-1], data.offsets[1:])):
        phi, gx = cache if shared else dictionary.evaluate(data.w[a:b, d:], gradient=True)
        prediction = np.einsum('ck,mk->mc', f[subject], phi, optimize=True)
        y = data.y[a:b]; sse_all += np.sum((prediction-y)**2, axis=0)
        if subject in train_set:
            train_sum += y.sum(0); train_sq += np.sum(y*y, axis=0); nt += len(y)
        else:
            pred_valid = np.einsum('ck,mk->mc', heldout[subject], phi, optimize=True)
            sse_valid += np.sum((pred_valid-y)**2, axis=0); nv += len(y)
        grad_u = np.einsum('cki,mk->mci', gu[subject], phi, optimize=True)
        grad_x = np.einsum('ck,mkj->mcj', f[subject], gx, optimize=True)
        huu = np.einsum('ckij,mk->mcij', hu[subject], phi, optimize=True)
        hux = np.einsum('cki,mkj->mcij', gu[subject], gx, optimize=True)
        gradient = np.concatenate((grad_u, grad_x), axis=-1)
        h_u = np.concatenate((huu, hux), axis=-1)
        fields = dict(raw_global=hux, **lu.fields(gradient, h_u, d))
        for profile, field in fields.items():
            covariance[profile][subject] = np.einsum('mcix,mcjx->cxij', field, field, optimize=True)/(b-a)
        rows_used += int(b-a)
        if progress and ((subject+1) % 1000 == 0 or subject+1 == data.n):
            progress(subject+1, rows_used)
    variance = np.maximum(train_sq/nt-(train_sum/nt)**2, 1e-12)
    if rows_used != data.m:
        raise ValueError('Raw-W/Y evaluation omitted original trial rows')
    return covariance, dict(raw_heldout_subject_validation_nmse=(sse_valid/nv/variance).tolist(),
        raw_all_rows_nmse=(sse_all/data.m/variance).tolist(), raw_validation_rows=nv,
        raw_train_subjects=len(bank.train), all_subjects=data.n, all_original_rows=rows_used,
        shared_x_cache_numerically_verified=shared, derivative_X_units='training-standardized X',
        derivative_U_units='common-scalar-standardized U')
