"""Evaluate only the U-X coupling of a WY-fitted polynomial/kernel function.

fUX = f - empirical-U mean(f) - empirical-X mean(f) + overall product mean.
This is an explicit product reference between the whole U and X blocks,
without assuming independence of coordinates inside either block.
"""
import json
from pathlib import Path

import numpy as np

import run as wave
import joint_bank
import fitted_bank
import prepare_response_bank
import product_response as product
import all_trial_moments as am
import fast_projection as fp


def shared_x(data):
    counts = np.diff(data.offsets)
    if not np.all(counts == counts[0]):
        return None
    common = np.asarray(data.w[:counts[0], data.d:])
    for a, b in zip(data.offsets[:-1], data.offsets[1:]):
        if not np.array_equal(data.w[a:b, data.d:], common):
            return None
    return common


def centered_rows(data, coefficients, phi, heldout_coefficients=None):
    common = shared_x(data)
    cached = phi(common) if common is not None else None
    if cached is not None:
        mean_x = cached.mean(0)
    else:
        total = np.zeros(coefficients.shape[-1])
        for a in range(0, data.m, 8192):
            total += phi(data.w[a:min(a+8192, data.m), data.d:]).sum(0)
        mean_x = total/data.m
    centered = coefficients-coefficients.mean(0, keepdims=True)
    target = np.empty_like(data.y); error = np.zeros(len(data.channels))
    val_error = error.copy(); nvalid = 0
    heldout, validation_ids = heldout_coefficients if heldout_coefficients is not None else (None, set())
    for n, (a, b) in enumerate(zip(data.offsets[:-1], data.offsets[1:])):
        features = cached if cached is not None else phi(data.w[a:b, data.d:])
        target[a:b] = (features-mean_x) @ centered[n].T
        predicted = features @ coefficients[n].T
        error += np.sum((predicted-data.y[a:b])**2, axis=0)
        if n in validation_ids:
            val_error += np.sum((features @ heldout[n].T-data.y[a:b])**2, axis=0); nvalid += b-a
    variance = np.maximum(data.y.var(0), 1e-12)
    audit = dict(raw_all_rows_teacher_nmse=(error/data.m/variance).tolist(),
        shared_X_grid_verified=common is not None, coefficient_U_centering_error=float(np.max(np.abs(centered.mean(0)))),
        all_original_rows_evaluated=data.m, all_subjects=data.n,
        reference='product of whole-block empirical U and pooled joint X; not coordinate independence')
    if nvalid:
        audit.update(raw_subject_validation_nmse=(val_error/nvalid/variance).tolist(), raw_validation_rows=int(nvalid))
    return target, audit


def polynomial(data, degree=4):
    folder = data.folder.parent/'step5_matrix_metrics'/'joint_bank_seed391'
    bank = joint_bank.JointBank(data.folder, folder); saved = bank.load(degree)
    if bank.data.n != data.n or bank.data.channels != data.channels:
        raise ValueError('Polynomial teacher does not match physical subject/response rows')
    up = wave.sur.powers_for(data.d, degree); xp = wave.sur.powers_for(data.dx, degree)
    ul = {tuple(p): i for i, p in enumerate(up)}; xl = {tuple(p): i for i, p in enumerate(xp)}
    tensor = np.zeros((len(up), len(xp), len(data.channels)))
    for power, weight, coefficient in zip(saved['powers'], saved['weights'], saved['coef']):
        tensor[ul[tuple(power[:data.d])], xl[tuple(power[data.d:])]] += weight*coefficient
    z = (bank.data.u-saved['mean'][:data.d])/saved['scale'][:data.d]
    uphi = wave.sur.monomials(z, up, np.ones(len(up)))
    coefficients = (uphi @ tensor.reshape(len(up), -1)).reshape(data.n, len(xp), len(data.channels)).transpose(0, 2, 1)
    scaler = am.FrozenScaler(saved['mean'][data.d:], saved['scale'][data.d:])
    xmodel = fp.Polynomial(scaler, degree)
    def phi(x):
        return xmodel.features(x)/xmodel.weights
    target, audit = centered_rows(data, coefficients, phi)
    audit.update(teacher=f'joint_poly{degree}', source_manifest_sha256=bank.protocol['manifest_sha256'],
        validation_nmse=[float(bank.quality[f'poly{degree}', c]['validation_nmse']) for c in data.channels])
    return target, audit


def kernel(data, name='rbf2'):
    folder = data.folder.parent/'step5_matrix_metrics'/'UX_complexity_source_seed383'
    if not folder.exists():
        prepare_response_bank.prepare(data.folder, folder, estimators=('poly3', 'poly5', 'rbf1', 'rbf2'))
    bank = fitted_bank.FittedBank(data.folder, folder); bank.model(name)
    with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
        nodes = saved['x']
    common = shared_x(data)
    if common is not None and np.array_equal(common, nodes):
        with np.load(bank.folder/f'model_{name}.npz', allow_pickle=False) as saved:
            scaler = object.__new__(wave.sur.Scaler); scaler.mean, scaler.scale = saved['mean'].copy(), saved['scale'].copy()
            model = wave.sur.RBF(scaler, saved['centers'], float(name[3:])); model.set_coef(saved['coef'])
        prediction = (model.features(bank.u) @ model.coef).reshape(data.n, len(data.channels), len(nodes))
        coupling = prediction-prediction.mean(0, keepdims=True)-prediction.mean(2, keepdims=True)+prediction.mean((0, 2), keepdims=True)
        target = coupling.transpose(0, 2, 1).reshape(data.m, len(data.channels))
        raw = prediction.transpose(0, 2, 1).reshape(data.y.shape)
        audit = dict(teacher=f'{name}_observed_shared_X_bank', source_manifest_sha256=bank.source_digest,
            validation_nmse=[float(bank.quality[name, c]['validation_nmse']) for c in data.channels],
            raw_all_rows_teacher_nmse=np.mean((raw-data.y)**2, axis=0).tolist(),
            all_original_rows_evaluated=data.m, all_subjects=data.n, shared_X_grid_verified=True,
            reference='empirical whole U and joint-X blocks; complete recorded grid',
            U_centering_error=float(np.max(np.abs(coupling.mean(0)))),
            X_centering_error=float(np.max(np.abs(coupling.mean(2)))))
        audit['raw_all_rows_teacher_nmse'] = (np.mean((raw-data.y)**2, axis=0)/np.maximum(data.y.var(0), 1e-12)).tolist()
        return target, audit
    dictionary = product.XDictionary(bank.data, seed=383, centers=128, degree=4)
    mapping, quality = product.x_projection(bank, dictionary, nodes, seed=401)
    coefficients, _, _, heldout = product.prepare_function(bank, name, mapping, need_derivatives=False)
    validation_ids = set(range(data.n))-set(map(int, bank.train))
    target, audit = centered_rows(data, coefficients, dictionary.evaluate, (heldout, validation_ids))
    audit.update(teacher=f'{name}_times_generic_X_dictionary', source_manifest_sha256=bank.source_digest,
        X_interpolation_quality=quality, validation_nmse=audit['raw_subject_validation_nmse'])
    return target, audit
