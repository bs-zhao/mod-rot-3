"""UX support at every observed point, before trial/subject aggregation."""
import numpy as np

import local_support as local
import joint_fields
import fast_projection
from all_trial_moments import FrozenScaler


TRIAL_POOLS = ('mean', 'q25', 'median', 'q75')
X_POOLS = ('mean', 'worst')


def make_field(data, saved, path, chunk=16384):
    _, _, _, powers, _, coefficients = joint_fields.derivative_coefficients(
        saved['powers'], saved['weights'], saved['coef'])
    model = fast_projection.Polynomial(FrozenScaler(saved['mean'], saved['scale']), int(powers.sum(1).max()))
    if not np.array_equal(powers, model.powers):
        raise ValueError('Derivative polynomial basis mismatch')
    coef = coefficients[:, :data.d, data.d:]
    field = np.lib.format.open_memmap(path, mode='w+', dtype='float64',
        shape=(data.m, len(data.channels), data.dx, data.d))
    for a in range(0, data.m, chunk):
        b = min(a+chunk, data.m)
        values = model.features(data.w[a:b]) @ coef.reshape(len(powers), -1)
        field[a:b] = values.reshape(b-a, data.d, data.dx, len(data.channels)).transpose(0, 3, 2, 1)
    field.flush()
    return field


def point_scores(field, q):
    # Rescale before squaring/rotation to retain support under extreme
    # positive changes of pointwise X-column units.
    peak = np.max(np.abs(field), axis=-1, keepdims=True)
    unit = np.divide(field, peak, out=np.zeros_like(field), where=peak > 0)
    return local.row_scores((unit @ q)**2)


def subject_statistics(field, offsets, q, subjects_per_chunk=32, progress=None):
    output, keys = None, None
    for start in range(0, len(offsets)-1, subjects_per_chunk):
        stop = min(start+subjects_per_chunk, len(offsets)-1)
        a, b = offsets[start], offsets[stop]
        table = point_scores(np.asarray(field[a:b]), q)
        if keys is None:
            keys = tuple(table)
            output = np.empty((len(offsets)-1, len(TRIAL_POOLS), field.shape[1], len(X_POOLS), len(keys)))
        values = np.stack([table[key] for key in keys], axis=-1)
        for subject in range(start, stop):
            lo, hi = offsets[subject:subject+2]-a
            row = values[lo:hi]
            trial = np.concatenate((row.mean(0, keepdims=True), np.quantile(row, [.25, .5, .75], axis=0)), axis=0)
            output[subject, :, :, 0] = trial.mean(axis=2)
            output[subject, :, :, 1] = trial.max(axis=2)
        if progress:
            progress(stop)
    return output, keys


def scores(statistics, keys, channel):
    out = {}
    for trial, label in enumerate(TRIAL_POOLS):
        for x, x_label in enumerate(X_POOLS):
            values = statistics[:, trial, channel, x]
            for subject_label, pooled in (('mean', values.mean(0)), ('median', np.median(values, axis=0))):
                for key, loss in zip(keys, pooled):
                    out[f'UX_columns_X{x_label}_trial{label}_subject{subject_label}_{key}'] = float(loss)
    return out
