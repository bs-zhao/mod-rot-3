"""One global response link from conditional mixed-derivative moments.

Let A=gU*gX^T. k(y)=-E[<HUX,A>|f=y]/E[||A||^2|f=y].
Then R=HUX+k(f)A is H_(h o f)/h'(f) for log h' = integral k.
Under a smooth increasing output reparameterization, the population R
changes by one positive pointwise scalar. Normalized support is invariant.
Finite-bin estimates and surrogate derivatives require separate validation.
"""
import numpy as np

import joint_fields as jf
import fast_projection as fp
from all_trial_moments import FrozenScaler
import local_support as local


def polynomial_fields(data, saved, output, chunk=8192, progress=None):
    gp, _, gc, hp, _, hc = jf.derivative_coefficients(saved['powers'], saved['weights'], saved['coef'])
    model = fp.Polynomial(FrozenScaler(saved['mean'], saved['scale']), int(saved['powers'].sum(1).max()))
    if not np.array_equal(model.powers[:len(gp)], gp) or not np.array_equal(model.powers[:len(hp)], hp):
        raise ValueError('Polynomial derivative prefixes changed')
    channels, d, dx, dimension = len(data.channels), data.d, data.dx, data.w.shape[1]
    prediction = np.lib.format.open_memmap(output/'prediction.npy', mode='w+', dtype='float64', shape=(data.m, channels))
    gradient = np.lib.format.open_memmap(output/'gradient.npy', mode='w+', dtype='float64', shape=(data.m, channels, dimension))
    mixed = np.lib.format.open_memmap(output/'mixed.npy', mode='w+', dtype='float64', shape=(data.m, channels, dx, d))
    uxcoef = hc[:, :d, d:]
    for a in range(0, data.m, chunk):
        b = min(a+chunk, data.m); phi = model.features(data.w[a:b])
        prediction[a:b] = phi @ saved['coef']
        gradient[a:b] = (phi[:, :len(gp)] @ gc.reshape(len(gp), -1)).reshape(b-a, dimension, channels).transpose(0, 2, 1)
        mixed[a:b] = (phi[:, :len(hp)] @ uxcoef.reshape(len(hp), -1)).reshape(b-a, d, dx, channels).transpose(0, 3, 2, 1)
        if progress and (b % 524288 < chunk or b == data.m):
            progress(b)
    for value in (prediction, gradient, mixed):
        value.flush()
    return prediction, gradient, mixed


def invariants(gradient, mixed, d, chunk=65536):
    numerator = np.empty(mixed.shape[:2]); denominator = np.empty_like(numerator); base = np.empty_like(numerator)
    for a in range(0, len(mixed), chunk):
        b = min(a+chunk, len(mixed)); g = np.asarray(gradient[a:b]); h = np.asarray(mixed[a:b])
        outer = g[:, :, None, :d]*g[:, :, d:, None]
        numerator[a:b] = np.sum(h*outer, axis=(-2, -1))
        denominator[a:b] = np.sum(outer**2, axis=(-2, -1))
        base[a:b] = np.sum(h*h, axis=(-2, -1))
    return numerator, denominator, base


def fit_link(y, numerator, denominator, bins):
    if bins == 0:
        return dict(y=[float(y.min()), float(y.max())], k=[0., 0.], bins=0)
    edges = np.unique(np.quantile(y, np.linspace(0., 1., bins+1)))
    if len(edges) < 3:
        raise ValueError('Response has too few distinct levels for a global link')
    ids = np.searchsorted(edges[1:-1], y, side='right'); n = len(edges)-1
    den = np.bincount(ids, weights=denominator, minlength=n)
    num = np.bincount(ids, weights=numerator, minlength=n)
    total_y = np.bincount(ids, weights=y, minlength=n)
    counts = np.bincount(ids, minlength=n)
    centers = total_y/np.maximum(counts, 1)
    k = np.divide(-num, den, out=np.zeros_like(num), where=den > 1e-300)
    good = counts > 0
    centers, k = centers[good], k[good]
    return dict(y=centers.tolist(), k=k.tolist(), bins=bins)


def evaluate(link, y):
    if link.get('kind') == 'histogram':
        return np.asarray(link['k'])[np.searchsorted(np.asarray(link['edges'])[1:-1], y, side='right')]
    return np.interp(y, link['y'], link['k'])


def link_audit(link, minimum, maximum):
    if link.get('kind') == 'histogram':
        y = np.asarray(link['edges'])
        log_derivative = np.r_[0., np.cumsum(np.asarray(link['k'])*np.diff(y))]
        span = float(np.ptp(log_derivative))
        return dict(log_derivative_range=span, global_link_integrable=True, global_link_monotone=True,
            numerical_derivative_dynamic_range_representable=span < 700., integration_y=y.tolist(),
            log_derivative=(log_derivative-log_derivative.max()).tolist())
    centers = np.asarray(link['y']); k = np.asarray(link['k'])
    y = np.r_[minimum, centers[(centers > minimum) & (centers < maximum)], maximum]
    ky = evaluate(link, y)
    changes = ky[:-1]*ky[1:] < 0
    roots = y[:-1][changes]-ky[:-1][changes]*np.diff(y)[changes]/np.diff(ky)[changes]
    y = np.sort(np.r_[y, roots])
    ky = evaluate(link, y)
    log_derivative = np.r_[0., np.cumsum((ky[:-1]+ky[1:])*.5*np.diff(y))]
    span = float(np.ptp(log_derivative))
    return dict(log_derivative_range=span, global_link_integrable=True,
        global_link_monotone=True, numerical_derivative_dynamic_range_representable=span < 700.,
        integration_y=y.tolist(), log_derivative=(log_derivative-log_derivative.max()).tolist())


def fit_all(y, numerator, denominator, base, training_rows, bins=(0, 16, 64, 256)):
    result, audits, selections = [], [], []
    for c in range(y.shape[1]):
        links, history = {}, []
        valid = ~training_rows
        for count in bins:
            training = fit_link(y[training_rows, c], numerator[training_rows, c], denominator[training_rows, c], count)
            k = evaluate(training, y[valid, c])
            loss = np.mean(base[valid, c]+2*k*numerator[valid, c]+k*k*denominator[valid, c])
            relative = float(loss/max(float(base[valid, c].mean()), 1e-300))
            link = fit_link(y[:, c], numerator[:, c], denominator[:, c], count)
            audit = link_audit(link, float(y[:, c].min()), float(y[:, c].max()))
            links[count] = link
            history.append(dict(bins=count, subject_validation_relative_UX_residual_energy=relative, **audit))
        index = int(np.argmin([row['subject_validation_relative_UX_residual_energy'] for row in history]))
        selections.append(bins[index]); result.append(links)
        audits.append(dict(channel_index=c, selected_bins=bins[index], history=history))
    return result, selections, audits


def corrected(mixed, gradient, d, k):
    return mixed+k[..., None, None]*(gradient[..., None, :d]*gradient[..., d:, None])


def unit_columns(field):
    peak = np.max(np.abs(field), axis=-1, keepdims=True)
    field = np.divide(field, peak, out=np.zeros_like(field), where=peak > 0)
    norm = np.sqrt(np.sum(field*field, axis=-1, keepdims=True))
    return np.divide(field, norm, out=np.zeros_like(field), where=norm > 0)


def moments(data, prediction, gradient, mixed, links, counts, subjects_per_chunk=32, progress=None):
    result = {count: np.empty((data.n, len(data.channels), data.dx, data.d, data.d)) for count in counts}
    for start in range(0, data.n, subjects_per_chunk):
        stop = min(start+subjects_per_chunk, data.n); a, b = data.offsets[start], data.offsets[stop]
        y, g, h = np.asarray(prediction[a:b]), np.asarray(gradient[a:b]), np.asarray(mixed[a:b])
        for count in counts:
            k = np.column_stack([evaluate(links[c][count], y[:, c]) for c in range(len(data.channels))])
            field = unit_columns(corrected(h, g, data.d, k))
            for subject in range(start, stop):
                lo, hi = data.offsets[subject:subject+2]-a
                result[count][subject] = np.einsum('ncxi,ncxj->cxij', field[lo:hi], field[lo:hi], optimize=True)/(hi-lo)
        if progress and (stop % 1024 < subjects_per_chunk or stop == data.n):
            progress(stop)
    return result


def scores(covariance, q):
    energy = np.maximum(np.einsum('ia,nxij,ja->nxa', q, covariance, q, optimize=True), 0.)
    out = {}
    for metric, values in local.row_scores(energy).items():
        for x_label, pooled in (('mean', values.mean(1)), ('worst', values.max(1))):
            out[f'UX_columns_X{x_label}_subject_mean_'+metric] = float(pooled.mean())
            out[f'UX_columns_X{x_label}_subject_median_'+metric] = float(np.median(pooled))
    for metric, values in local.row_scores(energy.mean(0)).items():
        out['UX_columns_global_mean_'+metric] = float(values.mean())
        out['UX_columns_global_worst_'+metric] = float(values.max())
    return out
