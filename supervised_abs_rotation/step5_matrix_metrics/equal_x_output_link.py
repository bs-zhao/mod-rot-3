"""Global response-only correction with conditional X-column balancing."""
import numpy as np

import global_output_link as global_link


def column_invariants(gradient, mixed, d, chunk=65536):
    shape = mixed.shape[:-1]
    numerator, denominator, base = [np.empty(shape) for _ in range(3)]
    for a in range(0, len(mixed), chunk):
        b = min(a+chunk, len(mixed)); g, h = np.asarray(gradient[a:b]), np.asarray(mixed[a:b])
        outer = g[:, :, None, :d]*g[:, :, d:, None]
        numerator[a:b] = np.sum(h*outer, axis=-1)
        denominator[a:b] = np.sum(outer**2, axis=-1)
        base[a:b] = np.sum(h*h, axis=-1)
    return numerator, denominator, base


def edges_for(y, bins, bounds):
    if bins == 0:
        return np.asarray(bounds)
    return np.unique(np.r_[bounds[0], np.quantile(y, np.linspace(0., 1., bins+1))[1:-1], bounds[1]])


def balanced_rows(y, numerator, denominator, base, training, reference_bins=64):
    """One reference weighting is shared by every candidate resolution."""
    edges = edges_for(y[training], reference_bins, (float(y.min()), float(y.max())))
    ids = np.searchsorted(edges[1:-1], y, side='right'); count = len(edges)-1
    counts = np.bincount(ids[training], minlength=count)
    means = np.column_stack([np.bincount(ids[training], weights=denominator[training, j], minlength=count)
                             /np.maximum(counts, 1) for j in range(denominator.shape[1])])
    result = [np.zeros(len(y)) for _ in range(3)]
    for j in range(denominator.shape[1]):
        normalizer = means[ids, j]
        for target, values in zip(result, (numerator, denominator, base)):
            target += np.divide(values[:, j], normalizer, out=np.zeros(len(y)), where=normalizer > 0)
    if not all(np.isfinite(value).all() for value in result):
        raise ArithmeticError('Nonfinite conditional column balancing')
    return result, dict(reference_bins=reference_bins, edges=edges.tolist(),
        per_bin_training_rows=counts.tolist(), conditional_outer_energy_by_X=means.tolist())


def fit_histogram(y, numerator, denominator, bins, bounds):
    edges = edges_for(y, bins, bounds)
    ids = np.searchsorted(edges[1:-1], y, side='right'); count = len(edges)-1
    if bins:
        num = np.bincount(ids, weights=numerator, minlength=count)
        den = np.bincount(ids, weights=denominator, minlength=count)
        k = np.divide(-num, den, out=np.zeros_like(num), where=den > 0)
    else:
        k = np.zeros(1)
    return dict(kind='histogram', edges=edges.tolist(), k=k.tolist(), bins=bins)


def fit_all(y, numerator, denominator, base, training_rows, bins=(0, 16, 64, 256), reference_bins=64):
    links, selections, audits = [], [], []
    for c in range(y.shape[1]):
        yc = np.asarray(y[:, c]); bounds = (float(yc.min()), float(yc.max()))
        (nr, dr, br), reference = balanced_rows(yc, numerator[:, c], denominator[:, c], base[:, c], training_rows, reference_bins)
        valid = ~training_rows; history = []
        for count in bins:
            fitted = fit_histogram(yc[training_rows], nr[training_rows], dr[training_rows], count, bounds)
            k = global_link.evaluate(fitted, yc[valid])
            loss = float(np.mean(br[valid]+2*k*nr[valid]+k*k*dr[valid])/max(float(br[valid].mean()), 1e-300))
            history.append(dict(bins=count, subject_validation_balanced_UX_residual_energy=loss))
        chosen = bins[int(np.argmin([r['subject_validation_balanced_UX_residual_energy'] for r in history]))]
        (nr, dr, br), full_reference = balanced_rows(yc, numerator[:, c], denominator[:, c], base[:, c], np.ones(len(yc), dtype=bool), reference_bins)
        fitted = {count: fit_histogram(yc, nr, dr, count, bounds) for count in bins}
        for row in history:
            row.update(global_link.link_audit(fitted[row['bins']], *bounds))
        links.append(fitted); selections.append(chosen)
        audits.append(dict(channel_index=c, selected_bins=chosen, history=history,
            CV_reference=reference, all_subject_refit_reference=full_reference))
    return links, selections, audits
