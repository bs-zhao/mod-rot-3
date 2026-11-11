"""Group sparse U-polynomial regression with unrestricted observed-X profiles."""
import numpy as np
from scipy.linalg import eigh


def full_error(gram, target_gram, energy, cutoff=1e-12):
    values, vectors = eigh((gram+gram.T)/2)
    good = values > max(values.max()*cutoff, 1e-15)
    inverse = (vectors[:, good]/values[good]) @ vectors[:, good].T
    captured = float(np.sum(inverse*target_gram))
    error = (energy-captured)/energy
    if error < -1e-6:
        raise ArithmeticError('Full polynomial projection has negative prediction energy')
    return max(error, 0.), int(good.sum())


def compress(gram, target_gram, energy, maximum=256, audit=True):
    """Greedy orthogonal least squares from exact all-subject statistics.

    target_gram[a,b]=<phi_a, F><F, phi_b>, where F can have arbitrarily
    many response/X-profile columns. Every selected U monomial gets its
    own unconstrained X profile. No X polynomial or link form is imposed.
    """
    scale = np.sqrt(np.maximum(np.diag(gram), 0.))
    active = scale > 0.
    scale[~active] = 1.
    original_g = gram/scale[:, None]/scale[None, :]
    original_s = target_gram/scale[:, None]/scale[None, :]
    g, s = original_g.copy(), original_s.copy()
    selected, curve, checks = [], [1.], []
    captured = 0.; maximum = min(maximum, len(gram))
    for step in range(maximum):
        diagonal = np.maximum(np.diag(g), 0.)
        valid = active & (diagonal > 1e-10)
        if not np.any(valid):
            break
        gain = np.divide(np.maximum(np.diag(s), 0.), diagonal, out=np.zeros(len(g)), where=valid)
        gain[~valid] = -np.inf
        j = int(np.argmax(gain)); pivot = diagonal[j]
        if gain[j] < energy*1e-14:
            break
        gc, sc, sjj = g[:, j].copy(), s[:, j].copy(), s[j, j]
        captured += sjj/pivot
        error = (energy-captured)/energy
        if error < -1e-6 or not np.isfinite(error):
            raise ArithmeticError('Greedy U-profile projection lost positive prediction energy')
        curve.append(max(0., error)); selected.append(j); active[j] = False
        s -= (gc[:, None]*sc[None, :]+sc[:, None]*gc[None, :])/pivot
        s += gc[:, None]*gc[None, :]*(sjj/pivot**2)
        g -= gc[:, None]*gc[None, :]/pivot
        if audit and (len(selected) in (2, 8, 32, 128, maximum)):
            gg = original_g[np.ix_(selected, selected)]
            ss = original_s[np.ix_(selected, selected)]
            exact = (energy-float(np.trace(np.linalg.solve(gg, ss))))/energy
            difference = abs(error-exact)
            if difference > 1e-6:
                raise ArithmeticError(f'Greedy profile loss disagrees with direct selected-span OLS: {difference}')
            checks.append(dict(budget=len(selected), direct_OLS_error=max(exact, 0.), absolute_difference=difference))
    return np.asarray(curve), dict(selected_term_indices=selected, direct_OLS_checks=checks,
        retained_terms=len(selected), candidate_terms=len(gram), residual_relative_mse=float(curve[-1]))


def scores(curves, maximum=256):
    """Curves[X,k] are each normalized by that X contrast's own energy."""
    result = {}
    for budget in (2, 4, 8, 16, 32, 64, 128, 256):
        if budget > maximum:
            continue
        tail = np.sqrt([curve[min(budget, len(curve)-1)] for curve in curves])
        result[f'UX_U_profile_terms_tail{budget}_mean_l2'] = float(tail.mean())
        result[f'UX_U_profile_terms_tail{budget}_worst_l2'] = float(tail.max())
    for eps in (.01, .03, .1):
        counts = []
        for curve in curves:
            hits = np.flatnonzero(curve <= eps**2)
            if len(hits):
                k = int(hits[0])
                count = k-1+(curve[k-1]-eps**2)/max(curve[k-1]-curve[k], 1e-300) if k else 0.
            else:
                count = maximum+1.
            counts.append(count)
        result[f'UX_U_profile_terms_fractional_count_eps{int(eps*1000):03d}_mean'] = float(np.mean(counts))
        result[f'UX_U_profile_terms_fractional_count_eps{int(eps*1000):03d}_worst'] = float(np.max(counts))
    return result
