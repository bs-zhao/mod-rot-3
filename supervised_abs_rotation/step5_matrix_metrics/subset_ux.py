"""Minimum coordinate subsets explaining held-out UX contrasts."""
from itertools import combinations

import numpy as np


def subspace_classes(maps, tolerance=1e-6):
    """Cache W-only projection subspaces, including signed/permuted bases."""
    bases, projectors, membership = [], [], []
    d = len(maps[0])
    for q in maps:
        rows = []
        for size in range(1, d+1):
            for subset in combinations(range(d), size):
                basis = q[:, subset]; projector = basis @ basis.T
                distances = np.linalg.norm(np.asarray(projectors)-projector, axis=(1, 2)) if projectors else []
                match = np.flatnonzero(np.asarray(distances) < tolerance)
                if len(match):
                    index = int(match[0])
                else:
                    index = len(bases); bases.append(basis); projectors.append(projector)
                rows.append(dict(size=size, class_id=index, subset=subset))
        membership.append(rows)
    return bases, membership


def budget_curves(errors, memberships, dimension):
    """Error ratios[space,channel,X]; best held-out error at each budget.

    Selection of subsets uses pooled out-of-fold prediction error. These
    are descriptive cross-validated compression costs, not a nested-wrapper
    unbiased generalization estimate. A full-dimensional adequacy guard is
    applied separately.
    """
    curves = np.ones((len(memberships), dimension+1, *errors.shape[1:]))
    for case, rows in enumerate(memberships):
        for size in range(1, dimension+1):
            candidates = [errors[r['class_id']] for r in rows if r['size'] == size]
            curves[case, size] = np.minimum(curves[case, size-1], np.min(candidates, axis=0))
    return np.maximum(curves, 0.)


def scores(curve):
    """curve[budget,X] is squared relative prediction error, no amplitudes."""
    d = len(curve)-1; out = {}
    for k in range(1, d):
        tails = np.sqrt(curve[k])
        out[f'UX_minimum_U_subset_tail{k}_mean_l2'] = float(tails.mean())
        out[f'UX_minimum_U_subset_tail{k}_worst_l2'] = float(tails.max())
    area = curve[:-1].sum(0)
    out['UX_subset_budget_area_mean'] = float(area.mean())
    out['UX_subset_budget_area_worst'] = float(area.max())
    for eps in (.01, .03, .1, .3):
        counts, fractional = [], []
        for j in range(curve.shape[1]):
            hits = np.flatnonzero(curve[:, j] <= eps**2)
            if len(hits):
                k = int(hits[0]); counts.append(k)
                fractional.append(float(k-1+(curve[k-1, j]-eps**2)/max(curve[k-1, j]-curve[k, j], 1e-300)) if k else 0.)
            else:
                counts.append(d+1); fractional.append(d+1.)
        for name, values in (('count', counts), ('fractional_count', fractional)):
            out[f'UX_subset_{name}_eps{int(eps*1000):03d}_mean'] = float(np.mean(values))
            out[f'UX_subset_{name}_eps{int(eps*1000):03d}_worst'] = float(np.max(values))
    return out
