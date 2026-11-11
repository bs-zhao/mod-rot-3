"""Prespecified pure-UX and U-degree-resolved projection support descriptors."""
from collections import defaultdict

import numpy as np

import tail_support as ts


def degree_support(powers, coef, d):
    order = powers[:, :d].sum(1)
    has_x = powers[:, d:].sum(1) > 0
    out, audit = {}, {}
    for scope, scope_mask in (('U', order > 0), ('UX', (order > 0) & has_x)):
        tables = defaultdict(list)
        for k in range(1, int(order.max())+1):
            keep = scope_mask & (order == k)
            if not np.any(keep):
                continue
            p, e = powers[keep], coef[keep]**2
            groups = defaultdict(float)
            blocks = defaultdict(list)
            for power, energy in zip(p, e):
                groups[tuple(np.flatnonzero(power[:d]))] += energy
                blocks[tuple(power[d:])].append(energy)
            level = dict(terms=ts.scores(e), U_axis_subsets=ts.scores(list(groups.values())))
            total = float(e.sum())
            block_scores = [ts.scores(v) for v in blocks.values() if sum(v) > 1e-12*total]
            if block_scores:
                level['X_profile_equal_blocks'] = {key: float(np.mean([row[key] for row in block_scores]))
                                                   for key in block_scores[0]}
            for group, scores in level.items():
                for metric, loss in scores.items():
                    key = f'{scope}_{group}_{metric}'
                    out[f'Uorder{k}_'+key] = loss
                    tables[key].append(loss)
            audit[f'{scope}_order{k}'] = dict(coefficient_energy=total,
                active_x_blocks=len(block_scores), total_x_blocks=len(blocks))
        for key, values in tables.items():
            out['equal_orders_'+key] = float(np.mean(values))
    return out, audit


def scores(model, orthogonal_coef):
    powers, d = model.powers, model.d
    has_u = powers[:, :d].sum(1) > 0
    has_x = powers[:, d:].sum(1) > 0
    out, audit = degree_support(powers, orthogonal_coef, d)
    for scope, mask in (('U_dependent', has_u), ('UX_coupling', has_u & has_x)):
        projected = np.where(mask, orthogonal_coef, 0.)
        block_scores, block_audit = model.scores(projected)
        for key, value in block_scores.items():
            if key.startswith('control_'):
                out['control_'+scope+'_'+key[8:]] = value
            else:
                out[scope+'_'+key] = value
        audit[scope] = dict(block_spectra=block_audit)
    return out, audit
