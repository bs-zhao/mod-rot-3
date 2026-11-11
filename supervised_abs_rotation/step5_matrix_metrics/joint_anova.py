"""Joint U/X ANOVA component support under an explicit product reference."""
from collections import defaultdict

import numpy as np

import tail_support as ts


def scores(model, orthogonal_coef):
    powers, d = model.powers, model.d
    components = defaultdict(float)
    for p, value in zip(powers, orthogonal_coef**2):
        components[tuple(np.flatnonzero(p))] += float(value)
    labels = [s for s in components if s]
    energy = np.array([components[s] for s in labels])
    nu = np.array([sum(i < d for i in s) for s in labels])
    nx = np.array([sum(i >= d for i in s) for s in labels])
    groups = dict(all_components=np.ones(len(labels), dtype=bool), U_components=nu > 0,
                  UU_components=nu >= 2, UX_components=(nu > 0) & (nx > 0),
                  pair_UX_components=(nu == 1) & (nx == 1),
                  pair_UU_components=(nu == 2) & (nx == 0),
                  U_higher_order_components=(nu > 0) & (nu+nx >= 3))
    out = {}
    for group, mask in groups.items():
        for key, value in ts.scores(energy[mask]).items():
            out[group+'_'+key] = value
    # Marginal co-occurrence graph: a hyperedge contributes to every
    # contained pair. These are deliberately not exclusive pair Sobol shares.
    ux_edges, uu_edges = [], []
    for i in range(d):
        for j in range(d, powers.shape[1]):
            ux_edges.append(sum(components[s] for s in labels if i in s and j in s))
    for i in range(d):
        for j in range(i+1, d):
            uu_edges.append(sum(components[s] for s in labels if i in s and j in s))
    for group, values in (('UX_cooccurrence_edges', ux_edges), ('UU_cooccurrence_edges', uu_edges)):
        for key, value in ts.scores(values).items():
            out[group+'_'+key] = value
    column_scores = []
    for j in range(d, powers.shape[1]):
        u_groups = defaultdict(float)
        for s in labels:
            u = tuple(i for i in s if i < d)
            if u and j in s:
                u_groups[u] += components[s]
        column_scores.append(ts.scores(list(u_groups.values())))
    for key in column_scores[0]:
        out['X_columns_U_subsets_mean_'+key] = float(np.mean([r[key] for r in column_scores]))
        out['X_columns_U_subsets_worst_'+key] = float(np.max([r[key] for r in column_scores]))
    out['control_joint_variance'] = float(energy.sum())
    out['control_joint_function_norm2'] = float(orthogonal_coef @ orthogonal_coef)
    return out, dict(component_subsets=[[int(i) for i in s] for s in labels], component_energy=energy.tolist(),
                     constant_energy=components.get((), 0.))
