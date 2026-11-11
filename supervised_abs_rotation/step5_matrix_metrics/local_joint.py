"""Local support of actual UU/UX derivative entries, before pooling."""
import numpy as np

import local_support as ls


def evaluate(g, h, q, d):
    ii, jj = np.triu_indices(d, 1)
    gu = g[..., :d] @ q
    hu = q.T @ h[..., :d, :d] @ q
    hx = q.T @ h[..., :d, d:]
    diagonal = np.diagonal(hu, axis1=-2, axis2=-1)**2
    edges = hu[..., ii, jj]**2
    groups = dict(UU_edges=edges, UU_all=np.concatenate((diagonal, 2*edges), axis=-1),
                  U_curvature=diagonal, U_gradient=gu**2, UX_entries=(hx**2).reshape(*hx.shape[:-2], -1))
    out, individual = {}, {}
    for group, energy in groups.items():
        individual[group] = ls.pooled_scores(energy)
        for metric, value in individual[group].items():
            out[group+'_'+metric] = value
    columns = [ls.pooled_scores(hx[..., j]**2) for j in range(hx.shape[-1])]
    for key in columns[0]:
        out['UX_columns_'+key] = float(np.mean([row[key] for row in columns]))
    # One unit per effective edge: a prospective sum, with no learned weights.
    for key in individual['UU_edges']:
        out['UU_UX_total_'+key] = individual['UU_edges'][key]+individual['UX_entries'][key]
    out['control_UU_fro2'] = float(np.mean(np.sum(hu**2, axis=(-2, -1))))
    out['control_UX_fro2'] = float(np.mean(np.sum(hx**2, axis=(-2, -1))))
    out['control_U_gradient_fro2'] = float(np.mean(np.sum(gu**2, axis=-1)))
    return out
