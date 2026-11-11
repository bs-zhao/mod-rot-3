"""UX support with each X column and local reference carrying equal weight."""
import numpy as np

import tail_support as ts


def scores(energy):
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 3 or not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError('Finite nonnegative energy[reference,U,X] required')
    # Compute support before pooling any reference or X-column magnitudes.
    table = [[ts.scores(energy[a, :, j]) for j in range(energy.shape[2])]
             for a in range(energy.shape[0])]
    out = {}
    for key in table[0][0]:
        values = np.array([[row[key] for row in anchor] for anchor in table])
        out['local_mean_UX_columns_mean_'+key] = float(values.mean())
        out['local_mean_UX_columns_worst_'+key] = float(values.max(1).mean())
        out['local_worst_UX_columns_mean_'+key] = float(values.mean(1).max())
    # A second descriptor pools normalized edge patterns, retaining which U
    # directions repeatedly interact with each X across local references.
    peak = energy.max(axis=1, keepdims=True)
    normalized = np.divide(energy, peak, out=np.zeros_like(energy), where=peak > 0)
    total = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, total, out=np.zeros_like(normalized), where=total > 0)
    matrix = normalized.mean(0)
    for key, value in ts.scores(matrix).items():
        if key == 'tail2_l2':
            # A global budget of two would compare top-entry dominance once
            # several X columns each contain multiple genuine interactions.
            # Keep the same two-per-column budget when flattening the graph.
            p = np.sort(matrix.ravel())[::-1]
            value = float(np.sqrt(p[2*matrix.shape[1]:].sum()/p.sum())) if p.sum() else 0.
            key = 'tail2_per_X_budget_l2'
        out['column_balanced_UX_entries_'+key] = value
    columns = [ts.scores(matrix[:, j]) for j in range(matrix.shape[1])]
    for key in columns[0]:
        out['column_balanced_UX_columns_mean_'+key] = float(np.mean([v[key] for v in columns]))
        out['column_balanced_UX_columns_worst_'+key] = float(np.max([v[key] for v in columns]))
    return out
