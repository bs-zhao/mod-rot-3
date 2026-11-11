"""Local linear contrasts on recorded X nodes; no X-response interpolation."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

import run as wave
import link_ux as lu


def contrast_operator(x, scale, neighbors=32, rcond=1e-3):
    z = np.asarray(x, dtype=float)/scale
    m, dx = z.shape; k = min(neighbors, m-1)
    distance, indices = cKDTree(z).query(z, k=k+1)
    rows, columns, values, ranks, conditions = [], [], [], [], []
    for i in range(m):
        keep = indices[i] != i; nodes, radius = indices[i][keep][:k], distance[i][keep][:k]
        delta = z[nodes]-z[i]
        bandwidth = max(float(radius.max(initial=0.)), 1e-12)
        root_weight = np.exp(-.25*(radius/bandwidth)**2)
        u, s, vt = np.linalg.svd(root_weight[:, None]*delta, full_matrices=False)
        good = s > rcond*max(float(s.max(initial=0.)), np.finfo(float).tiny)
        inverse = np.zeros_like(s); inverse[good] = 1/s[good]
        operator = (vt.T*inverse) @ (u.T*root_weight[None, :])
        ranks.append(int(good.sum())); conditions.append(float(s[-1]/s[0]) if s[0] else 0.)
        for j in range(dx):
            row = i*dx+j
            rows.extend([row]*len(nodes)+[row]); columns.extend(nodes.tolist()+[i])
            values.extend(operator[j].tolist()+[-float(operator[j].sum())])
    matrix = csr_matrix((values, (rows, columns)), shape=(m*dx, m))
    return matrix, dict(requested_neighbors=neighbors, actual_neighbors=k, rcond=rcond,
        full_rank_node_fraction=float(np.mean(np.asarray(ranks) == dx)), rank_min=min(ranks),
        singular_ratio_min=min(conditions), singular_ratio_median=float(np.median(conditions)),
        X_nodes=m, X_dimension=dx, coordinate_units='all-original-row X standard deviations',
        interpretation='weighted local linear contrasts, not unique pointwise X derivatives')


def source_gradient(bank, name):
    bank.model(name)  # includes stored derivative reconstruction verification
    with np.load(bank.folder/f'model_{name}.npz', allow_pickle=False) as saved:
        scaler = object.__new__(wave.sur.Scaler)
        scaler.mean, scaler.scale = saved['mean'].copy(), saved['scale'].copy()
        coef, centers = saved['coef'].copy(), saved['centers'].copy()
    if name.startswith('poly'):
        model = wave.sur.Polynomial(scaler, int(name[4:])); model.set_coef(coef)
        gradient, _ = model.derivatives(bank.u)
    else:
        model = wave.sur.RBF(scaler, centers, float(name[3:])); model.set_coef(coef)
        z = scaler(bank.u); k = model.kernel(z); total = k @ model.beta
        first = (k @ model.first.reshape(len(model.centers), -1)).reshape(len(z), bank.data.d, -1)
        gradient = ((first-z[:, :, None]*total[:, None, :])/model.lengthscale**2).transpose(0, 2, 1)/scaler.scale
    return gradient.reshape(bank.data.n, len(bank.data.channels), bank.bank.shape[2], bank.data.d)


def moments(gradient, operator, dx, chunk=128):
    n, channels, m, d = gradient.shape
    out = {profile: np.empty((n, channels, dx, d, d)) for profile in ('global', 'point', 'column')}
    for a in range(0, n, chunk):
        b = min(a+chunk, n)
        rhs = gradient[a:b].transpose(2, 0, 1, 3).reshape(m, -1)
        transformed = operator @ rhs
        field = transformed.reshape(m, dx, b-a, channels, d).transpose(2, 3, 0, 4, 1)
        for profile in out:
            current = field if profile == 'global' else lu.balanced(field, profile)
            out[profile][a:b] = np.einsum('ncmix,ncmjx->ncxij', current, current, optimize=True)/m
    return out
