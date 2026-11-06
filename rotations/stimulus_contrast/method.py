"""Orthogonal rotation from stimulus variation of decoder effect covariances.

G[n,i,x] = d logit(F(z_n,x)) / dq_i. C[x] = E_n G G^T.
M[x] = C[x] - E_x C[x]. Minimize sum ||off(A M[x] A^T)||² /
sum ||M[x]||². The denominator is invariant under orthogonal A.
This removes a shared latent metric, not subject-average gradients.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


def matrices(g, contrast=True):
    g = np.asarray(g, dtype=float)
    if g.ndim != 3 or not np.isfinite(g).all():
        raise ValueError("Expected finite subject x axis x stimulus gradients")
    c = np.einsum('nix,njx->xij', g, g) / len(g)
    return c - c.mean(axis=0) if contrast else c


def loss(a, m):
    r = np.einsum('ia,xab,jb->xij', a, m, a, optimize=True)
    diag = np.diagonal(r, axis1=1, axis2=2)
    total = np.sum(m*m)
    if total <= 1e-24:
        return 0.0
    return float(max(0., (np.sum(r*r)-np.sum(diag*diag))/total))


def from_angles(v, k):
    v = np.asarray(v, dtype=float)
    s = np.zeros((k, k))
    i, j = np.triu_indices(k, 1)
    s[i,j] = v
    s[j,i] = -v
    return expm(s)


def optimize(m, seed=31, starts=6, maxiter=250):
    k = m.shape[-1]
    rng = np.random.default_rng(seed)
    runs, rotations = [], []
    for start in range(starts):
        v = np.zeros(k*(k-1)//2) if start == 0 else rng.normal(0, .7, k*(k-1)//2)
        result = minimize(lambda t: loss(from_angles(t,k),m), v,
                          method='BFGS', options={'maxiter':maxiter, 'gtol':1e-7})
        a = from_angles(result.x,k)
        rotations.append(a)
        runs.append(dict(start=start, loss=loss(a,m), success=bool(result.success),
                         message=str(result.message), nfev=int(result.nfev)))
    best = int(np.argmin([r['loss'] for r in runs]))
    return rotations[best], runs, np.stack(rotations)
