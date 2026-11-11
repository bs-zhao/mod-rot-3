"""Observed-support forward projection compression, with X-only nuisance free.

The target is a frozen full-data projection, not the true response function.
Compression coefficients use training subjects; errors use heldout subjects.
"""
import numpy as np

import coefficient_support as cs
import kernel_anova as ka


def symmetric(a):
    return (a + a.T)*.5


def nuisance_system(g, v, coef, has_u):
    """Partial out the common X-only span using a scale-aware pseudoinverse."""
    u, x = np.flatnonzero(has_u), np.flatnonzero(~has_u)
    xx = g[np.ix_(x, x)]
    scale = np.sqrt(np.maximum(np.diag(xx), 0.))
    active = scale > 0
    inv_scale = np.zeros_like(scale); inv_scale[active] = 1/scale[active]
    inverse = (inv_scale[:, None] * np.linalg.pinv(
        symmetric(xx*inv_scale[:, None]*inv_scale[None, :]), rcond=1e-10, hermitian=True)
        * inv_scale[None, :])
    a = inverse @ g[np.ix_(x, u)]
    b = inverse @ (g[x] @ coef)
    c = symmetric(g[np.ix_(u, u)] - g[np.ix_(u, x)] @ a)
    vv = symmetric(v[np.ix_(u, u)] - a.T @ v[np.ix_(x, u)] - v[np.ix_(u, x)] @ a
                   + a.T @ v[np.ix_(x, x)] @ a)
    ct = g[u] @ coef - g[np.ix_(u, x)] @ b
    cv = v[u] @ coef - v[np.ix_(u, x)] @ b - a.T @ (v[x] @ coef - v[np.ix_(x, x)] @ b)
    def residual_ss(gram):
        return np.einsum('pc,pq,qc->c', coef, gram, coef, optimize=True) - 2*np.sum(b*(gram[x] @ coef), axis=0) + np.sum(b*(gram[np.ix_(x, x)] @ b), axis=0)
    return dict(c=c, v=vv, ct=ct, cv=cv, ss_train=residual_ss(g), ss_validation=residual_ss(v),
                full_train=np.einsum('pc,pq,qc->c', coef, g, coef, optimize=True),
                x_rank=int(np.linalg.matrix_rank(xx*inv_scale[:, None]*inv_scale[None, :], tol=1e-10)))


class Transport:
    def __init__(self, powers, d, seed=392, interaction_only=False):
        self.structure = cs.Structure(powers, d, seed)
        candidate = self.structure.has_u & self.structure.has_x if interaction_only else self.structure.has_u
        self.u = np.flatnonzero(candidate)
        self.lookup = {row: i for i, row in enumerate(self.u)}

    def blocks(self, q):
        s = self.structure
        transform = ka.rotation_representation(s.up, s.uw, q, s.probes)
        out = []
        for pairs in s.blocks.values():
            if pairs[0][1] not in self.lookup:
                continue
            uid, rows = map(list, zip(*pairs))
            out.append((np.array([self.lookup[r] for r in rows]), transform[np.ix_(uid, uid)]))
        return out

    @staticmethod
    def square(matrix, blocks):
        left, out = np.empty_like(matrix), np.empty_like(matrix)
        for rows, transform in blocks:
            left[rows] = transform @ matrix[rows]
        for rows, transform in blocks:
            out[:, rows] = left[:, rows] @ transform.T
        return symmetric(out)

    @staticmethod
    def vector(matrix, blocks):
        out = np.empty_like(matrix)
        for rows, transform in blocks:
            out[rows] = transform @ matrix[rows]
        return out


def compress(c, v, ct, cv, ss_train, ss_validation, max_terms=128):
    """Forward orthogonal least squares; validation uses the selected train fit.

    Columns are normalized in the observed training metric. Relative singular
    thresholds remove numerical nulls, not small response effects. Exact
    prediction aliases are redundant and cannot increase the selected support.
    """
    if ss_train <= 0 or ss_validation <= 0:
        raise ValueError('Zero conditional target is unidentifiable')
    diag = np.maximum(np.diag(c), 0.)
    scale = np.sqrt(diag)
    active = diag > 1e-12*max(float(diag.max()), np.finfo(float).tiny)
    inverse = np.zeros_like(scale); inverse[active] = 1/scale[active]
    c = symmetric(c*inverse[:, None]*inverse[None, :])
    v = symmetric(v*inverse[:, None]*inverse[None, :])
    ct = ct*inverse/np.sqrt(ss_train); cv = cv*inverse/np.sqrt(ss_train)
    initial = float(ss_validation/ss_train)
    n, limit = len(c), min(max_terms, int(active.sum()))
    factors = np.zeros((n, limit)); basis = np.zeros((n, limit))
    corr = ct.copy(); residual_diag = np.diag(c).copy()
    chosen, coefficients, raw, train_tail = [], np.zeros(n), [initial], [1.]
    for k in range(limit):
        ok = active & (residual_diag > 1e-10)
        if not np.any(ok):
            break
        gains = np.full(n, -np.inf); gains[ok] = corr[ok]**2/residual_diag[ok]
        j = int(np.argmax(gains)); gain = float(gains[j])
        if gain < 1e-16:
            break
        root = np.sqrt(residual_diag[j])
        ell = (c[:, j] - factors[:, :k] @ factors[j, :k])/root
        b = -basis[:, :k] @ factors[j, :k]; b[j] += 1.; b /= root
        h = corr[j]/root
        coefficients += h*b
        chosen.append(j)
        ids = np.asarray(chosen)
        a = coefficients[ids]
        error = initial - 2*a @ cv[ids] + a @ v[np.ix_(ids, ids)] @ a
        if error < -1e-6*max(initial, 1.):
            raise ArithmeticError('Negative validation residual beyond roundoff')
        raw.append(max(0., float(error)))
        train_tail.append(max(0., train_tail[-1]-gain))
        factors[:, k], basis[:, k] = ell, b
        residual_diag -= ell**2; corr -= ell*h; active[j] = False
    # Best prefix is an explicit validation-error envelope, not a refit on test.
    tail = np.minimum.accumulate(raw)/initial
    out = {}
    for budget in (2, 4, 8, 16, 32, 64, 128):
        out[f'prediction_tail{budget}_l2'] = float(np.sqrt(tail[min(budget, len(tail)-1)]))
    censored = {}
    for eps in (.01, .03, .1):
        key = f'prediction_count_eps{int(round(eps*1000)):03d}'
        hit = np.flatnonzero(tail <= eps**2)
        censored[key] = not len(hit)
        if len(hit):
            k = int(hit[0]); before, after = tail[k-1], tail[k]
            out[key] = float(k-1+(before-eps**2)/max(before-after, 1e-300))
        else:
            out[key] = float(max_terms+1)
    audit = dict(selected=chosen, raw_relative_mse=(np.asarray(raw)/initial).tolist(),
                 train_relative_mse=train_tail, count_censored=censored,
                 training_conditional_ss=float(ss_train), validation_conditional_ss=float(ss_validation))
    return out, audit
