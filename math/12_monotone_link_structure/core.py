"""WY-only and link-invariant structural scores; NO generator imports.

Two mathematically different invariants of Y=h(g(Z,X)), h strictly increasing:
1. stimulus gradient projector: q q.T, q=grad_X Y / ||grad_X Y||
2. stimulus preference ordering: sign(Y(Z,Xa)-Y(Z,Xb)).
Both are unchanged by a shared unknown increasing link (up to finite-step error).
"""
from __future__ import annotations
import numpy as np


def orthogonal_plane(d: int, i: int, j: int, deg: float) -> np.ndarray:
    if d < 2 or not (0 <= i < d and 0 <= j < d and i != j):
        raise ValueError('two distinct latent axes are required')
    r = np.eye(d, dtype=float)
    a = np.deg2rad(deg)
    r[i, i] = r[j, j] = np.cos(a)
    r[i, j] = np.sin(a)
    r[j, i] = -np.sin(a)
    return r


def transform(W: np.ndarray, R: np.ndarray, zdim: int, mu: np.ndarray, sd: np.ndarray):
    w = np.asarray(W, dtype=float).copy()
    z = (w[:, :zdim] - mu) / sd
    w[:, :zdim] = z @ R.T
    return w


def inverse(Wr: np.ndarray, R: np.ndarray, zdim: int, mu: np.ndarray, sd: np.ndarray):
    w = np.asarray(Wr, dtype=float).copy()
    w[:, :zdim] = (w[:, :zdim] @ R) * sd + mu
    return w


def _pred(predict, W: np.ndarray):
    ans = np.asarray(predict(W), dtype=float).reshape(-1)
    if len(ans) != len(W) or not np.all(np.isfinite(ans)):
        raise ValueError('prediction must be one finite scalar per input')
    return ans


def _unit_stimulus_gradient(predict, W, zdim, step_x, eps=1e-12):
    n, p = W.shape
    dx = p-zdim
    g = np.empty((n, dx), dtype=float)
    for j in range(dx):
        plus = W.copy(); minus = W.copy()
        plus[:, zdim+j] += step_x[j]
        minus[:, zdim+j] -= step_x[j]
        g[:, j] = (_pred(predict, plus)-_pred(predict, minus))/(2*step_x[j])
    norm = np.linalg.norm(g, axis=1)
    unit = g / np.maximum(norm[:, None], eps)
    return unit, norm


def gradient_direction_strength(predict, W, zdim, step_x, step_z,
                                valid_cutoff=None, min_fraction=.60):
    """RMS Z-derivative of normalized stimulus-gradient projector.

    Common scalar h' cancels in infinite-sample, infinitesimal-step limit.
    Does NOT identify additive/multiplicative modules uniquely.
    """
    W = np.asarray(W, dtype=float)
    if W.shape[1]-zdim < 2:
        raise ValueError('need at least two continuously varying X columns')
    _, norm = _unit_stimulus_gradient(predict, W, zdim, step_x)
    if valid_cutoff is None:
        # Absolute cutoffs do not survive arbitrary monotone output changes.
        # A quantile gives scale invariance for the set of retained anchors.
        valid_cutoff = float(np.quantile(norm, .05))
    ok = (norm > max(valid_cutoff, 1e-10))
    if ok.mean() < min_fraction:
        raise RuntimeError('Too many points have vanishing stimulus gradients; directional score undefined')
    score = np.zeros(zdim, dtype=float)
    for j in range(zdim):
        pp = W[ok].copy(); mm = W[ok].copy()
        pp[:, j] += step_z[j]; mm[:, j] -= step_z[j]
        qp, _ = _unit_stimulus_gradient(predict, pp, zdim, step_x)
        qm, _ = _unit_stimulus_gradient(predict, mm, zdim, step_x)
        # Projective tangent: q and -q describe the same relative effect line.
        # This removes spurious apparent complexity from a sign-changing
        # common multiplicative amplitude (e.g. z1 in toy A).
        Pp = np.einsum("ni,nj->nij", qp, qp)
        Pm = np.einsum("ni,nj->nij", qm, qm)
        derivative = (Pp-Pm)/(2*step_z[j])
        score[j] = float(np.sqrt(np.mean(np.sum(derivative**2, axis=1))))
    return score, {'valid_anchor_fraction':float(ok.mean()),
                   'min_gradient_norm_quantile':float(valid_cutoff),
                   'n_eval':int(len(W)), 'n_accepted':int(ok.sum())}


def ordinal_contrast_strength(predict, Z, XA, XB, step_z):
    """Share of pairwise stimulus rankings that flip when latent axis j changes.

    XA/XB must be actual valid stimulus rows. This has no derivatives in X.
    Ordinal contrast is preserved under ANY shared strictly increasing output link.
    Counts can be zero; then this metric cannot identify axis relevance.
    """
    Z = np.asarray(Z, dtype=float)
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)
    if Z.ndim != 2 or XA.shape != XB.shape or len(Z) != len(XA):
        raise ValueError('Z, XA, XB must have the same row count')
    zdim = Z.shape[1]
    out = np.empty(zdim)
    def sign_at(z):
        a = _pred(predict, np.c_[z, XA]); b = _pred(predict, np.c_[z, XB])
        return np.sign(a-b)
    for j in range(zdim):
        plus = Z.copy(); minus = Z.copy()
        plus[:, j] += step_z[j]; minus[:, j] -= step_z[j]
        positive = sign_at(plus)
        negative = sign_at(minus)
        out[j] = float(np.mean(positive != negative))
    return out, {'n_pairs':int(len(Z)), 'n_positive_strength_axes':int(np.sum(out>0)),
                 'zero_signal':bool(np.all(out==0))}


def complexity(strength):
    a = np.asarray(strength, dtype=float).reshape(-1)
    if a.ndim != 1 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError('strengths must be finite nonnegative')
    energy = a*a
    tot = float(energy.sum())
    if tot < 1e-18:
        return {'l1':0.0,'effective_axes':None,'active_axes':0,
                'score_valid':False,'reason':'zero directional/ordinal signal'}
    weights = energy/tot
    nz = weights>0
    entropy = -float(np.sum(weights[nz]*np.log(weights[nz])))
    return {'l1':float(a.sum()),'effective_axes':float(np.exp(entropy)),
            'active_axes':int(np.count_nonzero(a>0)), 'score_valid':True}


def signed_permutation_angle_error(deg, target):
    return abs(((deg-target+45.) % 90.)-45.)
