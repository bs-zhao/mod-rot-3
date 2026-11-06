# Adapted verbatim function bodies from user-supplied rot1_config.zip.
# Source hashes and method mapping are recorded in source_manifest.json.
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler
SEED=31


def within_subject_center(A, subj_idx):
    A = np.asarray(A, float)
    out = A.copy()
    for s in np.unique(subj_idx):
        mask = subj_idx == s
        out[mask] -= out[mask].mean(axis=0, keepdims=True)
    return out

def linear_xz_signatures(X, G, subj_idx):
    Xc = within_subject_center(X, subj_idx)
    Gc = within_subject_center(G, subj_idx)
    scale = Xc.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    Xz = Xc / scale
    B, *_ = np.linalg.lstsq(Xz, Gc, rcond=None)
    pred = Xz @ B
    ss_res = np.sum((Gc - pred) ** 2, axis=0)
    ss_tot = np.sum((Gc - Gc.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return B, r2

def level_signature(G, subj_idx):
    # Estimate subject mean effects, then average across subjects. This preserves
    # X-independent information that the centered XZ procedure removes.
    means = []
    for s in np.unique(subj_idx):
        means.append(G[subj_idx == s].mean(axis=0))
    return np.mean(np.vstack(means), axis=0, keepdims=True)

def nonlinear_basis_signatures(X, G, subj_idx, n_knots=5):
    """Ground-truth-blind rich basis: univariate splines + pairwise raw-X interactions."""
    X = np.asarray(X, float)
    basis_parts, names = [], []
    for j in range(X.shape[1]):
        spl = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
        Bj = spl.fit_transform(X[:, [j]])
        basis_parts.append(Bj)
        names.extend([f"x{j+1}_spline{r+1}" for r in range(Bj.shape[1])])
    Xstd = StandardScaler().fit_transform(X)
    for j in range(X.shape[1]):
        for l in range(j + 1, X.shape[1]):
            basis_parts.append((Xstd[:, j] * Xstd[:, l])[:, None])
            names.append(f"x{j+1}:x{l+1}")
    Phi = np.concatenate(basis_parts, axis=1)
    Phi = within_subject_center(Phi, subj_idx)
    Gc = within_subject_center(G, subj_idx)
    scaler = StandardScaler()
    Phi_z = scaler.fit_transform(Phi)
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 20), fit_intercept=False)
    ridge.fit(Phi_z, Gc)
    # sklearn multi-output coef_: [K, P]; transpose to functional-channel x latent.
    B = np.asarray(ridge.coef_).T
    pred = ridge.predict(Phi_z)
    ss_res = np.sum((Gc - pred) ** 2, axis=0)
    ss_tot = np.sum((Gc - Gc.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return B, r2, names, float(ridge.alpha_)

def contrast_signatures(X, G, subj_idx, n_bins=4):
    """Nonparametric context signatures: quantile-bin mean gradient contrasts."""
    Gc = within_subject_center(G, subj_idx)
    rows, names = [], []
    for j in range(X.shape[1]):
        edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        # make interior edges strictly usable despite ties
        for b in range(n_bins):
            if b == n_bins - 1:
                mask = (X[:, j] >= edges[b]) & (X[:, j] <= edges[b+1])
            else:
                mask = (X[:, j] >= edges[b]) & (X[:, j] < edges[b+1])
            if mask.sum() >= 5:
                rows.append(Gc[mask].mean(axis=0))
                names.append(f"x{j+1}_q{b+1}")
    return np.vstack(rows), names

def rich_x_basis_signatures(X, G, subj_idx, n_knots=5):
    """Generic X signature library.

    - cubic splines for each nonconstant observed X coordinate
    - every pairwise product of standardized nonconstant X coordinates

    This is ground-truth-blind.  In M4 it can represent the shared-X linear /
    quadratic channels and the X3*X4 interaction without using b.
    """
    X = np.asarray(X, float)
    Gc = within_subject_center(G, subj_idx)

    sd = X.std(axis=0, ddof=0)
    varying = np.where(sd > 1e-8)[0]
    parts, names = [], []

    for j in varying:
        spl = SplineTransformer(
            n_knots=int(n_knots),
            degree=3,
            include_bias=False,
        )
        Bj = spl.fit_transform(X[:, [j]])
        parts.append(Bj)
        names += [f"x{j+1}_spline{r+1}" for r in range(Bj.shape[1])]

    Xv = X[:, varying]
    Xz = StandardScaler().fit_transform(Xv)
    for a in range(len(varying)):
        for b in range(a + 1, len(varying)):
            parts.append((Xz[:, a] * Xz[:, b])[:, None])
            names.append(f"x{varying[a]+1}:x{varying[b]+1}")

    if not parts:
        raise RuntimeError("No varying X columns found.")

    Phi = np.concatenate(parts, axis=1)
    Phi = within_subject_center(Phi, subj_idx)
    Phi_z = StandardScaler().fit_transform(Phi)

    ridge = RidgeCV(
        alphas=np.logspace(-5, 3, 25),
        fit_intercept=False,
    )
    ridge.fit(Phi_z, Gc)

    # Multioutput RidgeCV coef_ = K x P.
    B = np.asarray(ridge.coef_, float).T
    pred = ridge.predict(Phi_z)
    ss_res = np.sum((Gc - pred) ** 2, axis=0)
    ss_tot = np.sum((Gc - Gc.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return B, r2, names, float(ridge.alpha_), varying
