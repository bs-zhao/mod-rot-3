"""M2 estimator E01/E02: rotation-equivariant polynomial sieve fitted on numeric (W, Y).

h_hat(v, x) = sum_{a,b} c_ab * phi_a(v) * psi_b(x)
  phi : ALL monomials of total degree <= du in v (span is invariant under any
        invertible linear map of v  =>  OLS fit is exactly equivariant: fitting
        once in base coordinates and rotating derivatives == refitting in every
        rotated coordinate system; the tiny ridge breaks this only at 1e-8 level)
  psi : 'legendre'  tensor Legendre polynomials of total degree <= dx in x
        'onehot'    one indicator per row of a fixed shared stimulus bank (CR3)

The estimator never sees the generating formula; it only receives arrays
(v, x or trial id, y).  `binary` y is fitted by least squares => estimate of
E[y|W] = p (linear-probability sieve).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch


def exponents(dim, deg):
    out = []
    for total in range(deg + 1):
        for c in itertools.combinations_with_replacement(range(dim), total):
            e = [0] * dim
            for k in c:
                e[k] += 1
            out.append(e)
    return torch.tensor(out, dtype=torch.long)


def monomials(v, E):
    # v (n, d), E (m, d) -> (n, m); differentiable
    return torch.prod(v.unsqueeze(1) ** E.to(v.device).unsqueeze(0), dim=-1)


def legendre_1d(x, deg):
    P = [torch.ones_like(x), x]
    for n in range(1, deg):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return torch.stack(P[:deg + 1], -1)  # (..., deg+1)


def legendre_tensor(x, E):
    deg = int(E.max()) if E.numel() else 0
    L = legendre_1d(x, max(deg, 1))  # (n, p, deg+1)
    idx = E.to(x.device).T.unsqueeze(0).expand(x.shape[0], -1, -1)  # (n, p, m)
    return torch.prod(torch.gather(L, 2, idx), dim=1)


class Sieve:
    def __init__(self, d, p, du, dx, xbasis="legendre", n_bank=None, ridge=1e-8, device="cpu"):
        self.Eu = exponents(d, du)
        self.xbasis = xbasis
        self.Ex = exponents(p, dx) if xbasis == "legendre" else None
        self.n_bank = n_bank
        self.ridge = ridge
        self.device = device
        self.coef = None

    def features(self, v, x=None, trial=None):
        Fu = monomials(v, self.Eu)
        if self.xbasis == "legendre":
            Fx = legendre_tensor(x, self.Ex)
            return (Fu.unsqueeze(2) * Fx.unsqueeze(1)).reshape(len(v), -1)
        raise RuntimeError("use fit_bank for onehot basis")

    # ---------- continuous X ----------
    def fit(self, v, x, y, chunk=20000):
        dev = self.device
        m = None
        G = b = None
        for i in range(0, len(v), chunk):
            F = self.features(torch.as_tensor(v[i:i + chunk], dtype=torch.float64, device=dev),
                              torch.as_tensor(x[i:i + chunk], dtype=torch.float64, device=dev))
            yy = torch.as_tensor(y[i:i + chunk], dtype=torch.float64, device=dev)
            if G is None:
                m = F.shape[1]
                G = torch.zeros(m, m, dtype=torch.float64, device=dev)
                b = torch.zeros(m, dtype=torch.float64, device=dev)
            G += F.T @ F
            b += F.T @ yy
        lam = self.ridge * torch.trace(G) / m
        self.coef = torch.linalg.solve(G + lam * torch.eye(m, dtype=G.dtype, device=dev), b)
        return self

    def h(self):
        coef, Eu, Ex = self.coef, self.Eu, self.Ex
        mu, mx = len(Eu), len(Ex)

        def h(v, x):  # single point
            Fu = monomials(v.unsqueeze(0), Eu)[0]
            Fx = legendre_tensor(x.unsqueeze(0), Ex)[0]
            return (coef.to(v).reshape(mu, mx) @ Fx) @ Fu
        return h

    # ---------- fixed bank (onehot trial basis) ----------
    def fit_bank(self, v_subj, trial_y):
        """v_subj (S, d); trial_y (S, K) with NaN for missing. One poly per trial."""
        dev = self.device
        F = monomials(torch.as_tensor(v_subj, dtype=torch.float64, device=dev), self.Eu)
        Y = torch.as_tensor(trial_y, dtype=torch.float64, device=dev)
        mask = ~torch.isnan(Y)
        if bool(mask.all()):
            G = F.T @ F
            lam = self.ridge * torch.trace(G) / G.shape[0]
            self.coef = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=dev), F.T @ Y)
        else:
            cols = []
            for k in range(Y.shape[1]):
                mk = mask[:, k]
                Fk = F[mk]
                G = Fk.T @ Fk
                lam = self.ridge * torch.trace(G) / G.shape[0]
                cols.append(torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=dev), Fk.T @ Y[mk, k]))
            self.coef = torch.stack(cols, 1)
        return self

    def h_bank(self):
        """Returns h(v, onehot_x) where x is the one-hot trial vector (so 'x-derivatives' are meaningless)."""
        coef, Eu = self.coef, self.Eu

        def h(v, x):
            Fu = monomials(v.unsqueeze(0), Eu)[0]
            return Fu @ (coef.to(v) @ x)
        return h
