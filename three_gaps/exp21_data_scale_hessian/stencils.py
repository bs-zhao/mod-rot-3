"""EXP21 finite-scale second derivatives of the frozen decoder (probability channel), recomputed for every rotation R.

Coordinates: u = D_R^{-1} R V, A = D_R R (frozen EXP13 semantics), so a unit step in u_i is the physical displacement dV_i = A^T e_i (row i of A).
  UU diagonal : [f(+h e_i) - 2 f(0) + f(-h e_i)] / h^2
  UU off diag : [f(++) - f(+-) - f(-+) + f(--)] / (4 h^2)
  UX hybrid (F2, U finite / X autograd): [d_x f(v + h e_i, x) - d_x f(v - h e_i, x)] / (2h)
  UX ambient (F1, both finite, X step in raw X units): four-corner mixed difference / (4 h_U h_Xj)   <- extrapolating decoder query
The finite-difference tensors are NOT second-order tensors: R changes the query positions, so they are rebuilt for every R (no R H R^T shortcut).
Centres are a FIXED grid (subjects x bank stimuli) shared by every R and by the matched autograd baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp20_latent_integration"))
import x7core as X  # noqa: E402
import losses2 as L2  # noqa: E402
from common.rotlib import haar  # noqa: E402


class FD:
    def __init__(self, st, ch="prob", nsub=64, nx=64, seed=3, dtype=torch.float32):
        assert ch == "prob", "finite-difference stencils are implemented for the probability channel"
        self.st, self.dtype, self.d, self.p = st, dtype, st.d, st.p
        rng = np.random.default_rng(seed)
        sub = st.cache_ids[:nsub]
        xi = np.sort(rng.choice(len(st.X), nx, replace=False))
        self.sub, self.xi = sub, xi
        V0 = torch.tensor(st.V[sub], dtype=dtype)
        X0 = st.X[xi].to(dtype)
        n = len(sub) * len(xi)
        self.Vc = V0.repeat_interleave(len(xi), 0).to(X.DEV)
        self.Xc = X0.repeat(len(sub), 1).to(X.DEV)
        self.n = n
        import copy
        self.net = torch.nn.Module()
        self.net.decoder = copy.deepcopy(st.net.decoder).to(dtype).to(X.DEV)     # never cast the shared frozen network in place
        self.mu = torch.tensor(st.mu_z, dtype=dtype, device=X.DEV)
        self.MinvT = torch.tensor(st.Minv, dtype=dtype, device=X.DEV).T
        self.C = torch.tensor(X.population_cov(st.V[st.tr]), dtype=dtype, device=X.DEV)   # fixed full-TRAIN reference covariance
        self.sdX = st.X.std(0).to(dtype).to(X.DEV)
        self.lo, self.hi = st.X.min(0).values.to(dtype).to(X.DEV), st.X.max(0).values.to(dtype).to(X.DEV)

    def f(self, V, Xs):
        z = self.mu + V @ self.MinvT
        lg = self.net.decoder(torch.cat([Xs, z], 1))
        return torch.sigmoid(lg[:, 1] - lg[:, 0])

    def energies(self, M, hU, mode="hybrid", hXc=None):
        """(e_ux (d,p), e_uu (d,d)) energies for rotation M; mode 'hybrid' (U finite, X autograd) or 'ambient' (X step = hXc * SD(X_j))."""
        M = M.to(self.dtype).to(X.DEV)
        A = X.eff_A(M, self.C)
        steps = hU * A                                          # rows: dV for +h in u_i (V = R^T D U => dV = A^T e_i)
        d, p = self.d, self.p
        Vc, Xc = self.Vc, self.Xc
        f0 = self.f(Vc, Xc)
        e_uu = torch.zeros(d, d, dtype=self.dtype, device=X.DEV)
        fp = [self.f(Vc + steps[i], Xc) for i in range(d)]
        fm = [self.f(Vc - steps[i], Xc) for i in range(d)]
        for i in range(d):
            e_uu[i, i] = (((fp[i] + fm[i] - 2 * f0) / hU ** 2) ** 2).mean()
        for i in range(d):
            for k in range(i + 1, d):
                D = (self.f(Vc + steps[i] + steps[k], Xc) - self.f(Vc + steps[i] - steps[k], Xc)
                     - self.f(Vc - steps[i] + steps[k], Xc) + self.f(Vc - steps[i] - steps[k], Xc)) / (4 * hU ** 2)
                e_uu[i, k] = e_uu[k, i] = (D ** 2).mean()
        e_ux = torch.zeros(d, p, dtype=self.dtype, device=X.DEV)
        if mode == "hybrid":
            def gradx(Vq):
                with torch.enable_grad():
                    Xr = Xc.clone().requires_grad_(True)
                    g = torch.autograd.grad(self.f(Vq, Xr).sum(), Xr, create_graph=torch.is_grad_enabled() or True)[0]
                return g
            for i in range(d):
                D = (gradx(Vc + steps[i]) - gradx(Vc - steps[i])) / (2 * hU)
                e_ux[i] = (D ** 2).mean(0)
        else:
            hX = hXc * self.sdX
            for i in range(d):
                for j in range(p):
                    t = torch.zeros(p, dtype=self.dtype, device=X.DEV); t[j] = hX[j]
                    D = (self.f(Vc + steps[i], Xc + t) - self.f(Vc + steps[i], Xc - t) - self.f(Vc - steps[i], Xc + t) + self.f(Vc - steps[i], Xc - t)) / (4 * hU * hX[j])
                    e_ux[i, j] = (D ** 2).mean()
        return e_ux, e_uu

    def support_audit(self, hXc):
        """fraction of ambient X-stencil queries that stay inside the observed per-column range of the stimulus bank (no clipping applied)."""
        hX = hXc * self.sdX
        ok = []
        for j in range(self.p):
            for s in (-1, 1):
                x = self.Xc[:, j] + s * hX[j]
                ok.append(((x >= self.lo[j]) & (x <= self.hi[j])).float().mean().item())
        return float(np.mean(ok)), float(np.min(ok))


class FDLoss:
    names = ("UX_neff", "UUd_share")

    def __init__(self, fd, hU, mode="hybrid", hXc=None, scales=None, n_haar=200, haar_seed=1):
        self.fd, self.hU, self.mode, self.hXc = fd, hU, mode, hXc
        self.d = fd.d
        if scales is None:
            rng = np.random.default_rng(haar_seed)
            Ms = [torch.tensor(haar(self.d, rng), dtype=torch.float32) for _ in range(n_haar)]
            with torch.no_grad():
                vals = [self.terms(M) for M in Ms]
            self.s = {k: float(np.std([v[k] for v in vals])) + 1e-9 for k in self.names}
        else:
            self.s = dict(scales)

    def energies(self, M):
        return self.fd.energies(M, self.hU, self.mode, self.hXc)

    def terms(self, M):
        e = self.energies(M)
        return {k: float(L2.LOSSES[k](*e)) for k in self.names}

    def total(self, M):
        e = self.energies(M)
        return sum(L2.LOSSES[k](*e) / self.s[k] for k in self.names)

    def search(self, starts=8, steps=120, seed=7, lr=0.05):
        rng = np.random.default_rng(seed)
        finals = []
        for _ in range(starts):
            Q0 = torch.tensor(haar(self.d, rng), dtype=torch.float32, device=X.DEV)
            W = torch.zeros(self.d, self.d, device=X.DEV, requires_grad=True)
            opt = torch.optim.Adam([W], lr=lr)
            for _ in range(steps):
                Q = torch.matrix_exp(W - W.T) @ Q0
                L = self.total(Q)
                opt.zero_grad()
                L.backward()
                if not torch.isfinite(W.grad).all():
                    break
                opt.step()
            with torch.no_grad():
                Q = torch.matrix_exp(W - W.T) @ Q0
                finals.append((float(self.total(Q)), Q.cpu().numpy()))
        finals.sort(key=lambda z: z[0])
        return finals
