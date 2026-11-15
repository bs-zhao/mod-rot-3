"""cld-simplicity-v2 shared library.

READ-ONLY use of the frozen EXP13 and of the wave-07 / EXP24 helpers (x7core, catalog); nothing there is modified.
No role detection: no gain / shift axes, no parameter-specific or design-category cells.  Contexts are data-driven (k-means of the stimulus bank).
Success rule (user): a candidate is a REAL improvement over EXP13 only if the CR3 leakage (mean OR max) drops by >= 0.03 on BOTH NP repeats
(dev repeat 1 and held-out repeat 2) and C1-C3 MCC does not fall by more than 0.01.  T / Oracle are used only for post-hoc scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "three_gaps" / "exp20_latent_integration"))
sys.path.insert(0, str(ROOT / "three_gaps" / "exp24_sparsity_catalog"))
import x7core as X  # noqa: E402
import catalog as C  # noqa: E402
import losses2 as L2  # noqa: E402
from common.rotlib import haar  # noqa: E402

OUT = ROOT / "outputs" / "cld-simplicity-v2"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["CR3", "C1", "C2", "C3"]
# frozen EXP13 reference (historical, p channel): (mcc, leak_mean, leak_max)
EXP13 = {("CR3", 1): (0.8173, 0.1646, 0.3677), ("CR3", 2): (0.8122, 0.1832, 0.3781), ("C1", 1): (0.9688, 0.0212, 0.0576), ("C1", 2): (0.9689, 0.0202, 0.0611),
         ("C2", 1): (0.9645, 0.0599, 0.1523), ("C2", 2): (0.9677, 0.0427, 0.1450), ("C3", 1): (0.9600, 0.0464, 0.1143), ("C3", 2): (0.9615, 0.0478, 0.1525)}
THRESH = 0.03


def kmeans_labels(Xb, K, seed=0):
    """data-driven contexts: k-means of the standardised stimulus bank (no design labels)."""
    from sklearn.cluster import KMeans
    Z = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-12)
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Z)


class Dev:
    """one (model, repeat) development object on the historical 300-subject grid (frozen semantics) with extra context moments."""

    def __init__(self, model, rep):
        self.model, self.rep = model, rep
        self.st = X.Setting(model, rep)
        self.c = torch.load(self.st.cache_path, weights_only=False)
        self.d = self.st.d
        SX, T4 = L2.moments(self.c)
        self.ml = X.MomentLoss(SX.numpy(), T4.numpy(), X.population_cov(self.st.V[self.c["ids"]]), self.d, scales={"UX_neff": 1.0, "UUd_share": 1.0})
        self.g = self.c["g"].double()
        self.Xb = self.st.X.numpy()
        self._ctx = {}
        self.ident = X.evaluate(self.st, np.eye(self.d))
        self.oracle = X.evaluate(self.st, self.st.oracle)

    def ctx_moments(self, K, floor=0.0):
        """S2c (K,d,d): first-order gradient second moments per k-means context; optional generic noise-floor subtraction (frac * min_c lambda_min)."""
        key = (K, floor)
        if key not in self._ctx:
            lab = torch.tensor(kmeans_labels(self.Xb, K))
            S2 = torch.stack([torch.einsum("ska,skb->ab", self.g[:, lab == k], self.g[:, lab == k]) / (self.g.shape[0] * int((lab == k).sum())) for k in range(K)])
            if floor > 0:
                sig2 = floor * torch.stack([torch.linalg.eigvalsh(e)[0] for e in S2]).min()
                out = []
                for e in S2:
                    ev, U = torch.linalg.eigh(e)
                    out.append((U * (ev - sig2).clamp_min(0)) @ U.T)
                S2 = torch.stack(out)
            self._ctx[key] = S2.float()
        return self._ctx[key]

    def energies(self, M, K=None, floor=0.0):
        """(e_ux (d,p), e_uu (d,d), e_ctx (d,K) or None)."""
        A = X.eff_A(M, self.ml.C)
        ex, eu = L2.energies(A, self.ml.SX, self.ml.T4)
        e1 = None
        if K:
            S2 = self.ctx_moments(K, floor)
            e1 = torch.einsum("id,cde,ie->ic", A, S2, A).clamp_min(0)
        return ex, eu, e1


def search_terms(dev, terms, K=None, floor=0.0, starts=8, steps=150, seed=7, lr=0.05, n_haar=200):
    """minimise sum_t f_t(e_ux, e_uu, e_ctx)/s_t (each term Haar-standardised, equal weights, frozen semantics); returns (best_loss, Q)."""
    d = dev.d
    rng = np.random.default_rng(1)
    HM = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(n_haar)]
    with torch.no_grad():
        vals = [[float(f(*dev.energies(M, K, floor))) for f in terms] for M in HM]
    sc = np.std(np.array(vals), 0)
    if (sc < 1e-9).any():
        return None, None
    w = 1.0 / sc
    rngs = np.random.default_rng(seed)
    best = (1e30, None)
    for _ in range(starts):
        Q0 = torch.tensor(haar(d, rngs), dtype=torch.float32)
        W = torch.zeros(d, d, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(steps):
            Q = torch.matrix_exp(W - W.T) @ Q0
            ex, eu, e1 = dev.energies(Q, K, floor)
            L = sum(f(ex, eu, e1) * wi for f, wi in zip(terms, w))
            opt.zero_grad()
            L.backward()
            if not torch.isfinite(W.grad).all():
                break
            opt.step()
        with torch.no_grad():
            Q = torch.matrix_exp(W - W.T) @ Q0
            L = float(sum(f(*dev.energies(Q, K, floor)) * wi for f, wi in zip(terms, w)))
        if np.isfinite(L) and L < best[0]:
            best = (L, Q.numpy())
    return best


# term factories: every term takes (ex, eu, e1)
def T_ux(name): return lambda ex, eu, e1: C.UX[name](ex)
def T_uu(name): return lambda ex, eu, e1: C.UU[name](eu)
def T_ctx(name): return lambda ex, eu, e1: C.UX[name](e1)           # catalogue measure applied to the (axes x contexts) first-order energy matrix
FROZEN = [T_ux("neff"), T_uu("diagshare")]


def verdict(rows):
    """rows: list of dict(model, rep, mcc, leak, leak_max) for one candidate. returns dict of per-repeat deltas and the REAL-improvement flag."""
    out, ok = {}, True
    for rep in (1, 2):
        cr = [r for r in rows if r["model"] == "CR3" and r["rep"] == rep]
        if not cr:
            return dict(real=None)
        r = cr[0]
        b = EXP13[("CR3", rep)]
        out[f"d_mean_r{rep}"], out[f"d_max_r{rep}"] = b[1] - r["leak"], b[2] - r["leak_max"]
        ok &= (out[f"d_mean_r{rep}"] >= THRESH) or (out[f"d_max_r{rep}"] >= THRESH)
        for m in ("C1", "C2", "C3"):
            cm = [q for q in rows if q["model"] == m and q["rep"] == rep]
            if cm:
                out[f"dmcc_{m}_r{rep}"] = cm[0]["mcc"] - EXP13[(m, rep)][0]
                ok &= out[f"dmcc_{m}_r{rep}"] >= -0.01
    out["real"] = bool(ok)
    return out
