"""Shared core for the doc-07 wave (EXP20/21/22/23).

Frozen EXP13 pieces (losses2, common/*) are IMPORTED, never modified.
Provides: setting loading (checkpoint / latent / X bank), per-centre second-moment computation (autograd),
the moment-based EXP13 loss with the frozen semantics (float32 search, Haar-200 seed-1 scales, 24 starts seed 7,
Adam 250 steps lr .05) and post-hoc evaluation (T is used only in `evaluate`).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "three_gaps" / "exp13_pure_simplicity", ROOT / "three_gaps"):
    sys.path.insert(0, str(_p))
import losses2 as L2  # noqa: E402  frozen EXP13 losses
from common import io  # noqa: E402
from common.decoder import decoder_h  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402
from common.rotlib import haar, match_report  # noqa: E402

OUT = ROOT / "outputs" / "three_gaps" / "exp20_wave07"
OUT.mkdir(parents=True, exist_ok=True)
BETA = {"CR3": (0.00005, 4), "C1": (0.0005, 4), "C2": (0.0005, 4), "C3": (0.0005, 6)}
DEV = io.pick_device("auto")


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


class Setting:
    """checkpoint / latent / X bank / split of one (model, repeat).

    X bank = the frozen EXP13 cache grid X (CR3: the fixed 500-trial bank; C1-C3: the frozen 400-point grid)."""

    def __init__(self, model, rep):
        self.model, self.rep = model, rep
        beta, zd = BETA[model]
        self.data, self.truth = io.load_all(model, beta, zd, 0, rep)
        self.net, _ = io.load_decoder(self.data["paths"]["checkpoint"], DEV)
        self.cond = self.data["condition"]
        self.cache_path = ROOT / "outputs" / "three_gaps" / "exp13" / f"cache2_{model}_{self.cond}_prob.pt"
        c = torch.load(self.cache_path, weights_only=False)
        self.X = c["X"].double()
        self.cache_ids = c["ids"]
        self.V = self.data["V"]
        self.Tst = self.truth.std()
        self.tr, self.va, self.te = self.data["train_ids"], self.data["val_ids"], self.data["test_ids"]
        self.d, self.p = self.V.shape[1], self.X.shape[1]
        self.mu_z, self.Minv = self.data["mu_z"], self.data["Minv"]
        self.oracle = np.load(ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{self.cond}_pca0" / "R_oracle_raw.npy")

    def h(self, ch):
        return decoder_h(self.net, self.mu_z, self.Minv, ch)


def point_moments(Htt, Htx):
    """second moments averaged over the K stimuli of each centre: SX (B,p,d,d), T4 (B,d,d,d,d), float64."""
    Htt, Htx = Htt.double(), Htx.double()
    K = Htt.shape[1]
    d = Htt.shape[-1]
    SX = torch.einsum("bkaj,bkcj->bjac", Htx, Htx) / K
    Hf = Htt.reshape(Htt.shape[0], K, d * d)
    T4 = (torch.einsum("bkm,bkn->bmn", Hf, Hf) / K).reshape(-1, d, d, d, d)
    return SX, T4


def centre_moments(st, ch, centres_V, X=None, bs=16, chunk=2048, h=None):
    """autograd second moments for arbitrary centres (N,d) in V coordinates crossed with an X bank.

    returns SX (N,p,d,d), T4 (N,d,d,d,d) as float64 numpy arrays (accumulated in float64; the decoder itself runs in float32)."""
    X = st.X if X is None else X
    h = st.h(ch) if h is None else h
    SXs, T4s = [], []
    for i in range(0, len(centres_V), bs):
        grid = torch.tensor(np.asarray(centres_V[i:i + bs]), dtype=torch.float64, device=DEV)
        b = base_derivatives(h, grid, X.to(DEV), which=("Htt", "Htx"), chunk=chunk)
        SX, T4 = point_moments(b["Htt"].cpu(), b["Htx"].cpu())
        SXs.append(SX.numpy())
        T4s.append(T4.numpy())
    return np.concatenate(SXs), np.concatenate(T4s)


def eff_A(M, C):
    """A = D_M M^{-T}, D_M = diag(sqrt(diag(M C M^T))): equals the frozen effective_A for orthogonal M.

    C = population covariance (ddof=0) of the reference V."""
    s = torch.sqrt(torch.clamp(torch.diagonal(M @ C @ M.T), min=1e-30))
    return s[:, None] * torch.linalg.inv(M).T


class MomentLoss:
    """EXP13 loss L = UX_neff/s_UX + UUd_share/s_UU from weighted second moments SX (p,d,d), T4 (d,d,d,d) and C_ref."""

    names = ("UX_neff", "UUd_share")

    def __init__(self, SX, T4, Cref, d, scales=None, haar_seed=1, n_haar=200):
        self.SX = torch.as_tensor(SX, dtype=torch.float32)
        self.T4 = torch.as_tensor(T4, dtype=torch.float32)
        self.C = torch.as_tensor(Cref, dtype=torch.float32)
        self.d = d
        if scales is None:
            rng = np.random.default_rng(haar_seed)
            Ms = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(n_haar)]
            with torch.no_grad():
                vals = {k: np.array([float(self.term(k, M)) for M in Ms]) for k in self.names}
            self.s = {k: float(v.std()) + 1e-9 for k, v in vals.items()}
        else:
            self.s = dict(scales)

    def energies(self, M):
        return L2.energies(eff_A(M, self.C), self.SX, self.T4)

    def term(self, k, M):
        return L2.LOSSES[k](*self.energies(M))

    def total(self, M):
        e = self.energies(M)
        return sum(L2.LOSSES[k](*e) / self.s[k] for k in self.names)

    def search(self, starts=24, steps=250, seed=7, lr=0.05):
        rng = np.random.default_rng(seed)
        finals = []
        for _ in range(starts):
            Q0 = torch.tensor(haar(self.d, rng), dtype=torch.float32)
            W = torch.zeros(self.d, self.d, requires_grad=True)
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
                finals.append((float(self.total(Q)), Q.numpy()))
        finals.sort(key=lambda z: z[0])
        return finals


def population_cov(V):
    Vc = V - V.mean(0)
    return Vc.T @ Vc / len(V)


def evaluate(st, M):
    """post-hoc TEST (mcc, mean leakage, max leakage); T is used here only."""
    r = match_report(st.V[st.te] @ np.asarray(M).T, st.Tst[st.te])
    return r["mcc"], r["off_mean"], r["off_max"]


def signed_perm_angle(Q, R):
    from scipy.optimize import linear_sum_assignment
    C = np.abs(np.asarray(Q) @ np.asarray(R).T)
    r, c = linear_sum_assignment(-C)
    return float(np.degrees(np.arccos(np.clip(C[r, c], 0, 1))).max())
