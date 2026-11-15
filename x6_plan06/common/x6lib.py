"""x6_plan06 common library.  READ-ONLY use of the frozen EXP13 baseline and its dependencies:
three_gaps/exp13_pure_simplicity (losses2, screen/cache2) and three_gaps/exp11_cr3_search/harness are imported, never modified.
New helpers here: caches, frozen baseline loss with Haar scales, double-precision energies, rotation utilities, evaluation.
T / Oracle are used only for post-hoc evaluation and diagnostics."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "three_gaps" / "exp13_pure_simplicity", ROOT / "three_gaps" / "exp11_cr3_search", ROOT / "three_gaps"):
    sys.path.insert(0, str(p))
import harness as H          # frozen (exp11)
import losses2 as L2          # frozen (exp13)
import screen as SC           # frozen (exp13): cache loading
from common.rotlib import effective_A, haar, givens, match_report   # noqa: E402

OUT = ROOT / "outputs" / "x6_plan06"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["CR3", "C1", "C2", "C3"]
BETA = {"CR3": (0.00005, 4), "C1": (0.0005, 4), "C2": (0.0005, 4), "C3": (0.0005, 6)}


def load(model, rep, ch):
    """frozen cache2 field + oracle rotation (diagnostics only)."""
    c, cond = SC.load(model, rep, ch)
    R = np.load(ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
    return c, cond, R


class Baseline:
    """Frozen EXP13 loss  L = UX_neff/s_UX + UUd_share/s_UU  (Haar sd over 200 rotations, seed 1), float32 exactly as exp13 combos.py,
    plus a float64 evaluator for finite-difference diagnostics."""
    def __init__(self, c, names=("UX_neff", "UUd_share"), standardize_x=False, haar_seed=1, n_haar=200):
        self.c, self.names = c, names
        self.d = c["g"].shape[-1]
        self.SX, self.T4 = L2.moments(c, standardize_x=standardize_x)
        self.t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
        self.SXd, self.T4d, self.t_d = self.SX.double(), self.T4.double(), self.t_dev.double()
        self.term = {k: (lambda A, k=k: L2.LOSSES[k](*L2.energies(A, self.SX, self.T4))) for k in names}
        rng = np.random.default_rng(haar_seed)
        Ms = [torch.tensor(haar(self.d, rng), dtype=torch.float32) for _ in range(n_haar)]
        with torch.no_grad():
            self.s = {k: float(np.std([float(self.term[k](effective_A(M, self.t_dev))) for M in Ms])) + 1e-9 for k in names}
            tot = np.array([sum(float(self.term[k](effective_A(M, self.t_dev))) / self.s[k] for k in names) for M in Ms])
        self.haar_total_sd, self.haar_total_mean = float(tot.std()), float(tot.mean())

    # float32 loss, exactly the frozen search objective
    def loss32(self, A):
        return sum(self.term[k](A) / self.s[k] for k in self.names)

    # float64 evaluation of terms / total for a rotation matrix M (numpy or torch)
    def terms64(self, M):
        M = torch.as_tensor(np.asarray(M), dtype=torch.float64)
        u = self.t_d @ M.T
        A = u.std(0, unbiased=False)[:, None] * torch.linalg.inv(M).T
        e_ux = torch.einsum("ia,jab,ib->ij", A, self.SXd, A).clamp_min(0)
        e_uu = torch.einsum("ia,kb,abcd,ic,kd->ik", A, A, self.T4d, A, A).clamp_min(0)
        return {"UX_neff": float(L2.neff(e_ux)), "UUd_share": float(torch.diagonal(e_uu).sum() / (e_uu.sum() + L2.EPS))}

    def total64(self, M):
        t = self.terms64(M)
        return sum(t[k] / self.s[k] for k in self.names)

    def search(self, starts=24, steps=250, seed=7):
        f = lambda b, A, M, cc: self.loss32(A)
        return H.optimize(self.c, f, "SO", F=None, starts=starts, steps=steps, seed=seed)


def evaluate(c, M):
    """post-hoc TEST (mcc, mean leak, max leak)."""
    return H.evaluate(c, M)


def axis_angles(Q, R):
    """degrees between matched rows of Q and R (signed permutation ambiguity handled by Hungarian on |Q R^T|)."""
    C = np.abs(np.asarray(Q) @ np.asarray(R).T)
    r, c_ = linear_sum_assignment(-C)
    return np.degrees(np.arccos(np.clip(C[r, c_], 0, 1))), c_


def skew_basis(d):
    out = []
    for i in range(d):
        for j in range(i + 1, d):
            X = np.zeros((d, d)); X[i, j], X[j, i] = 1.0, -1.0; out.append(X)
    return out


def expm_np(X):
    return torch.linalg.matrix_exp(torch.as_tensor(X, dtype=torch.float64)).numpy()
