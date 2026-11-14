"""EXP13: generic W-Y simplicity losses from second moments of NN-decoder derivatives (no role detection, no design flags).

Rotated coordinates u = Q v with per-axis z-scoring: A = effective_A(Q).  Energies of the rotated derivative tensors:
  e_UX[i,j] = E[(d^2 h / du_i dx_j)^2]       (U-X interaction strength)      -> sparsity of UX interactions
  e_UU[i,k] = E[(d^2 h / du_i du_k)^2]       i != k: UU interactions (non-additivity); i == k: curvature of u_i (nonlinearity)
Every loss is a scale-free statistic of these energies, invariant to per-axis scale, sign and permutation.  Lower = simpler.
"""
from __future__ import annotations
import torch

EPS = 1e-12


def moments(c, standardize_x=False):
    Htx = c["Htx"].double(); Htt = c["Htt"].double()
    if standardize_x:   # derivatives w.r.t. unit-sd stimulus coordinates: d/d(x/sd) = sd * d/dx  (generic unit choice for X)
        sd = c["X"].double().std(0).clamp_min(1e-9)
        Htx = Htx * sd
    S, K, d, p = Htx.shape
    SX = torch.einsum("skaj,skbj->jab", Htx, Htx) / (S * K)                  # (p,d,d)
    c["S2"] = (torch.einsum("ska,skb->ab", c["g"].double(), c["g"].double()) / (S * K)).float()
    Hf = Htt.reshape(S, K, d * d)
    T4 = (torch.einsum("skm,skn->mn", Hf, Hf) / (S * K)).reshape(d, d, d, d)  # E[H_ab H_cd]
    return SX.float(), T4.float()


def energies(A, SX, T4, S2=None):
    e_ux = torch.einsum("ia,jab,ib->ij", A, SX, A)                            # (d,p)
    e_uu = torch.einsum("ia,kb,abcd,ic,kd->ik", A, A, T4, A, A)               # (d,d)
    return e_ux.clamp_min(0), e_uu.clamp_min(0)


def _norm(e):
    return e / (e.sum() + EPS)


def quartimax(e):      # concentration: low value = energy on few entries
    q = _norm(e).flatten(); return -(q * q).sum()


def l1l2(e):           # (sum sqrt e) / sqrt(sum e): classic sparsity of amplitudes
    return e.clamp_min(0).sqrt().sum() / (e.sum().sqrt() + EPS)


def neff(e):           # exp(entropy): effective number of active entries
    q = _norm(e).flatten() + EPS; return (-(q * q.log()).sum()).exp()


def softL0(e, tau=0.1):
    return (e / (e + tau * tau * e.mean() + EPS)).sum()


def row_neff(e):       # per row (axis): how many columns it acts on, energy weighted
    q = e / (e.sum(1, keepdim=True) + EPS) + EPS; ent = -(q * q.log()).sum(1)
    w = e.sum(1) / (e.sum() + EPS); return (w * ent.exp()).sum()


def col_neff(e):       # per column (X input): how many axes act on it
    return row_neff(e.T)


def offmask(d, dev=None):
    return 1 - torch.eye(d, device=dev)


LOSSES = {
    # UX interaction sparsity
    "UX_quartimax": lambda ux, uu: quartimax(ux), "UX_l1l2": lambda ux, uu: l1l2(ux), "UX_neff": lambda ux, uu: neff(ux),
    "UX_softL0": lambda ux, uu: softL0(ux), "UX_rowNeff": lambda ux, uu: row_neff(ux), "UX_colNeff": lambda ux, uu: col_neff(ux),
    # UU interaction sparsity (off-diagonal) and non-additivity share
    "UUo_quartimax": lambda ux, uu: quartimax(uu * offmask(uu.shape[0])), "UUo_l1l2": lambda ux, uu: l1l2((uu * offmask(uu.shape[0]))[offmask(uu.shape[0]) > 0]),
    "UUo_neff": lambda ux, uu: neff((uu * offmask(uu.shape[0]))[offmask(uu.shape[0]) > 0]),
    "UUo_share": lambda ux, uu: (uu * offmask(uu.shape[0])).sum() / (uu.sum() + EPS),
    # U curvature (nonlinearity): concentrated on few axes / small overall share
    "UUd_quartimax": lambda ux, uu: quartimax(torch.diagonal(uu)), "UUd_neff": lambda ux, uu: neff(torch.diagonal(uu)),
    "UUd_share": lambda ux, uu: torch.diagonal(uu).sum() / (uu.sum() + EPS),
}


def _diag(uu):
    return torch.diagonal(uu)


LOSSES.update({
    "UUd_sum": lambda ux, uu: _diag(uu).sum(),                                       # total squared self-curvature (unit-sd axes)
    "UUd_l1l2": lambda ux, uu: l1l2(_diag(uu)),                                      # curvature amplitudes sparse over axes
    "UUd_rms": lambda ux, uu: _diag(uu).sum().sqrt(),
    "UUd_max": lambda ux, uu: _diag(uu).max() / (_diag(uu).sum() + EPS),            # worst single axis share (flat -> small)
})


def _natural(ux, uu):
    return ux.sum(), uu.sum(), torch.diagonal(uu).sum()


def make_natural(mu):
    """log(effective number of active UX interactions) + mu * (self-curvature share of ALL second-order energy).
    Both terms live on natural scales (log-count, share in [0,1]); no data-dependent weight normalisation."""
    def f(ux, uu):
        tot = ux.sum() + uu.sum() + EPS
        return torch.log(neff(ux)) + mu * torch.diagonal(uu).sum() / tot
    return f
