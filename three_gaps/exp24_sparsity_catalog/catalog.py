"""EXP24 catalogue of sparsity / simplicity measures for the two parts of the frozen EXP13 loss.

Inputs are the rotated second-derivative ENERGIES (non-negative): e_ux (d x p) = E[(d2f/du_i dx_j)^2] and e_uu (d x d) = E[(d2f/du_i du_k)^2].
Every measure returns a scalar where LOWER = SIMPLER / SPARSER, is invariant to a global rescaling of the energies (checked in audit.py), and is
differentiable almost everywhere.  `a` denotes amplitudes sqrt(e).

Sources of the measure families (see README.md): Hurley & Rickard 2009 (Gini, L_p norms, Hoyer; criteria D1-D4, P1-P2), Hoyer 2004 (L1/L2 sparseness),
Hill numbers / Renyi entropies (effective numbers, inverse participation ratio), Candes-Wakin-Boyd 2008 (log-sum penalty), Crawford-Ferguson / orthomax
rotation criteria (row + column parsimony; quartimax / varimax), tail-L2 and fractional effective support (earlier waves of this project).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_pure_simplicity"))
import losses2 as L2  # noqa: E402  frozen EXP13 losses (neff, row/col neff reference implementations)

EPS = 1e-12


def _flat(e):
    return e.reshape(-1)


def _q(e):
    e = _flat(e)
    return e / (e.sum() + EPS)


def _amp(e):
    return (_flat(e) + 1e-24).sqrt()


# ---------------------------------------------------------------- entry-level measures on a vector of energies
def neff(e):                       # Hill number order 1 (exp Shannon entropy): frozen EXP13 UX measure
    q = _q(e) + EPS
    return (-(q * q.log()).sum()).exp()


def shannon(e):
    q = _q(e) + EPS
    return -(q * q.log()).sum()


def renyi(e, alpha):               # Renyi entropy; alpha=2 is minus log of the inverse participation ratio / quartimax-type concentration
    q = _q(e) + EPS
    return torch.log((q ** alpha).sum()) / (1 - alpha)


def maxshare(e):                   # -(largest share): 'one dominant entry'
    return -_q(e).max()


def tail_l2(e, k):                 # relative L2 tail outside the k largest entries (earlier projects' tail-L2); distinguishes [3,2,.01,.01] from [1,1,.5,.1]
    q = torch.sort(_q(e), descending=True).values
    return (1 - q[:k].sum()).clamp_min(0).sqrt()


def l1l2(e):                       # L1/L2 of amplitudes (Hoyer's sparseness is an affine function of the reciprocal)
    a = _amp(e)
    return a.sum() / ((a * a).sum().sqrt() + EPS)


def gini(e):                       # minus Gini index of the amplitudes (Hurley & Rickard: the only measure meeting all six criteria in the arXiv version)
    a = torch.sort(_amp(e)).values
    n = len(a)
    k = torch.arange(1, n + 1, dtype=a.dtype, device=a.device)
    g = 1 - 2 * ((a / (a.sum() + EPS)) * ((n - k + 0.5) / n)).sum()
    return -g


def lp_ratio(e, p):                # ||a||_p / ||a||_2 for p < 1 (quasi-norm sparsity)
    a = _amp(e)
    return ((a ** p).sum()) ** (1.0 / p) / ((a * a).sum().sqrt() + EPS)


def logsum(e, eps):                # log-sum penalty (Candes-Wakin-Boyd), scale-free through the mean energy
    e = _flat(e)
    return torch.log(1 + e / (eps * e.mean() + EPS)).sum()


def softl0(e, tau):                # smooth count of entries above a relative threshold
    e = _flat(e)
    return (e / (e + tau ** 2 * e.mean() + EPS)).sum()


def support_frac(e, mass=0.99):    # fractional number of entries needed to hold `mass` of the energy
    q = torch.sort(_q(e), descending=True).values
    prev = torch.cumsum(q, 0) - q
    return ((mass - prev) / (q + EPS)).clamp(0, 1).sum()


def softcount_rel(e, t, w):        # smooth count of entries above t * max
    e = _flat(e)
    return torch.sigmoid((e / (e.max() + EPS) - t) / w).sum()


# ---------------------------------------------------------------- structured measures on a matrix of energies (rows = axes)
def row_neff(e):
    return L2.row_neff(e)


def col_neff(e):
    return L2.col_neff(e)


def cf(e, kappa):                  # Crawford-Ferguson: (1-kappa) row parsimony + kappa column parsimony on squared loadings q
    q = e / (e.sum() + EPS)
    rt = (q.sum(1) ** 2).sum() - (q ** 2).sum()
    ct = (q.sum(0) ** 2).sum() - (q ** 2).sum()
    return (1 - kappa) * rt + kappa * ct


def varimax_cols(e):               # -(sum over columns of the variance of squared loadings): orthomax with gamma = 1
    q = e / (e.sum() + EPS)
    d = q.shape[0]
    return -((q ** 2).sum(0) - q.sum(0) ** 2 / d).sum()


def axis_neff(e):                  # effective number of axes carrying any energy
    return neff(e.sum(1))


def axis_maxshare(e):              # mean over axes of the largest within-row share: each axis acts through one input (simple structure)
    q = e / (e.sum(1, keepdim=True) + EPS)
    return -q.max(1).values.mean()


UX = {  # name -> f(e_ux)
    "neff": neff, "shannon": shannon, "renyi0.5": lambda e: renyi(e, 0.5), "renyi2": lambda e: renyi(e, 2.0), "renyi4": lambda e: renyi(e, 4.0),
    "maxshare": maxshare, "tail1": lambda e: tail_l2(e, 1), "tail2": lambda e: tail_l2(e, 2), "tail4": lambda e: tail_l2(e, 4),
    "l1l2": l1l2, "gini": gini, "lp0.5": lambda e: lp_ratio(e, 0.5), "lp0.25": lambda e: lp_ratio(e, 0.25),
    "logsum0.01": lambda e: logsum(e, 0.01), "logsum0.1": lambda e: logsum(e, 0.1),
    "soft0.03": lambda e: softl0(e, 0.03), "soft0.1": lambda e: softl0(e, 0.1), "soft0.3": lambda e: softl0(e, 0.3),
    "support99": lambda e: support_frac(e, 0.99), "support90": lambda e: support_frac(e, 0.90),
    "count1%": lambda e: softcount_rel(e, 0.01, 0.005), "count10%": lambda e: softcount_rel(e, 0.10, 0.05),
    "rowneff": row_neff, "colneff": col_neff, "cf0": lambda e: cf(e, 0.0), "cf0.5": lambda e: cf(e, 0.5), "cf1": lambda e: cf(e, 1.0),
    "varimax": varimax_cols, "axisneff": axis_neff, "axismaxshare": axis_maxshare,
}


def _upper(e):
    d = e.shape[0]
    iu = torch.triu_indices(d, d, 1)
    return e[iu[0], iu[1]]


def _diag(e):
    return torch.diagonal(e)


UU = {  # name -> f(e_uu)
    "diagshare": lambda e: _diag(e).sum() / (e.sum() + EPS),                       # frozen EXP13 UU part
    "sqrt_diagshare": lambda e: (_diag(e).sum() / (e.sum() + EPS)).sqrt(),
    "log_diagshare": lambda e: torch.log(_diag(e).sum() / (e.sum() + EPS) + 1e-3),
    "offshare": lambda e: _upper(e).sum() * 2 / (e.sum() + EPS),
    "off_neff": lambda e: neff(_upper(e)), "off_l1l2": lambda e: l1l2(_upper(e)), "off_tail1": lambda e: tail_l2(_upper(e), 1),
    "off_gini": lambda e: gini(_upper(e)), "off_lp0.5": lambda e: lp_ratio(_upper(e), 0.5), "off_soft0.1": lambda e: softl0(_upper(e), 0.1),
    "diag_neff": lambda e: neff(_diag(e)), "diag_l1l2": lambda e: l1l2(_diag(e)), "diag_gini": lambda e: gini(_diag(e)), "diag_tail1": lambda e: tail_l2(_diag(e), 1),
    "diag_soft0.1": lambda e: softl0(_diag(e), 0.1),
    "full_neff": lambda e: neff(torch.cat([_upper(e), _diag(e)])), "full_gini": lambda e: gini(torch.cat([_upper(e), _diag(e)])),
    "full_l1l2": lambda e: l1l2(torch.cat([_upper(e), _diag(e)])),
}

JOINT = {  # name -> f(e_ux, e_uu): replace BOTH parts by one measure over the concatenated entries (UX entries + UU upper triangle incl. diagonal)
    "J_neff": lambda ux, uu: neff(torch.cat([_flat(ux), _upper(uu), _diag(uu)])),
    "J_l1l2": lambda ux, uu: l1l2(torch.cat([_flat(ux), _upper(uu), _diag(uu)])),
    "J_gini": lambda ux, uu: gini(torch.cat([_flat(ux), _upper(uu), _diag(uu)])),
    "J_tail2": lambda ux, uu: tail_l2(torch.cat([_flat(ux), _upper(uu), _diag(uu)]), 2),
    "J_soft0.1": lambda ux, uu: softl0(torch.cat([_flat(ux), _upper(uu), _diag(uu)]), 0.1),
    "UUtotal_share": lambda ux, uu: uu.sum() / (ux.sum() + uu.sum() + EPS),
}

# measures that need a matrix (rows = axes): not defined on a bare vector in the audit
STRUCTURED = {"rowneff", "colneff", "cf0", "cf0.5", "cf1", "varimax", "axisneff", "axismaxshare"}
