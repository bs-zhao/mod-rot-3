"""cld-crossfit-v3 common library: cross-fitted (noise-floor-free) UX / UU interaction energies and the losses built on them.

Idea.  EXP13 measures  E[(d2h/du_i du_k)^2]  from ONE estimate of the function, so every estimation error is squared and adds a
positive floor to all cells (exact zeros disappear).  Here the function is estimated K >= 2 times on DISJOINT sets of subjects
(per-stimulus polynomial-in-U fits on the observed choices, or independently trained decoders) and every "energy" is a product of
two DIFFERENT estimates:   E_hat = mean_{a != b} H^(a) H^(b).   Independent errors cancel in expectation, so a true zero has
expectation zero - there is no floor to subtract and no tuning constant.

Everything below is T-free.  `evaluate` (post-hoc) is the only function that touches the truth.
The frozen EXP13 code (three_gaps/exp13_pure_simplicity, exp11 harness) is imported read-only through x6_plan06/common/x6lib.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "x6_plan06" / "common"))
import x6lib as X                                        # noqa: E402  (frozen baseline, caches, oracle rotation for diagnostics)
from common import io                                    # noqa: E402  three_gaps.common
from common.rotlib import effective_A, haar, match_report, axis_error   # noqa: E402
from common.sieve import exponents                       # noqa: E402

EPS = 1e-12
OUT = ROOT / "outputs" / "cld-crossfit-v3"
OUT.mkdir(parents=True, exist_ok=True)
BETA = {"CR3": (0.00005, 4), "C1": (0.0005, 4), "C2": (0.0005, 4), "C3": (0.0005, 6)}


# ============================================================ polynomial-in-U features with exact derivatives
def _pw(v, E):
    """prod_j v_j ** E_mj  for v (n,d), E (m,d) >= 0  ->  (n,m)"""
    return torch.prod(v.unsqueeze(1) ** E.to(v.device).unsqueeze(0).to(v.dtype), dim=-1)


def mono_feats(v, E, order=0):
    """Monomials and their exact first / second derivatives.  v (n,d), E (m,d) long.
    Returns F (n,m) [, dF (n,m,d) [, d2F (n,m,d,d)]]."""
    E = E.to(v.device); d = v.shape[1]
    F = _pw(v, E)
    if order == 0:
        return F
    eye = torch.eye(d, dtype=torch.long, device=v.device)
    dF = torch.stack([E[:, a].to(v.dtype) * _pw(v, (E - eye[a]).clamp_min(0)) for a in range(d)], -1)
    if order == 1:
        return F, dF
    d2 = v.new_zeros(v.shape[0], E.shape[0], d, d)
    for a in range(d):
        for b in range(a, d):
            if a == b:
                co = (E[:, a] * (E[:, a] - 1)).to(v.dtype)
            else:
                co = (E[:, a] * E[:, b]).to(v.dtype)
            t = co * _pw(v, (E - eye[a] - eye[b]).clamp_min(0))
            d2[:, :, a, b] = t; d2[:, :, b, a] = t
    return F, dF, d2


# ============================================================ per-stimulus fits (one polynomial in U per item of a shared bank)
def _orthobasis(F):
    """Orthonormalise the feature columns w.r.t. the empirical measure:  F = Q Rm,  Q^T Q = n I.
    A ridge on the coefficients of Q[:, 1:] is a penalty on E[(eta - mean eta)^2]: a function-space norm, hence the fit is exactly
    equivariant under any invertible linear change of U coordinates (the monomial span is invariant)."""
    Q, Rm = torch.linalg.qr(F)
    n = F.shape[0]
    return Q * n ** 0.5, Rm / n ** 0.5


def fit_lpm(F, Y, ridge=1e-8):
    """Linear-probability polynomial: least squares of Y (n,K) (binary or probabilities) on F (n,m)  ->  coef (m,K)."""
    Q, Rm = _orthobasis(F)
    n, m = Q.shape
    pen = torch.ones(m, dtype=F.dtype, device=F.device) * ridge; pen[0] = 0.0
    cq = (Q.T @ Y) / n / (1.0 + pen)[:, None]
    return torch.linalg.solve_triangular(Rm, cq, upper=True)


def fit_logistic(F, Y, ridge=1e-4, iters=40, tol=1e-9):
    """Per-item logistic polynomial:  logit P(y=1 | u, item k) = sum_a coef[a,k] phi_a(u).  Batched damped Newton over the K items.
    Y (n,K) may be binary or probabilities (soft targets).  ridge * E[(eta - mean)^2] keeps near-separated items finite."""
    Q, Rm = _orthobasis(F)
    n, m = Q.shape; K = Y.shape[1]
    pen = torch.full((m,), ridge, dtype=F.dtype, device=F.device); pen[0] = 0.0
    c = torch.zeros(m, K, dtype=F.dtype, device=F.device)
    ybar = Y.mean(0).clamp(1e-6, 1 - 1e-6); c[0] = torch.log(ybar / (1 - ybar))

    def nll(c):
        eta = Q @ c
        return (torch.nn.functional.softplus(eta) - Y * eta).mean(0) + 0.5 * (pen[:, None] * c * c).sum(0)
    cur = nll(c)
    for it in range(iters):
        eta = Q @ c; p = torch.sigmoid(eta)
        g = Q.T @ (p - Y) / n + pen[:, None] * c                        # (m,K)
        W = (p * (1 - p)).clamp_min(1e-10)
        Hk = torch.einsum("nm,nk,nl->kml", Q, W, Q) / n + torch.diag(pen + 1e-10)
        step = torch.linalg.solve(Hk, g.T.unsqueeze(-1)).squeeze(-1).T   # (m,K)
        t = torch.ones(K, dtype=F.dtype, device=F.device)
        for _ in range(12):                                             # per-item backtracking
            new = nll(c - t * step); bad = new > cur + 1e-12
            if not bool(bad.any()):
                break
            t = torch.where(bad, t * 0.5, t)
        c = c - t * step; new = nll(c)
        done = float((cur - new).abs().max()); cur = new
        if done < tol:
            break
    return torch.linalg.solve_triangular(Rm, c, upper=True)


def poly_fields(coef, v_grid, E, link="identity"):
    """Derivative fields of the fitted per-item polynomials on a grid of latent points.
    coef (m,K), v_grid (S,d).  link 'identity': h is the polynomial itself; 'logit': the polynomial is eta.
    Returns dict with eta-channel and prob-channel fields:  g (S,K,d), Htt (S,K,d,d), h (S,K)."""
    F, dF, d2F = mono_feats(v_grid, E, order=2)
    h = F @ coef
    g = torch.einsum("sma,mk->ska", dF, coef)
    H = torch.einsum("smab,mk->skab", d2F, coef)
    out = {}
    if link == "identity":
        out["prob"] = dict(h=h, g=g, Htt=H)
    else:
        out["logit"] = dict(h=h, g=g, Htt=H)
        p = torch.sigmoid(h); s1 = p * (1 - p); s2 = s1 * (1 - 2 * p)
        out["prob"] = dict(h=p, g=s1[..., None] * g,
                           Htt=s1[..., None, None] * H + s2[..., None, None] * g[..., :, None] * g[..., None, :])
    return out


def split_folds(ids, n_folds, seed):
    rng = np.random.default_rng(seed); p = rng.permutation(np.asarray(ids))
    return [np.sort(p[i::n_folds]) for i in range(n_folds)]


def sieve_bank_fields(V, Y, fold_ids, v_grid, du=2, link="logit", ridge=None, dev="cpu"):
    """One fit per fold.  V (N,d) latent of every subject, Y (N,K) choices on the shared bank, fold_ids list of subject-id arrays.
    Returns (list over folds of {channel: fields}, exponent matrix)."""
    d = V.shape[1]; E = exponents(d, du)
    vg = torch.as_tensor(v_grid, dtype=torch.float64, device=dev)
    out = []
    for ids in fold_ids:
        F = mono_feats(torch.as_tensor(V[ids], dtype=torch.float64, device=dev), E)
        Yf = torch.as_tensor(Y[ids], dtype=torch.float64, device=dev)
        if link == "logit":
            coef = fit_logistic(F, Yf, ridge=1e-4 if ridge is None else ridge)
        else:
            coef = fit_lpm(F, Yf, ridge=1e-8 if ridge is None else ridge)
        out.append({ch: {k: t.cpu() for k, t in f.items()} for ch, f in poly_fields(coef, vg, E, link).items()})
    return out, E


# ============================================================ cross-fitted moments
def _pairs(ts, eq):
    """mean over ordered pairs a != b of einsum(eq, t_a, t_b)   ( = (sum_a t_a)(sum_b t_b) - sum_a t_a t_a, normalised )."""
    Kf = len(ts); tot = sum(ts)
    M = torch.einsum(eq, tot, tot) - sum(torch.einsum(eq, t, t) for t in ts)
    return M / (Kf * (Kf - 1))


def _plain(ts, eq):
    """mean over folds of einsum(eq, t_a, t_a): the usual squared-then-averaged moment (keeps the noise floor)."""
    return sum(torch.einsum(eq, t, t) for t in ts) / len(ts)


def moments(fields, cross=True, per_item=True):
    """fields: list over folds of dict(g (S,K,d), Htt (S,K,d,d) [, Htx (S,K,d,p)]).   Second moments used by all losses:
      T4  (d,d,d,d)    E_{s,k}[H_ab H_cd]            pooled UU                    (EXP13's T4)
      SX  (p,d,d)      E_{s,k}[Hx_aj Hx_bj]          pooled UX w.r.t. X columns   (EXP13's SX; only if Htx is given)
      T4I (K,d,d,d,d)  E_s[H_ab H_cd] per item       item-level UU
      SG  (K,d,d)      E_s[g_a g_b] per item         item-level first order
      SGc (K,d,d)      same with g centred over items within subject (variation of the U effect across stimuli = U x stimulus interaction)
    cross=True: products of two different folds only.  With a single fold, or cross=False: plain squares."""
    f = _pairs if (cross and len(fields) > 1) else _plain
    g = [x["g"].double() for x in fields]; H = [x["Htt"].double() for x in fields]
    S, K = g[0].shape[:2]
    m = dict(T4=f(H, "skab,skcd->abcd") / (S * K))
    if "Htx" in fields[0]:
        m["SX"] = f([x["Htx"].double() for x in fields], "skaj,skbj->jab") / (S * K)
    if per_item:
        m["T4I"] = f(H, "skab,skcd->kabcd") / S
        m["SG"] = f(g, "ska,skb->kab") / S
        m["SGc"] = f([t - t.mean(1, keepdim=True) for t in g], "ska,skb->kab") / S
    return m


def pool_items(m, labels):
    """Average the item-level moments within stimulus contexts (labels: one integer per item, built from X only).
    Energies are linear in the moments, so this equals averaging the cross-fitted energies over the items of a context: their
    zero-mean noise averages out, which a plain squared energy cannot do."""
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long); C = int(lab.max()) + 1
    P = torch.zeros(C, len(lab), dtype=torch.float64); P[lab, torch.arange(len(lab))] = 1.0
    P = P / P.sum(1, keepdim=True)
    out = dict(m)
    for k in ("T4I", "SG", "SGc"):
        if k in m:
            out[k] = torch.einsum("ck,k...->c...", P, m[k])
    return out


# ============================================================ energies and losses (A = effective_A(M, grid latents))
def e_uu(A, m):   return torch.einsum("ia,kb,abcd,ic,kd->ik", A, A, m["T4"], A, A)
def e_ux(A, m):   return torch.einsum("ia,jab,ib->ij", A, m["SX"], A)
def e_uui(A, m):  return torch.einsum("ia,jb,kabcd,ic,jd->kij", A, A, m["T4I"], A, A)      # (K,d,d)
def e_gi(A, m):   return torch.einsum("ia,kab,ib->ik", A, m["SG"], A)                      # (d,K)
def e_gci(A, m):  return torch.einsum("ia,kab,ib->ik", A, m["SGc"], A)


def neff(e):
    q = e.clamp_min(0); q = q / (q.sum() + EPS); q = q.flatten() + EPS
    return (-(q * q.log()).sum()).exp()


def soft_l0(e, k):
    """soft count of active cells; threshold relative to the mean energy (scale free).  Negative cross-fitted energies count as zero."""
    e = e.clamp_min(0); tau2 = k * k * e.mean()
    return (e / (e + tau2 + EPS)).sum()


def make_terms(m, clamp_share=True):
    """name -> f(A).  Terms needing a moment that is absent are skipped."""
    T = {}
    def uud(A):
        e = e_uu(A, m); e = e.clamp_min(0) if clamp_share else e
        return torch.diagonal(e).sum() / (e.sum() + EPS)
    T["UUd_share"] = uud
    T["UUo_neff"] = lambda A: neff(e_uu(A, m) * (1 - torch.eye(A.shape[0], dtype=A.dtype, device=A.device)))
    if "SX" in m:
        T["UX_neff"] = lambda A: neff(e_ux(A, m))
    if "T4I" in m:
        T["UI_neff"] = lambda A: neff(e_gci(A, m))                       # U x stimulus interaction, effective number of active (axis, item) cells
        def uudi(A):                                                     # item-wise self-curvature share, energy weighted = pooled share (kept for clarity)
            e = e_uui(A, m).clamp_min(0)
            return torch.diagonal(e, dim1=1, dim2=2).sum() / (e.sum() + EPS)
        T["UUdI_share"] = uudi
        for k in (0.03, 0.1, 0.3):
            tag = f"{int(round(k * 100)):02d}"
            T[f"UUI_softL0_{tag}"] = lambda A, k=k: soft_l0(e_uui(A, m), k)              # per-item support count of UU entries (EXP03's HUUP softL0)
            T[f"GI_softL0_{tag}"] = lambda A, k=k: soft_l0(e_gi(A, m), k)                # per-item support count of first-order effects
            T[f"TOT_softL0_{tag}"] = lambda A, k=k: soft_l0(e_gi(A, m) + 0.5 * e_uui(A, m).sum(2).T, k)   # "axis i is inactive at item k" (1st + 2nd order)
    return T


# ============================================================ search (same recipe as the frozen EXP13: Haar scales seed 1, 24 starts seed 7, Adam 250 x 0.05)
def haar_scales(terms, t_grid, n=200, seed=1):
    d = t_grid.shape[1]; rng = np.random.default_rng(seed)
    Ms = [torch.tensor(haar(d, rng), dtype=t_grid.dtype) for _ in range(n)]
    with torch.no_grad():
        vals = {k: np.array([float(f(effective_A(M, t_grid))) for M in Ms]) for k, f in terms.items()}
    return {k: float(v.std()) + 1e-9 for k, v in vals.items()}, vals


def combo(terms, scales, names):
    return lambda A: sum(terms[k](A) / scales[k] for k in names)


def _batch_A(M, t_grid):
    """effective_A for a batch of rotations M (B,d,d):  A = sd(M t) * inv(M)^T."""
    u = torch.einsum("sd,bid->bsi", t_grid, M)
    return u.std(1, unbiased=False)[:, :, None] * torch.linalg.inv(M).transpose(1, 2)


def search(loss, t_grid, starts=24, steps=250, lr=0.05, seed=7):
    """minimise loss(A(M)) over M in SO(d): `starts` Haar starts (same rng order as the frozen harness), Adam, R = exp(W - W^T) Q0.
    All starts are advanced together (Adam is element-wise, so this equals running them one by one); a start whose gradient
    becomes non-finite is frozen, as in the frozen harness.  Returns [(loss, M)] sorted."""
    d = t_grid.shape[1]; rng = np.random.default_rng(seed)
    Q0 = torch.stack([torch.tensor(haar(d, rng), dtype=t_grid.dtype) for _ in range(starts)])
    W = torch.zeros(starts, d, d, dtype=t_grid.dtype, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    vloss = torch.func.vmap(loss)
    alive = torch.ones(starts, dtype=torch.bool)
    build = lambda: torch.matrix_exp(W - W.transpose(1, 2)) @ Q0
    for _ in range(steps):
        L = vloss(_batch_A(build(), t_grid))
        opt.zero_grad(); L.sum().backward()
        ok = torch.isfinite(W.grad).flatten(1).all(1)
        alive &= ok
        W.grad[~alive] = 0.0
        if not bool(alive.any()):
            break
        Wold = W.detach().clone(); opt.step()
        with torch.no_grad():
            W[~alive] = Wold[~alive]
    with torch.no_grad():
        M = build(); Lf = vloss(_batch_A(M, t_grid))
    fin = sorted(((float(Lf[i]), M[i].numpy()) for i in range(starts) if np.isfinite(float(Lf[i]))), key=lambda z: z[0])
    return fin


def loss_at(loss, t_grid, M):
    with torch.no_grad():
        return float(loss(effective_A(torch.as_tensor(np.asarray(M), dtype=t_grid.dtype), t_grid)))


# ============================================================ post-hoc evaluation (the ONLY place where T is used)
def evaluate(W, Tst, te, M):
    """TEST matched |corr| statistics of U = W M^T against the standardised true parameters."""
    r = match_report(W[te] @ np.asarray(M).T, Tst[te])
    return r["mcc"], r["off_mean"], r["off_max"]


def run_losses(m, t_grid, combos, W, Tst, te, M_ref=None, starts=24, steps=250, clamp_share=True, extra=None):
    """Search every combo (tuple of term names).  M_ref: reference rotation (Oracle / truth) used ONLY to report the loss gap."""
    terms = make_terms(m, clamp_share)
    need = sorted({k for c in combos for k in c})
    terms = {k: terms[k] for k in need if k in terms}
    sc, hv = haar_scales(terms, t_grid)
    rows = []
    for names in combos:
        if any(k not in terms for k in names):
            continue
        f = combo(terms, sc, names); fin = search(f, t_grid, starts, steps)
        mcc, leak, lmax = evaluate(W, Tst, te, fin[0][1])
        row = dict(loss="+".join(names), mcc=mcc, leak=leak, leak_max=lmax, L_found=fin[0][0])
        if M_ref is not None:
            tot = np.sum([hv[k] / sc[k] for k in names], 0)
            row["delta_sd"] = (loss_at(f, t_grid, M_ref) - fin[0][0]) / (tot.std() + 1e-12)     # > 0: the reference is NOT the minimum
            row["ang_max"] = float(X.axis_angles(fin[0][1], M_ref)[0].max())
        if extra:
            row.update(extra)
        rows.append((row, fin[0][1]))
    return rows
