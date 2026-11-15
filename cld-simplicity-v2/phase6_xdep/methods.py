"""Phase 6: estimating the X-dependence of the first-order gradient g_U(x) (hence H_UX = dg/dx) WITHOUT x-autograd, with progressively more flexible estimators.

Input: G (S,K,d) gradients dh/dv of the frozen decoder at every (subject, stimulus) of the shared bank X (K,p).  Output: Hx (S,K,d,p) in RAW X units.
All hyper-parameters are chosen by cross-validated PREDICTION error of g (never by the downstream score, never by T).
Methods: krr (shared RBF ridge, isotropic or ARD by analytic LOO), shared-basis (SVD / functional PCA + KRR on the shared X-profiles), cp (rank-r CP-ALS + KRR on the trial factor),
local linear (kNN-bandwidth, also unweighted 'chord' form), spline-GAM (cubic truncated-power main effects + pairwise linear products, ridge), sobolev MLP (values + gradients).
"""
from __future__ import annotations

import numpy as np
import torch

DT = torch.float64


def _std(Xb):
    mu, sd = Xb.mean(0), Xb.std(0) + 1e-12
    return (Xb - mu) / sd, sd


def _rbf(Xs, ell):
    diff = Xs[:, None, :] - Xs[None, :, :]
    D2 = ((diff / ell) ** 2).sum(-1)
    return torch.exp(-0.5 * D2), diff


def krr_fit(Xs, G, ell, lam):
    """returns alpha and the gradient tensor of the kernel at the training points: Hx_std (K,p,M) = d fhat/d xs."""
    K, p = Xs.shape
    ell = ell * torch.ones(p, dtype=DT) if torch.is_tensor(ell) is False else ell
    Km, diff = _rbf(Xs, ell)
    alpha = torch.linalg.solve(Km + lam * torch.eye(K, dtype=DT), G)
    gk = -Km.unsqueeze(-1) * diff / (ell ** 2)                      # d k(x_k, x_m)/d x_k  (K,K,p)
    return alpha, torch.einsum("kmj,mc->kjc", gk, alpha)


def krr_loo(Xs, G, ell, lam):
    p = Xs.shape[1]
    ell = ell * torch.ones(p, dtype=DT) if torch.is_tensor(ell) is False else ell
    Km, _ = _rbf(Xs, ell)
    Ki = torch.linalg.inv(Km + lam * torch.eye(len(Xs), dtype=DT))
    return float((((Ki @ G) / torch.diagonal(Ki).unsqueeze(1)) ** 2).mean())


def select_krr(Xs, G, ard=False):
    """isotropic grid + optional ARD refinement of log length-scales by Adam on the analytic LOO error (T-free)."""
    best = (1e30, None, None)
    for ell in (0.5, 1.0, 2.0, 4.0, 8.0):
        for lam in (1e-6, 1e-4, 1e-2, 1e-1):
            e = krr_loo(Xs, G, ell, lam)
            if e < best[0]:
                best = (e, ell, lam)
    if not ard:
        return best[1], best[2]
    ell0, lam = best[1], best[2]
    le = torch.full((Xs.shape[1],), float(np.log(ell0)), dtype=DT, requires_grad=True)
    opt = torch.optim.Adam([le], lr=0.05)
    for _ in range(120):
        Km, _ = _rbf(Xs, le.exp())
        Ki = torch.linalg.inv(Km + lam * torch.eye(len(Xs), dtype=DT))
        loss = (((Ki @ G) / torch.diagonal(Ki).unsqueeze(1)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return le.detach().exp(), lam


def to_hx(Hstd, sd, S, d):
    """(K,p,S*d) derivatives wrt standardised x -> (S,K,d,p) wrt raw x"""
    K, p, M = Hstd.shape
    H = (Hstd / sd.view(1, p, 1)).reshape(K, p, S, d).permute(2, 0, 3, 1)
    return H


def m_krr(G, Xb, ard=False):
    S, K, d = G.shape
    Xs, sd = _std(Xb)
    Gm = G.permute(1, 0, 2).reshape(K, S * d)
    ell, lam = select_krr(Xs, Gm, ard)
    _, Hs = krr_fit(Xs, Gm, ell, lam)
    return to_hx(Hs, sd, S, d), dict(ell=(ell.tolist() if torch.is_tensor(ell) else ell), lam=lam)


def m_shared_basis(G, Xb, rank=8):
    """functional PCA: G (K x S*d) ~ sum_r U_r s_r V_r^T ; the shared X-profiles U_r are smoothed by KRR; derivatives of the smoothed profiles."""
    S, K, d = G.shape
    Xs, sd = _std(Xb)
    Gm = G.permute(1, 0, 2).reshape(K, S * d)
    U, s, Vt = torch.linalg.svd(Gm, full_matrices=False)
    Ur = U[:, :rank]
    ell, lam = select_krr(Xs, Ur * s[:rank].sqrt())              # CV on the profiles
    _, dU = krr_fit(Xs, Ur, ell, lam)                            # (K,p,r)
    Hs = torch.einsum("kjr,r,rc->kjc", dU, s[:rank], Vt[:rank])
    ev = float((s[:rank] ** 2).sum() / (s ** 2).sum())
    return to_hx(Hs, sd, S, d), dict(rank=rank, explained=ev, ell=ell, lam=lam)


def m_cp(G, Xb, rank=8, iters=60):
    """CP-ALS of the (subject x axis x trial) gradient tensor, then KRR on the trial factor."""
    S, K, d = G.shape
    Xs, sd = _std(Xb)
    T = G
    g = torch.Generator().manual_seed(0)
    A, B, Cf = torch.randn(S, rank, dtype=DT, generator=g), torch.randn(d, rank, dtype=DT, generator=g), torch.randn(K, rank, dtype=DT, generator=g)
    for _ in range(iters):
        A = torch.einsum("ski,ir,kr->sr", T, B, Cf) @ torch.linalg.pinv((B.T @ B) * (Cf.T @ Cf))
        B = torch.einsum("ski,sr,kr->ir", T, A, Cf) @ torch.linalg.pinv((A.T @ A) * (Cf.T @ Cf))
        Cf = torch.einsum("ski,sr,ir->kr", T, A, B) @ torch.linalg.pinv((A.T @ A) * (B.T @ B))
    ell, lam = select_krr(Xs, Cf)
    _, dC = krr_fit(Xs, Cf, ell, lam)                           # (K,p,r)
    H = torch.einsum("sr,ir,kjr->skij", A, B, dC) / sd.view(1, 1, 1, -1)
    rec = float(((torch.einsum("sr,ir,kr->ski", A, B, Cf) - T) ** 2).sum() / (T ** 2).sum())
    return H, dict(rank=rank, rel_recon_err=rec, ell=ell, lam=lam)


def m_local_linear(G, Xb, m=None, weighted=True):
    """local linear regression of g on x in kNN neighbourhoods (weighted: tricube; unweighted = 'chord' difference regression); m chosen by 5-fold CV."""
    S, K, d = G.shape
    Xs, sd = _std(Xb)
    Gm = G.permute(1, 0, 2).reshape(K, S * d)
    D = torch.cdist(Xs, Xs)

    def fit_at(xq, idx_train, mm, want_slope):
        dist = torch.cdist(xq, Xs[idx_train])
        h = dist.sort(1).values[:, mm - 1:mm].clamp_min(1e-9)
        w = (1 - (dist / h).clamp(max=1) ** 3) ** 3 if weighted else (dist <= h).double()
        out_i, out_s = [], []
        for q in range(len(xq)):
            Dm = torch.cat([torch.ones(len(idx_train), 1, dtype=DT), Xs[idx_train] - xq[q]], 1)
            W = w[q]
            A = Dm.T @ (Dm * W[:, None]) + 1e-6 * torch.eye(Dm.shape[1], dtype=DT)
            Bc = torch.linalg.solve(A, Dm.T @ (Gm[idx_train] * W[:, None]))
            out_i.append(Bc[0]); out_s.append(Bc[1:])
        return torch.stack(out_i), torch.stack(out_s)

    if m is None:
        rng = np.random.default_rng(0)
        perm = torch.tensor(rng.permutation(K))
        folds = perm.chunk(5)
        best = (1e30, 40)
        for mm in (20, 40, 80, 160):
            err = 0.0
            for f in folds:
                tr = torch.tensor(np.setdiff1d(np.arange(K), f.numpy()))
                pi, _ = fit_at(Xs[f], tr, min(mm, len(tr) - 1), False)
                err += float(((pi - Gm[f]) ** 2).mean())
            if err < best[0]:
                best = (err, mm)
        m = best[1]
    _, slopes = fit_at(Xs, torch.arange(K), m, True)             # (K,p,M)
    return to_hx(slopes, sd, S, d), dict(m=m, weighted=weighted)


def _gam_basis(Xs, knots):
    cols = []
    p = Xs.shape[1]
    for j in range(p):
        x = Xs[:, j]
        cols += [x, x ** 2, x ** 3] + [torch.clamp(x - k, min=0) ** 3 for k in knots[j]]
    for j in range(p):
        for l in range(j + 1, p):
            cols.append(Xs[:, j] * Xs[:, l])
    return torch.stack([torch.ones_like(Xs[:, 0])] + cols, 1)


def m_gam(G, Xb):
    S, K, d = G.shape
    Xs, sd = _std(Xb)
    Gm = G.permute(1, 0, 2).reshape(K, S * d)
    knots = [[float(np.quantile(Xs[:, j].numpy(), q)) for q in (0.25, 0.5, 0.75)] for j in range(Xs.shape[1])]
    Phi = _gam_basis(Xs, knots)
    nb = Phi.shape[1]
    best = (1e30, 1e-2)
    rng = np.random.default_rng(1)
    perm = torch.tensor(rng.permutation(K)).chunk(5)
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        err = 0.0
        for f in perm:
            tr = torch.tensor(np.setdiff1d(np.arange(K), f.numpy()))
            B = torch.linalg.solve(Phi[tr].T @ Phi[tr] + lam * len(tr) * torch.eye(nb, dtype=DT), Phi[tr].T @ Gm[tr])
            err += float(((Phi[f] @ B - Gm[f]) ** 2).mean())
        if err < best[0]:
            best = (err, lam)
    lam = best[1]
    B = torch.linalg.solve(Phi.T @ Phi + lam * K * torch.eye(nb, dtype=DT), Phi.T @ Gm)
    from torch.func import jacrev, vmap
    Jb = vmap(jacrev(lambda x: _gam_basis(x.unsqueeze(0), knots)[0]))(Xs)       # (K,nb,p)
    Hs = torch.einsum("kbj,bc->kjc", Jb, B)
    return to_hx(Hs, sd, S, d), dict(lam=lam, n_basis=nb)


def m_sobolev_mlp(G, Hval, Vgrid, Xb, Vsc, epochs=40, width=128, device="cuda", seed=0):
    """MLP h_hat(v/Vsc, x_std) trained on decoder VALUES and v-gradients; H_UX_hat = d(grad_v h_hat)/dx by autograd."""
    from torch.func import jacrev, jacfwd, vmap
    S, K, d = G.shape
    p = Xb.shape[1]
    Xs, sd = _std(Xb)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(d + p, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 1)).double().to(device)
    z_all = torch.cat([(Vgrid / Vsc).repeat_interleave(K, 0), Xs.repeat(S, 1)], 1).to(device)            # (S*K, d+p)
    h_all = Hval.reshape(-1).to(device)
    g_all = (G.reshape(-1, d) * Vsc).to(device)                                                         # gradient wrt scaled v
    hs, gs = h_all.std(), g_all.std()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(z_all)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            z = z_all[b].clone().requires_grad_(True)
            out = net(z).squeeze(-1)
            gz = torch.autograd.grad(out.sum(), z, create_graph=True)[0][:, :d]
            loss = ((out - h_all[b]) ** 2).mean() / hs ** 2 + ((gz - g_all[b]) ** 2).mean() / gs ** 2
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    f = lambda z: net(z.unsqueeze(0)).squeeze()
    gfun = lambda z: jacrev(f)(z)[:d]
    Hfun = vmap(jacfwd(gfun))
    Hx = torch.zeros(S * K, d, p, dtype=DT)
    for i in range(0, n, 2000):
        Hx[i:i + 2000] = Hfun(z_all[i:i + 2000])[:, :, d:].cpu()                                       # d g_scaled / d xs
    Hx = Hx.reshape(S, K, d, p) * Vsc / Vsc / sd.view(1, 1, 1, p)       # g_scaled = Vsc * g  -> divide by Vsc to restore physical units: net outputs d(Vsc g)/dxs
    Hx = Hx / Vsc * 1.0
    return Hx, dict(epochs=epochs, width=width)
