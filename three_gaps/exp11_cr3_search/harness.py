"""EXP11 harness: optimise candidate losses over SO(d) or GL(d) on a cached decoder/analytic gradient field.
T is used ONLY in `evaluate` (post-hoc)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.equivariant import detect  # noqa: E402
from common.losses import LOSSES  # noqa: E402
from common.rotlib import apply_view, effective_A, haar, match_report, bank_categories  # noqa: E402
from common.screen import complement  # noqa: E402

OUT = io.ROOT / "outputs" / "three_gaps" / "exp11"


def load(model, cond, source="decoder", channel="logit"):
    c = torch.load(OUT / f"cache_{model}_{cond}_{source}_{channel}.pt", weights_only=False)
    c["cat"] = bank_categories(c["X"]) if model == "CR3" else None
    import os
    if os.environ.get("GENERIC_CTX"):   # W-only contexts: mode-flag patterns of X columns; none exist for continuous X
        ids = mode_contexts(c["X"].numpy())
        c["cat"] = torch.tensor(ids, dtype=torch.long) if ids.max() > 0 else None
    return c


def mode_contexts(X, freq=0.10, min_group=0.06):
    """Partition trials by which X columns sit at their (frequent) modal value, e.g. amount 0 or probability .5.
    Uses only the stimulus columns of W.  Groups smaller than min_group are merged into one remainder group."""
    n, p = X.shape
    flags = np.zeros((n, p), int)
    for j in range(p):
        v, cn = np.unique(X[:, j], return_counts=True); i = int(np.argmax(cn))
        if cn[i] / n >= freq:
            flags[:, j] = (X[:, j] == v[i])
    keys = [tuple(r) for r in flags]; uk = sorted(set(keys)); ids = np.array([uk.index(k) for k in keys])
    sizes = np.bincount(ids); out = np.full(n, -1); k = 0
    for g in np.argsort(-sizes, kind="stable"):
        if sizes[g] / n >= min_group:
            out[ids == g] = k; k += 1
    out[out < 0] = k
    return out


def pure_cells(X, freq=0.10, min_group=0.06, min_flags=2):
    """Loss cells from W only: contexts in which at least `min_flags` X columns sit at their modal value simultaneously
    (e.g. both amounts of the loss pair at 0, both probabilities at .5) are kept as separate cells; every other trial forms one cell."""
    n, p = X.shape
    flags = np.zeros((n, p), int)
    for j in range(p):
        v, cn = np.unique(X[:, j], return_counts=True); i = int(np.argmax(cn))
        if cn[i] / n >= freq:
            flags[:, j] = (X[:, j] == v[i])
    keys = [tuple(r) for r in flags]; uk = sorted(set(keys)); ids = np.array([uk.index(k) for k in keys])
    out = np.full(n, 0); k = 1
    for g in np.argsort(-np.bincount(ids), kind="stable"):
        if sum(uk[g]) >= min_flags and (ids == g).mean() >= min_group:
            out[ids == g] = k; k += 1
    return out


def base_of(c, dev="cpu"):
    if c["cat"] is not None:
        return {"gP": c["g"].to(dev).float(), "cat_index": c["cat"].to(dev)}
    from common.rotlib import legendre_basis  # continuous X: project onto Legendre basis (total degree <= 3)
    Psi, _ = legendre_basis(c["X"].double(), 3)
    Pm = torch.linalg.pinv(Psi)
    return {"gP": torch.einsum("bk,skd->sbd", Pm, c["g"].double()).float().to(dev)}


def stageA(c, channel="logit"):
    axes, rep = detect(c["g"].double().numpy(), c["h"].double().numpy(), channel)
    return np.array([a["w"] for a in axes]).reshape(-1, c["g"].shape[-1]), rep


def evaluate(c, M):
    """post-hoc TEST report of U = V M^T vs Tstd."""
    te = c["te"]
    r = match_report(c["V"][te] @ np.asarray(M).T, c["Tst"][te])
    return r["mcc"], r["off_mean"], r["off_max"]


def optimize(c, loss_fn, family="SO", F=None, starts=12, steps=250, lr=0.05, seed=0, dev="cpu", extra=None):
    """loss_fn(B_view_inputs) -> callable taking (base, A, M) -> scalar. family 'SO' or 'GL'.
    F: fixed leading unit rows (stage A) for SO."""
    rng = np.random.default_rng(seed)
    base = base_of(c, dev)
    t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32, device=dev)
    d = t_dev.shape[1]
    F = np.zeros((0, d)) if F is None else np.asarray(F).reshape(-1, d)
    m = len(F); k = d - m
    Bp = torch.tensor(complement(F), dtype=torch.float32, device=dev)
    Ft = torch.tensor(F, dtype=torch.float32, device=dev)
    fin = []
    for s in range(starts):
        Q0 = torch.tensor(haar(k, rng), dtype=torch.float32, device=dev)
        W = torch.zeros(k, k, device=dev, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)

        def build():
            if family == "SO":
                Rr = torch.matrix_exp(W - W.T) @ Q0
            else:  # GL: unconstrained exp map, rows renormalised (scale-free)
                Rr = torch.matrix_exp(W) @ Q0
                Rr = Rr / Rr.norm(dim=1, keepdim=True)
            return torch.cat([Ft, Rr @ Bp.T], 0)
        for _ in range(steps):
            M = build()
            L = loss_fn(base, effective_A(M, t_dev), M, c)
            opt.zero_grad(); L.backward()
            if not torch.isfinite(W.grad).all():
                break
            opt.step()
        with torch.no_grad():
            M = build(); Lf = float(loss_fn(base, effective_A(M, t_dev), M, c))
        fin.append((Lf, M.cpu().numpy()))
    fin.sort(key=lambda z: z[0])
    return fin


def L_view(name, view="GCat"):
    f = LOSSES[name]
    return lambda base, A, M, c: f(apply_view(base, view, A))
