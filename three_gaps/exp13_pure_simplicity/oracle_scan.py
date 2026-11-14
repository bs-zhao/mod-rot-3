"""EXP13 scan: for a catalogue of generic simplicity losses, how many Haar-null standard deviations below the random-rotation mean
is the loss at the supervised Oracle rotation?   z = (mean_Haar - L(oracle)) / sd_Haar.   (T / Oracle used only as a DIAGNOSTIC to find which
losses carry signal; a loss with z <= 0 cannot be minimised at the Oracle.)
Energy granularities: UX(i,j), UX(i,j,context), G(i,context) first-order, UUoff(i<k), UUdiag(i), UU(i,k,context), joint UX+UU.
Contexts = W-only cells (X modal-flag rule) for CR3, a single context for continuous X.
python oracle_scan.py"""
import sys, numpy as np, torch, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, losses2 as L2, screen as SC
from common.rotlib import effective_A, haar
torch.set_num_threads(4)
EPS = 1e-12


def ctx_moments(c):
    d = c["g"].shape[-1]
    ids = H.pure_cells(c["X"].numpy()) if c["X"].shape[1] == 6 else np.zeros(c["X"].shape[0], int)
    Htx, Htt, g = c["Htx"].double(), c["Htt"].double(), c["g"].double()
    SXc, T4c, S2c = [], [], []
    for k in range(int(ids.max()) + 1):
        m = torch.tensor(ids == k)
        a, b, gg = Htx[:, m], Htt[:, m], g[:, m]
        n = a.shape[0] * a.shape[1]
        SXc.append(torch.einsum("skaj,skbj->jab", a, a) / n)
        Hf = b.reshape(b.shape[0], b.shape[1], -1)
        T4c.append((torch.einsum("skm,skn->mn", Hf, Hf) / n).reshape(d, d, d, d))
        S2c.append(torch.einsum("ska,skb->ab", gg, gg) / n)
    w = torch.tensor([(ids == k).mean() for k in range(int(ids.max()) + 1)], dtype=torch.float32)
    S2 = torch.stack(S2c)
    if c["X"].shape[1] != 6:       # continuous X: first-order cells = Legendre-basis projections of the gradient field (generic)
        gP = H.base_of(c)["gP"].double()
        S2 = torch.einsum("sbd,sbe->bde", gP, gP) / gP.shape[0]
    lam = torch.stack([torch.linalg.eigvalsh(e)[0] for e in S2]); sig2 = lam.min()      # noise floor: smallest cell eigenvalue
    S2f = []
    for e in S2:
        ev, U = torch.linalg.eigh(e); S2f.append((U * (ev - sig2).clamp_min(0)) @ U.T)
    SXs, T4s = torch.stack(SXc), torch.stack(T4c)
    fux = torch.stack([torch.stack([torch.linalg.eigvalsh(SXs[k, j])[0] for j in range(SXs.shape[1])]) for k in range(SXs.shape[0])]).clamp_min(0)   # (C,p)
    d = T4s.shape[1]; iu = torch.triu_indices(d, d); sc = torch.where(iu[0] == iu[1], 1.0, 2.0 ** 0.5).double()
    fuu = []
    for k in range(T4s.shape[0]):                  # vech (symmetric Hessian entries) energy matrix; smallest eigenvalue = floor
        Mv = T4s[k][iu[0][:, None], iu[1][:, None], iu[0][None, :], iu[1][None, :]] * sc[:, None] * sc[None, :]
        fuu.append(torch.linalg.eigvalsh(Mv)[0].clamp_min(0))
    return (SXs.float(), T4s.float(), S2.float(), w, torch.stack(S2f).float(), fux.float(), torch.stack(fuu).float())


def tensors(A, SXc, T4c, S2c):
    e_ux = torch.einsum("ia,cjab,ib->ijc", A, SXc, A).clamp_min(0)          # (d,p,C)
    e_uu = torch.einsum("ia,kb,cabde,id,ke->ikc", A, A, T4c, A, A).clamp_min(0) if False else \
        torch.einsum("ia,kb,cabuv,iu,kv->ikc", A, A, T4c, A, A).clamp_min(0)  # (d,d,C)
    e_g = torch.einsum("ia,cab,ib->ic", A, S2c, A).clamp_min(0)             # (d,C)
    return e_ux, e_uu, e_g


def stat(e, kind):
    e = e.flatten()
    q = e / (e.sum() + EPS)
    if kind == "neff": return torch.log((-(q + EPS) * (q + EPS).log()).sum().exp())
    if kind == "quart": return -(q * q).sum()
    if kind == "l1l2": return e.sqrt().sum() / (e.sum().sqrt() + EPS)
    if kind == "soft": return (e / (e + 0.01 * e.mean() + EPS)).sum()


def catalogue(A, M):
    SXc, T4c, S2c, w, S2f, fux, fuu = M
    ux, uu, g = tensors(A, SXc, T4c, S2c)
    d = ux.shape[0]; off = (1 - torch.eye(d)).unsqueeze(-1)
    iu = torch.triu_indices(d, d, 1)
    out = {}
    cands = {"UX(i,j)": ux.sum(2), "UX(i,j,c)": ux, "G(i,c)": g,
             "UUoff(i<k)": (uu.sum(2))[iu[0], iu[1]], "UUdiag(i)": torch.diagonal(uu.sum(2)),
             "UU(i,k,c)": uu, "UUdiag(i,c)": torch.diagonal(uu, dim1=0, dim2=1).T,
             "joint UX+UUall": torch.cat([ux.sum(2).flatten(), uu.sum(2).flatten()])}
    for nm, e in cands.items():
        for kind in ["neff", "quart", "l1l2", "soft"]:
            out[f"{nm}|{kind}"] = stat(e, kind)
    gf = torch.einsum("ia,cab,ib->ic", A, S2f, A).clamp_min(0)
    n2 = (A * A).sum(1)                                                  # |a_i|^2
    uxf = (ux - fux.T.unsqueeze(0)[:, :, :ux.shape[2]] * n2[:, None, None]).clamp_min(0) if fux.shape[0] == ux.shape[2] else ux
    uuf = (uu - fuu.view(1, 1, -1) * n2[:, None, None] * n2[None, :, None]).clamp_min(0) if fuu.shape[0] == uu.shape[2] else uu
    iu2 = torch.triu_indices(ux.shape[0], ux.shape[0], 1)
    for nm, e in {"UXf(i,j,c)": uxf, "UXf(i,j)": uxf.sum(2), "UUoff-f(i<k)": uuf.sum(2)[iu2[0], iu2[1]], "UUoff-f(i,k,c)": (uuf * (1 - torch.eye(ux.shape[0])).unsqueeze(-1)),
                  "UUdiag-f(i)": torch.diagonal(uuf.sum(2)), "joint-f UX+UU": torch.cat([uxf.sum(2).flatten(), uuf.sum(2).flatten()])}.items():
        for kind in ["neff", "quart", "l1l2", "soft"]:
            out[f"{nm}|{kind}"] = stat(e, kind)
    out["share-f:UUdiag/UU"] = torch.diagonal(uuf.sum(2)).sum() / (uuf.sum() + EPS)
    def collog(e, delta):
        col = e.sum(0); q = e / (col + EPS); wt = col / (col.sum() + EPS)
        return (wt * torch.log(q + delta).sum(0)).sum()
    for nm, e in {"Gfloor(i,c)": gf}.items():
        for kind in ["neff", "quart", "l1l2", "soft"]:
            out[f"{nm}|{kind}"] = stat(e, kind)
    def collog_u(e, delta):      # every context counts equally (not energy-weighted)
        col = e.sum(0); q = e / (col + EPS)
        return torch.log(q + delta).sum(0).mean()
    for delta, tag in [(1e-1, "1"), (1e-2, "2"), (1e-3, "3"), (1e-4, "4")]:
        out[f"colLogU{tag} G floor"] = collog_u(gf, delta); out[f"colLogU{tag} G raw"] = collog_u(g, delta)
    for delta, tag in [(1e-1, "1"), (1e-2, "2"), (1e-3, "3")]:
        out[f"colLog{tag}  G raw"] = collog(g, delta); out[f"colLog{tag}  G floor"] = collog(gf, delta)
        out[f"colLog{tag}  UX(i,c) raw"] = collog(ux.sum(1), delta)
    # block shares (scale-free ratios)
    tot = ux.sum() + uu.sum() + EPS
    out["share:UUdiag/UU"] = torch.diagonal(uu.sum(2)).sum() / (uu.sum() + EPS)
    out["share:UUoff/UU"] = (uu.sum(2) * (1 - torch.eye(d))).sum() / (uu.sum() + EPS)
    out["share:UU/(UX+UU)"] = uu.sum() / tot
    return out


rows = []
import os
SPEC = [x.split(":") for x in os.environ.get("SCAN", "CR3:logit,CR3:prob,C1:logit,C1:prob,C2:logit,C2:prob,C3:logit,C3:prob").split(",")]
for model, ch in SPEC:
  if True:
    if True:
        c, cond = SC.load(model, 1, ch); d = c["g"].shape[-1]
        M = ctx_moments(c); t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
        R = np.load(SC.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
        rng = np.random.default_rng(1)
        with torch.no_grad():
            H_ = [catalogue(effective_A(torch.tensor(haar(d, rng), dtype=torch.float32), t_dev), M) for _ in range(150)]
            O = catalogue(effective_A(torch.tensor(R, dtype=torch.float32), t_dev), M)
        for k in O:
            v = np.array([float(h[k]) for h in H_]); rows.append(dict(model=model, channel=ch, loss=k, z=(v.mean() - float(O[k])) / (v.std() + 1e-12)))
        print(model, ch, "cells:", M[0].shape[0], flush=True)
df = pd.DataFrame(rows); df.to_csv(SC.OUT / "oracle_scan.csv", index=False)
p = df.pivot_table(index="loss", columns=["model", "channel"], values="z")
p["min"] = p.min(1); p["mean"] = p.mean(1)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(p.sort_values("min", ascending=False).round(2).to_string())
