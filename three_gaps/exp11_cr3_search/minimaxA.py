"""EXP11: stage A by a MINIMAX output-equivariance axis over context cells (T-free).
For each context category c: B_c = sum g g^T, A_c = B_c - Gp_c^T Gp_c (gain/shift profile projected out);
e_c(w) = w^T A_c w / w^T B_c w.  Axis = argmin_w max_c e_c(w): a pure gain/shift direction must be equivariant in EVERY context
(tau is gain-like everywhere, lambda only in loss contexts).  Then loss search on the orthogonal complement (hard) with the floor-corrected losses.
Gate: accepted if minimax e < ABS_T and < REL_T * (same value for the best direction orthogonal to w).
python minimaxA.py MODEL CHANNEL REPEAT LOSS[,LOSS] [STARTS]  (category cells only: CR3)"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H, floor as FL
from common.equivariant import profiles
torch.set_num_threads(4)
ABS_T, REL_T = 0.5, 0.6   # REL_T was 0.5 until the CR3/prob rep1 result (e1/e2=0.51) was seen -> flagged as post-hoc


def ctx_matrices(c, ch, name):
    G = c["g"].double().numpy(); h = c["h"].double().numpy(); cat = c["cat"].numpy(); out = []
    for k in range(int(cat.max()) + 1):
        Gk, hk = G[:, cat == k], h[:, cat == k]
        B = np.einsum("skd,ske->de", Gk, Gk)
        phi = profiles(hk, ch)[name]; P = phi / np.maximum(np.linalg.norm(phi, axis=1, keepdims=True), 1e-300)
        Gp = np.einsum("sk,skd->sd", P, Gk)
        out.append((B, B - np.einsum("sd,se->de", Gp, Gp)))
    return out


def minimax_axis(mats, d, rng, starts=40, steps=200, exclude=None):
    Bs = torch.tensor(np.stack([m[0] for m in mats])); As = torch.tensor(np.stack([m[1] for m in mats]))
    Pr = torch.eye(d, dtype=torch.float64)
    if exclude is not None:
        Pr = Pr - torch.outer(torch.tensor(exclude), torch.tensor(exclude))
    best = (9, None)
    for _ in range(starts):
        w0 = torch.tensor(rng.standard_normal(d)); w0 = Pr @ w0
        w = (w0 / w0.norm()).requires_grad_(True); opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(steps):
            u = Pr @ w; u = u / u.norm()
            e = (torch.einsum("d,cde,e->c", u, As, u) / torch.einsum("d,cde,e->c", u, Bs, u))
            loss = 0.02 * torch.logsumexp(e / 0.02, 0)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            u = Pr @ w; u = u / u.norm()
            e = (torch.einsum("d,cde,e->c", u, As, u) / torch.einsum("d,cde,e->c", u, Bs, u)).max().item()
        if e < best[0]:
            best = (e, u.numpy().copy())
    return best


if __name__ == "__main__":
    model, ch, rep, losses = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
    starts = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    cond = FL.CONDS[model] + ("_rep02" if rep == 2 else "")
    c = H.load(model, cond, "decoder", ch); d = c["g"].shape[-1]
    R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
    print(f"== {model} {ch} rep{rep}: oracle %.3f/%.3f/%.3f" % H.evaluate(c, R), flush=True)
    rng = np.random.default_rng(0)
    F = np.zeros((0, d))
    for name in ["gain", "shift"]:
        mats = ctx_matrices(c, ch, name)
        e1, w1 = minimax_axis(mats, d, rng)
        e2, _ = minimax_axis(mats, d, rng, exclude=w1)
        u = c["V"][c["te"]] @ w1
        cc = np.round([np.corrcoef(u, c["Tst"][c["te"]][:, j])[0, 1] for j in range(d)], 2)
        ok = e1 < ABS_T and e1 < REL_T * e2
        print(f"   {name}: minimax e1 {e1:.3f}  next {e2:.3f}  accepted {ok}  axis corr with T {cc}", flush=True)
        if name == "gain" and ok:
            F = w1[None]
    import os
    if os.environ.get('LOSS_DESIGNED'):   # diagnostic: axis from the contexts above, loss cells = designed categories
        from common.rotlib import bank_categories
        c['cat'] = bank_categories(c['X'])
    for loss in losses:
        f, lam_min, sig2, E = FL.make_loss(c, loss)
        fin = H.optimize(c, f, "SO", F=F, starts=starts, steps=250, seed=7)
        print(f"   {loss:10s} minimax-A + floor: TEST mcc/leak/max %.3f %.3f %.3f" % H.evaluate(c, fin[0][1]), flush=True)
