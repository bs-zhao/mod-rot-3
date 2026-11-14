"""EXP13: optimise the role-free first-order U x context simplicity loss with a noise-floor multiple (no stage A, no roles).
cells: CR3 -> W-only cells (modal-flag rule, pure_cells); continuous X -> Legendre-basis projections of the gradient field.
energy e[i,c] = (A E'_c A^T)_ii,  E'_c = clip(E_c - frac*min_c lambda_min(E_c), 0);  loss from common/losses.py applied to e (soft L0, neff, colLog3...)
python cells_opt.py MODEL CHANNEL REPEAT FRAC LOSS[,LOSS] [STARTS]"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, screen as SC
from common.losses import LOSSES
from common.rotlib import effective_A
torch.set_num_threads(4)


def cell_matrices(c):
    g = c["g"].double()
    if c["X"].shape[1] == 6:
        ids = H.pure_cells(c["X"].numpy())
        return torch.stack([torch.einsum("ska,skb->ab", g[:, ids == k], g[:, ids == k]) / (g.shape[0] * int((ids == k).sum())) for k in range(int(ids.max()) + 1)])
    gP = H.base_of(c)["gP"].double()
    return torch.einsum("sbd,sbe->bde", gP, gP) / gP.shape[0]


def floored(E, frac):
    lam = torch.stack([torch.linalg.eigvalsh(e)[0] for e in E]); sig2 = frac * lam.min()
    out = []
    for e in E:
        ev, U = torch.linalg.eigh(e); out.append((U * (ev - sig2).clamp_min(0)) @ U.T)
    return torch.stack(out).float()


def make_loss(c, loss, frac):
    Ef = floored(cell_matrices(c), frac); fn = LOSSES[loss]
    return lambda b, A, M, cc: fn(torch.einsum("id,cde,ie->ic", A, Ef, A).clamp_min(0).sqrt().unsqueeze(0))


if __name__ == "__main__":
    model, ch, rep, frac = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    losses = sys.argv[5].split(","); starts = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    c, cond = SC.load(model, rep, ch); d = c["g"].shape[-1]
    R = np.load(SC.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
    print(f"== {model} {ch} rep{rep} floor x{frac}: oracle %.3f/%.3f/%.3f" % H.evaluate(c, R), flush=True)
    for loss in losses:
        t = time.time(); fin = H.optimize(c, make_loss(c, loss, frac), "SO", F=None, starts=starts, steps=250, seed=7)
        print(f"{model} {ch} rep{rep} x{frac:<3g} {loss:9s} TEST mcc/leak/max %.3f %.3f %.3f ({time.time()-t:.0f}s)" % H.evaluate(c, fin[0][1]), flush=True)
