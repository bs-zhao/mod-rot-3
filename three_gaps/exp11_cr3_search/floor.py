"""EXP11: gradient-noise-floor-corrected W-Y simplicity losses.
Cell energy matrices E_c = mean g g^T (V-gradient space) per context cell c (CR3: design categories; C1-C3: Legendre basis cells).
Floor sigma^2 = min_c lambda_min(E_c)  (T-free rule; floor = what an axis-free null cell still carries).
Corrected E'_c = U max(Lambda - sigma^2, 0) U^T ;  axis energy e[i,c] = (A E'_c A^T)_ii with A = effective_A(M).
The loss functions of common/losses.py that only use cell energies are then applied unchanged.
python floor.py MODEL CHANNEL REPEAT LOSS[,LOSS] [STARTS] [fraction_of_floor] """
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H
from common.losses import LOSSES
from common.rotlib import effective_A
torch.set_num_threads(4)
CONDS = {"CR3": "beta5em05_lambda0_ctx0p25to0p8_L4", "C1": "beta0p0005_lambda0_ctx0p25to0p8_L4",
         "C2": "beta0p0005_lambda0_ctx0p25to0p8_L4", "C3": "beta0p0005_lambda0_ctx0p25to0p8_L6"}


def cell_energies(c):
    if c["cat"] is not None:
        g = c["g"].double(); cat = c["cat"]
        return torch.stack([torch.einsum("skd,ske->de", g[:, cat == k], g[:, cat == k]) / (g.shape[0] * int((cat == k).sum()))
                            for k in range(int(cat.max()) + 1)])
    gP = H.base_of(c)["gP"].double()
    return torch.einsum("sbd,sbe->bde", gP, gP) / gP.shape[0]


def correct(E, frac=1.0):
    lam_min = torch.stack([torch.linalg.eigvalsh(e)[0] for e in E])
    sig2 = frac * lam_min.min()
    out = []
    for e in E:
        w, u = torch.linalg.eigh(e)
        out.append((u * (w - sig2).clamp_min(0)) @ u.T)
    return torch.stack(out).float(), lam_min, float(sig2)


def make_loss(c, loss, frac=1.0):
    E = cell_energies(c)
    Ec, lam_min, sig2 = correct(E, frac)
    fn = LOSSES[loss]

    def f(_b, A, M, _c):
        e2 = torch.einsum("id,cde,ie->ic", A, Ec, A)            # (d, C) axis x cell energies
        return fn(e2.clamp_min(0).sqrt().unsqueeze(0))            # losses use mean(B^2) -> reproduces e2
    return f, lam_min, sig2, E


if __name__ == "__main__":
    model, ch, rep = sys.argv[1], sys.argv[2], int(sys.argv[3])
    losses = sys.argv[4].split(",")
    starts = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    frac = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    cond = CONDS[model] + ("_rep02" if rep == 2 else "")
    c = H.load(model, cond, "decoder", ch)
    F, _ = H.stageA(c, ch)
    R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
    print(f"== {model} {ch} rep{rep} frac={frac}: oracle TEST mcc/leak/max %.3f %.3f %.3f" % H.evaluate(c, R), flush=True)
    for loss in losses:
        f, lam_min, sig2, E = make_loss(c, loss, frac)
        if loss == losses[0]:
            print("   per-cell lambda_min(E_c):", np.round(lam_min.numpy(), 5), " floor sigma^2 = %.5f ; mean cell trace %.4f" % (sig2, float(torch.stack([e.trace() for e in E]).mean())), flush=True)
        for nm, FF in [("hardA", F), ("noA", None)]:
            t = time.time(); fin = H.optimize(c, f, "SO", F=FF, starts=starts, steps=250, seed=7)
            print(f"   {loss:10s} {nm:6s} TEST mcc/leak/max %.3f %.3f %.3f (%.0fs)" % (H.evaluate(c, fin[0][1]) + (time.time() - t,)), flush=True)
