"""EXP11: stage A as a SOFT penalty instead of a hard constraint.
Each salient output-equivariant type t (gate: e1<0.3 and e1<0.3*e2, as in common/equivariant.py) adds
    pen_t(M) = softmin_i e_t(M_i),   e_t(w) = w^T A_t w / w^T B w   (relative residual of 'axis acts as pure shift/gain')
to a W-Y simplicity loss, with fixed normalisation by its spread over Haar rotations.  No T anywhere.
python softA.py MODEL CHANNEL REPEAT LOSS[,LOSS] MU[,MU] [STARTS]"""
import sys, time, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H
from common.equivariant import profiles, ABS_T, REL_T
from common.losses import LOSSES
from common.rotlib import apply_view, effective_A, haar
import scipy.linalg as sl
torch.set_num_threads(4)
CONDS = {"CR3": "beta5em05_lambda0_ctx0p25to0p8_L4", "C1": "beta0p0005_lambda0_ctx0p25to0p8_L4",
         "C2": "beta0p0005_lambda0_ctx0p25to0p8_L4", "C3": "beta0p0005_lambda0_ctx0p25to0p8_L6"}


def eq_matrices(c, ch):
    G = c["g"].double().numpy(); h = c["h"].double().numpy()
    B = np.einsum("skd,ske->de", G, G); act = {}
    ph = profiles(h, "prob" if ch == "prob" else "logit")
    for name, phi in ph.items():
        P = phi / np.maximum(np.linalg.norm(phi, axis=1, keepdims=True), 1e-300)
        Gp = np.einsum("sk,skd->sd", P, G)
        A = B - np.einsum("sd,se->de", Gp, Gp)
        ev, _ = sl.eigh(A, B)
        if ev[0] < ABS_T and ev[0] < REL_T * ev[1]:
            act[name] = torch.tensor(A, dtype=torch.float32)
    return torch.tensor(B, dtype=torch.float32), act


def make_loss(c, ch, loss, mu, view="GCat", temp=0.05):
    B, act = eq_matrices(c, ch)

    def pen(M):
        tot = 0.0
        for A in act.values():
            e = ((M @ A) * M).sum(1) / ((M @ B) * M).sum(1)
            tot = tot - temp * torch.logsumexp(-e / temp, 0)
        return tot
    base = H.base_of(c); t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32); d = t_dev.shape[1]
    rng = np.random.default_rng(1)
    Ms = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(200)]
    with torch.no_grad():
        a = np.array([float(LOSSES[loss](apply_view(base, view, effective_A(M, t_dev)))) for M in Ms])
        p = np.array([float(pen(M)) for M in Ms]) if act else np.array([1.0, 0.0])
    sa, sp = a.std(), p.std() + 1e-9
    return (lambda b, A, M, cc: LOSSES[loss](apply_view(b, view, A)) / sa + mu * pen(M) / sp), list(act)


if __name__ == "__main__":
    model, ch, rep = sys.argv[1], sys.argv[2], int(sys.argv[3])
    losses, mus = sys.argv[4].split(","), [float(x) for x in sys.argv[5].split(",")]
    starts = int(sys.argv[6]) if len(sys.argv) > 6 else 12
    cond = CONDS[model] + ("_rep02" if rep == 2 else "")
    c = H.load(model, cond, "decoder", ch)
    d = c["g"].shape[-1]
    R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
    print(f"== {model} {ch} rep{rep}: oracle %.3f/%.3f/%.3f" % H.evaluate(c, R), flush=True)
    for loss in losses:
        for mu in mus:
            f, act = make_loss(c, ch, loss, mu)
            t = time.time(); fin = H.optimize(c, f, "SO", F=None, starts=starts, steps=300, seed=7)
            print(f"{model} {ch} rep{rep} {loss:9s} mu={mu:<4g} active={act} TEST mcc/leak/max %.3f %.3f %.3f (%.0fs)" %
                  (H.evaluate(c, fin[0][1]) + (time.time() - t,)), flush=True)
