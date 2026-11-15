"""EXP13 tuning / robustness of the role-free combination  L = UX_stat/s1 + w * CURV/s2   (s = Haar-null sd of each term, 200 rotations).
Config = (ux_stat, curv, w, std_x).  ux_stat in {neff,l1l2,quartimax,softL0}, curv in {share,sum}.  std_x: standardise X before second moments.
T is used only to score TEST mcc/leak.   Selection protocol: choose on repeat A, evaluate on the held-out repeat B (and swap).
python tune.py REPEAT OUTCSV "stat,curv,w,std;stat,curv,w,std;..." [STARTS] [SEED_OPT] [SEED_HAAR] [SUBSAMPLE_FRAC] [SUBSAMPLE_SEED]"""
import sys, time, numpy as np, torch, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H, losses2 as L2, screen as SC
from common.rotlib import effective_A, haar
torch.set_num_threads(4)
rep = int(sys.argv[1]); out = sys.argv[2]
cfgs = [tuple(x.split(",")) for x in sys.argv[3].split(";")]
starts = int(sys.argv[4]) if len(sys.argv) > 4 else 12
seed_opt = int(sys.argv[5]) if len(sys.argv) > 5 else 7
seed_haar = int(sys.argv[6]) if len(sys.argv) > 6 else 1
sub_frac = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
sub_seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
UXF = {"neff": L2.neff, "l1l2": L2.l1l2, "quartimax": L2.quartimax, "softL0": L2.softL0}
CURV = {"share": lambda uu: torch.diagonal(uu).sum() / (uu.sum() + L2.EPS), "sum": lambda uu: torch.diagonal(uu).sum()}


def subsample(c, frac, seed):
    if frac >= 1.0: return c
    S = c["g"].shape[0]; k = max(int(S * frac), 20); idx = np.sort(np.random.default_rng(seed).choice(S, k, replace=False))
    c2 = dict(c)
    for key in ("g", "Htt", "Htx", "h"): c2[key] = c[key][idx]
    c2["ids"] = c["ids"][idx]
    return c2


rows = []
for model in ["CR3", "C1", "C2", "C3"]:
    for ch in ["logit", "prob"]:
        c0, cond = SC.load(model, rep, ch); c = subsample(c0, sub_frac, sub_seed); d = c["g"].shape[-1]
        R = np.load(SC.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{model}_{cond}_pca0" / "R_oracle_raw.npy")
        om = H.evaluate(c0, R)
        t_dev = torch.tensor(c["V"][c["ids"]], dtype=torch.float32)
        mom = {s: L2.moments(c, standardize_x=bool(int(s))) for s in {x[3] for x in cfgs}}
        rng = np.random.default_rng(seed_haar); Ms = [torch.tensor(haar(d, rng), dtype=torch.float32) for _ in range(200)]
        for (stat, curv, w, std) in cfgs:
            SX, T4 = mom[std]; w = float(w)
            fu = lambda A: UXF[stat](L2.energies(A, SX, T4)[0]); fc = lambda A: CURV[curv](L2.energies(A, SX, T4)[1])
            with torch.no_grad():
                su = np.std([float(fu(effective_A(M, t_dev))) for M in Ms]) + 1e-9; sc_ = np.std([float(fc(effective_A(M, t_dev))) for M in Ms]) + 1e-9
            f = lambda b, A, M, cc: fu(A) / su + (w * fc(A) / sc_ if w > 0 else 0.0)
            fin = H.optimize(c0, f, "SO", F=None, starts=starts, steps=250, seed=seed_opt)
            m = H.evaluate(c0, fin[0][1])
            rows.append(dict(rep=rep, model=model, channel=ch, stat=stat, curv=curv, w=w, std=std, mcc=m[0], leak=m[1], leak_max=m[2],
                             oracle_mcc=om[0], oracle_leak=om[1], seed_opt=seed_opt, seed_haar=seed_haar, sub_frac=sub_frac, sub_seed=sub_seed))
        print(model, ch, "done", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
print("finished")
