"""EXP21 sanity: finite-difference energies vs autograd energies on the SAME centres for a fixed rotation, h -> 0, float32 vs float64 decoder."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stencils as S, x7core as X
from common.rotlib import haar
st = X.Setting("CR3", 1)
Cfull = X.population_cov(st.V[st.tr])
for dt in (torch.float32, torch.float64):
    fd = S.FD(st, "prob", dtype=dt)
    SXs, T4s = X.centre_moments(st, "prob", st.V[fd.sub], X=st.X[fd.xi])
    ml = X.MomentLoss(SXs.mean(0), T4s.mean(0), Cfull, st.d)
    M = torch.tensor(haar(st.d, np.random.default_rng(5)), dtype=torch.float32)
    eux_a, euu_a = ml.energies(M)
    for h in (0.5, 0.25, 0.1, 0.05, 0.01):
        eux, euu = fd.energies(M, h, "hybrid")
        r1 = float(torch.linalg.norm(eux.cpu().float() - eux_a) / torch.linalg.norm(eux_a)); r2 = float(torch.linalg.norm(euu.cpu().float() - euu_a) / torch.linalg.norm(euu_a))
        print(f"{str(dt)[-7:]} h={h:<5} rel err e_UX {r1:.3f}  e_UU {r2:.3f}")
