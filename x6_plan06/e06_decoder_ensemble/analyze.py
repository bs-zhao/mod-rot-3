"""E6 step 2 - derivative fields of the retrained decoders (frozen Z, observed-choice training) and the frozen EXP13 loss on them.
For each (model, repeat, channel): fields g, H_UX, H_UU of K decoders on the same grid as the exp13 cache; replicate variance (noise-floor estimate,
T-free); frozen baseline loss search on (a) the original NP decoder field, (b) single new decoders, (c) ensemble means of size 2 / 5 / K.
python analyze.py [CHANNELS=prob] [MODELS=CR3,C1,C2,C3] [REPS=1,2] [TAG=base] [K=10]"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
import x6lib as X
from common import io                       # three_gaps.common
from common.decoder import decoder_h
from common.derivs import base_derivatives
import torch.nn as nn
torch.set_num_threads(4)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
channels = (sys.argv[1] if len(sys.argv) > 1 else "prob").split(",")
models = (sys.argv[2] if len(sys.argv) > 2 else "CR3,C1,C2,C3").split(",")
reps = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2").split(",")]
tag = sys.argv[4] if len(sys.argv) > 4 else "base"
K = int(sys.argv[5]) if len(sys.argv) > 5 else 10
dev = io.pick_device("auto")


def load_net(path, p, d):
    layers, dims = [], [p + d, 128, 256, 128, 2]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2: layers.append(nn.GELU())
    m = nn.Module(); m.decoder = nn.Sequential(*layers); m.decoder.load_state_dict(torch.load(path, map_location="cpu")); return m.to(dev).eval().requires_grad_(False)


def field(net, data, grid, Xb, ch):
    h = decoder_h(net, data["mu_z"], data["Minv"], ch)
    b = base_derivatives(h, grid, Xb.to(dev), which=("g", "Htt", "Htx"), chunk=2048)
    return b["g"].cpu(), b["Htt"].cpu(), b["Htx"].cpu()


rows = []
for ch in channels:
    for model in models:
        for rep in reps:
            t0 = time.time(); c, cond, R = X.load(model, rep, ch); d = c["g"].shape[-1]
            beta, zd = X.BETA[model]
            data, truth = io.load_all(model, beta, zd, 0, rep, need_checkpoint=False)
            edir = X.ROOT / "outputs" / "x6_plan06" / "e06" / f"{model}_{data['condition']}_{tag}"
            fp = edir / f"fields_{ch}.pt"
            grid = torch.tensor(data["V"][c["ids"]], dtype=torch.float64, device=dev)
            if fp.exists(): F = torch.load(fp)
            else:
                gs, hs, xs = [], [], []
                for k in range(K):
                    net = load_net(edir / f"dec_{k}.pt", c["X"].shape[1], d); g, Htt, Htx = field(net, data, grid, c["X"].double(), ch)
                    gs.append(g); hs.append(Htt); xs.append(Htx)
                F = dict(g=torch.stack(gs), Htt=torch.stack(hs), Htx=torch.stack(xs)); torch.save(F, fp)
            # replicate variance (noise floor) relative to total energy
            rv = {}
            for nm in ("g", "Htx", "Htt"):
                A = F[nm].double(); mean = A.mean(0)
                rv[nm] = float(((A - mean) ** 2).sum(0).mean() / (K - 1) / (mean ** 2).mean().clamp_min(1e-30) / 1.0) if False else \
                    float(((A - mean) ** 2).mean() * K / (K - 1) / (A ** 2).mean())
            def run(name, g, Htt, Htx):
                c2 = dict(c); c2["g"], c2["Htt"], c2["Htx"] = g, Htt, Htx
                b = X.Baseline(c2); fin = b.search(); m = X.evaluate(c2, fin[0][1])
                rows.append(dict(channel=ch, model=model, rep=rep, field=name, mcc=m[0], leak=m[1], leak_max=m[2],
                                 oracle_mcc=X.evaluate(c, R)[0], oracle_leak=X.evaluate(c, R)[1],
                                 delta_sd=(b.total64(R) - b.total64(fin[0][1])) / b.haar_total_sd, repvar_g=rv["g"], repvar_Htx=rv["Htx"], repvar_Htt=rv["Htt"]))
                return m
            m0 = run("original_decoder", c["g"], c["Htt"], c["Htx"])
            ms = [run(f"single_{k}", F["g"][k], F["Htt"][k], F["Htx"][k]) for k in range(3)]
            me = {n: run(f"ensemble_{n}", F["g"][:n].mean(0), F["Htt"][:n].mean(0), F["Htx"][:n].mean(0)) for n in (2, 5, K)}
            print(f"{ch:5s} {model:3s} rep{rep}: orig {m0[0]:.3f}/{m0[1]:.3f} | singles " + " ".join(f"{m[0]:.3f}/{m[1]:.3f}" for m in ms) +
                  " | ensembles " + " ".join(f"K{n}: {m[0]:.3f}/{m[1]:.3f}" for n, m in me.items()) +
                  f" | replicate var share g/Htx/Htt {rv['g']:.3f}/{rv['Htx']:.3f}/{rv['Htt']:.3f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(RES / f"ensemble_{tag}.csv", index=False)
print("done")
