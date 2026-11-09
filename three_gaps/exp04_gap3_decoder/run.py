"""EXP04 - Gap 3: frozen NP decoder replaces the formula; search R WITHOUT T.

Coordinates v = V (centred Z, or TRAIN-PCA scores), u = R v.  The decoder is
queried through the exact inverse map z = mu_Z + C^T v (README 02 sec 1.5).

sources:
  autodiff : derivative fields of the decoder itself (diagnostic: network's own
             derivatives, no statistical estimation)
  sieve    : decoder generates numeric Y (logit / prob / Bernoulli binary) on the
             REAL TRAIN trials; the SAME M2 sieve as EXP02 is fitted on (v, X, Y)
             and differentiated  (this is Gap 3 proper)
The unsupervised optimiser only sees V, X and decoder outputs.  T is loaded in
a separate object and used ONLY after R_hat is fixed (TEST correlations) and
to compare with the saved supervised oracle R (EXP00).

python three_gaps/exp04_gap3_decoder/run.py --model C3 --beta 0.0005 --zdim 6 --pca 0 \
       --candidates "logit|GP|rmsL1;prob|GP|rmsL1" --source sieve
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.analytic import X_DIM  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402
from common.rotlib import add_centered, add_projected, axis_error, match_report  # noqa: E402
from common.screen import optimize  # noqa: E402
from common.sieve import Sieve  # noqa: E402
from common.decoder import decoder_eta, decoder_h  # noqa: E402


def main():
    ap = io.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--source", default="sieve", choices=["sieve", "autodiff"])
    ap.add_argument("--subjects", type=int, default=0)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--du", type=int, default=0)
    ap.add_argument("--dx", type=int, default=3)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    dev = io.pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    data, truth = io.load_all(a.model, a.beta, a.zdim, a.pca, a.repeat, whiten=a.whiten)
    net, cfg = io.load_decoder(data["paths"]["checkpoint"], dev)
    d, p = io.D_T[a.model], X_DIM[a.model]
    bank = a.model == "CR3"
    V = data["V"]
    tr, te = data["train_ids"], data["test_ids"]
    # ---- evaluation grid (subjects' V, stimuli) - no T involved ----
    S = a.subjects or (300 if bank else 400)
    ids = np.sort(rng.choice(tr, S, replace=False))
    v_grid = torch.tensor(V[ids], dtype=torch.float64)
    m0 = data["subject_index"] == 0
    if bank:
        X = torch.tensor(data["x"][m0], dtype=torch.float64)
    else:
        X0 = rng.uniform(-1, 1, (a.nx, p))
        X = torch.tensor(np.concatenate([X0, -X0]), dtype=torch.float64)
    v_dev = v_grid.to(dev, torch.float32)
    du = a.du or (4 if bank else 3)
    out = io.ROOT / "outputs" / "three_gaps" / "exp04" / a.tag / f"{a.model}_{data['condition']}_pca{a.pca}_{a.source}"
    out.mkdir(parents=True, exist_ok=True)
    cands = [tuple(c.split("|")) for c in a.candidates.split(";")]
    # sanity: decoder through inverse map reproduces direct decode
    hz = decoder_h(net, data["mu_z"], data["Minv"], "logit")
    xs = torch.tensor(data["x"][data["subject_index"] == ids[0]][:5], dtype=torch.float64, device=dev)
    lg = net.decode(xs.float().unsqueeze(0), torch.tensor(data["z"][ids[0:1]], dtype=torch.float32, device=dev))[0]
    e_direct = (lg[:, 1] - lg[:, 0]).double()
    e_inv = torch.stack([hz(v_grid[0].to(dev), x) for x in xs])
    print(f"[sanity] inverse-map decoder max diff {float((e_direct - e_inv).abs().max()):.2e}", flush=True)

    sel = np.isin(data["subject_index"], tr)
    si, xw = data["subject_index"][sel], data["x"][sel]
    Ys = None
    if a.source == "sieve":
        eta = decoder_eta(net, data["z"], xw, si, dev)
        pr = 1 / (1 + np.exp(-eta))
        Ys = dict(logit=eta, prob=pr,
                  binary=(np.random.default_rng(a.seed + 2).uniform(size=len(pr)) < pr).astype(float))
    oracle_dir = io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{a.model}_{data['condition']}_pca{a.pca}"
    R_or = np.load(oracle_dir / "R_oracle_raw.npy") if (oracle_dir / "R_oracle_raw.npy").exists() else None
    T = truth.std()
    rep_id = match_report(V[te], T[te])
    rep_or = match_report(V[te] @ R_or.T, T[te]) if R_or is not None else None
    rows = []
    for (ch, view, loss) in cands:
        t0 = time.time()
        need = ("g", "Htt") if bank else ("g", "Htt", "Htx")
        if a.source == "autodiff":
            if ch == "binary":
                continue
            h = decoder_h(net, data["mu_z"], data["Minv"], ch)
            base = base_derivatives(h, v_grid.to(dev), X.to(dev), which=need, chunk=4096)
        else:
            sv = Sieve(d, p, du, a.dx, "onehot" if bank else "legendre", device=dev)
            if bank:
                K = int(m0.sum()); subj = si[::K]
                sv.fit_bank(V[subj], Ys[ch].reshape(len(subj), K))
                base = base_derivatives(sv.h_bank(), v_grid.to(dev), torch.eye(K, dtype=torch.float64, device=dev),
                                        which=("g", "Htt"), chunk=2048)
            else:
                sv.fit(V[si], xw, Ys[ch])
                base = base_derivatives(sv.h(), v_grid.to(dev), X.to(dev), which=need, chunk=4096)
        base = add_projected(add_centered(base), X.to(dev), bank, a.deg)
        fin = optimize(base, v_dev, view, loss, d, rng, a.starts, a.steps)
        Lb = fin[0][0]
        Rh = fin[0][1]
        spread = [axis_error(Q @ Rh.T)[0] for L, Q in fin if L <= Lb + 1e-4 * abs(Lb)]
        rep = match_report(V[te] @ Rh.T, T[te])  # ---- post-hoc only ----
        r = dict(model=a.model, source=a.source, channel=ch, view=view, loss=loss,
                 test_mcc=rep["mcc"], test_worst=rep["worst"], test_off_mean=rep["off_mean"], test_off_max=rep["off_max"],
                 identity_mcc=rep_id["mcc"], oracle_mcc=rep_or["mcc"] if rep_or else np.nan,
                 angle_to_oracle=axis_error(Rh @ R_or.T)[0] if R_or is not None else np.nan,
                 n_near_best=len(spread), near_best_spread=max(spread))
        if R_or is not None:
            with torch.no_grad():
                from common.screen import make_L
                Lo = float(make_L(base, v_dev)(R_or, view, loss))
            r["L_oracle_rel"] = (Lo - Lb) / abs(Lb)
        rows.append(r)
        np.save(out / f"R_hat_{ch}_{view.replace('/', '_')}_{loss}.npy", Rh)
        print(f"  {ch:6s} {view:10s} {loss:10s} TEST mcc={rep['mcc']:.3f} worst={rep['worst']:.3f} "
              f"offmax={rep['off_max']:.3f} | angle-to-oracle={r['angle_to_oracle']:.1f} "
              f"| oracle {r['oracle_mcc']:.3f} identity {rep_id['mcc']:.3f} | L(oracle) rel +{r.get('L_oracle_rel', np.nan):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        print("     C=\n" + np.array2string(rep["C"], precision=2), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "gap3.csv", index=False)
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# EXP04 Gap3 decoder ({a.source}) - {a.model}\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
    json.dump(vars(a), open(out / "manifest.json", "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
