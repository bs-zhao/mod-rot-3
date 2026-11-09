"""EXP05 - diagnostic (uses T / oracle R; NOT an unsupervised method).

Q1  how nonlinear is the Z -> T relation?  TEST R^2 of Tstd ~ poly_k(V), k=1..4
Q2  which structural zeros of the analytic CR3 function survive in the frozen
    decoder, viewed in the supervised oracle axes U = R_or V (z-scored)?
      * first-order trial support: RMS of dh/du_i on gain-only / gamma-neutral /
        loss-only / other trials, relative to that axis' overall RMS
      * second-order: RMS of d2h/du_i^2 relative to RMS of the row of H
    Same numbers for the analytic function in true Tstd axes for reference.

python three_gaps/exp05_cr3_structure_diag/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import io  # noqa: E402
from common.analytic import make_h  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402
from common.rotlib import effective_A, transform  # noqa: E402
from common.sieve import exponents, monomials  # noqa: E402
from common.decoder import decoder_h  # noqa: E402


def poly_r2(V, T, tr, te, k):
    E = exponents(V.shape[1], k)
    F = monomials(torch.tensor(V / V[tr].std(0)), E).numpy()
    A, *_ = np.linalg.lstsq(F[tr], T[tr], rcond=None)
    P = F[te] @ A
    return 1 - ((P - T[te]) ** 2).sum(0) / ((T[te] - T[te].mean(0)) ** 2).sum(0)


def categories(X):
    ga, la, pa, gb, lb, pb = X.T
    gain_only = (la == 0) & (lb == 0)
    loss_only = (ga == 0) & (gb == 0)
    gneutral = (pa == 0.5) & (pb == 0.5)
    return dict(gain_only=gain_only, loss_only=loss_only, gamma_neutral=gneutral,
                other=~(gain_only | loss_only | gneutral))


def report(base, A, cats, names, label):
    g = transform(base["g"], 1, A)          # (S,K,d)
    H = transform(base["Htt"], 2, A)        # (S,K,d,d)
    rms = (g ** 2).mean((0, 1)).sqrt()
    print(f"\n[{label}] first-order RMS by trial category / overall RMS of that axis")
    print("axis      " + " ".join(f"{c:>13s}" for c in cats))
    for i, n in enumerate(names):
        vals = [float((g[:, m, i] ** 2).mean().sqrt() / rms[i]) for m in cats.values()]
        print(f"{n:8s}  " + " ".join(f"{v:13.3f}" for v in vals))
    diag = torch.diagonal(H, dim1=2, dim2=3)
    row = (H ** 2).mean((0, 1)).sum(-1).sqrt()
    print(f"[{label}] RMS d2h/du_i^2 / RMS ||H_i.||:  " +
          "  ".join(f"{n}={float((diag[..., i]**2).mean().sqrt()/row[i]):.3f}" for i, n in enumerate(names)))


def main():
    ap = io.add_common_args(argparse.ArgumentParser())
    a = ap.parse_args()
    dev = io.pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    data, truth = io.load_all(a.model, a.beta, a.zdim, a.pca, a.repeat, whiten=a.whiten)
    V, tr, te, T = data["V"], data["train_ids"], data["test_ids"], truth.std()
    for k in (1, 2, 3, 4):
        print(f"poly{k} TEST R2 Tstd~V:", np.round(poly_r2(V, T, tr, te, k), 3))
    od = io.ROOT / "outputs" / "three_gaps" / "exp00" / f"{a.model}_{data['condition']}_pca{a.pca}"
    R = np.load(od / "R_oracle_raw.npy")
    from common.rotlib import match_report
    rep = match_report(V[te] @ R.T, T[te])
    perm = rep["perm"]  # oracle row i <-> parameter perm[i]
    names = [["alpha", "lambda", "gamma", "tau"][j] for j in perm]
    ids = np.sort(rng.choice(tr, 300, replace=False))
    X = torch.tensor(data["x"][data["subject_index"] == 0], dtype=torch.float64)
    cats = {k: torch.tensor(v) for k, v in categories(X.numpy()).items()}
    print({k: int(v.sum()) for k, v in cats.items()})
    net, _ = io.load_decoder(data["paths"]["checkpoint"], dev)
    for ch in ("logit", "prob"):
        t = torch.tensor(T[ids], dtype=torch.float64, device=dev)
        b = base_derivatives(make_h(a.model, ch, truth.mu, truth.sd), t, X.to(dev), which=("g", "Htt"))
        report(b, effective_A(torch.eye(4, dtype=torch.float32, device=dev), t.float()),
               cats, ["alpha", "lambda", "gamma", "tau"], f"analytic {ch}, true axes")
        v = torch.tensor(V[ids], dtype=torch.float64, device=dev)
        b = base_derivatives(decoder_h(net, data["mu_z"], data["Minv"], ch), v, X.to(dev), which=("g", "Htt"), chunk=4096)
        report(b, effective_A(torch.tensor(R, dtype=torch.float32, device=dev), v.float()),
               cats, names, f"decoder {ch}, oracle axes")


if __name__ == "__main__":
    main()
