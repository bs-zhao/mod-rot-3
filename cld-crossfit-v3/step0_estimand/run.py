"""Step 0 - is the CR3 decoder's second-order distortion an NN artefact or a property of the estimand P(Y | Z, x)?
DIAGNOSTIC ONLY: uses T, the analytic formula and the supervised Oracle rotation.  Nothing here is an unsupervised method.

The decoder is trained to predict choices from the posterior-mean latent, so its target is
    p*(v, x) = E[ sigmoid(eta(T, x)) | V = v ]            (average over what Z does not know about T),
not sigmoid(eta(T_hat(v), x)).  Averaging happens on the PROBABILITY scale, so "eta is linear in tau / lambda" need not survive in
logit(p*).  EXP12 smoothed eta itself (mean of logits), which keeps that linearity by construction and therefore could not test this.

Fields (all in the frozen-Z coordinates V, derivatives by autodiff, same 300 x 500 grid as the EXP13 cache):
  decoder     : frozen NP decoder                                             (EXP13 cache)
  plugin      : eta(T_hat(v), x), T_hat = TRAIN least squares of Tstd on V    (no posterior noise)
  eta_avg(s)  : mean_m eta(T_hat(v) + s*eps_m, x)                             (EXP12-style, logit averaged)
  marg(s)     : logit( mean_m sigmoid(eta(T_hat(v) + s*eps_m, x)) )           (the actual estimand at s = 1)
eps_m = resampled TRAIN residuals of Tstd on V (homoscedastic approximation of T | V).

Reported per field and channel, in the Oracle axes:
  * self-curvature ratio per axis  RMS(H_ii) / RMS(||H_i.||)   (EXP05 definition; analytic logit has 0 for lambda and tau)
  * relative distance to the decoder field for g / H_UU / H_UX
  * the frozen EXP13 loss run on that field: TEST MCC / leakage, and (L(oracle) - L(found)) / Haar sd

python cld-crossfit-v3/step0_estimand/run.py [REP=1] [M=64]
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "x6_plan06" / "common"))
import x6lib as X                                   # read-only: frozen EXP13 loss + caches
from common import io
from common.analytic import ETA
from common.derivs import base_derivatives
from common.rotlib import effective_A, transform, match_report
torch.set_num_threads(4)

rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
M = int(sys.argv[2]) if len(sys.argv) > 2 else 64
RES = HERE / "results"; RES.mkdir(exist_ok=True)
dev = io.pick_device("auto")
NAMES = ["alpha", "lambda", "gamma", "tau"]

caches = {ch: X.load("CR3", rep, ch) for ch in ("logit", "prob")}
c0, cond, R = caches["logit"]
data, truth = io.load_all("CR3", 0.00005, 4, 0, rep, need_checkpoint=False)
V, Tst, tr, te = data["V"], truth.std(), data["train_ids"], data["test_ids"]
ids, Xb = c0["ids"], c0["X"].double()
perm = match_report(V[te] @ R.T, Tst[te])["perm"]; names = [NAMES[j] for j in perm]

D1 = np.c_[V[tr], np.ones(len(tr))]
B = np.linalg.lstsq(D1, Tst[tr], rcond=None)[0]; A_lin, b_lin = B[:-1].T, B[-1]
resid = Tst[tr] - (V[tr] @ A_lin.T + b_lin)
print(f"== CR3 rep{rep}: residual sd of Tstd | V per parameter {resid.std(0).round(3)}  (R2 {(1 - resid.var(0) / Tst[tr].var(0)).round(3)})", flush=True)
rng = np.random.default_rng(0)
eps = torch.tensor(resid[rng.choice(len(resid), M, replace=False)], dtype=torch.float64, device=dev)
At, bt = torch.tensor(A_lin, dtype=torch.float64, device=dev), torch.tensor(b_lin, dtype=torch.float64, device=dev)
mu, sd = torch.tensor(truth.mu, device=dev), torch.tensor(truth.sd, device=dev)


def make_h(kind, s, ch):
    def h(v, x):
        t = At @ v + bt
        if kind == "plugin":
            e = ETA["CR3"](mu + sd * t, x)
        elif kind == "eta_avg":
            e = ETA["CR3"](mu + sd * (t + s * eps), x).mean()
        elif kind == "marg":
            p = torch.sigmoid(ETA["CR3"](mu + sd * (t + s * eps), x)).mean().clamp(1e-12, 1 - 1e-12)
            return torch.log(p / (1 - p)) if ch == "logit" else p
        return e if ch == "logit" else torch.sigmoid(e)
    return h


def selfcurv(c):
    A = effective_A(torch.tensor(R, dtype=torch.float32), torch.tensor(V[ids], dtype=torch.float32))
    H = transform(c["Htt"].float(), 2, A)
    diag = torch.diagonal(H, dim1=2, dim2=3); row = (H ** 2).mean((0, 1)).sum(-1).sqrt()
    return {n: float((diag[..., i] ** 2).mean().sqrt() / row[i]) for i, n in enumerate(names)}


def rel(a, b):          # relative squared distance, and the same after the best scalar gain
    a, b = a.double(), b.double(); k = (a * b).sum() / (b * b).sum()
    return float(((a - b) ** 2).mean() / (a ** 2).mean()), float(((a - k * b) ** 2).mean() / (a ** 2).mean()), float(k)


grid = torch.tensor(V[ids], dtype=torch.float64, device=dev)
rows = []
FIELDS = [("decoder", 0.0), ("plugin", 0.0), ("eta_avg", 1.0), ("marg", 0.5), ("marg", 1.0), ("marg", 1.5)]
for ch in ("logit", "prob"):
    c, _, _ = caches[ch]
    for kind, s in FIELDS:
        t0 = time.time()
        if kind == "decoder":
            c2 = c
        else:
            b = base_derivatives(make_h(kind, s, ch), grid, Xb.to(dev), which=("g", "Htt", "Htx"), chunk=1024)
            c2 = dict(c); c2["g"], c2["Htt"], c2["Htx"], c2["h"] = b["g"].cpu(), b["Htt"].cpu(), b["Htx"].cpu(), b["h"].cpu()
        sc = selfcurv(c2)
        base = X.Baseline(c2); fin = base.search(); m = X.evaluate(c2, fin[0][1])
        dsd = (base.total64(R) - base.total64(fin[0][1])) / base.haar_total_sd
        t_or = base.terms64(R)
        row = dict(channel=ch, field=kind, s=s, mcc=m[0], leak=m[1], leak_max=m[2], delta_sd=dsd, UUd_share_oracle=t_or["UUd_share"], UX_neff_oracle=t_or["UX_neff"])
        row.update({f"selfcurv_{n}": v for n, v in sc.items()})
        for nm in ("g", "Htt", "Htx"):
            r0, r1, k = rel(c[nm], c2[nm]); row[f"rel_{nm}"], row[f"relgain_{nm}"], row[f"gain_{nm}"] = r0, r1, k
        rows.append(row)
        print(f"{ch:5s} {kind:8s} s={s:.1f}: selfcurv " + " ".join(f"{n}={v:.3f}" for n, v in sc.items()) +
              f" | UUd_share@oracle {t_or['UUd_share']:.3f} | EXP13 loss -> mcc/leak/max {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}, delta {dsd:+.2f} sd" +
              f" | dist to decoder (after gain) g {row['relgain_g']:.3f} Huu {row['relgain_Htt']:.3f} Hux {row['relgain_Htx']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(RES / f"estimand_rep{rep}.csv", index=False)
print("oracle TEST mcc/leak/max %.3f %.3f %.3f" % X.evaluate(c0, R))
