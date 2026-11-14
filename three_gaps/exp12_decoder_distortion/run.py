"""EXP12 - is the CR3 decoder failure caused by posterior averaging (decoder ~ E[f(T,x)|z])?   (DIAGNOSTIC: uses T / analytic f)

Fields in the frozen-Z coordinates V (all gradients w.r.t. V):
  decoder        : the NN decoder logit (autodiff)                          [eligible Y source]
  plug-in(s=0)   : eta(mu + sd*(A V + b), x)  - analytic f at the linear estimate of T   [auxiliary only]
  smoothed(s)    : mean_m eta(mu + sd*(A V + b + s*eps_m), x), eps_m resampled TRAIN residuals of T on V  [auxiliary only]
A,b: TRAIN least squares Tstd ~ V.   The frozen two-stage method (stage A + absL1, GCat) is then run on each field.
Question 1: does smoothed(s=1) reproduce the decoder's logit values better than plug-in / true-T analytic?
Question 2: does the method's TEST leakage grow with s and reach the decoder level at s~1?
python three_gaps/exp12_decoder_distortion/run.py [--repeat 1]
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "exp11_cr3_search"))
import harness as H
from common import io
from common.analytic import ETA
from common.decoder import decoder_eta
from common.derivs import base_derivatives
torch.set_num_threads(4)

ap = argparse.ArgumentParser(); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--M", type=int, default=32)
ap.add_argument("--scales", default="0,0.5,1,1.5"); a = ap.parse_args()
cond = "beta5em05_lambda0_ctx0p25to0p8_L4" + ("_rep02" if a.repeat == 2 else "")
dev = io.pick_device("auto")
c = H.load("CR3", cond, "decoder", "logit")
data, truth = io.load_all("CR3", 0.00005, 4, 0, a.repeat)
net, _ = io.load_decoder(data["paths"]["checkpoint"], dev)
V, Tst, tr, te = data["V"], truth.std(), data["train_ids"], data["test_ids"]
mu = torch.tensor(truth.mu); sd = torch.tensor(truth.sd)
D1 = np.c_[V[tr], np.ones(len(tr))]
B = np.linalg.lstsq(D1, Tst[tr], rcond=None)[0]; A, b = B[:-1].T, B[-1]          # Tstd ~ A V + b
resid = Tst[tr] - (V[tr] @ A.T + b)
print("residual sd per param", resid.std(0).round(3), " R2 per param", (1 - resid.var(0) / Tst[tr].var(0)).round(3), flush=True)
rng = np.random.default_rng(0)
Xb = c["X"].double()
eps_all = torch.tensor(resid[rng.choice(len(resid), a.M, replace=False)], dtype=torch.float64)   # (M,d)
At, bt = torch.tensor(A, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)

def make_h(scale):
    At_, bt_, ep_, mu_, sd_ = (z.to(dev) for z in (At, bt, eps_all, mu, sd))
    def h(v, x):
        t = At_ @ v + bt_ + scale * ep_                            # (M,d) standardised parameters
        return ETA["CR3"](mu_ + sd_ * t, x).mean()
    return h

# ---- Q1: how well does each analytic variant reproduce the decoder logit on TEST subjects?
sel = te[:300]; K = Xb.shape[0]
x_rows = np.tile(Xb.numpy(), (len(sel), 1)); si = np.repeat(np.arange(len(sel)), K)
z_sel = data["z"][sel]
eta_dec = decoder_eta(net, z_sel, x_rows, si, dev).reshape(len(sel), K)
def eta_from(tstd):  # (S,d) -> (S,K) analytic
    T = mu + sd * torch.tensor(tstd, dtype=torch.float64)
    return ETA["CR3"](T[:, None, :], Xb[None]).numpy()
fits = {"true T": eta_from(Tst[sel]), "plug-in A V": eta_from(V[sel] @ A.T + b)}
for s in [0.5, 1.0, 1.5]:
    acc = 0
    for m in range(a.M):
        acc = acc + eta_from(V[sel] @ A.T + b + s * eps_all[m].numpy())
    fits[f"smoothed s={s}"] = acc / a.M
print("\nQ1: decoder logit vs analytic variants (TEST subjects x all trials)")
for k, v in fits.items():
    r = np.corrcoef(eta_dec.ravel(), v.ravel())[0, 1]
    # best affine calibration, then relative RMSE
    cc = np.polyfit(v.ravel(), eta_dec.ravel(), 1); rr = eta_dec.ravel() - np.polyval(cc, v.ravel())
    print(f"  {k:16s} corr {r:.4f}  slope {cc[0]:.3f}  rel RMSE after affine {rr.std() / eta_dec.std():.3f}")
# per-subject: how does the variance across subjects (at fixed trial) compare?
print("  sd of eta over trials/subjects: decoder %.3f | true %.3f | plug-in %.3f | smoothed1 %.3f" %
      (eta_dec.std(), fits["true T"].std(), fits["plug-in A V"].std(), fits["smoothed s=1.0"].std()), flush=True)

# ---- Q2: run the method on each field
Rr = np.load(io.ROOT / "outputs" / "three_gaps" / "exp00" / f"CR3_{cond}_pca0" / "R_oracle_raw.npy")
print("\noracle TEST mcc/leak/max %.3f %.3f %.3f" % H.evaluate(c, Rr), flush=True)
grid = torch.tensor(V[c["ids"]], dtype=torch.float64, device=dev)
out = {}
def run(name, cc):
    F, rep = H.stageA(cc, "logit")
    rows = []
    for loss in ["absL1", "quartimax"]:
        fin = H.optimize(cc, H.L_view(loss), "SO", F=F, starts=12, steps=250, seed=7)
        rows.append((loss, H.evaluate(cc, fin[0][1]), fin[0][1]))
    print(f"{name:18s} stageA axes {len(F)} " + "  ".join(f"{l}: mcc {m[0]:.3f} leak {m[1]:.3f}/{m[2]:.2f}" for l, m, _ in rows), flush=True)
    return rows
out["decoder"] = run("decoder (NN)", c)
for s in [float(x) for x in a.scales.split(",")]:
    t0 = time.time()
    base = base_derivatives(make_h(s), grid, Xb.to(dev), which=("g",), chunk=2048)
    cc = dict(c); cc["g"] = base["g"].cpu(); cc["h"] = base["h"].cpu()
    out[f"s={s}"] = run(f"analytic smoothed s={s}", cc)
    print(f"   ({time.time() - t0:.0f}s)", flush=True)
