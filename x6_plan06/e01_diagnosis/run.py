"""E1 - diagnosis: is the frozen EXP13 loss mis-specified at the true axes, or did the optimiser miss them?
For every (model, repeat, channel): frozen search -> Q_found; compare L(R_oracle) with L(Q_found) in Haar-sd units; per-plane loss curves
around both; Riemannian Hessian (so(d) coordinates) at both; per-axis angle errors.  Oracle / T are used only here, as diagnostics.
python x6_plan06/e01_diagnosis/run.py [CHANNELS=prob,logit]"""
import sys, itertools, json
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import x6lib as X
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(4)
RES = Path(__file__).resolve().parent / "results"; RES.mkdir(exist_ok=True)
channels = (sys.argv[1] if len(sys.argv) > 1 else "prob,logit").split(",")
H_STEP = 0.02


def hessian(F, Q, d, h=H_STEP):
    B = X.skew_basis(d); n = len(B)
    def f(xi):
        S = sum(x * b for x, b in zip(xi, B)); return F(X.expm_np(S) @ Q)
    f0 = f(np.zeros(n)); g = np.zeros(n); Hm = np.zeros((n, n))
    e = np.eye(n) * h
    fp = [f(e[a]) for a in range(n)]; fm = [f(-e[a]) for a in range(n)]
    for a in range(n):
        g[a] = (fp[a] - fm[a]) / (2 * h); Hm[a, a] = (fp[a] - 2 * f0 + fm[a]) / h ** 2
    for a in range(n):
        for b in range(a + 1, n):
            Hm[a, b] = Hm[b, a] = (f(e[a] + e[b]) - f(e[a] - e[b]) - f(-e[a] + e[b]) + f(-e[a] - e[b])) / (4 * h ** 2)
    return g, np.linalg.eigvalsh(Hm)


rows, curves = [], {}
for ch in channels:
    for model in X.MODELS:
        for rep in (1, 2):
            c, cond, R = X.load(model, rep, ch); d = c["g"].shape[-1]
            b = X.Baseline(c); fin = b.search(); Q = fin[0][1]
            mq, lq, lxq = X.evaluate(c, Q); mo, lo, lxo = X.evaluate(c, R)
            Lq, Lo = b.total64(Q), b.total64(R)
            ang, perm = X.axis_angles(Q, R)
            gq, eq = hessian(b.total64, Q, d); go, eo = hessian(b.total64, R, d)
            tq, to = b.terms64(Q), b.terms64(R)
            # plane curves around Q_found and R_oracle
            th = np.radians(np.arange(-90, 91, 2))
            cur = {}
            for (i, j) in itertools.combinations(range(d), 2):
                cur[(i, j)] = (np.array([b.total64(X.givens(d, i, j, t) @ Q) for t in th]), np.array([b.total64(X.givens(d, i, j, t) @ R) for t in th]))
            curves[(ch, model, rep)] = (th, cur)
            rows.append(dict(channel=ch, model=model, rep=rep, mcc=mq, leak=lq, leak_max=lxq, oracle_mcc=mo, oracle_leak=lo, oracle_leak_max=lxo,
                             L_found=Lq, L_oracle=Lo, delta_sd=(Lo - Lq) / b.haar_total_sd, haar_sd=b.haar_total_sd,
                             UX_found=tq["UX_neff"], UX_oracle=to["UX_neff"], UU_found=tq["UUd_share"], UU_oracle=to["UUd_share"],
                             max_axis_angle_deg=float(ang.max()), mean_axis_angle_deg=float(ang.mean()), axis_angles_deg=json.dumps(np.round(ang, 1).tolist()),
                             hess_min_found=float(eq[0]), hess_max_found=float(eq[-1]), hess_min_oracle=float(eo[0]), hess_max_oracle=float(eo[-1]),
                             grad_norm_found=float(np.linalg.norm(gq)), grad_norm_oracle=float(np.linalg.norm(go))))
            r = rows[-1]
            print(f"{ch:5s} {model:3s} rep{rep}: mcc {mq:.3f} leak {lq:.3f} | L_oracle-L_found = {r['delta_sd']:+.2f} sd | max axis angle {r['max_axis_angle_deg']:.1f} deg "
                  f"| hess min/max found {eq[0]:.2f}/{eq[-1]:.1f} oracle {eo[0]:.2f}/{eo[-1]:.1f} | |grad| oracle {r['grad_norm_oracle']:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(RES / "diagnosis.csv", index=False)
# figures: planes for CR3 (and C3 if present)
for (ch, model, rep), (th, cur) in curves.items():
    if model != "CR3" or ch != "prob": continue
    n = len(cur); fig, axs = plt.subplots(2, 3, figsize=(12, 6.5)); axs = axs.ravel()
    for ax, ((i, j), (cq, co)) in zip(axs, cur.items()):
        ax.plot(np.degrees(th), cq, label="around Q_found"); ax.plot(np.degrees(th), co, label="around R_oracle")
        ax.axvline(0, color="grey", lw=.5); ax.set_title(f"plane ({i},{j})", fontsize=9); ax.set_xlabel("Givens angle (deg)"); ax.set_ylabel("L")
    axs[0].legend(fontsize=7); fig.suptitle(f"CR3 {ch} rep{rep}: frozen EXP13 loss along rotation planes", fontsize=10); fig.tight_layout()
    fig.savefig(RES / f"planes_{model}_{ch}_rep{rep}.png", dpi=120); plt.close(fig)
print("done")
