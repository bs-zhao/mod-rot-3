"""E4 - epsilon-constraint (lexicographic) combination, no weight tuning:
  stage 1: U_min = min_Q UX_neff(Q) (24 starts);  stage 2: minimise C = UUd_share subject to UX_neff <= U_min + eps*s_U
  (penalty continuation mu = 1,10,100,1000).  Also reports how fast UX_neff rises away from its optimum (sharpness / near-optimal set width).
Frozen EXP13 pieces are only imported.  T is used only to score TEST mcc/leak.
python x6_plan06/e04_epsilon/run.py [CHANNELS=prob,logit] [EPS=0.25,0.5,1.0] [REPS=1,2]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import x6lib as X
torch.set_num_threads(4)
RES = Path(__file__).resolve().parent / "results"; RES.mkdir(exist_ok=True)
channels = (sys.argv[1] if len(sys.argv) > 1 else "prob,logit").split(",")
epss = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.25,0.5,1.0").split(",")]
reps = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2").split(",")]
MUS = (1.0, 10.0, 100.0, 1000.0)


def penalty_search(b, Q0s, eps, u_min, steps=150, lr=0.03):
    d = b.d; out = []
    for Q0 in Q0s:
        Q = torch.tensor(np.asarray(Q0), dtype=torch.float32)
        for mu in MUS:
            W = torch.zeros(d, d, requires_grad=True); opt = torch.optim.Adam([W], lr=lr); Qs = Q
            for _ in range(steps):
                M = torch.matrix_exp(W - W.T) @ Qs
                A = X.effective_A(M, b.t_dev)
                u = b.term["UX_neff"](A) / b.s["UX_neff"]; cc = b.term["UUd_share"](A) / b.s["UUd_share"]
                J = cc + mu * torch.relu(u - u_min - eps) ** 2
                opt.zero_grad(); J.backward()
                if not torch.isfinite(W.grad).all(): break
                opt.step()
            with torch.no_grad(): Q = torch.matrix_exp(W - W.T) @ Qs
        with torch.no_grad():
            A = X.effective_A(Q, b.t_dev)
            u = float(b.term["UX_neff"](A) / b.s["UX_neff"]); cc = float(b.term["UUd_share"](A) / b.s["UUd_share"])
        out.append((u, cc, Q.numpy()))
    return out


def sharpness(b, Q, u_min, radii=(5, 10, 20, 30, 45), n=100, seed=0):
    rng = np.random.default_rng(seed); B = X.skew_basis(b.d); res = {}
    for r in radii:
        du = []
        for _ in range(n):
            xi = rng.standard_normal(len(B)); xi = xi / np.linalg.norm(xi) * np.radians(r)
            M = X.expm_np(sum(x * bb for x, bb in zip(xi, B))) @ Q
            du.append(b.terms64(M)["UX_neff"] / b.s["UX_neff"] - u_min)
        res[r] = float(np.median(du))
    return res


rows = []
for ch in channels:
    for model in X.MODELS:
        for rep in reps:
            t0 = time.time(); c, cond, R = X.load(model, rep, ch); b = X.Baseline(c); d = b.d
            # frozen equal-weight baseline and UX-only
            fb = b.search(); mb = X.evaluate(c, fb[0][1])
            f_u = lambda bb, A, M, cc: b.term["UX_neff"](A) / b.s["UX_neff"]
            fu = X.H.optimize(c, f_u, "SO", F=None, starts=24, steps=250, seed=7); mu_ = X.evaluate(c, fu[0][1])
            u_min = float(fu[0][0]); Qu = fu[0][1]
            sh = sharpness(b, Qu, u_min)
            q0s = [f[1] for f in fu[:6]] + [X.haar(d, np.random.default_rng(100 + k)) for k in range(6)]
            for eps in epss:
                res = penalty_search(b, q0s, eps, u_min)
                feas = [r for r in res if r[0] - u_min <= eps + 0.05]
                u, cc, Q = min(feas, key=lambda r: r[1]) if feas else min(res, key=lambda r: r[0] + r[1])
                m = X.evaluate(c, Q)
                rows.append(dict(channel=ch, model=model, rep=rep, eps=eps, mcc=m[0], leak=m[1], leak_max=m[2], feasible=bool(feas), u_gap=u - u_min, C=cc,
                                 base_mcc=mb[0], base_leak=mb[1], uxonly_mcc=mu_[0], uxonly_leak=mu_[1], oracle_mcc=X.evaluate(c, R)[0], oracle_leak=X.evaluate(c, R)[1],
                                 **{f"dU_at_{r}deg": v for r, v in sh.items()}))
                np.save(RES / f"Q_{model}_rep{rep}_{ch}_eps{eps}.npy", Q)
            r0 = rows[-len(epss):]
            print(f"{ch:5s} {model:3s} rep{rep}: base {mb[0]:.3f}/{mb[1]:.3f}  UXonly {mu_[0]:.3f}/{mu_[1]:.3f}  eps-constraint " +
                  "  ".join(f"e{r['eps']}: {r['mcc']:.3f}/{r['leak']:.3f}" for r in r0) + f"  | dU(5/20/45 deg) {sh[5]:.2f}/{sh[20]:.2f}/{sh[45]:.2f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(RES / "epsilon_constraint.csv", index=False)
print("done")
