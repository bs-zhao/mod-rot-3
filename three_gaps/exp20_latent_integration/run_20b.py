"""EXP20B - continuous Gaussian latent integration of the frozen EXP13 moments.

python run_20b.py MODEL REPEAT CHANNEL nodes   # decoder queries -> per-node second moments (GPU, cached; resumable)
python run_20b.py MODEL REPEAT CHANNEL eval    # aggregation, search, convergence + measure comparison (CPU)
Phase 1 (same measure P1 = N(mu_train, Sigma_train), C_ref = full TRAIN): MC (5 seeds), RQMC (scrambled Sobol, 8 scrambles), Gauss-Hermite orders.
Phase 2 (converged RQMC budget): P0 empirical TRAIN, P1, P2 diag, P3 N(0,I) in original Z, P4 spherical, P5 truncated 95%, P6 c^2 Sigma.
No labels / T enter the integration or the optimisation; T only in the post-hoc scores.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x7core as X
import nodes as N
from common.rotlib import haar

model, rep, ch, stage = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
st = X.Setting(model, rep)
S, Lch, eigs, jitter = N.cholesky_cov(st.V[st.tr])
d = st.d
NODEDIR = X.OUT / f"nodes_{model}_{st.cond}_{ch}"
NODEDIR.mkdir(exist_ok=True)
MC_N, RQ_N, RQ_SCR, P2_N, P2_SCR = 8192, 8192, 8, 4096, 4
GH_M = [3, 5, 7, 9] if d == 4 else [3, 5]            # d=6: m=7 (117,649 nodes) is run only after a low-order check


def node_sets():
    sets = {}
    for s in range(5):
        sets[f"P1_mc_s{s}"] = ("P1", N.mc_eps(MC_N, d, 100 + s), None, {})
    for s in range(RQ_SCR):
        eps, bad = N.rqmc_eps(RQ_N, d, 200 + s)
        sets[f"P1_rqmc_s{s}"] = ("P1", eps, None, dict(clipped=bad))
    for m in GH_M:
        eps, w = N.gh_eps(m, d)
        sets[f"P1_gh_m{m}"] = ("P1", eps, w, {})
    for kind, c in (("P2", 1.0), ("P3", 1.0), ("P4", 1.0), ("P6", 0.5), ("P6", 0.75), ("P6", 1.25), ("P6", 1.5)):
        for s in range(P2_SCR):
            eps, _ = N.rqmc_eps(P2_N, d, 300 + s)
            sets[f"{kind}{'' if kind != 'P6' else '_c' + str(c)}_rqmc_s{s}"] = (kind, eps, None, dict(c=c))
    return sets


def stage_nodes():
    for name, (kind, eps, w, info) in node_sets().items():
        f = NODEDIR / f"{name}.npz"
        if f.exists():
            continue
        t0 = time.time()
        V = N.measure_V(kind, eps, S, Lch, st.mu_z, c=info.get("c", 1.0))
        SX, T4 = X.centre_moments(st, ch, V, bs=32)
        tmp = NODEDIR / f"{name}.tmp.npz"
        np.savez(tmp, SX=SX, T4=T4, eps=eps, V=V, w=np.ones(len(V)) if w is None else w)
        tmp.replace(f)
        print(f"{name}: {len(V)} nodes in {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(model=model, rep=rep, channel=ch, cov_eigs=eigs.tolist(), jitter=jitter, device=str(X.DEV),
                   checkpoint_sha256=X.sha256(st.data["paths"]["checkpoint"]), x_bank=str(st.cache_path)), open(NODEDIR / "COMPLETE.json", "w"), indent=1)


def load_set(name):
    z = np.load(NODEDIR / f"{name}.npz")
    return z["SX"], z["T4"], z["w"], z["V"], z["eps"]


def wavg(SX, T4, w, n=None):
    n = len(w) if n is None else n
    ww = w[:n] / w[:n].sum()
    return np.tensordot(ww, SX[:n], 1), np.tensordot(ww, T4[:n], 1)


def stage_eval():
    mom = np.load(X.OUT / f"moments_{model}_{st.cond}_{ch}.npz")
    tr = st.tr
    Cfull = X.population_cov(st.V[tr])
    rngh = np.random.default_rng(11)
    HAAR = [torch.tensor(haar(d, rngh), dtype=torch.float32) for _ in range(1000)]
    rows = []
    # reference: pooled RQMC at the largest budget (all scrambles) = converged P1 integral (no labels)
    SXr = np.mean([wavg(*load_set(f"P1_rqmc_s{s}")[:3])[0] for s in range(RQ_SCR)], 0)
    T4r = np.mean([wavg(*load_set(f"P1_rqmc_s{s}")[:3])[1] for s in range(RQ_SCR)], 0)
    mlr = X.MomentLoss(SXr, T4r, Cfull, d)
    with torch.no_grad():
        Lref = np.array([float(mlr.total(M)) for M in HAAR])
    finr = mlr.search()
    Qref = finr[0][1]

    def rec(phase, name, integrator, budget, nodes, SX_, T4_, Cref, ref=True, extra=None, scales=None):
        t0 = time.time()
        ml = X.MomentLoss(SX_, T4_, Cref, d, scales=scales)
        fin = ml.search()
        Q = fin[0][1]
        mcc, leak, lmax = X.evaluate(st, Q)
        with torch.no_grad():
            L = np.array([float(ml.total(M)) for M in HAAR])
        r = dict(model=model, rep=rep, channel=ch, phase=phase, name=name, integrator=integrator, budget=budget, nodes=nodes, mcc=mcc, leak=leak,
                 leak_max=lmax, loss=fin[0][0], err_SX=float(np.linalg.norm(SX_ - SXr) / np.linalg.norm(SXr)), err_T4=float(np.linalg.norm(T4_ - T4r) / np.linalg.norm(T4r)),
                 spearman_vs_ref=float(spearmanr(L, Lref)[0]), angle_to_ref_deg=X.signed_perm_angle(Q, Qref), oracle_angle_deg=X.signed_perm_angle(Q, st.oracle),
                 sUX=ml.s["UX_neff"], sUU=ml.s["UUd_share"], secs=time.time() - t0)
        if extra:
            r.update(extra)
        rows.append(r)
        print(f"{phase} {name:16s} {integrator:5s} budget {budget:6d} mcc {mcc:.4f} leak {leak:.4f} errSX {r['err_SX']:.3f} spearman {r['spearman_vs_ref']:.3f} "
              f"angle-to-ref {r['angle_to_ref_deg']:.1f}", flush=True)
        pd.DataFrame(rows).to_csv(RES / f"exp20b_{model}_rep{rep}_{ch}.csv", index=False)

    # ---- phase 1: P1 convergence
    for nb in (256, 512, 1024, 2048, 4096, 8192):
        for s in range(5):
            SX, T4, w, V, eps = load_set(f"P1_mc_s{s}")
            a, b = wavg(SX, T4, w, nb); rec("P1", f"mc_s{s}", "MC", nb, nb, a, b, Cfull)
        for s in range(5):
            SX, T4, w, V, eps = load_set(f"P1_rqmc_s{s}")
            a, b = wavg(SX, T4, w, nb); rec("P1", f"rqmc_s{s}", "RQMC", nb, nb, a, b, Cfull)
    for m in GH_M:
        SX, T4, w, V, eps = load_set(f"P1_gh_m{m}")
        a, b = wavg(SX, T4, w); rec("P1", f"gh_m{m}", "GH", m, len(w), a, b, Cfull)
    # ---- phase 2: measure comparison at the converged RQMC budget (4 scrambles x 4096 pooled)
    def pooled(prefix, n=P2_N, scr=P2_SCR):
        out = [wavg(*load_set(f"{prefix}_rqmc_s{s}")[:3], n) for s in range(scr)]
        return np.mean([o[0] for o in out], 0), np.mean([o[1] for o in out], 0)
    SX0, T40 = mom["SX"][tr].mean(0), mom["T4"][tr].mean(0)
    rec("P2", "P0_empirical", "emp", len(tr), len(tr), SX0, T40, Cfull)
    for nm, pre, cinfo in (("P1_gauss", "P1", None), ("P2_diag", "P2", None), ("P3_N01", "P3", None), ("P4_spherical", "P4", None),
                           ("P6_c0.5", "P6_c0.5", None), ("P6_c0.75", "P6_c0.75", None), ("P6_c1.25", "P6_c1.25", None), ("P6_c1.5", "P6_c1.5", None)):
        if pre == "P1":
            a, b = pooled("P1")
        else:
            a, b = pooled(pre)
        rec("P2", nm, "RQMC", P2_N * P2_SCR, P2_N * P2_SCR, a, b, Cfull)
    # P5 truncated Gaussian (95%): weights on P1 RQMC nodes
    ws = []
    for s in range(P2_SCR):
        SX, T4, w, V, eps = load_set(f"P1_rqmc_s{s}")
        wt = N.truncation_weights(eps[:P2_N], 0.95)
        ws.append((wavg(SX, T4, wt, P2_N), float(wt.mean())))
    rec("P2", "P5_trunc95", "RQMC", P2_N * P2_SCR, P2_N * P2_SCR, np.mean([x[0][0] for x in ws], 0), np.mean([x[0][1] for x in ws], 0), Cfull,
        extra=dict(trunc_rate=1 - float(np.mean([x[1] for x in ws]))))
    # measure-native reference covariance (axis scaling follows the measure)
    a, b = pooled("P1")
    rec("P2", "P1_native_C", "RQMC", P2_N * P2_SCR, P2_N * P2_SCR, a, b, S)
    a, b = pooled("P4")
    rec("P2", "P4_native_C", "RQMC", P2_N * P2_SCR, P2_N * P2_SCR, a, b, np.trace(S) / d * np.eye(d))
    a, b = pooled("P3")
    rec("P2", "P3_native_C", "RQMC", P2_N * P2_SCR, P2_N * P2_SCR, a, b, np.eye(d))
    # tail / support diagnostics from the P1 RQMC nodes: share of moment energy by empirical-Mahalanobis strata of the TRAIN latent
    Vtr = st.V[tr] - st.V[tr].mean(0)
    Sinv = np.linalg.inv(S)
    mdist = np.einsum("nd,de,ne->n", Vtr, Sinv, Vtr)
    qs = np.quantile(mdist, [0.5, 0.9, 0.95, 0.99])
    SX, T4, w, V, eps = load_set("P1_rqmc_s0")
    md = np.einsum("nd,de,ne->n", V, Sinv, V)
    edges = [0.0] + list(qs) + [np.inf]
    strata = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (md >= lo) & (md < hi)
        strata.append(dict(lo=float(lo), hi=float(hi), node_share=float(m.mean()), T4_energy_share=float(np.linalg.norm(T4[m].reshape(m.sum(), -1)) ** 2 / np.linalg.norm(T4.reshape(len(T4), -1)) ** 2),
                           SX_energy_share=float((SX[m] ** 2).sum() / (SX ** 2).sum())))
    # Gaussianity diagnostics of the TRAIN latent (no T): skew/kurtosis per axis, Mahalanobis vs chi-square_d quantiles
    from scipy import stats as ss
    diag = dict(cov_eigs=eigs.tolist(), skew=ss.skew(Vtr, 0).tolist(), excess_kurtosis=ss.kurtosis(Vtr, 0).tolist(),
                mahalanobis_q=dict(zip(["50", "90", "95", "99"], qs.tolist())), chi2_q=dict(zip(["50", "90", "95", "99"], ss.chi2.ppf([0.5, 0.9, 0.95, 0.99], d).tolist())),
                energy_strata=strata, jitter=jitter)
    json.dump(diag, open(RES / f"exp20b_{model}_rep{rep}_{ch}_diagnostics.json", "w"), indent=1)
    json.dump(dict(model=model, rep=rep, channel=ch), open(RES / f"exp20b_{model}_rep{rep}_{ch}.COMPLETE.json", "w"))
    print("done eval")


if __name__ == "__main__":
    {"nodes": stage_nodes, "eval": stage_eval}[stage]()
