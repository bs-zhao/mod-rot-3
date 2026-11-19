"""POST-FREEZE full-population Oracle R vs blind Gaussian-integrated R*.

Oracle is an ALL-subject supervised rotation used ONLY to evaluate whether
the Gaussian Hessian simplicity target prefers parameter-aligned axes. No
theta enters unsupervised quadrature/optimization. All reported correlations
are IN-SAMPLE; there is no rotation-stage participant TEST split.
"""
from __future__ import annotations
import csv
import json
import numpy as np
import torch
from scipy.optimize import minimize,linear_sum_assignment
from .optimizer import givens,pairs
from .loss import GaussianHessianLoss


def corr(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    ac=a-a.mean(0);bc=b-b.mean(0)
    den=np.sqrt(np.sum(ac*ac,axis=0)[:,None]*np.sum(bc*bc,axis=0)[None,:])
    return np.divide(ac.T@bc,den,out=np.zeros((a.shape[1],b.shape[1])),
                     where=den>1e-15)


def procrustes(q,theta):
    """Orthogonal L2 target after z-scoring actual theta on ALL subjects."""
    if q.shape!=theta.shape or len(q)<=q.shape[1]:
        raise ValueError("Oracle requires all-subject square-dimensional q/theta")
    a=q-q.mean(0)
    b=theta-theta.mean(0)
    sd=b.std(0,ddof=1)
    if np.any(sd<1e-12):raise ValueError("Degenerate Oracle theta variance")
    U,_,Vt=np.linalg.svd((b/sd).T@a)
    fix=np.eye(q.shape[1])
    fix[-1,-1]=np.sign(np.linalg.det(U@Vt))
    return U@fix@Vt


def corr_oracle(q,theta,initial,restarts=20,maxiter=300,seed=41):
    """Max mean named-axis Pearson r² subject to R in SO(d), ALL people."""
    d=q.shape[1]
    qc=q-q.mean(0);tc=theta-theta.mean(0)
    covariance=torch.tensor(qc.T@qc,dtype=torch.float64)
    cross=torch.tensor(tc.T@qc,dtype=torch.float64)
    energy=torch.tensor(np.sum(tc*tc,axis=0),dtype=torch.float64)
    base=torch.tensor(initial,dtype=torch.float64)
    if np.min(np.linalg.eigvalsh(covariance.numpy()))<1e-10:
        raise ValueError("Oracle q covariance degenerate")
    def fun(x):
        angles=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        R=givens(angles,d)@base
        top=torch.einsum("ia,ia->i",R,cross)
        bottom=torch.einsum("ia,ab,ib->i",R,covariance,R)*energy
        val=-(top*top/bottom.clamp(min=1e-14)).mean()
        g=torch.autograd.grad(val,angles)[0]
        return float(val.detach()),g.detach().numpy()
    rng=np.random.default_rng(seed)
    k=len(pairs(d))
    starts=[np.zeros(k)]+[
        rng.uniform(-np.pi,np.pi,size=k) for _ in range(int(restarts)-1)]
    results=[]
    best=None
    for idx,x0 in enumerate(starts):
        sol=minimize(fun,x0,method="L-BFGS-B",jac=True,
                     bounds=[(-4*np.pi,4*np.pi)]*k,
                     options=dict(maxiter=int(maxiter),ftol=1e-13,gtol=1e-9))
        R=(givens(torch.tensor(sol.x,dtype=torch.float64),d)@base).detach().numpy()
        score=float(np.mean(np.diag(corr(theta,q@R.T))**2))
        results.append(dict(start=idx,all_subject_mean_named_r2=score,
                            success=bool(sol.success),iterations=int(sol.nit)))
        if best is None or score>best[1]:
            best=(R,score,idx)
    return best[0],dict(selected_start=best[2],
                        mean_named_all_r2=best[1],
                        restarts=results,
                        theta_used_for_oracle=True,
                        hessian_used_for_oracle=False,
                        no_held_out_subjects=True)


def subject_correlations(theta,q,R,labels,oracle=False):
    correlation=corr(theta,q@R.T)
    if oracle:assignment={name:i for i,name in enumerate(labels)}
    else:
        i,j=linear_sum_assignment(-correlation**2)
        assignment={labels[a]:int(b) for a,b in zip(i,j)}
    r2={name:float(correlation[i,assignment[name]]**2)
        for i,name in enumerate(labels)}
    return dict(correlation=correlation.tolist(),
                matched_axis=assignment,
                parameter_r2=r2,
                mean_r2=float(np.mean(list(r2.values()))),
                n_subjects=len(theta),
                scope="ALL subjects; descriptive in-sample; no TEST set")


def postfreeze(source,R,fit_mom,valid_mom,config,out):
    if not (out/"fit_complete.json").is_file() or not (out/"rotation.npz").is_file():
        raise RuntimeError("Oracle cannot run until blind rotation is frozen")
    freeze=json.loads((out/"fit_complete.json").read_text(encoding="utf-8"))
    if freeze.get("theta_used_for_blind_rotation") is not False:
        raise RuntimeError("Blind theta-free rotation provenance missing")
    with np.load(source["dataset"],allow_pickle=False) as f:
        theta=np.asarray(f["parameters"],dtype=np.float64)
    q=source["q"]
    d=q.shape[1]
    labels=["alpha","lambda_loss","gamma","tau"] if source["model"]=="CR3" else ["lambda_loss","gamma","tau"]
    if theta.shape!=(len(q),d):
        raise RuntimeError("ALL-subject parameter shape mismatch")
    Rpro=procrustes(q,theta)
    Roracle,oracle_meta=corr_oracle(q,theta,Rpro,
        restarts=int(config["oracle_restarts"]),
        maxiter=int(config["oracle_maxiter"]),seed=int(config["seed"])+603)
    fitloss=GaussianHessianLoss(fit_mom,mode=config["derivatives"],
        epsilon=float(config["loss_epsilon"]),
        uu_weight=float(config["uu_weight"]),ux_weight=float(config["ux_weight"]))
    valloss=GaussianHessianLoss(valid_mom,mode=config["derivatives"],
        epsilon=float(config["loss_epsilon"]),
        uu_weight=float(config["uu_weight"]),ux_weight=float(config["ux_weight"]))
    matrices={"identity_PCA":np.eye(d),"optimized_R_star":R,
              "oracle_procrustes":Rpro,"oracle_correlation":Roracle}
    compares={}
    for name,T in matrices.items():
        compares[name]=dict(FIT=fitloss.scalar(T),VALID=valloss.scalar(T),
            ALL=subject_correlations(theta,q,T,labels,
                                     oracle=name.startswith("oracle_")))
    gaps={
        key:compares["oracle_correlation"][key]["total"]-
            compares["optimized_R_star"][key]["total"]
        for key in ("FIT","VALID")}
    report=dict(
        model=source["model"],
        method="Gaussian-integrated fixed Decoder; user-independent q points",
        gaussian_scope=source["gaussian_scope"],
        participant_scope="ALL-subject in-sample oracle, no TEST",
        oracle_definition="ALL-theta Pearson² SO(d) multistart; supervised after freeze only",
        oracle_optimization=oracle_meta,
        theta_used_for_blind_rotation=False,
        oracle_used_to_select_blind_R=False,
        comparison=compares,
        oracle_minus_Rstar=gaps,
        R_optimized=R.tolist(),R_oracle=Roracle.tolist(),
        R_oracle_procrustes=Rpro.tolist(),
        axis_overlap_abs=np.abs(Roracle@R.T).tolist(),
        caveat="An SO(d) Oracle is only the best linear orthogonal alignment, not a guarantee of recovering nonlinear theta(z).")
    with np.load(out/"rotation.npz",allow_pickle=False) as f:
        if not np.allclose(f["R"],R):
            raise RuntimeError("Frozen R differs from Oracle evaluation input")
    np.savez_compressed(out/"oracle_rotations.npz",
                        R_optimized=R,R_oracle=Roracle,R_procrustes=Rpro)
    with (out/"oracle_loss_comparison.csv").open("w",newline="",encoding="utf-8") as file:
        cols=["R","FIT_loss","VALID_loss","ALL_subject_mean_r2"]
        writer=csv.DictWriter(file,fieldnames=cols)
        writer.writeheader()
        for name,entry in compares.items():
            writer.writerow(dict(R=name,FIT_loss=entry["FIT"]["total"],
                                 VALID_loss=entry["VALID"]["total"],
                                 ALL_subject_mean_r2=entry["ALL"]["mean_r2"]))
    with (out/"oracle_comparison.json").open("w",encoding="utf-8") as file:
        json.dump(report,file,indent=2,allow_nan=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=list(compares)
    fig,ax=plt.subplots(figsize=(9,4.3))
    xx=np.arange(len(names))
    ax.bar(xx-.18,[compares[k]["FIT"]["total"] for k in names],width=.36,label="Gaussian FIT")
    ax.bar(xx+.18,[compares[k]["VALID"]["total"] for k in names],width=.36,label="Gaussian VALID")
    ax.set_xticks(xx,["PCA","Blind R*","Oracle Procrustes","Oracle Corr"])
    ax.tick_params(axis="x",labelrotation=18)
    ax.set_ylabel("Original role-free Hessian simplicity loss")
    ax.set_title(f"{source['model']}: blind Hessian R* vs all-subject Oracle R")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"oracle_loss_comparison.png",dpi=165)
    plt.close(fig)
    return report
