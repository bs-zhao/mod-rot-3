"""CR2: independent 4P/5P CPT mixed/zero/sure fixed-battery design and oracle pilot.

Run from repo root, no editable install:
 python experiment/choice/research_based/s0_cr2_design_recovery.py --mode all --subjects 300

No CR1 code/datasets are modified. All random design reference thetas are
independent of the held-out oracle-recovery subjects. Numerical results are
not validation until independently replicated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist, gamma as gamma_dist, truncnorm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "choice" / "CR2" / "s0_design_recovery"
NAMES = {4: ["alpha", "lambda_loss", "gamma", "tau"],
         5: ["alpha", "lambda_loss", "gamma", "delta", "tau"]}
# Gamma is shape/scale. These are proposed population distributions,
# NOT empirical posterior estimates or priors copied from Pachur (2017).
BOUNDS = {4: np.array([[.02,.999],[.3,5],[.02,.999],[.5,4.]]),
          5: np.array([[.02,.999],[.3,5],[.02,.999],[.3,3.],[.5,4.]])}
FAMILIES = ("mixed_mixed", "mixed_zero", "mixed_gain_sure",
            "mixed_loss_sure", "gain_sure", "loss_sure", "gain_gain",
            "loss_loss", "mixed_gain_risk", "mixed_loss_risk")
# Each lottery is (gain, positive loss magnitude, probability of gain).
# (g,0,p) and (0,l,p) encode positive or negative outcome versus zero.
# A sure gain uses p=1, a sure loss uses p=0.
def sample_prior(n, dim, rng):
    a = beta_dist.rvs(12, 3, size=n, random_state=rng)
    lam = gamma_dist.rvs(a=9, scale=.22, size=n, random_state=rng)
    # Resample at [0.3,5], not clipping (clipping creates artificial atoms).
    bad = (lam < .3) | (lam > 5.)
    while bad.any():
        lam[bad] = gamma_dist.rvs(a=9, scale=.22, size=int(bad.sum()), random_state=rng)
        bad = (lam < .3) | (lam > 5.)
    g = beta_dist.rvs(9, 4, size=n, random_state=rng)
    tau = truncnorm.rvs((.5-2.2)/.55, (4.-2.2)/.55,
                        loc=2.2, scale=.55, size=n, random_state=rng)
    if dim == 4:
        return np.column_stack([a,lam,g,tau])
    delta = np.exp(rng.normal(0,.35,size=n))
    bad = (delta < .3) | (delta > 3.)
    while bad.any():
        delta[bad] = np.exp(rng.normal(0,.35,size=int(bad.sum())))
        bad = (delta < .3) | (delta > 3.)
    return np.column_stack([a,lam,g,delta,tau])

def weighting(p, gamma, delta):
    p = np.asarray(p)
    return expit(gamma*logit(np.clip(p,1e-12,1-1e-12)) + np.log(delta))

def lottery_value(lot, theta):
    # Supports theta shape (...,d), lot shape (...,3), broadcasting.
    a, lam, gamma = theta[...,0], theta[...,1], theta[...,2]
    delta = theta[...,3] if theta.shape[-1] == 5 else 1.
    gain, loss, p = lot[...,0], lot[...,1], lot[...,2]
    wg = weighting(p,gamma,delta)
    wl = weighting(1-p,gamma,delta)
    # At p=0/1 continuity applies; p=0 with (gain>0,loss=0) is zero lottery.
    return wg*np.power(gain/100.,a) - lam*wl*np.power(loss/100.,a)

def logits(x, theta):
    # x (...,6), theta (...,d), broadcast leading dimensions.
    dv = lottery_value(x[...,:3],theta)-lottery_value(x[...,3:],theta)
    return theta[...,-1]*dv

def categories_sample(family,rng,amounts,probs):
    def g(): return float(rng.choice(amounts))
    def l(): return float(rng.choice(amounts))
    def p(): return float(rng.choice(probs))
    def mixed(): return (g(),l(),p())
    def zero(): return (0.,0.,.5)
    def gsure(): return (g(),0.,1.)
    def lsure(): return (0.,l(),0.)
    def grisk(): return (g(),0.,p())
    def lrisk(): return (0.,l(),p())
    if family=="mixed_mixed": return mixed(),mixed()
    if family=="mixed_zero": return mixed(),zero()
    if family=="mixed_gain_sure": return mixed(),gsure()
    if family=="mixed_loss_sure": return mixed(),lsure()
    if family=="gain_sure": return grisk(),gsure()
    if family=="loss_sure": return lrisk(),lsure()
    if family=="gain_gain": return grisk(),grisk()
    if family=="loss_loss": return lrisk(),lrisk()
    if family=="mixed_gain_risk": return mixed(),grisk()
    return mixed(),lrisk()

def make_candidates(n,rng):
    amounts = np.array([5,10,15,20,30,40,50,60,80,100,125,150,175,200.])
    probs = np.array([.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95])
    arr, labels, seen = [], [], set()
    attempts=0
    while len(arr)<n:
        attempts+=1
        if attempts > n*100: raise RuntimeError("Candidate generation stalled")
        j=int(rng.integers(len(FAMILIES)))
        aa,bb = categories_sample(FAMILIES[j],rng,amounts,probs)
        # Randomize A/B, but canonicalize mirrors and omit exact duplicates.
        a=np.array(aa); b=np.array(bb)
        if np.array_equal(a,b): continue
        sig1=tuple(np.concatenate([a,b]))
        sig2=tuple(np.concatenate([b,a]))
        key=min(sig1,sig2)
        if key in seen: continue
        seen.add(key)
        if rng.random()<.5: a,b=b,a
        arr.append(np.r_[a,b]); labels.append(j)
    return np.asarray(arr,dtype=float),np.asarray(labels,dtype=int)

def fisher_and_effects(xs, refs, step=250):
    """Vectorized central finite differences of logit, verified against analytic
    tau derivative in selftest. coords are prior SD standardized.
    Returns (candidate,reference,parameter,parameter) and squared probability grads.
    """
    sd=refs.std(axis=0,ddof=1)
    if np.any(sd<=0): raise ValueError("degenerate prior scale")
    d=refs.shape[1]; n=len(xs); m=len(refs)
    fisher=np.empty((n,m,d,d),dtype=np.float32)
    effects=np.empty((n,m,d),dtype=np.float32)
    for start in range(0,n,step):
        x=xs[start:start+step,None,:]
        t=refs[None,:,:]
        h=logits(x,t)
        p=expit(h)
        grad=np.empty(h.shape+(d,),dtype=float)
        for j in range(d):
            eps=1e-5*max(sd[j],1e-3)
            plus=refs.copy(); minus=refs.copy()
            plus[:,j]+=eps; minus[:,j]-=eps
            grad[...,j]=(logits(x,plus[None,:,:])-logits(x,minus[None,:,:]))/(2*eps)
            grad[...,j]*=sd[j]
        info=np.maximum(p*(1-p),1e-14)
        fisher[start:start+step]=(info[...,None,None]*grad[..., :,None]*grad[...,None,:]).astype("float32")
        effects[start:start+step]=(np.square(info[...,None]*grad)).astype("float32")
    return fisher,effects,sd

def objective(total, total_e, n_selected, robustness=.2, balance=.18):
    # Inversion is PER reference person (not inverse of population-average F).
    m,d=total.shape[:2]
    posterior=np.linalg.inv(total + np.eye(d)[None,:,:])
    diag=np.diagonal(posterior,axis1=-2,axis2=-1)
    mean=diag.mean(axis=0)
    loss=float(mean.mean()+.75*mean.max())
    worst=np.max(diag,axis=1)
    k=max(1,int(np.ceil(.1*m)))
    loss += robustness*float(np.partition(worst,m-k)[m-k:].mean())
    # Avoid logarithm of tiny gradients for empty / low-info designs.
    if n_selected>0:
        ef=np.maximum(total_e.mean(axis=0)/n_selected,1e-15)
        z=np.log(ef)
        loss += balance*float(np.mean((z-z.mean())**2))
    return loss

def design(xs,fam,info,effect,select_n,rng,shortlist=96,balance=.18):
    n,m,d,_=info.shape
    if n<select_n: raise ValueError("Fewer candidates than selected trials")
    selected=np.zeros(n,dtype=bool)
    total=np.zeros((m,d,d),dtype=np.float64)
    total_e=np.zeros((m,d),dtype=np.float64)
    # Stratified initial coverage: ensure representation across families.
    for t in range(select_n):
        remaining=np.flatnonzero(~selected)
        if t<len(FAMILIES):
            eligible=remaining[fam[remaining]==t]
            if len(eligible): remaining=eligible
        if len(remaining)>shortlist:
            trial=rng.choice(remaining,size=shortlist,replace=False)
        else:
            trial=remaining
        scores=np.array([objective(total+info[k],total_e+effect[k],
                                   t+1,balance=balance) for k in trial])
        best=int(trial[np.argmin(scores)])
        selected[best]=True
        total += info[best]
        total_e += effect[best]
        if (t+1)%50==0:
            print(f"  greedy {t+1}/{select_n}, objective={scores.min():.6g}",flush=True)
    return np.flatnonzero(selected)

def refine(selected, info, effect, rng, n_swaps=600, balance=.18):
    chosen=np.asarray(selected).copy()
    n,m,d,_=info.shape
    flag=np.zeros(n,bool); flag[chosen]=True
    total=info[chosen].sum(axis=0,dtype=np.float64)
    te=effect[chosen].sum(axis=0,dtype=np.float64)
    score=objective(total,te,len(chosen),balance=balance)
    accepted=0
    for it in range(n_swaps):
        pos=int(rng.integers(len(chosen)))
        old=int(chosen[pos])
        new=int(rng.choice(np.flatnonzero(~flag)))
        nt=total-info[old]+info[new]
        ne=te-effect[old]+effect[new]
        cand=objective(nt,ne,len(chosen),balance=balance)
        if cand+1e-12<score:
            flag[old]=False;flag[new]=True;chosen[pos]=new
            total,te,score=nt,ne,cand;accepted+=1
    return chosen,score,accepted

def metrics(xs,refs):
    fi,eff,sd=fisher_and_effects(xs,refs,step=100)
    total=fi.sum(axis=0,dtype=np.float64)
    diag=np.diagonal(total,axis1=-2,axis2=-1).mean(axis=0)
    impact=eff.mean(axis=(0,1))
    cov=np.linalg.inv(total+np.eye(len(sd))[None,:,:])
    conditional=np.diagonal(cov,axis1=-2,axis2=-1).mean(axis=0)
    return dict(fisher_diagonal=diag.tolist(),
                fisher_balance_ratio=float(diag.max()/max(diag.min(),1e-12)),
                probability_gradient_sq=impact.tolist(),
                probability_gradient_ratio=float(impact.max()/max(impact.min(),1e-12)),
                regularized_inverse_fisher_diag=conditional.tolist(),
                median_condition=float(np.median(np.linalg.cond(total+1e-9*np.eye(len(sd))))))

def fit_one(xs,y,dim,rng,starts=4):
    bounds=[tuple(b) for b in BOUNDS[dim]]
    def nll(t):
        h=logits(xs,t)
        return float(np.logaddexp(0,h).sum()-np.dot(y,h))
    seeds=sample_prior(max(starts-1,1),dim,rng)
    initial=np.concatenate([np.mean(seeds,axis=0,keepdims=True),seeds[:starts-1]],axis=0)
    best=None
    for seed in initial:
        res=minimize(nll,np.clip(seed,BOUNDS[dim][:,0]+1e-7,BOUNDS[dim][:,1]-1e-7),
                     method="L-BFGS-B",bounds=bounds,
                     options={"maxiter":450,"ftol":1e-11})
        if best is None or res.fun<best.fun: best=res
    return best.x

def evaluate(xs,dim,n,seed,starts=4):
    rng=np.random.default_rng(seed)
    truth=sample_prior(n,dim,rng)
    h=logits(xs[None,:,:],truth[:,None,:])
    p=expit(h)
    choices=(rng.random(p.shape)<p).astype(np.uint8)
    estimated=np.zeros_like(truth)
    for i in range(n):
        estimated[i]=fit_one(xs,choices[i],dim,rng,starts)
        if (i+1)%50==0: print(f"  oracle {i+1}/{n}",flush=True)
    error=estimated-truth
    r=np.array([np.corrcoef(truth[:,j],estimated[:,j])[0,1] for j in range(dim)])
    rmse=np.sqrt(np.mean(error**2,axis=0))
    sd=truth.std(axis=0,ddof=1)
    r2=1-np.sum(error**2,axis=0)/np.sum((truth-truth.mean(axis=0))**2,axis=0)
    hit=np.mean(((estimated-BOUNDS[dim][:,0])<1e-4)|
                ((BOUNDS[dim][:,1]-estimated)<1e-4),axis=0)
    summary=dict(model=f"CR2_{dim}P",parameter_names=NAMES[dim],subjects=n,
        trials=len(xs),seed=seed,pearson_r=r.tolist(),r2=r2.tolist(),
        rmse=rmse.tolist(),normalized_rmse=(rmse/sd).tolist(),
        bias=error.mean(axis=0).tolist(),boundary_hit=hit.tolist(),
        error_correlation=np.corrcoef(error,rowvar=False).tolist(),
        oracle_r_gate_pass=bool(np.all(np.isfinite(r)) and np.all(r>=.8)),
        mean_choice_entropy=float(np.mean(-p*np.log(np.maximum(p,1e-15))-
                                      (1-p)*np.log(np.maximum(1-p,1e-15)))),
        note="Unregularized bounded individual MLE; truth used ONLY for evaluation.")
    return summary,truth,estimated,choices

def dump(path,data):
    def serial(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(str(type(x)))
    path.write_text(json.dumps(data,indent=2,default=serial),encoding="utf-8")

def selftest():
    rng=np.random.default_rng(44)
    for dim in (4,5):
        th=sample_prior(25,dim,rng)
        assert th.shape==(25,dim)
        assert np.all((th>BOUNDS[dim][:,0])&(th<BOUNDS[dim][:,1]))
        x,fam=make_candidates(40,rng)
        assert np.isfinite(logits(x[None,:,:],th[:,None,:])).all()
        fi,ef,sd=fisher_and_effects(x,th)
        assert np.isfinite(fi).all() and (np.diagonal(fi,axis1=-2,axis2=-1)>=0).all()
        # For tau: d logit / d tau = delta subjective lottery value.
        grad=(logits(x,th[0]+np.eye(dim)[-1]*1e-5)-
              logits(x,th[0]-np.eye(dim)[-1]*1e-5))/(2e-5)
        assert np.allclose(grad,logits(x,th[0])/th[0,-1],atol=1e-7)
        # delta=1 5P reduces exactly to 4P.
        if dim==5:
            t5=th[0].copy();t5[3]=1.
            t4=np.delete(t5,3)
            assert np.allclose(logits(x,t5),logits(x,t4))
    print("CR2 selftest PASS")

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--mode",choices=["all","design","evaluate","selftest"],default="all")
    ap.add_argument("--variants",nargs="+",type=int,choices=[4,5],default=[4,5])
    ap.add_argument("--subjects",type=int,default=300)
    ap.add_argument("--candidate-count",type=int,default=6000)
    ap.add_argument("--reference-count",type=int,default=48)
    ap.add_argument("--test-reference-count",type=int,default=128)
    ap.add_argument("--shortlist",type=int,default=96)
    ap.add_argument("--swaps",type=int,default=600)
    ap.add_argument("--balance-weight",type=float,default=.18)
    ap.add_argument("--starts",type=int,default=4)
    ap.add_argument("--seed",type=int,default=20261023)
    ap.add_argument("--eval-seed",type=int,default=20261101)
    args=ap.parse_args()
    selftest()
    if args.mode=="selftest":return
    if args.candidate_count<500 or args.reference_count<8 or args.subjects<3:
        ap.error("Need >=500 candidates, >=8 design reference thetas, >=3 subjects.")
    OUT.mkdir(parents=True,exist_ok=True)
    for dim in args.variants:
        stem=f"CR2_{dim}P_seed{args.seed}"
        bank=OUT/f"{stem}_fixed500.npz"
        if args.mode in ("all","design"):
            rng=np.random.default_rng(args.seed+dim)
            xs,fam=make_candidates(args.candidate_count,rng)
            ref=sample_prior(args.reference_count,dim,rng)
            info,effect,scale=fisher_and_effects(xs,ref)
            ix=design(xs,fam,info,effect,500,rng,args.shortlist,args.balance_weight)
            ix,score,accepted=refine(ix,info,effect,rng,args.swaps,args.balance_weight)
            # Separate references assess design generalization; do not use true
            # evaluation subjects for stimulus selection.
            testref=sample_prior(args.test_reference_count,dim,
                                 np.random.default_rng(args.seed+10000+dim))
            fixed=xs[ix]
            diagnostics=metrics(fixed,testref)
            np.savez_compressed(bank,X=fixed,family_ids=fam[ix],
                                parameter_names=np.asarray(NAMES[dim]))
            dump(OUT/f"{stem}_design.json",dict(
                model=f"CR2_{dim}P",config=vars(args),objective=score,
                swaps_accepted=accepted,heldout_reference_diagnostics=diagnostics,
                family_counts={v:int((fam[ix]==j).sum()) for j,v in enumerate(FAMILIES)},
                population_priors="alpha Beta(12,3); lambda truncated Gamma(shape=9,scale=.22,[.3,5]); gamma Beta(9,4); tau TN(2.2,.55,[.5,4]); 5P delta truncated LogNormal(0,.35,[.3,3])",
                note="Fixed stimuli shared by ALL subjects; independent reference theta for design vs diagnostics."))
            print(f"{dim}P design written: {bank}; metrics={diagnostics}",flush=True)
        if args.mode in ("all","evaluate"):
            if not bank.exists(): raise FileNotFoundError(f"Run --mode design first: {bank}")
            with np.load(bank) as obj: fixed=obj["X"]
            if fixed.shape!=(500,6):raise ValueError(f"Invalid stimulus bank: {fixed.shape}")
            result,truth,estimate,choices=evaluate(
                fixed,dim,args.subjects,args.eval_seed+dim,args.starts)
            # Evaluate information balance on independent reference parameters.
            eval_refs=sample_prior(args.test_reference_count,dim,
                                   np.random.default_rng(args.seed+20000+dim))
            result["heldout_fisher_diagnostics"]=metrics(fixed,eval_refs)
            result["fisher_balance_gate_pass"]=(result["heldout_fisher_diagnostics"]
                                                ["fisher_balance_ratio"]<4)
            result["joint_gate_pass"]=bool(result["fisher_balance_gate_pass"] and
                                          result["oracle_r_gate_pass"])
            suffix=f"_eval{args.eval_seed}_n{args.subjects}"
            dump(OUT/f"{stem}{suffix}_recovery.json",result)
            np.savez_compressed(OUT/f"{stem}{suffix}_subjects.npz",
                                theta_true=truth,theta_mle=estimate,choices=choices)
            print(json.dumps(result,indent=2),flush=True)
    print("CR2 finished. Existing CR1 files untouched.")

if __name__=="__main__":
    main()
