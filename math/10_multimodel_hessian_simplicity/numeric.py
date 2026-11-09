"""Gap 2 and 3: paired numeric WY -> Hessian estimation.

Exactly SAME supervised TRAIN-only Z->T projection, standardized W, subject
split, fit/eval points, estimator, hyperparameter selection protocol for:
  Gap2: true formula at T_hat -> numeric Y
  Gap3: frozen NP at real Z -> numeric Y
NN is used for Y ONLY; no torch gradient/Hessian.
"""
from __future__ import annotations
import hashlib
import itertools
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.spatial.distance import cdist
from oracle import DIMENSIONS, values, valid_x_mask


def number_tag(x):
    return f"{float(x):g}".replace("-","m").replace(".","p").replace("+","")


def artifact_paths(root,model,beta,zdim,repeat=1,lamb=0,ctxlo=.25,ctxhi=.8):
    condition=(f"beta{number_tag(beta)}_lambda{number_tag(lamb)}_"
               f"ctx{number_tag(ctxlo)}to{number_tag(ctxhi)}_L{zdim}")
    base=Path(root)/"outputs"/"choice"/model
    return (base/"s1_generate"/"s1_generate__dataset.npz",
            base/"s3_train"/condition/f"repeat{repeat:02d}"/"s3_train__checkpoint.pt",
            base/"s4_evaluate"/condition/f"repeat{repeat:02d}"/"s4_evaluate__latent.npz",
            condition)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_latent(root,model,beta,zdim,repeat=1,lamb=0,ctxlo=.25,ctxhi=.8,pca=False):
    ds,ck,latent,condition=artifact_paths(root,model,beta,zdim,repeat,lamb,ctxlo,ctxhi)
    for p in (ds,ck,latent):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}. Run corresponding S1/S3/S5 stage; no automatic retraining.")
    with np.load(latent,allow_pickle=False) as f:
        must={"z","theta_true","train_ids","validation_ids","test_ids",
              "dataset_sha256","checkpoint_sha256","context_mode"}
        missing=must-set(f.files)
        if missing:
            raise ValueError(f"Invalid S5 latent artifact missing {missing}: {latent}")
        z=np.asarray(f["z"],float)
        t=np.asarray(f["theta_true"],float)
        ids={k:np.asarray(f[k+"_ids"],int) for k in ("train","validation","test")}
        if str(f["context_mode"])!="all_observed_trials":
            raise ValueError("Only S5 ALL-trial inferred Z allowed")
        if str(f["dataset_sha256"])!=digest(ds) or str(f["checkpoint_sha256"])!=digest(ck):
            raise ValueError("S5 Z belongs to a different dataset or checkpoint")
    d,p=DIMENSIONS[model]
    if z.shape!=t.shape or z.ndim!=2 or z.shape[1]!=zdim or zdim!=d:
        raise ValueError(f"{model}: require equal and true parameter dimension d={d}, got Z {z.shape}")
    all_ids=np.concatenate(list(ids.values()))
    if not np.array_equal(np.sort(all_ids),np.arange(len(z))):
        raise ValueError("Saved TRAIN/validation/TEST IDs are not a disjoint partition")
    if len(ids["train"])<d+3 or min(len(ids[k]) for k in ("validation","test"))<2:
        raise ValueError("Need enough split subjects for fit/validation/test")
    z_raw=z.copy()
    if pca:
        train_mean=z[ids["train"]].mean(axis=0)
        _,_,vt=np.linalg.svd(z[ids["train"]]-train_mean,full_matrices=False)
        v=(z-train_mean)@vt.T
        # Full d-component PCA only: exact invertible orthogonal re-expression.
    else:
        v=z
        train_mean=np.zeros(zdim);vt=np.eye(zdim)
    coef,*_=np.linalg.lstsq(np.c_[v[ids["train"]],np.ones(len(ids["train"]))],
                           t[ids["train"]],rcond=None)
    A=coef[:d,:].T;b=coef[d,:]
    singular=np.linalg.svd(A,compute_uv=False)
    cond=float(singular[0]/max(singular[-1],1e-30))
    if not np.isfinite(cond) or cond>1e10:
        raise ValueError(f"{model}: TRAIN Z->T regression is singular/ill-conditioned: {cond:g}")
    that=v@A.T+b
    mapping={}
    for k,sel in ids.items():
        residual=that[sel]-t[sel]
        var=np.mean((t[sel]-t[sel].mean(axis=0))**2,axis=0)
        r2=1-np.mean(residual*residual,axis=0)/np.maximum(var,1e-15)
        mapping[k]=dict(r2_per_parameter=r2.tolist(),
                        rmse=float(np.sqrt(np.mean(residual*residual))))
    return dict(z=z_raw,v=v,truth=t,that=that,splits=ids,regression_A=A,
                regression_b=b,mapping=mapping,condition=condition,
                z_condition=cond,pca=bool(pca),pca_train_mean=train_mean,
                pca_components=vt,data=ds,checkpoint=ck,latent=latent,
                dataset_hash=digest(ds),checkpoint_hash=digest(ck),latent_hash=digest(latent))


def read_original_arrays(path,model,expected_t):
    with np.load(path,allow_pickle=False) as f:
        x=np.asarray(f["x"],float)
        counts=np.asarray(f["trial_counts"],int)
        ids=np.asarray(f["subject_index"],int)
        original_t=np.asarray(f["parameters"],float)
    if not np.array_equal(ids,np.repeat(np.arange(len(counts)),counts)):
        raise ValueError("S1 observations are not contiguous by subject")
    if not np.array_equal(original_t,expected_t):
        raise ValueError("S5 theta_true does not match S1 original subject parameter matrix")
    if int(counts.sum())!=len(x):
        raise ValueError("S1 trial_counts inconsistent")
    return x,counts,np.r_[0,np.cumsum(counts)]


def cr3_theta_valid(t):
    return (np.isfinite(t).all(axis=1)&(t[:,0]>.05)&(t[:,0]<2)
            &(t[:,1]>.1)&(t[:,1]<8)
            &(t[:,2]>.05)&(t[:,2]<2)&(t[:,3]>.05)&(t[:,3]<8))


def choose_subject_trials(model,mapping,args,root):
    """TRAIN/val/TEST groups independent; identical fixed X indices for CR3."""
    x,counts,offsets=read_original_arrays(mapping["data"],model,mapping["truth"])
    rng=np.random.default_rng(args.seed+sum(map(ord,model)))
    that=mapping["that"]
    valid_param=cr3_theta_valid(that) if model=="CR3" else np.isfinite(that).all(axis=1)
    st=(np.std(mapping["truth"][mapping["splits"]["train"]],axis=0)
        if model=="CR3" else np.full(DIMENSIONS[model][0],.5))
    if model=="CR3":
        fixed=x[:500].copy()
        bank=Path(root)/"outputs"/"choice"/"CR3"/"s0_design"/"CR3_4P_fixed500.npz"
        if not bank.is_file():
            raise FileNotFoundError(f"Missing official CR3 bank: {bank}")
        with np.load(bank,allow_pickle=False) as f:
            if not np.allclose(f["X"],fixed,atol=1e-12,rtol=0):
                raise ValueError("CR3 official bank does not match S1")
        xm=fixed.mean(axis=0);sx=np.std(fixed,axis=0)
        common=np.flatnonzero(valid_x_mask(fixed,"CR3"))
        if len(common)<args.trials_per_subject:
            raise ValueError("Not enough differentiable CR3 official fixed-X trials")
        common=np.sort(rng.choice(common,args.trials_per_subject,replace=False))
    else:
        xm=np.zeros(x.shape[1]);sx=np.full(x.shape[1],1/np.sqrt(3))
        common=None
    if np.min(st)<=0 or np.min(sx)<=0:
        raise ValueError("Invalid normalization scale")
    parts={};selected_subject_ids={}
    for split,max_subjects in (("train",args.train_subjects),("validation",args.validation_subjects),
                               ("test",args.test_subjects)):
        usable=mapping["splits"][split]
        usable=usable[valid_param[usable]]
        if len(usable)<min(3,max_subjects):
            raise RuntimeError(f"{model}: only {len(usable)} domain-valid {split} subjects")
        sub=np.sort(rng.choice(usable,min(max_subjects,len(usable)),replace=False))
        xx=[];zz=[];tt=[];sid=[]
        for i in sub:
            if model=="CR3":
                if int(counts[i])!=500:
                    raise ValueError("CR3 S1 must contain 500 trials per subject")
                chosen=common
                if not np.allclose(x[offsets[i]+chosen],fixed[chosen],atol=1e-12,rtol=0):
                    raise ValueError("CR3 shared-X property violated")
            else:
                v=np.flatnonzero(valid_x_mask(x[offsets[i]:offsets[i+1]],model))
                if len(v)<args.trials_per_subject:
                    raise ValueError("Not enough trials for requested sample")
                chosen=np.sort(rng.choice(v,args.trials_per_subject,replace=False))
            xx.append(x[offsets[i]+chosen]);zz.append(mapping["z"][i])
            tt.append(mapping["that"][i]);sid.append(int(i))
        xx=np.stack(xx,axis=0);zz=np.stack(zz);tt=np.stack(tt)
        # T-hat and X both define W. Decoded Y uses original Z exactly.
        v=(tt-mapping["truth"][mapping["splits"]["train"]].mean(axis=0))/st
        w=np.concatenate((np.broadcast_to(v[:,None,:],(*xx.shape[:2],len(st))),
                           (xx-xm)/sx),axis=-1)
        parts[split]=dict(w=w.reshape(-1,w.shape[-1]),t=tt,x=xx,z=zz,
                          nsubjects=len(sub),ntrials=xx.shape[1],sid=sid)
        selected_subject_ids[split]=sid
    return parts,dict(selected_subject_ids=selected_subject_ids,
                      st=st.tolist(),sx=sx.tolist(),xm=xm.tolist(),
                      original_parameter_mean=mapping["truth"][mapping["splits"]["train"]].mean(axis=0).tolist(),
                      train_subject_eligibility=int(np.sum(valid_param[mapping["splits"]["train"]])),
                      validation_subject_eligibility=int(np.sum(valid_param[mapping["splits"]["validation"]])),
                      test_subject_eligibility=int(np.sum(valid_param[mapping["splits"]["test"]])),
                      x_support="CR3: same valid subset of official 500" if model=="CR3"
                      else "C1/C2/C3: individual sampled X rows")


def numeric_channels(t,x,z,model,checkpoint,device="cpu",batch_size=32):
    """Generate both formula and decoder Y on paired identical (T_hat,X,Z).

    NOTE: no neural derivative is ever computed. Returns (subject,trial,4).
    """
    ns,nt,p=x.shape
    eta,ptrue=values(np.repeat(t,nt,axis=0),x.reshape(-1,p),model)
    if not np.isfinite(eta).all() or not np.isfinite(ptrue).all():
        raise FloatingPointError("Nonfinite formula Y at T_hat")
    eta=eta.reshape(ns,nt);ptrue=ptrue.reshape(ns,nt)
    # Avoid import of heavy torch modules when testing mathematical inference.
    import sys
    root=Path(__file__).resolve().parents[2]
    np_src=root/"np"/"src"
    if str(np_src) not in sys.path:
        sys.path.insert(0,str(np_src))
    import torch
    from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
    from modlearn_rot.training.continuous import load_checkpoint
    ck=load_checkpoint(checkpoint)
    model_np=DiscreteNeuralProcess(DiscreteNPConfig(**ck["model_config"]))
    model_np.load_state_dict(ck["model_state_dict"],strict=True)
    dev=torch.device("cuda" if device=="auto" and torch.cuda.is_available() else "cpu"
                     if device=="auto" else device)
    model_np=model_np.to(dev).eval()
    if int(ck["model_config"]["latent_dim"])!=z.shape[1] or int(ck["model_config"]["x_dim"])!=p:
        raise ValueError("NP checkpoint latent/X dimension mismatch")
    en=np.empty((ns,nt))
    pn=np.empty_like(en)
    with torch.no_grad():
        for k in range(0,ns,batch_size):
            b=min(ns,k+batch_size)
            logits=model_np.decode(torch.as_tensor(x[k:b],dtype=torch.float32,device=dev),
                                   torch.as_tensor(z[k:b],dtype=torch.float32,device=dev))
            en[k:b]=(logits[...,1]-logits[...,0]).cpu().numpy()
            pn[k:b]=torch.softmax(logits,dim=-1)[...,1].cpu().numpy()
    if np.max(np.abs(expit(en)-pn))>2e-6:
        raise ValueError("NN p inconsistent with sigmoid(two-class log odds)")
    return np.stack((eta,ptrue,en,pn),axis=-1)


def feature_exponents(p,degree):
    """All monomials up to total degree, exact 0.5 Hessian convention."""
    return [()] + [v for k in range(1,degree+1)
                   for v in itertools.combinations_with_replacement(range(p),k)]


def polynomial_design(w,exponents,deriv=()):
    w=np.asarray(w,float)
    p=w.shape[1]
    dcount=np.bincount(deriv,minlength=p)
    arr=np.empty((len(w),len(exponents)),float)
    for j,e in enumerate(exponents):
        powers=np.bincount(e,minlength=p)
        if np.any(powers<dcount):
            arr[:,j]=0
            continue
        fac=1.
        for i,v in enumerate(dcount):
            if v:
                for k in range(v):
                    fac*=powers[i]-k
        z=np.ones(len(w))
        for i,n in enumerate(powers-dcount):
            if n:
                z*=w[:,i]**n
        arr[:,j]=z*fac
    return arr


def sieve_fit(parts,y,degree=3,ridge=1e-5,eval_rows=None):
    """Group-selected polynomial ridge with exact differentiated basis."""
    train=parts["train"]["w"];valid=parts["validation"]["w"]
    expo=feature_exponents(train.shape[1],degree)
    F=polynomial_design(train,expo)
    Y=y["train"]
    # Explicit intercept unconstrained; ridge regularizes all OTHER coefficients.
    normal=F.T@F
    penalty=np.eye(len(expo))*ridge*len(F)
    penalty[0,0]=0
    coef=np.linalg.solve(normal+penalty+1e-10*np.eye(len(expo)),F.T@Y)
    pred=polynomial_design(valid,expo)@coef
    rmse=np.sqrt(np.mean((pred-y["validation"])**2,axis=0))
    w=parts["test"]["w"] if eval_rows is None else np.asarray(eval_rows,float)
    d=next(iter(parts.values()))["t"].shape[1]
    n=len(w);p=parts["train"]["x"].shape[-1]
    hzz=np.empty((n,4,d,d))
    hzx=np.empty((n,4,d,p))
    for i in range(d):
        for j in range(i,d):
            v=polynomial_design(w,expo,(i,j))@coef
            hzz[:,:,i,j]=v;hzz[:,:,j,i]=v
        for j in range(p):
            hzx[:,:,i,j]=polynomial_design(w,expo,(i,j+d))@coef
    return dict(zz=hzz,zx=hzx,validation_rmse=rmse,coefficients=coef,degree=degree,ridge=ridge)


def rbf_fit(parts,y,landmarks=128,bandwidth=2.,ridge=1e-4,seed=0,eval_rows=None):
    """Isotropic RBF ridge on standardized (T_hat,X); analytic RBF Hessian."""
    w=parts["train"]["w"];v=parts["validation"]["w"]
    rng=np.random.default_rng(seed)
    centers=w[np.sort(rng.choice(len(w),min(landmarks,len(w)),replace=False))]
    h=float(bandwidth)
    phi=np.exp(-cdist(w,centers,"sqeuclidean")/(2*h*h))
    phiv=np.exp(-cdist(v,centers,"sqeuclidean")/(2*h*h))
    reg=np.eye(len(centers))*(ridge*len(w))
    coef=np.linalg.solve(phi.T@phi+reg+1e-10*np.eye(len(centers)),phi.T@y["train"])
    rmse=np.sqrt(np.mean((phiv@coef-y["validation"])**2,axis=0))
    test=parts["test"]["w"] if eval_rows is None else np.asarray(eval_rows,float)
    d=parts["train"]["t"].shape[1];p=parts["train"]["x"].shape[-1]
    dz=test[:,None,:]-centers[None,:,:]
    phie=np.exp(-np.sum(dz*dz,axis=-1)/(2*h*h))
    weighted=phie[:,:,None]*coef[None,:,:]  # N,M,4
    prod=np.einsum("nmc,nmi,nmj->ncij",weighted,dz,dz,optimize=True)/h**4
    # Negative diagonal term on the full Hessian:
    strength=np.sum(weighted,axis=1)/h**2
    prod-=strength[:,:,None,None]*np.eye(test.shape[1])[None,None,:,:]
    return dict(zz=prod[:,:,:d,:d],zx=prod[:,:,:d,d:d+p],
                validation_rmse=rmse,degree=0,ridge=ridge,bandwidth=h)


def analytic_test_hessian(parts,model,st,sx):
    """Analytic oracle at *same T_hat/X* TEST rows in standardized W units."""
    t=parts["test"]["t"];x=parts["test"]["x"]
    ns,nt,p=x.shape
    o=__import__("oracle").derivatives(np.repeat(t,nt,axis=0),x.reshape(-1,p),model)
    result={}
    st=np.asarray(st,float);sx=np.asarray(sx,float)
    for ch in ("logit","probability"):
        result[ch]=(
            o["tt_"+ch]*st[None,:,None]*st[None,None,:],
            o["tx_"+ch]*st[None,:,None]*sx[None,None,:])
    return result


def compare_field(reference,fit):
    """Numerical and directional error, splitting ZZ diag/offdiag and ZX."""
    zz,zx=reference
    az,bz=fit
    d=zz.shape[1]
    mask=~np.eye(d,dtype=bool)
    out={}
    for name,t,pred in (("ZZ",zz,az),("ZZ_diag",np.diagonal(zz,axis1=1,axis2=2),
                                      np.diagonal(az,axis1=1,axis2=2)),
                        ("ZZ_offdiag",zz[:,mask],az[:,mask]),("ZX",zx,bz)):
        a=np.asarray(t,float).reshape(-1);b=np.asarray(pred,float).reshape(-1)
        den=np.linalg.norm(a)*np.linalg.norm(b)
        out[name]=dict(mae=float(np.mean(abs(a-b))),rmse=float(np.sqrt(np.mean((a-b)**2))),
                       relative_frobenius=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),
                       cosine=float((a@b)/den) if den>1e-14 else None,
                       reference_rms=float(np.sqrt(np.mean(a*a))),
                       fitted_rms=float(np.sqrt(np.mean(b*b))))
    return out
