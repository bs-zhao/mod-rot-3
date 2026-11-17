"""Supplementary B/C/D/E metrics and genuine full-W field assembly.

Extends the immutable A-F core registry without changing historical modules.
Computes expensive full-W spectral statistics on a clearly labeled common
deterministic subset; whole-population block energies remain separately exact.
"""
from __future__ import annotations
import numpy as np
from .metrics import EPS, _sparsity, _rank_stats, _entropy_neff, block_metrics


def _joint_diag(operators):
    """Return orthogonal JD residual and identity residual; no true parameters."""
    if len(operators)<2:
        return {"joint_diag.status":"not_applicable"}
    d=operators[0].shape[0]
    mats=np.asarray(operators,dtype=float)
    mats=mats/(np.linalg.norm(mats,axis=(1,2),keepdims=True)+EPS)
    def score(Q):
        z=Q@mats@Q.T
        return float(np.sum(z*z)-np.sum(np.diagonal(z,axis1=-2,axis2=-1)**2))
    q0=np.eye(d)
    try:
        from scipy.linalg import expm
        from scipy.optimize import minimize
        pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
        def theta_to_Q(theta):
            skew=np.zeros((d,d))
            for t,(i,j) in zip(theta,pairs):
                skew[i,j]=t;skew[j,i]=-t
            return expm(skew)
        res=minimize(lambda v:score(theta_to_Q(v)),np.zeros(len(pairs)),
                     method="BFGS",options={"maxiter":200})
        return {"joint_diag.identity_offdiag":score(q0),
                "joint_diag.optimized_offdiag":score(theta_to_Q(res.x)),
                "joint_diag.success":float(res.success)}
    except ImportError:
        return {"joint_diag.status":"scipy_unavailable"}


def _sample(field,n=16000):
    n0,n1=field.shape[:2]
    if n0*n1<=n:return np.asarray(field,dtype=float)
    take=np.unique(np.linspace(0,n0*n1-1,n,dtype=int))
    a=np.asarray(field).reshape(n0*n1,*field.shape[2:])[take]
    return a[None,:,:]  # metrics sample deliberately loses subject decomposition


def enrich(field,block,vals):
    """Append additional indicators without silently interpreting a rectangle as symmetric."""
    a=_sample(field)
    x=a.reshape(-1,*a.shape[2:])
    if len(block)==1:
        windows=np.array_split(np.arange(a.shape[1]),min(4,a.shape[1]))
        mats=[np.einsum("ni,nj->ij",x[w],x[w])/max(len(w),1) for w in windows if len(w)]
        vals.update({f"B13.{k}":v for k,v in _joint_diag(mats).items()})
        vals["B09.gram_condition_number"]=vals.get("B07.gram_spectrum.condition",float("nan"))
        return vals
    if len(block)==2:
        energy=np.mean(x*x,axis=0)
        vals["C15.row_entropy"]=np.array([_entropy_neff(row) for row in energy])
        vals["C15.col_entropy"]=np.array([_entropy_neff(col) for col in energy.T])
        vals["C16.stable_rank"]=float(np.sum(np.linalg.svd(np.sqrt(energy),compute_uv=False)**2)/
               (np.linalg.svd(np.sqrt(energy),compute_uv=False)[0]**2+EPS))
        if block in ("UU","XX","WW"):
            eig=np.linalg.eigvalsh(x)
            vals["C06.mean_eigenvalues"]=np.mean(eig,axis=0)
            vals["C09.positive_eigenvalue_energy"]=float(np.mean(np.maximum(eig,0)**2))
            vals["C09.negative_eigenvalue_energy"]=float(np.mean(np.minimum(eig,0)**2))
            mats=[np.mean(xx,axis=0) for xx in np.array_split(x,min(4,len(x))) if len(xx)]
            vals.update({f"C19.{k}":v for k,v in _joint_diag(mats).items()})
            directions=np.eye(x.shape[-1])
            vals["C20.directional_basis_curvature_rms"]=np.array(
                [np.sqrt(np.mean(np.einsum("i,nij,j->n",v,x,v)**2)) for v in directions])
        return vals
    if len(block)==3:
        energy=np.mean(x*x,axis=0)
        mode_ranks=[]
        for axis in range(3):
            m=np.moveaxis(np.sqrt(energy),axis,0).reshape(energy.shape[axis],-1)
            singular=np.linalg.svd(m,compute_uv=False)
            mode_ranks.append(_entropy_neff(singular))
        vals["D08.multilinear_effective_ranks"]=np.asarray(mode_ranks)
        if block in ("UUU","XXX","WWW"):
            # Symmetric eigen-direction contracts; explicit positional basis.
            n=x.shape[-1]
            diag=np.stack([x[:,i,i,i] for i in range(n)],axis=1)
            vals["D10.same_axis_directional_rms"]=np.sqrt(np.mean(diag**2,axis=0))
        vals["D13.nonzero_support_1pct"]=float(
            np.sum(energy>(.01*energy.max(initial=0))))
        # Tensorly is optional; CP rank approximation must never silently
        # be replaced with a heuristic alias.
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac
            tl.set_backend("numpy")
            tensor=np.sqrt(energy)
            errors=[]
            for rank in range(1,min(4,min(tensor.shape))+1):
                factors=parafac(tensor,rank=rank,init="svd",n_iter_max=100,
                                tol=1e-6,random_state=12)
                recon=tl.cp_to_tensor(factors)
                errors.append(float(np.linalg.norm(tensor-recon)/(np.linalg.norm(tensor)+EPS)))
            vals["D09.cp_reconstruction_error_ranks1to4"]=np.asarray(errors)
        except ImportError:
            vals["D09.cp_status"]="tensorly_unavailable"
        return vals
    return vals


def fullw_fields(fields,limit_subjects=12,limit_stimuli=32):
    """Reconstruct symmetric W tensors from canonical blocks on SAME points.

    Returns available full-W fields, not concatenated scalar energies.
    The sampling limit is recorded so reduced sample size is explicit.
    """
    if "U" not in fields or "X" not in fields: return {}
    shape=fields["U"].shape
    S=min(shape[0],limit_subjects);K=min(shape[1],limit_stimuli)
    def get(block):return np.asarray(fields[block][:S,:K],dtype=float)
    d=get("U").shape[-1];p=get("X").shape[-1]
    out={"W":np.concatenate([get("U"),get("X")],axis=-1)}
    if all(k in fields for k in ("UU","UX","XX")):
        H=np.zeros((S,K,d+p,d+p))
        H[:,:,:d,:d]=get("UU")
        H[:,:,:d,d:]=get("UX")
        H[:,:,d:,:d]=get("UX").transpose(0,1,3,2)
        H[:,:,d:,d:]=get("XX")
        out["WW"]=H
    if all(k in fields for k in ("UUU","UUX","UXX","XXX")):
        T=np.zeros((S,K,d+p,d+p,d+p))
        uuu=get("UUU"); uux=get("UUX");uxx=get("UXX");xxx=get("XXX")
        T[:,:,:d,:d,:d]=uuu
        T[:,:,:d,:d,d:]=uux
        T[:,:,:d,d:,:d]=uux.transpose(0,1,2,4,3)
        T[:,:,d:,:d,:d]=uux.transpose(0,1,4,2,3)
        T[:,:,:d,d:,d:]=uxx
        T[:,:,d:,:d,d:]=uxx.transpose(0,1,3,2,4)
        T[:,:,d:,d:,:d]=uxx.transpose(0,1,3,4,2)
        T[:,:,d:,d:,d:]=xxx
        out["WWW"]=T
    return out


def summarize_fullw(fields):
    arrays=fullw_fields(fields)
    out={}
    for key,a in arrays.items():
        results=block_metrics(a,key)
        results=enrich(a,key,results)
        for name,value in results.items():
            if isinstance(value,str):continue
            out[f"{key}.{name}"]=value
    return out
