"""Fast evaluation of a Gap1-frozen subset of the math10 2nd-order metrics.

Computes the SAME formulas as metrics.compute_metrics, but only requested
metric names. Unknown metric names are explicitly returned as unsupported,
never silently assigned zero. No data-based changes to the metric shortlist.
"""
from __future__ import annotations
import numpy as np
from metrics import (EPS,AGGREGATORS,aggregate,vector_scores,norm2)

AGG_ORDER=tuple(sorted(AGGREGATORS,key=len,reverse=True))


def compute_selected(zz,zx,wanted,*,threshold=.10):
    zz=np.asarray(zz,float);zx=np.asarray(zx,float)
    wanted=set(wanted)
    if zz.ndim!=3 or zx.ndim!=3 or zz.shape[:2]!=zx.shape[:2] or not np.isfinite(zz).all() or not np.isfinite(zx).all():
        raise ValueError("Expected finite (N,d,d), (N,d,p)")
    d=zz.shape[1]
    if zz.shape[2]!=d:
        raise ValueError("Bad ZZ")
    scz=float(np.sqrt(np.mean(zz*zz)));scx=float(np.sqrt(np.mean(zx*zx)))
    outs={}
    aggs={};vectors={}
    local_cache={}
    def matrices(block,agg):
        key=(block,agg)
        if key not in aggs:
            src=zz if block=="zz" else zx
            aggs[key]=aggregate(src)[agg]
        return aggs[key]
    def vectors_for(block,agg,orient):
        key=(block,agg,orient)
        if key not in vectors:
            M=matrices(block,agg)
            if block=="zz":
                rows=[M[i,np.arange(d)!=i] for i in range(d)]
                scl=scz
            else:
                rows=([M[i,:] for i in range(d)] if orient=="row"
                      else [M[:,j] for j in range(M.shape[1])])
                scl=scx
            per_row=[vector_scores(row,scl,relative_threshold=threshold) for row in rows]
            vectors[key]={name:float(np.mean([v[name] for v in per_row]))
                          for name in per_row[0]}
        return vectors[key]
    def local():
        if local_cache:
            return local_cache
        off=zz[:,~np.eye(d,dtype=bool)]
        diag=np.diagonal(zz,axis1=1,axis2=2)
        local_cache.update(
            zz_local_diag_abs_loss=float(np.mean(abs(diag))),
            zz_local_diag_rms_loss=float(np.sqrt(np.mean(diag**2))),
            zz_local_diag_variance_loss=float(np.mean(np.var(diag,axis=0))),
            zz_local_offdiag_abs_loss=float(np.mean(abs(off))),
            zz_local_offdiag_rms_loss=float(np.sqrt(np.mean(off**2))),
            zz_local_offdiag_energy_loss=float(np.mean(np.sum(off**2,axis=1))),
            zz_local_offdiag_fraction_loss=float(np.sum(off**2)/max(float(np.sum(zz**2)),EPS)),
            zx_local_abs_loss=float(np.mean(abs(zx))),
            zx_local_rms_loss=float(np.sqrt(np.mean(zx**2))),
            zx_local_variance_loss=float(np.mean(np.var(zx,axis=0))),
            zz_local_frobenius_sq_diagnostic=float(np.mean(np.sum(zz**2,axis=(1,2)))),
            zx_local_frobenius_sq_diagnostic=float(np.mean(np.sum(zx**2,axis=(1,2)))),
            zz_mean_cancellation_diagnostic=float(norm2(zz.mean(axis=0))/max(
                float(np.sqrt(np.mean(np.sum(zz**2,axis=(1,2))))),EPS)),
            zx_mean_cancellation_diagnostic=float(norm2(zx.mean(axis=0))/max(
                float(np.sqrt(np.mean(np.sum(zx**2,axis=(1,2))))),EPS)))
        return local_cache
    def localfirst(block):
        key=("localfirst",block)
        if key not in local_cache:
            n=len(zz)
            idx=np.linspace(0,n-1,min(n,24),dtype=int)
            arr=(zx if block=="zx" else np.stack(
                [zz[:,i,np.arange(d)!=i] for i in range(d)],axis=1))
            flat=arr[idx].reshape(-1,arr.shape[-1])
            vv={}
            for k in (1,2,3):
                if k>=arr.shape[-1]:
                    continue
                residual=np.sort(np.abs(flat),axis=1)[:,::-1][:,k:]
                sq=np.sum(residual**2,axis=1)
                vv[f"{block}_localfirst_tail2_k{k}_loss"]=float(np.mean(np.sqrt(sq)))
                vv[f"{block}_localfirst_tail2sq_k{k}_loss"]=float(np.mean(sq))
            scl=scx if block=="zx" else scz
            vv[f"{block}_localfirst_softl0_loss"]=float(np.mean([
                vector_scores(v,scl,relative_threshold=threshold)["softl0_rational_loss"]
                for v in flat]))
            local_cache[key]=vv
        return local_cache[key]
    def gram(block):
        key=("gram",block)
        if key not in local_cache:
            field=zz if block=="zz" else zx
            G=np.einsum("nab,ncb->ac",field,field,optimize=True)/len(field)
            off=G-np.diag(np.diag(G))
            ev=np.linalg.eigvalsh(G)
            prob=ev/max(float(ev.sum()),EPS)
            entropy=float(-np.sum(prob*np.log(np.maximum(prob,EPS)))) if ev.sum()>EPS else 0.
            local_cache[key]={
                f"{block}_gram_offdiag_l1_loss":float(np.sum(abs(off))),
                f"{block}_gram_offdiag_l2sq_loss":float(np.sum(off*off)),
                f"{block}_gram_spectral_entropy_diagnostic":entropy,
                f"{block}_gram_effective_rank_diagnostic":float(np.exp(entropy)),
                f"{block}_gram_trace_diagnostic":float(np.trace(G))}
        return local_cache[key]
    def spectral():
        key=("spectral","zz")
        if key not in local_cache:
            eig,eigvec=np.linalg.eigh(zz)
            M=np.einsum("nij,nj,nkj->ik",eigvec,np.abs(eig),eigvec,optimize=True)/len(zz)
            off=M-np.diag(np.diag(M))
            local_cache[key]=dict(
                zz_spectral_absolute_offdiag_l1_loss=float(np.sum(abs(off))),
                zz_spectral_absolute_offdiag_l2sq_loss=float(np.sum(off*off)))
        return local_cache[key]
    def cov():
        key=("cov","zz")
        if key not in local_cache:
            ids=np.triu_indices(d)
            M=np.cov(zz[:,ids[0],ids[1]],rowvar=False,bias=True)
            local_cache[key]=dict(
                zz_hessian_element_covariance_offdiag_l1_diagnostic=float(
                    np.sum(abs(M-np.diag(np.diag(M))))),
                zz_hessian_element_covariance_trace_diagnostic=float(np.trace(M)))
        return local_cache[key]
    for name in wanted:
        if name in local():
            outs[name]=local()[name]
            continue
        if "_localfirst_" in name:
            b=name.split("_",1)[0]
            if b in ("zz","zx") and name in localfirst(b):
                outs[name]=localfirst(b)[name]
            continue
        if "_gram_" in name:
            b=name.split("_",1)[0]
            if b in ("zz","zx") and name in gram(b):
                outs[name]=gram(b)[name]
            continue
        if "spectral_absolute" in name:
            outs.update({name:spectral()[name]}) if name in spectral() else None
            continue
        if name.startswith("zz_hessian_element_covariance"):
            outs.update({name:cov()[name]}) if name in cov() else None
            continue
        block,sep,rest=name.partition("_")
        if block not in ("zz","zx"):
            continue
        for agg in AGG_ORDER:
            if not rest.startswith(agg+"_"):
                continue
            sub=rest[len(agg)+1:]
            M=matrices(block,agg)
            if block=="zz":
                a=abs(M)
                diagonal=np.diag(a)
                off=np.array([a[i,j] for i in range(d) for j in range(i+1,d)])
                nums={
                    "diag_l1_loss":float(np.sum(diagonal)),
                    "diag_l2_loss":norm2(diagonal),
                    "offdiag_l1_loss":float(np.sum(off)),
                    "offdiag_l2_loss":norm2(off),
                    "offdiag_l2sq_loss":float(off@off),
                    "offdiag_over_total_loss":float(2*off@off/max(float(np.sum(M*M)),EPS)),
                    "offdiag_over_diag_loss":float(2*off@off/max(float(diagonal@diagonal),EPS)),
                }
                if sub in nums:
                    outs[name]=nums[sub]
                elif sub.startswith("row_off_"):
                    short=sub[len("row_off_"):]
                    v=vectors_for("zz",agg,"row")
                    if short in v:
                        outs[name]=v[short]
            else:
                if sub=="total_l1_loss":
                    outs[name]=float(np.abs(M).sum())
                elif sub=="total_l2_loss":
                    outs[name]=norm2(M)
                elif sub.startswith("row_") or sub.startswith("column_"):
                    orient="row" if sub.startswith("row_") else "column"
                    short=sub[len(orient)+1:]
                    v=vectors_for("zx",agg,orient)
                    if short in v:
                        outs[name]=v[short]
            break
    if not all(np.isfinite(x) for x in outs.values()):
        raise ArithmeticError("Nonfinite selected score")
    return outs
