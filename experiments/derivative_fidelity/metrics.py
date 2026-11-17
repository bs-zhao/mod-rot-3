"""Shared A-F metric registry: same functions for analytic and every estimator.

All input fields are (subject, stimulus, index...). Structural metrics are
functions of field distributions; scalar, vector and matrix outputs retain
their original shape. 'Full-W' energies use symmetric index multiplicities.
"""
from __future__ import annotations
import math
from itertools import combinations
import numpy as np
from .derivatives import BLOCKS

EPS = 1e-12


def _flat(a):
    a=np.asarray(a,dtype=np.float64)
    return a.reshape(-1,*a.shape[2:])


def _entropy_neff(e):
    v=np.maximum(np.asarray(e,dtype=float).ravel(),0)
    if v.sum()<=EPS: return 0.0
    q=v/v.sum(); q=q[q>0]
    return float(np.exp(-(q*np.log(q)).sum()))


def _gini(e):
    v=np.sort(np.abs(np.asarray(e,dtype=float).ravel()))
    if len(v)==0 or v.sum()<=EPS: return 0.0
    return float(np.dot(2*np.arange(1,len(v)+1)-len(v)-1,v)/(len(v)*v.sum()))


def _sparsity(e, prefix):
    e=np.maximum(np.asarray(e,dtype=float),0)
    f=e.ravel(); total=f.sum(); q=f/(total+EPS)
    amplitude=np.sqrt(f)
    vals={
        f"{prefix}.neff":_entropy_neff(f),
        f"{prefix}.gini":_gini(f),
        f"{prefix}.l1l2":float(amplitude.sum()/(np.linalg.norm(amplitude)+EPS)),
        f"{prefix}.quartimax":float(-(q*q).sum()),
        f"{prefix}.max_share":float(q.max(initial=0)),
        f"{prefix}.tail1":float(1-np.sort(q)[-1:].sum()),
        f"{prefix}.tail2":float(1-np.sort(q)[-2:].sum()),
        f"{prefix}.tail4":float(1-np.sort(q)[-4:].sum()),
        f"{prefix}.soft_l0":float(np.sum(f/(f+0.1*max(f.mean(),EPS)+EPS))),
        f"{prefix}.support_90":float(np.searchsorted(np.cumsum(np.sort(q)[::-1]),.9)+1) if total>EPS else 0.0,
    }
    if e.ndim==2:
        row=e.sum(1); col=e.sum(0)
        vals[f"{prefix}.row_neff"]=_entropy_neff(row)
        vals[f"{prefix}.col_neff"]=_entropy_neff(col)
        vals[f"{prefix}.row_distribution"]=row/(total+EPS)
        vals[f"{prefix}.col_distribution"]=col/(total+EPS)
    return vals


def _rank_stats(mat,prefix,symmetric=False):
    mat=np.asarray(mat,dtype=np.float64)
    if mat.ndim!=2: return {}
    if symmetric:
        singular=np.abs(np.linalg.eigvalsh((mat+mat.T)/2))
    else:
        singular=np.linalg.svd(mat,compute_uv=False)
    singular=np.sort(singular)[::-1]
    s2=singular**2
    return {
        f"{prefix}.singular_values":singular,
        f"{prefix}.effective_rank":_entropy_neff(singular),
        f"{prefix}.energy_rank":_entropy_neff(s2),
        f"{prefix}.stable_rank":float(s2.sum()/(s2.max(initial=0)+EPS)),
        f"{prefix}.condition":float(singular.max(initial=0)/(singular[singular>1e-9].min() if np.any(singular>1e-9) else EPS)),
        f"{prefix}.spectral_norm":float(singular.max(initial=0)),
        f"{prefix}.nuclear_norm":float(singular.sum()),
    }


def elementwise(field,block,zero_tol=1e-4):
    a=np.asarray(field,dtype=np.float64)
    x=_flat(a)
    m=np.mean(x,axis=0)
    rms=np.sqrt(np.mean(x*x,axis=0))
    var=np.var(x,axis=0)
    centered=x-m
    sd=np.sqrt(var)
    z=centered/(sd+EPS)
    return {
        "A01.mean":m, "A02.abs_mean":np.mean(np.abs(x),axis=0),
        "A03.rms":rms, "A04.variance":var, "A05.std":sd,
        "A06.q50":np.quantile(np.abs(x),.50,axis=0),
        "A06.q90":np.quantile(np.abs(x),.90,axis=0),
        "A06.q95":np.quantile(np.abs(x),.95,axis=0),
        "A06.q99":np.quantile(np.abs(x),.99,axis=0),
        "A07.max_abs":np.max(np.abs(x),axis=0),
        "A08.zero_fraction":np.mean(np.abs(x)<=zero_tol,axis=0),
        "A09.pos_fraction":np.mean(x>zero_tol,axis=0),
        "A09.neg_fraction":np.mean(x< -zero_tol,axis=0),
        "A10.sign_transition":np.mean(np.sign(a[:,1:])!=np.sign(a[:,:-1]),axis=(0,1)) if a.shape[1]>1 else np.zeros_like(m),
        "A11.skew":np.mean(z**3,axis=0),
        "A11.kurtosis":np.mean(z**4,axis=0)-3,
        "A12.between_subject":np.var(np.mean(a,axis=1),axis=0),
        "A12.within_subject":np.mean(np.var(a,axis=1),axis=0),
    }


def gradient(field,block):
    a=np.asarray(field,dtype=np.float64)
    x=_flat(a)
    mean=np.mean(x,axis=0)
    gram=x.T@x/len(x)
    cov=np.cov(x,rowvar=False,bias=True) if x.shape[1]>1 else np.array([[np.var(x)]])
    with np.errstate(divide="ignore",invalid="ignore"):
        corr=np.corrcoef(x,rowvar=False) if x.shape[1]>1 else np.array([[1.]])
    vals={
        "B01.gradient_norm":float(np.mean(np.linalg.norm(x,axis=1))),
        "B02.squared_energy":float(np.mean((x*x).sum(1))),
        "B03.covariance":cov,
        "B04.gram":gram,
        "B05.mean_unit_gradient":np.mean(x/(np.linalg.norm(x,axis=1,keepdims=True)+EPS),axis=0),
        "B06.correlation":np.nan_to_num(corr),
        "B10.axis_energy":np.mean(x*x,axis=0),
        "B11.offdiag_share":float((np.sum(gram*gram)-np.sum(np.diag(gram)**2))/(np.sum(gram*gram)+EPS)),
    }
    vals.update(_rank_stats(gram,"B07.gram_spectrum",True))
    vals.update(_rank_stats(cov,"B08.cov_spectrum",True))
    vals.update(_sparsity(np.mean(x*x,axis=0),"B10.axis_sparsity"))
    # Subject-mean operator versus trial-first operator.
    means=np.mean(a,axis=1)
    vals["B12.subject_mean_gram"]=means.T@means/len(means)
    vals["B12.trial_first_gram"]=gram
    return vals


def hessian(field,block):
    a=np.asarray(field,dtype=np.float64)
    x=_flat(a)
    mean=x.mean(0); energy=np.mean(x*x,axis=0)
    d0,d1=energy.shape; square=d0==d1 and block in ("UU","XX","WW")
    vals={
        "C01.mean_matrix":mean,
        "C02.rms_matrix":np.sqrt(energy),
        "C03.frob_energy":float(energy.sum()),
        "C04.mean_spectral_norm":float(np.mean(np.linalg.svd(x,compute_uv=False)[:,0])),
        "C05.mean_nuclear_norm":float(np.mean(np.linalg.svd(x,compute_uv=False).sum(1))),
        "C16.rms_spectrum":np.linalg.svd(np.sqrt(energy),compute_uv=False),
        "C17.matrix_variance":np.var(x,axis=0),
        "C18.between_subject_energy":np.mean(np.mean(a,axis=1)**2,axis=0),
    }
    vals.update(_sparsity(energy,"C13_C14.energy_sparsity"))
    vals.update(_rank_stats(np.sqrt(energy),"C16.energy_rank"))
    vals.update(_rank_stats(mean,"C16.mean_rank",square))
    if square:
        diag=np.diag(energy).sum()
        vals.update({
            "C07.mean_trace":float(np.mean(np.trace(x,axis1=-2,axis2=-1))),
            "C08.abs_trace":float(np.mean(np.abs(np.trace(x,axis1=-2,axis2=-1)))),
            "C10.diag_energy":float(diag),
            "C11.offdiag_energy":float(energy.sum()-diag),
            "C12.diag_share":float(diag/(energy.sum()+EPS)),
            "C20.mean_eig_pos_energy":float(np.mean(np.maximum(np.linalg.eigvalsh(x),0)**2)),
            "C20.mean_eig_neg_energy":float(np.mean(np.minimum(np.linalg.eigvalsh(x),0)**2)),
        })
    return vals


def third_order(field,block):
    a=np.asarray(field,dtype=np.float64)
    x=_flat(a)
    energy=np.mean(x*x,axis=0)
    vals={
        "D01.total_energy":float(energy.sum()),
        "D02.rms_tensor":np.sqrt(energy),
        "D03.signed_mean":np.mean(x,axis=0),
        "D06.energy_entropy":_entropy_neff(energy),
        "D09.third_variation":np.var(x,axis=0),
        "D10.directional_basis_mean":np.mean(x,axis=0),
        "D12.subject_mean_energy":np.mean(np.mean(a,axis=1)**2,axis=0),
    }
    vals.update(_sparsity(energy,"D06.sparsity"))
    for axis in range(3):
        unfolded=np.moveaxis(np.sqrt(energy),axis,0).reshape(energy.shape[axis],-1)
        vals.update(_rank_stats(unfolded,f"D07.mode{axis}_unfold"))
    # Repeated versus distinct *same-group* coordinate indices.
    if len(set(block))==1:
        sizes=energy.shape
        indices=list(np.ndindex(*sizes))
        repeat=[energy[ix] for ix in indices if len(set(ix))<3]
        distinct=[energy[ix] for ix in indices if len(set(ix))==3]
        vals["D04.repeated_index_energy"]=float(sum(repeat))
        vals["D05.distinct_index_energy"]=float(sum(distinct))
    return vals


def block_metrics(field,block):
    vals=elementwise(field,block)
    if len(block)==1: vals.update(gradient(field,block))
    elif len(block)==2: vals.update(hessian(field,block))
    else: vals.update(third_order(field,block))
    return vals


def assemble_energy(fields):
    """Combine energy summary arrays with exact symmetry multiplicities."""
    vals={}
    for order,specs in ((1,["U","X"]),(2,["UU","UX","XX"]),(3,["UUU","UUX","UXX","XXX"])):
        parts=[]; repeats=[]
        for key in specs:
            if key not in fields: continue
            tensor=np.asarray(fields[key],dtype=np.float64)
            en=np.mean(tensor*tensor,axis=(0,1))
            multiplicity={1:{"U":1,"X":1},2:{"UU":1,"UX":2,"XX":1},3:{"UUU":1,"UUX":3,"UXX":3,"XXX":1}}[order][key]
            parts.append(en.ravel()); repeats.append(np.repeat(multiplicity,en.size))
            vals[f"E.order{order}.{key}_energy"]=float(en.sum())
        if parts:
            combined=np.concatenate(parts); weights=np.concatenate(repeats)
            vals[f"E.order{order}.fullW_energy"]=float(np.dot(combined,weights))
            vals[f"E.order{order}.fullW_neff"]=_entropy_neff(np.repeat(combined,weights.astype(int)))
            vals[f"E.order{order}.fullW_gini"]=_gini(np.repeat(combined,weights.astype(int)))
            ukeys=[k for k in specs if "U" in k]
            vals[f"E.order{order}.U_related_energy"]=float(sum(vals.get(f"E.order{order}.{k}_energy",0)*
                   ({2:{"UX":2},3:{"UUX":3,"UXX":3}}.get(order,{}).get(k,1)) for k in ukeys))
    if "UU" in fields and "UX" in fields:
        uu=vals["E.order2.UU_energy"]; ux=vals["E.order2.UX_energy"]
        vals["E02.UU_to_UX_ratio"]=float(uu/(ux+EPS))
    if "UX" in fields and "XX" in fields:
        vals["E03.UX_to_XX_ratio"]=float(vals["E.order2.UX_energy"]/(vals["E.order2.XX_energy"]+EPS))
    if "U" in fields and "UX" in fields:
        vals["E01.U_plus_UX_energy"]=float(vals["E.order1.U_energy"]+vals["E.order2.UX_energy"])
    if all(f"E.order{k}.fullW_energy" in vals for k in (1,2,3)):
        vals["E10.order_energy_profile"]=np.array([vals[f"E.order{k}.fullW_energy"] for k in (1,2,3)])
        vals["E11.relative_second_to_first"]=float(vals["E.order2.fullW_energy"]/(vals["E.order1.fullW_energy"]+EPS))
        vals["E12.relative_third_to_second"]=float(vals["E.order3.fullW_energy"]/(vals["E.order2.fullW_energy"]+EPS))
    return vals


def metric_accuracy(est,ref,zero_tol=1e-10):
    a=np.asarray(est,dtype=float).ravel()
    b=np.asarray(ref,dtype=float).ravel()
    if a.shape!=b.shape: raise ValueError("metric shape mismatch")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return {"status":"nonfinite"}
    diff=a-b
    rmse=float(np.sqrt(np.mean(diff**2)))
    denom=np.sqrt(np.mean(b**2))
    return {"status":"ok","mae":float(np.mean(np.abs(diff))),
            "rmse":rmse,"relative_l2":float(np.linalg.norm(diff)/(np.linalg.norm(b)+EPS)) if denom>zero_tol else None,
            "reference_rms":float(denom),"estimate_rms":float(np.sqrt(np.mean(a*a)))}


def field_accuracy(est,ref,zero_tol=1e-10):
    a=np.asarray(est,dtype=float).ravel()
    b=np.asarray(ref,dtype=float).ravel()
    if a.size!=b.size: raise ValueError("field shape mismatch")
    valid=np.isfinite(a)&np.isfinite(b)
    if not valid.all(): return {"status":"nonfinite","valid_fraction":float(valid.mean())}
    result=metric_accuracy(a,b,zero_tol)
    result["bias"]=float(np.mean(a-b))
    result["zero_false_energy"]=float(np.mean(a[np.abs(b)<zero_tol]**2)) if np.any(np.abs(b)<zero_tol) else None
    if np.std(a)>zero_tol and np.std(b)>zero_tol:
        result["pearson"]=float(np.corrcoef(a,b)[0,1])
        result["r2_uncalibrated"]=float(1-np.sum((a-b)**2)/(np.sum((b-b.mean())**2)+EPS))
        result["slope_calibrated"]=float(np.cov(b,a,bias=True)[0,1]/np.var(b))
    else:
        result["pearson"]=None; result["r2_uncalibrated"]=None; result["slope_calibrated"]=None
    return result
