"""Catalogued LOCAL 2nd/3rd/4th-order Hessian complexity measures.

All input fields are in (U,X) coordinates; no generating formula is
required by this module. Order is (subjects, trials, tensor axes).
Metric columns ending in '_loss' should be minimized; '_diagnostic'
columns describe something that is not necessarily a simplicity loss.
Every metric is computed on exactly the same selected Cartesian grid
of U subjects and fixed X stimuli at all rotation angles.

The local means of the true/fitted Hessians must NEVER be mistaken for
a complete description of their dependence on U or X.
"""
from __future__ import annotations
from itertools import combinations_with_replacement
import numpy as np

# Imported only from 06; no analytic CR3 or NN details here.
from sparsity import rich_sparsity,vector_losses,PRIMARY

EPS=1e-15
STATS=("mean_abs","rms","q90_abs","q99_abs","max_abs","std",
       "mean_signed","sign_flip_fraction")


def stats(values,name,*,loss=True):
    """Summarize a whole derivative family, not only a mean Hessian."""
    v=np.asarray(values,dtype=float).ravel()
    v=v[np.isfinite(v)]
    if not len(v):raise ValueError("Derivative family has no finite values")
    a=np.abs(v)
    p=np.mean(v>0)
    n=np.mean(v<0)
    prefix=name+"_"
    suffix="_loss" if loss else "_diagnostic"
    return {
        prefix+"mean_abs"+suffix:float(a.mean()),
        prefix+"rms"+suffix:float(np.sqrt(np.mean(v*v))),
        prefix+"q90_abs"+suffix:float(np.quantile(a,.90)),
        prefix+"q99_abs"+suffix:float(np.quantile(a,.99)),
        prefix+"max_abs"+suffix:float(np.max(a)),
        prefix+"std"+suffix:float(np.std(v)),
        prefix+"mean_signed_diagnostic":float(v.mean()),
        prefix+"sign_flip_fraction_diagnostic":float(2*min(p,n)),
    }


def _one_field(fields,key):
    arr=np.asarray(fields[key],dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"Nonfinite local {key} tensor; choose interior X")
    return arr


def _scale(fields,us,xs):
    """Standardize coordinate derivative units using input SD products."""
    us=np.asarray(us,dtype=float).reshape(4)
    xs=np.asarray(xs,dtype=float).reshape(6)
    out=dict(value=_one_field(fields,"value"))
    out["zz"]=_one_field(fields,"zz")*us[None,None,:,None]*us[None,None,None,:]
    out["zx"]=_one_field(fields,"zx")*us[None,None,:,None]*xs[None,None,None,:]
    out["zzz"]=_one_field(fields,"zzz")*(
        us[None,None,:,None,None]*us[None,None,None,:,None]*
        us[None,None,None,None,:])
    out["zzx"]=_one_field(fields,"zzx")*(
        us[None,None,:,None,None]*us[None,None,None,:,None]*
        xs[None,None,None,None,:])
    out["zxx"]=_one_field(fields,"zxx")*(
        us[None,None,:,None,None]*xs[None,None,None,:,None]*
        xs[None,None,None,None,:])
    if "zzzz" in fields:
        u4=us[:,None,None,None]*us[None,:,None,None]*(
            us[None,None,:,None]*us[None,None,None,:])
        out["zzzz"]=_one_field(fields,"zzzz")*u4[None,None,:,:,:,:]
        out["zzxx"]=_one_field(fields,"zzxx")*(
            us[None,None,:,None,None]*us[None,None,None,:,None]*
            xs[None,None,None,None,:]**2)
        out["zxxx"]=_one_field(fields,"zxxx")*(
            us[None,None,:,None]*xs[None,None,None,:]**3)
    return out


def _anova2(a,prefix):
    """Empirical 2-way ANOVA variation of a tensor over subject × X bank."""
    # All samples use Cartesian subject×stimulus grid, NOT an ANOVA
    # of independent original Z coordinates.
    z=np.asarray(a,dtype=float)
    grand=z.mean(axis=(0,1))
    effect_u=z.mean(axis=1)-grand
    effect_x=z.mean(axis=0)-grand
    residual=z-grand-effect_u[:,None,...]-effect_x[None,...]
    var_total=float(np.mean((z-grand)**2))
    return {
        prefix+"_between_subject_variance_loss":float(np.mean(effect_u**2)),
        prefix+"_between_X_variance_loss":float(np.mean(effect_x**2)),
        prefix+"_subject_X_interaction_variance_loss":float(np.mean(residual**2)),
        prefix+"_total_local_variance_loss":var_total,
        prefix+"_subject_fraction_diagnostic":float(np.mean(effect_u**2)/max(var_total,EPS)),
        prefix+"_X_fraction_diagnostic":float(np.mean(effect_x**2)/max(var_total,EPS)),
    }


def _local_row_metrics(zx):
    """ZX row-wise support on each (subject,X) point; fixed deterministic subsample."""
    n,m=zx.shape[:2]
    idx_n=np.linspace(0,n-1,min(n,8),dtype=int)
    idx_m=np.linspace(0,m-1,min(m,8),dtype=int)
    scores={k:[] for k in PRIMARY}
    for i in idx_n:
        for j in idx_m:
            for row in zx[i,j]:
                v=vector_losses(row)
                for k in PRIMARY:scores[k].append(v[k])
    result={}
    for key,vals in scores.items():
        result[f"zx_local_row_mean_{key}"]=float(np.mean(vals))
        result[f"zx_local_row_q90_{key}"]=float(np.quantile(vals,.90))
        result[f"zx_local_row_max_{key}"]=float(np.max(vals))
    return result


def complexity_metrics(fields,scale_u,scale_x,*,include_fourth=True,
                       include_raw=True):
    """Compute 100s of LOCAL Hessian complexity indicators for one Y.

    This function is common to analytic, formula-Y, and NN-Y paths;
    the caller supplies derivatives evaluated in the SAME candidate U.
    """
    us=np.maximum(np.asarray(scale_u,dtype=float),1e-12)
    xs=np.maximum(np.asarray(scale_x,dtype=float),1e-12)
    F=_scale(fields,us,xs)
    zz=F["zz"];zx=F["zx"];zzz=F["zzz"];zzx=F["zzx"];zxx=F["zxx"]
    diag=np.arange(4)
    offi,offj=np.triu_indices(4,k=1)
    xi,xj=np.triu_indices(6,k=1)
    out={}
    # Magnitude / cancellation: mean Hessian can be 0 despite large
    # pointwise mixed derivatives, so compare mean vs RMS.
    for family,data in (("zz",zz),("zx",zx)):
        out.update(stats(data,f"{family}_local_hessian",loss=False))
        mean=np.mean(data,axis=(0,1))
        rms=np.sqrt(np.mean(data*data,axis=(0,1)))
        mean_norm=float(np.linalg.norm(mean))
        rms_norm=float(np.linalg.norm(rms))
        out[f"{family}_cancellation_ratio_diagnostic"]=float(
            mean_norm/max(rms_norm,EPS))
        out[f"{family}_local_rms_over_mean_diagnostic"]=float(
            rms_norm/max(mean_norm,EPS))
        # Existing 06 rich sparsity applied to both mean and RMS
        # in the SAME physically standardized derivative coordinates.
        block="ZZ" if family=="zz" else "ZX"
        for label,matrix in (("mean",mean),("rms",rms),
                             ("mean_abs",np.mean(np.abs(data),axis=(0,1)))):
            sparse=rich_sparsity(matrix,block)
            for key,val in sparse.items():
                if key.endswith("_loss"):
                    out[f"{family}_{label}_{key}"]=val
    out.update(_anova2(zz[...,diag,diag],"zz_diagonal_curvature"))
    out.update(_anova2(zz[...,offi,offj],"zz_offdiagonal_curvature"))
    out.update(_anova2(zx,"zx_cross_curvature"))
    out.update(stats(zz[...,diag,diag],"zz_diagonal_magnitude"))
    out.update(stats(zz[...,offi,offj],"zz_offdiagonal_magnitude"))
    out.update(stats(zx,"zx_magnitude"))
    out["zz_offdiag_over_diag_energy_loss"]=float(
        np.sqrt(np.mean(zz[...,offi,offj]**2))/
        max(np.sqrt(np.mean(zz[...,diag,diag]**2)),EPS))
    out["zz_offdiag_total_energy_fraction_loss"]=float(
        np.sum(zz[...,offi,offj]**2)*2/
        max(np.sum(zz**2),EPS))
    out["zz_nonzero_diagonal_fraction_diagnostic"]=float(
        np.mean(np.abs(zz[...,diag,diag])>1e-8))
    # 3rd Z derivatives: z^2 vs z^3, i^2*j interactions etc.
    self3=zzz[...,diag,diag,diag]
    other3=np.stack([zzz[...,i,i,j] for i in range(4)
                     for j in range(4) if j!=i],axis=-1)
    pair3=np.stack([zzz[...,i,j,k] for i,j,k in
                    combinations_with_replacement(range(4),3)
                    if len({i,j,k})>1],axis=-1)
    threeway=np.stack([zzz[...,i,j,k] for i,j,k in
                        combinations_with_replacement(range(4),3)
                        if len({i,j,k})==3],axis=-1)
    for label,data in (("zz_self_z_third",self3),
                       ("zz_other_z_third",other3),
                       ("zz_mixed_z_third",pair3),
                       ("zz_three_distinct_z_third",threeway)):
        out.update(stats(data,label))
    # 3rd X derivatives of ZZ: h_ii may depend on X even if f=z_i^2*g(X)
    zzx_diag=zzx[...,diag,diag,:]
    zzx_off=zzx[...,offi,offj,:]
    out.update(stats(zzx_diag,"zz_diagonal_X_modulation_third"))
    out.update(stats(zzx_off,"zz_offdiagonal_X_modulation_third"))
    # XZ derivative modulation wrt its own z, OTHER z, own X and other X.
    own_z=zzx[...,diag,diag,:]
    other_z=np.stack([zzx[...,i,j,:] for i in range(4)
                      for j in range(4) if i!=j],axis=-2)
    own_x=zxx[...,np.arange(4)[:,None],np.arange(6)[None,:],
              np.arange(6)[None,:]]
    other_x=zxx[..., :,xi,xj]
    out.update(stats(own_z,"zx_own_Z_modulation_third"))
    out.update(stats(other_z,"zx_other_Z_modulation_third"))
    out.update(stats(own_x,"zx_own_X_modulation_third"))
    out.update(stats(other_x,"zx_other_X_modulation_third"))
    # Dimensionless shape ratios: derivatives have already been
    # standardized by coordinate SD, and are compared to local Hessian
    # RMS on the same outcome channel.
    zz_ref=max(float(np.sqrt(np.mean(zz**2))),EPS)
    zx_ref=max(float(np.sqrt(np.mean(zx**2))),EPS)
    for label,data,den in (
        ("zz_self_z_higher_order_ratio",self3,zz_ref),
        ("zz_other_z_higher_order_ratio",other3,zz_ref),
        ("zz_X_modulation_ratio",zzx_diag,zz_ref),
        ("zx_own_Z_higher_order_ratio",own_z,zx_ref),
        ("zx_other_Z_higher_order_ratio",other_z,zx_ref),
        ("zx_own_X_higher_order_ratio",own_x,zx_ref),
        ("zx_other_X_higher_order_ratio",other_x,zx_ref)):
        out[label+"_loss"]=float(np.sqrt(np.mean(data**2))/den)
        out[label+"_l1_loss"]=float(np.mean(abs(data))/den)
    # Structured DGSM-like mixed derivative norms: no Sobol upper-bound
    # claim; U coordinates can be correlated and X has a fixed bank.
    out["zz_mixed_DGSM_like_loss"]=float(np.mean(zz[...,offi,offj]**2))
    out["zx_mixed_DGSM_like_loss"]=float(np.mean(zx*zx))
    out.update(_local_row_metrics(zx))
    # Rowwise/columnwise sparsity of the RMS interactions can reveal
    # nonlinear but high-impact terms canceled in the average.
    for label,mat in (("zx_rms",np.sqrt(np.mean(zx*zx,axis=(0,1)))),
                      ("zx_l1",np.mean(abs(zx),axis=(0,1)))):
        out.update({f"{label}_{k}":v for k,v in rich_sparsity(mat,"ZX").items()
                    if k.startswith(("zx_row","zx_col")) and k.endswith("_loss")})
    if include_fourth and "zzzz" in F:
        z4=F["zzzz"]
        self4=z4[...,diag,diag,diag,diag]
        cross4=np.stack([z4[...,i,i,j,j] for i in range(4)
                         for j in range(i+1,4)],axis=-1)
        threeone4=np.stack([z4[...,i,i,i,j] for i in range(4)
                            for j in range(4) if i!=j],axis=-1)
        zzxx=F["zzxx"]
        zxxx=F["zxxx"]
        out.update(stats(self4,"zz_self_z_fourth"))
        out.update(stats(cross4,"zz_Z_pair_fourth"))
        out.update(stats(threeone4,"zz_Z_three_one_fourth"))
        out.update(stats(zzxx[...,diag,diag,:],"zz_diagonal_X_twice_fourth"))
        out.update(stats(zzxx[...,offi,offj,:],"zz_offdiag_X_twice_fourth"))
        out.update(stats(zxxx,"zx_own_X_three_times_fourth"))
        out["zz_self_fourth_vs_third_rms_loss"]=float(
            np.sqrt(np.mean(self4**2))/
            max(np.sqrt(np.mean(self3**2)),EPS))
    if include_raw:
        # Non-standardized unit-dependent measures are explicitly
        # labeled raw; do not compare across lambda and alpha axes without
        # coordinate scaling.
        out.update(stats(_one_field(fields,"zzz")[...,diag,diag,diag],
                         "raw_zz_self_third",loss=False))
        out.update(stats(_one_field(fields,"zzx")[...,diag,diag,:],
                         "raw_zz_X_modulation",loss=False))
    return out


def metric_registry(fields=None,scale_u=None,scale_x=None,include_fourth=True):
    """Programmatic inventory; actual key names are authoritative."""
    if fields is None:
        return dict(meaning=(
            "zz_self_z: how diagonal Hzz changes with its OWN Z",
            "zz_other_z: how Hii depends on OTHER Z",
            "zz_X_modulation: how ZZ curvature changes with X",
            "zx_own_Z/other_Z: changes in cross-Hessian along Z",
            "zx_own_X/other_X: cross-Hessian changes along X",
            "fourth: 4th-order Sobolev-like roughness",
            "mean/rms/local row: sparse effects not erased by averaging",
            "loss=minimize; diagnostic=do not blindly minimize",
        ))
    return list(complexity_metrics(fields,scale_u,scale_x,
                                   include_fourth=include_fourth).keys())
