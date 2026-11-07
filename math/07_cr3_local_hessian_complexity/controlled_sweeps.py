"""Controlled coordinate sweeps: separate own-Z complexity from X modulation.

At a fixed anchor U_{-i} and fixed stimulus X, vary ONLY U_i
over a shared, linearly spaced grid. The fitted polynomial and original
analytic CR3 function are queried on that SAME grid.
The local Hessian H_ii(U,X) is retained at every grid point.

This explicitly addresses f=z_i²*g(X) vs f=z_i³*g(X):
both vary with X, but only the latter has H_ii varying with z_i.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import dct


def sweep_grids(U_all,anchor_indices,levels=7):
    U=np.asarray(U_all,dtype=float)
    anchors=U[np.asarray(anchor_indices,dtype=int)]
    if levels<6:raise ValueError("At least 6 levels for quadratic/cubic/quartic comparisons")
    grid=np.zeros((4,len(anchors),levels,4))
    positions=np.empty((4,levels))
    for i in range(4):
        lo,hi=np.quantile(U[:,i],[.10,.90])
        if lo>=hi:raise ValueError("Degenerate U sweep dimension")
        positions[i]=np.linspace(lo,hi,levels)
        grid[i]=anchors[:,None,:]
        grid[i,:,:,i]=positions[i][None,:]
    return grid,positions


def polynomial_sweeps(poly,U_all,anchor_idx,*,levels=7):
    grids,pos=sweep_grids(U_all,anchor_idx,levels=levels)
    C=poly.coef.shape[-1]; A=len(anchor_idx); M=len(poly.X)
    values=np.empty((C,4,A,levels,M))
    curvature=np.empty_like(values)
    for i in range(4):
        ugrid=grids[i].reshape(-1,4)
        values[:,i]=poly.evaluate(ugrid).reshape(C,A,levels,M)
        curvature[:,i]=poly.evaluate(ugrid,(i,i)).reshape(C,A,levels,M)
    return dict(value=values,curvature=curvature,position=pos)


def analytic_sweeps(U_all,b,R,anchor_idx,X,oracle_module,*,levels=7):
    """Analytic Y and exact Hessian on U grids; no NN differentiation."""
    grids,pos=sweep_grids(U_all,anchor_idx,levels=levels)
    X=np.asarray(X,dtype=float)
    A=len(anchor_idx);M=len(X)
    value=np.empty((3,4,A,levels,M))
    curvature=np.empty_like(value)
    for i in range(4):
        ugrid=grids[i].reshape(-1,4)
        # T_hat=b+R.T@(U-b). Row-vector equivalent: (U-b)@R+b.
        T=(ugrid-b)@R+b
        eta,p=oracle_module.oracle_outputs(T,X)
        h_eta,h_prob=oracle_module.oracle_hessians(T,X)
        w=R[i,:]
        c_eta=np.einsum("a,nmab,b->nm",w,h_eta[:,:,:4,:4],w,optimize=True)
        c_p=np.einsum("a,nmab,b->nm",w,h_prob[:,:,:4,:4],w,optimize=True)
        value[:,i]=np.stack((p,eta,p)).reshape(3,A,levels,M)
        curvature[:,i]=np.stack((c_p,c_eta,c_p)).reshape(3,A,levels,M)
    return dict(value=value,curvature=curvature,position=pos)


def _poly_residual(y,position,degree):
    """OLS per anchor-X function, no overfitting beyond grid points."""
    p=np.asarray(position,dtype=float)
    p=(p-p.mean())/max(float(p.std()),1e-12)
    design=np.stack([p**k for k in range(degree+1)],axis=1)
    Y=np.moveaxis(np.asarray(y,dtype=float),1,0).reshape(len(p),-1)
    coef=np.linalg.lstsq(design,Y,rcond=None)[0]
    pred=(design@coef).reshape((len(p),)+np.moveaxis(y,1,0).shape[1:])
    residual=np.moveaxis(pred,0,1)-y
    return residual


def sweep_metrics(one_source):
    """Input source value/curvature shape (4,anchors,levels,X)."""
    y=np.asarray(one_source["value"],dtype=float)
    H=np.asarray(one_source["curvature"],dtype=float)
    pos=np.asarray(one_source["position"],dtype=float)
    if y.ndim!=4 or H.shape!=y.shape or y.shape[0]!=4:
        raise ValueError("Sweep fields must be (4,anchors,levels,X)")
    nlev=y.shape[2]
    if nlev<6:raise ValueError("Need six or more levels")
    outputs={}
    records={}
    for i in range(4):
        f=y[i]  # anchors × levels × X
        h=H[i]
        xs=pos[i]
        # Variance along the OWN coordinate with other inputs held fixed.
        var_h=np.mean(np.var(h,axis=1))
        rms_h=float(np.sqrt(np.mean(h*h)))
        extent=float(xs[-1]-xs[0])
        normalized_h=np.diff(h,axis=1)
        curvature_tv=float(np.mean(np.abs(normalized_h))*max(nlev-1,1))
        slope=np.gradient(h,xs,axis=1)
        norm=float(np.mean(h*h))
        yy=float(np.mean(np.var(f,axis=1)))
        # Residual relative to the within-sweep variance, not to the
        # global response variance (which may be dominated by X effects).
        res={}
        for d in (1,2,3,4):
            rem=_poly_residual(f,xs,d)
            res[d]=float(np.mean(rem*rem)/max(yy,1e-15))
        # DCT is a frequency smoothness diagnostic, not a claim
        # about exact polynomial degree.
        spectral=dct(f,type=2,norm="ortho",axis=1)
        spectral_power=np.mean(spectral**2,axis=(0,2))
        freq_total=float(np.sum(spectral_power[1:]))
        highfreq=float(np.sum(spectral_power[3:])/max(freq_total,1e-15))
        records[i]=dict(
            self_curvature_variance_loss=float(var_h),
            self_curvature_variance_normalized_loss=float(var_h/max(norm,1e-15)),
            self_curvature_tv_loss=curvature_tv,
            self_curvature_tv_normalized_loss=float(curvature_tv/max(rms_h,1e-15)),
            self_curvature_slope_l1_loss=float(np.mean(abs(slope))*extent),
            self_curvature_slope_l2_loss=float(np.mean(slope*slope)*extent**2),
            quadratic_approx_relative_mse_loss=res[2],
            linear_approx_relative_mse_loss=res[1],
            cubic_approx_relative_mse_loss=res[3],
            quartic_approx_relative_mse_loss=res[4],
            high_frequency_dct_ratio_diagnostic=highfreq,
            sweep_curvature_rms_diagnostic=rms_h,
            sweep_range_diagnostic=extent,
        )
    for key in records[0]:
        vals=np.asarray([records[i][key] for i in range(4)],dtype=float)
        for i in range(4):outputs[f"sweep_z{i}_{key}"]=float(vals[i])
        outputs[f"sweep_mean_{key}"]=float(vals.mean())
        outputs[f"sweep_max_{key}"]=float(vals.max())
    return outputs
