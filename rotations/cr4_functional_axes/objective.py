"""Two functional invariants, completely blind to true theta and CPT formula.

I. An output-gain direction v has directional derivative G_v(s,x) proportional
   to eta(s,x) within each subject. Find the smallest generalized eigenvector
   of residual/total derivative operators. Normalize by directional energy,
   not raw residual, to reject artificially weak axes.
II. Among directions orthogonal to v, minimize mean directional squared
    Hessian (v^T H_qq v)^2. A low-curvature axis is functionally affine.

Neither condition is generally guaranteed for an arbitrary nonlinear NP
representation; require FIT/VALID gates and retain failed diagnoses.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def gain_operators(eta, gradient):
    e = np.asarray(eta, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if e.ndim != 2 or g.shape != (*e.shape, 3):
        raise ValueError("Expected eta (subjects,X), G (subjects,X,3)")
    power = np.sum(e*e, axis=1)
    coefficients = np.einsum("sx,sxi->si",e,g) / np.maximum(power[:,None],1e-12)
    residual = g - e[:,:,None]*coefficients[:,None,:]
    r = residual.reshape(-1,3)
    all_g = g.reshape(-1,3)
    A = r.T@r/len(r)
    B = all_g.T@all_g/len(all_g)
    return (A+A.T)/2, (B+B.T)/2


def rayleigh(v,A,B):
    v = np.asarray(v,np.float64)
    numerator = max(float(v@A@v),0.)
    denom = float(v@B@v)
    return numerator/max(denom,1e-15)


def gain_axis(eta,G, ridge=1e-4):
    A,B = gain_operators(eta,G)
    scale = max(float(np.trace(B))/3,1e-12)
    vals,vecs = eigh(A,B + ridge*scale*np.eye(3),check_finite=True)
    v = np.asarray(vecs[:,0],dtype=np.float64)
    v /= np.linalg.norm(v)
    if v[np.argmax(np.abs(v))] < 0: v = -v
    return dict(axis=v,operator_residual=A,operator_energy=B,
                eigenvalues=np.asarray(vals,float),
                fit_ratio=rayleigh(v,A,B),
                energy_share=float(v@B@v/max(np.trace(B),1e-15)),
                gap=float(vals[1]-vals[0]))


def perpendicular_frame(v):
    v=np.asarray(v,float)
    v=v/np.linalg.norm(v)
    # Deterministic basis in the full two-dimensional Euclidean complement.
    _,_,vt=np.linalg.svd(v.reshape(1,3),full_matrices=True)
    b=vt[1:].T.copy()
    if np.linalg.det(np.column_stack((b,v)))<0:
        b[:,1]*=-1
    return b


def curvature_value(H,v):
    H=np.asarray(H,np.float64)
    v=np.asarray(v,np.float64)
    dh=np.einsum("i,...ij,j->...",v,H,v,optimize=True)
    energy=float(np.mean(np.square(H).sum(axis=(-2,-1))))
    return float(np.mean(dh*dh)/max(energy,1e-15))


def angle_curvature(H,basis,angles):
    """Evaluate all SO(2) candidate axes cheaply via 2x2 quadratic form."""
    H=np.asarray(H,np.float64)
    basis=np.asarray(basis,np.float64)
    h=np.einsum("ia,...ij,jb->...ab",basis,H,basis,optimize=True)
    aa=h[...,0,0]; bb=h[...,0,1]; cc=h[...,1,1]
    out=[]
    scale=max(float(np.mean(np.square(H).sum(axis=(-2,-1)))),1e-15)
    for angle in np.atleast_1d(angles):
        c,s=np.cos(angle),np.sin(angle)
        d=c*c*aa+2*c*s*bb+s*s*cc
        out.append(float(np.mean(d*d)/scale))
    return np.asarray(out,float)


def fit_functional_axes(fit,val,config):
    for name,data in (("fit",fit),("valid",val)):
        if not all(key in data for key in ("eta","G","H")):
            raise ValueError(f"Missing {name} fields")
    eig = gain_axis(fit["eta"],fit["G"],float(config["gain_ridge"]))
    v_gain=eig["axis"]
    A_valid,B_valid = gain_operators(val["eta"],val["G"])
    valid_ratio=rayleigh(v_gain,A_valid,B_valid)
    frame=perpendicular_frame(v_gain)
    grid_n=int(config["curvature_grid"])
    if grid_n<15: raise ValueError("curvature_grid must be >=15")
    angles=np.linspace(0,np.pi,grid_n,endpoint=False)
    scores=angle_curvature(fit["H"],frame,angles)
    spacing=np.pi/grid_n
    candidates=np.argsort(scores)[:min(6,len(scores))]
    refined=[]
    for i in candidates:
        x0=float(angles[i])
        trial=minimize_scalar(
            lambda a:angle_curvature(fit["H"],frame,[a])[0],
            bounds=(x0-spacing,x0+spacing),method="bounded",
            options={"xatol":1e-9,"maxiter":80})
        refined.append((float(trial.fun),float(trial.x)%np.pi))
    _,best_angle=min(refined,key=lambda t:t[0])
    c,s=np.cos(best_angle),np.sin(best_angle)
    v_linear=frame@np.array([c,s])
    v_nonlinear=frame@np.array([-s,c])
    R=np.stack((v_linear,v_nonlinear,v_gain))
    if np.linalg.det(R)<0:
        R[1]*=-1
    if not np.allclose(R@R.T,np.eye(3),atol=1e-9) or abs(np.linalg.det(R)-1)>1e-9:
        raise RuntimeError("Invalid rotation geometry")
    fit_linear=curvature_value(fit["H"],R[0])
    fit_complement=curvature_value(fit["H"],R[1])
    val_linear=curvature_value(val["H"],R[0])
    val_complement=curvature_value(val["H"],R[1])
    val_curve=angle_curvature(val["H"],frame,angles)
    gain_pass=(eig["fit_ratio"]<=float(config["gain_fit_max_ratio"]) and
               valid_ratio<=float(config["gain_valid_max_ratio"]) and
               eig["gap"]>=float(config["gain_min_gap"]) and
               eig["energy_share"]>=float(config["gain_min_energy_share"]))
    curvature_pass=((fit_complement-fit_linear)>=float(config["min_curvature_gap"]) and
                    (val_complement-val_linear)>=float(config["min_curvature_gap"]))
    diag=dict(
        method="function-supported-output-gain_then_low-axial-curvature",
        fit_gain_ratio=float(eig["fit_ratio"]),
        validation_gain_ratio=float(valid_ratio),
        gain_eigenvalues=eig["eigenvalues"].tolist(),
        gain_eigen_gap=float(eig["gap"]),
        gain_axis_energy_fraction=float(eig["energy_share"]),
        gain_gate_pass=bool(gain_pass),
        fit_low_curvature=float(fit_linear),
        fit_complement_curvature=float(fit_complement),
        valid_low_curvature=float(val_linear),
        valid_complement_curvature=float(val_complement),
        curvature_gate_pass=bool(curvature_pass),
        both_gates_pass=bool(gain_pass and curvature_pass),
        interpretation=("supported axes" if (gain_pass and curvature_pass)
                        else "exploratory rotation only; do NOT claim recovered mechanisms"),
        fit_gain_residual_operator=eig["operator_residual"].tolist(),
        fit_gain_total_operator=eig["operator_energy"].tolist(),
        validation_gain_residual_operator=A_valid.tolist(),
        validation_gain_total_operator=B_valid.tolist(),
        selected_angle_deg=float(np.degrees(best_angle)),
        components=("low_axial_curvature","remaining","output_gain"))
    landscape=dict(angle_deg=np.degrees(angles).tolist(),
                   fit_curvature=scores.tolist(),validation_curvature=val_curve.tolist())
    return R,diag,landscape
