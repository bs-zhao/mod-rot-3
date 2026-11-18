"""CR3 4D extension of output-gain + low-axial-curvature suite.

This is a genuinely FOUR dimensional SO(4) optimization. It does not use
true parameter values or CR3 formula. The original 3D SO(2) angle search
is insufficient because the gain-orthogonal subspace is 3-dimensional.

1. Find a function-supported output gain direction by generalized eigen.
2. Find the minimum pure-U curvature in the perpendicular 3D subspace;
   search a deterministic full sphere and locally refine each best start.
3. In the remaining 2D plane, optimize a second curvature direction.
4. The last axis is the orthogonal complement; output is a proper SO(4) R.

Functional directions may fail to coincide with the four true parameters.
All validity gates are predeclared and evaluated without theta.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize, minimize_scalar
from rotations.cr4_functional_axes.objective import (
    curvature_value, angle_curvature, rayleigh
)


def gain_four(eta, G, ridge):
    e=np.asarray(eta,float)
    g=np.asarray(G,float)
    if e.ndim!=2 or g.shape!=(*e.shape,4):
        raise ValueError("Expected (subject,X) eta and (subject,X,4) gradient")
    coeff=np.einsum("sx,sxi->si",e,g)/np.maximum(np.sum(e*e,axis=1)[:,None],1e-12)
    residual=(g-e[:,:,None]*coeff[:,None,:]).reshape(-1,4)
    full=g.reshape(-1,4)
    A=(residual.T@residual)/len(residual)
    B=(full.T@full)/len(full)
    A=(A+A.T)*.5
    B=(B+B.T)*.5
    scale=max(float(np.trace(B))/4.,1e-12)
    eig,vec=eigh(A,B+float(ridge)*scale*np.eye(4),check_finite=True)
    v=np.asarray(vec[:,0],float)
    v/=np.linalg.norm(v)
    if v[np.argmax(np.abs(v))]<0:v=-v
    return v,A,B,eig


def sphere_starts(size):
    """Deterministic Fibonacci sphere, full 3D directional search."""
    if size<32:
        raise ValueError("sphere_points must be at least 32")
    t=np.arange(size,dtype=float)
    zz=1.-2.*(t+.5)/size
    phi=t*(np.pi*(3-np.sqrt(5)))
    rr=np.sqrt(np.maximum(0,1-zz*zz))
    return np.column_stack((rr*np.cos(phi),rr*np.sin(phi),zz))


def curvature_tensor(H,frame):
    """Fourth-order quartic tensor for mean[(v^T H v)^2]/E||H||^2."""
    H=np.asarray(H,float)
    h=np.einsum("ia,...ij,jb->...ab",frame,H,frame,optimize=True)
    mat=h.reshape(-1,9)
    divisor=max(float(np.mean((H*H).sum(axis=(-2,-1)))),1e-15)
    t=(mat.T@mat).reshape(3,3,3,3)/(len(mat)*divisor)
    return t


def quartic(T,v):
    return float(np.einsum("abcd,a,b,c,d->",T,v,v,v,v,optimize=True))


def fit_sphere(H,frame,npoints=512):
    T=curvature_tensor(H,frame)
    directions=sphere_starts(int(npoints))
    losses=np.array([quartic(T,v) for v in directions])
    best=(np.inf,None)
    indices=np.argsort(losses)[:min(12,len(losses))]
    def objective(raw):
        length=np.linalg.norm(raw)
        if length<1e-12:return 1e30,np.zeros(3)
        v=raw/length
        loss=quartic(T,v)
        d=4*np.einsum("abcd,b,c,d->a",T,v,v,v,optimize=True)
        derivative=(d-v*np.dot(d,v))/length
        return loss, derivative
    for idx in indices:
        local=minimize(objective,directions[idx],method="BFGS",jac=True,
                       options={"gtol":1e-9,"maxiter":250})
        size=np.linalg.norm(local.x)
        if size<1e-10:continue
        v=local.x/size
        loss=quartic(T,v)
        if loss<best[0]:best=(loss,v)
    if best[1] is None:raise RuntimeError("No valid sphere curvature optimum")
    v=best[1]
    if v[np.argmax(np.abs(v))]<0:v=-v
    stats=dict(sphere_points=int(npoints),
               fit_sphere_start_minimum=float(losses.min()),
               fit_sphere_start_median=float(np.median(losses)),
               fit_sphere_start_maximum=float(losses.max()),
               fit_sphere_refined_minimum=float(best[0]),
               starts_refined=int(len(indices)))
    return frame@v,stats


def fit_four_axes(fit,val,config):
    for key,fields in (("fit",fit),("valid",val)):
        if fields["G"].shape[-1]!=4 or fields["H"].shape[-2:]!=(4,4):
            raise ValueError(f"{key}: 4D derivative field expected")
    gain,A,B,eigenvalues=gain_four(fit["eta"],fit["G"],config["gain_ridge"])
    _,A_valid,B_valid,_=gain_four(val["eta"],val["G"],config["gain_ridge"])
    val_gain_ratio=rayleigh(gain,A_valid,B_valid)
    _,_,sv=np.linalg.svd(gain.reshape(1,4),full_matrices=True)
    frame=sv[1:].T  # four-by-three complement, not a plane

    first,sphere_diag=fit_sphere(fit["H"],frame,int(config.get("sphere_points",512)))
    first/=np.linalg.norm(first)
    # Independent 2D remaining subspace after first+gain.
    _,_,remain=np.linalg.svd(np.stack((gain,first)),full_matrices=True)
    plane=remain[2:].T
    if plane.shape!=(4,2):
        raise RuntimeError("CR3 gain+first curvature must leave a 2D plane")
    n=int(config["curvature_grid"])
    if n<15:raise ValueError("curvature_grid must be >=15")
    angle=np.linspace(0,np.pi,n,endpoint=False)
    scores=angle_curvature(fit["H"],plane,angle)
    spacing=np.pi/n
    refined=[]
    for idx in np.argsort(scores)[:min(6,n)]:
        start=float(angle[idx])
        trial=minimize_scalar(
            lambda t:angle_curvature(fit["H"],plane,[t])[0],
            bounds=(start-spacing,start+spacing),method="bounded",
            options={"xatol":1e-9,"maxiter":80})
        refined.append((float(trial.fun),float(trial.x)%np.pi))
    _,best_angle=min(refined,key=lambda x:x[0])
    c,s=np.cos(best_angle),np.sin(best_angle)
    second=plane@np.array([c,s])
    remaining=plane@np.array([-s,c])
    R=np.stack((first,second,remaining,gain),axis=0)
    if np.linalg.det(R)<0:R[2]*=-1
    if not np.allclose(R@R.T,np.eye(4),atol=1e-8) or not np.isclose(np.linalg.det(R),1.,atol=1e-8):
        raise RuntimeError("CR3 fit_four_axes must return SO(4)")
    fit_energy=[curvature_value(fit["H"],v) for v in R]
    valid_energy=[curvature_value(val["H"],v) for v in R]
    gap=float(config["min_curvature_gap"])
    gain_pass=(rayleigh(gain,A,B)<=config["gain_fit_max_ratio"] and
               val_gain_ratio<=config["gain_valid_max_ratio"] and
               float(eigenvalues[1]-eigenvalues[0])>=config["gain_min_gap"] and
               float(gain@B@gain)/max(float(np.trace(B)),1e-15)>=config["gain_min_energy_share"])
    first_pass=(fit_energy[1]-fit_energy[0]>=gap and valid_energy[1]-valid_energy[0]>=gap)
    second_pass=(fit_energy[2]-fit_energy[1]>=gap and valid_energy[2]-valid_energy[1]>=gap)
    curvature_pass=bool(first_pass and second_pass)
    diag=dict(
        method="CR3_SO4_output_gain_then_3D_sphere_curvature_then_SO2",
        fit_gain_ratio=float(rayleigh(gain,A,B)),
        validation_gain_ratio=float(val_gain_ratio),
        gain_eigenvalues=eigenvalues.tolist(),
        gain_eigen_gap=float(eigenvalues[1]-eigenvalues[0]),
        gain_axis_energy_fraction=float(gain@B@gain)/max(float(np.trace(B)),1e-15),
        gain_gate_pass=bool(gain_pass),
        fit_curvature_by_axis=fit_energy,
        validation_curvature_by_axis=valid_energy,
        first_curvature_gate_pass=bool(first_pass),
        second_curvature_gate_pass=bool(second_pass),
        curvature_gate_pass=curvature_pass,
        both_gates_pass=bool(gain_pass and curvature_pass),
        first_curvature_sphere=sphere_diag,
        selected_second_angle_deg=float(np.degrees(best_angle)),
        components=("lowest_curvature_in_gain_perp_3D",
                    "lowest_curvature_in_remaining_2D",
                    "final_orthogonal_complement","output_gain"),
        interpretation=("supported functional SO4 axes" if (gain_pass and curvature_pass)
                        else "exploratory SO4 rotation; NOT verified parameter recovery"),
        fit_gain_residual_operator=A.tolist(),fit_gain_total_operator=B.tolist(),
        validation_gain_residual_operator=A_valid.tolist(),
        validation_gain_total_operator=B_valid.tolist())
    landscape=dict(angle_deg=np.degrees(angle).tolist(),
                   fit_curvature=scores.tolist(),
                   validation_curvature=angle_curvature(val["H"],plane,angle).tolist())
    return R,diag,landscape
