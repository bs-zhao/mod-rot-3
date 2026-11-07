"""Affine Z->T anchor, pairwise SO(4) rotations, and exact Hessian pullbacks.

At theta=0, U=A_star Z+b=T_hat. At theta>0, A_theta=R_ij(theta)A_star,
U=A_theta Z+b=b+R_ij(theta)(T_hat-b). The original Z, X, SIX numerical
Y and generator function are FIXED during rotation.

If H_ZX is the original step-04 analytic mean Hessian at (Z,X),
J=diag(A_theta^{-1},I_6), and H_UX=J.T H_ZX J exactly.
This is an exact linear-coordinate change, NOT a new generating function.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np

PARAMETER_NAMES=("alpha","lambda","gamma","tau")
PLANES=tuple(combinations(range(4),2))


def fit_affine(Z,T,ids=None):
    """Return A, b for T[n]≈ A @ Z[n]+b; four-by-four, with intercept."""
    z=np.asarray(Z,dtype=np.float64)
    t=np.asarray(T,dtype=np.float64)
    if z.ndim!=2 or z.shape[1]!=4 or t.shape!=z.shape:
        raise ValueError("Expected Z,T shape (N,4)")
    sample=np.arange(len(z)) if ids is None else np.asarray(ids,dtype=int)
    if len(sample)<10:raise ValueError("At least ten subjects required")
    reg=np.column_stack((z[sample],np.ones(len(sample))))
    coefficients=np.linalg.lstsq(reg,t[sample],rcond=None)[0]
    A=np.asarray(coefficients[:4].T)
    b=np.asarray(coefficients[4])
    if np.linalg.matrix_rank(A)!=4:
        raise ValueError("The fitted A is not invertible")
    return A,b


def pair_rotation(i,j,angle_degrees):
    if not(0<=i<j<4):raise ValueError("Invalid 4D coordinate pair")
    angle=np.deg2rad(float(angle_degrees))
    c,s=np.cos(angle),np.sin(angle)
    R=np.eye(4)
    R[i,i]=c;R[j,j]=c
    R[i,j]=-s;R[j,i]=s
    return R


def candidate_mapping(A_star,b,Z,i,j,angle_degrees):
    A0=np.asarray(A_star,dtype=float)
    R=pair_rotation(i,j,angle_degrees)
    A=R@A0
    u=np.asarray(Z,dtype=float)@A.T+np.asarray(b,dtype=float)
    return A,R,u


def full_block_rotation(R):
    Q=np.eye(10)
    Q[:4,:4]=R
    return Q


def original_z_to_candidate_u(H_z,A_candidate):
    """Pull H_Z into U=A Z+b, including mixed and X-only Hessian blocks."""
    M=np.asarray(H_z,dtype=float)
    A=np.asarray(A_candidate,dtype=float)
    if M.shape!=(10,10) or A.shape!=(4,4):
        raise ValueError("Expected 10x10 Hessian and 4x4 A")
    J=np.eye(10)
    J[:4,:4]=np.linalg.inv(A)
    out=J.T@M@J
    return .5*(out+out.T)


def star_to_rotated(H_star,R):
    """Given H at baseline T_hat, produce same-function H at rotated U."""
    Q=full_block_rotation(R)
    return Q@np.asarray(H_star,dtype=float)@Q.T


def rotated_to_star(H_rotated,R):
    """Map estimator in U coordinates back into baseline T_hat coordinates."""
    Q=full_block_rotation(R)
    return Q.T@np.asarray(H_rotated,dtype=float)@Q


def parse_angles(spec):
    """Examples: '0:90:5', '0,10,20,90', '0:90:1' (inclusive)."""
    s=str(spec).strip()
    if ":" in s:
        fields=list(map(float,s.split(":")))
        if len(fields)!=3:raise ValueError("Angles must be start:stop:step")
        start,stop,step=fields
        if step<=0 or stop<start:raise ValueError("Invalid angle range")
        count=int(round((stop-start)/step))
        if count>1000:raise ValueError("Angle grid is too fine")
        values=[start+k*step for k in range(count+1)]
        if values[-1]>stop+1e-8:values=values[:-1]
        if not np.isclose(values[-1],stop):values.append(stop)
    else:
        values=[float(x) for x in s.split(",") if x.strip()]
    if not values or any(not np.isfinite(x) or x<0 or x>90 for x in values):
        raise ValueError("Angles must be between 0 and 90 degrees")
    if any(values[k]>=values[k+1] for k in range(len(values)-1)):
        raise ValueError("Angles must be strictly increasing")
    if not np.isclose(values[0],0):
        raise ValueError("The grid must contain 0 as the first angle")
    return np.asarray(values,dtype=float)


def parse_planes(spec):
    """Parse 'all', 'alpha:gamma', '0:2', '01,02,13', etc."""
    if spec is None or str(spec).lower()=="all":
        return PLANES
    result=[]
    for token in str(spec).split(","):
        token=token.strip().lower().replace("-" ,":")
        if not token:continue
        if len(token)==2 and token.isdigit():
            pair=(int(token[0]),int(token[1]))
        else:
            fields=token.split(":")
            if len(fields)!=2:raise ValueError("Expected e.g. 01 or alpha:gamma")
            pair=tuple(PARAMETER_NAMES.index(t) if t in PARAMETER_NAMES
                       else int(t) for t in fields)
        if pair not in PLANES:raise ValueError(f"Invalid pair {token!r}")
        if pair not in result:result.append(pair)
    if not result:raise ValueError("No valid rotation planes")
    return tuple(result)


def plane_name(i,j):
    return f"{i}{j}_{PARAMETER_NAMES[i]}_{PARAMETER_NAMES[j]}"
