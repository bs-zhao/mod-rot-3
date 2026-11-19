"""Integrate over a specified Gaussian measure, NOT observed participant Z.

Sobol nodes are deterministic fixed common random numbers; independent FIT
and VALID seeds diagnose quadrature convergence. Truncated Gaussian boxes
use conditional Gaussian inverse-CDF, NEVER uniform weighting in q.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import qmc


def gaussian_nodes(dim,method="sobol",n=512,order=7,cutoff=2.,seed=24):
    if dim not in (3,4):raise ValueError("Expected 3D or 4D q")
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff<=0):
        raise ValueError("Gaussian cutoff must be positive or null")
    if method=="sobol":
        if n<2 or n&(n-1):raise ValueError("Sobol n must be a power of two >=2")
        u=qmc.Sobol(d=dim,scramble=True,seed=seed).random_base2(int(np.log2(n)))
        u=np.clip(u,1e-12,1-1e-12)
        if cutoff is None:
            q=ndtri(u)
            kind="full_N0I"
        else:
            lo=ndtr(-cutoff);hi=ndtr(cutoff)
            q=ndtri(lo+u*(hi-lo))
            kind="coordinatewise_truncated_N0I"
        weights=np.full(n,1./n,dtype=float)
    elif method=="hermite":
        if cutoff is not None:
            raise ValueError("Gauss-Hermite integrates FULL Gaussian only: z_cutoff=null")
        if order<2 or order>15:
            raise ValueError("Hermite order should be 2..15")
        nodes,w=np.polynomial.hermite.hermgauss(order)
        axis=np.sqrt(2)*nodes
        wa=w/np.sqrt(np.pi)
        tuples=list(itertools.product(range(order),repeat=dim))
        q=np.array([[axis[j] for j in k] for k in tuples])
        weights=np.array([np.prod([wa[j] for j in k]) for k in tuples])
        kind="full_N0I_hermite"
    elif method=="grid":
        if cutoff is None:
            raise ValueError("Finite grid requires z_cutoff (e.g. 2)")
        if order<3 or order>21:raise ValueError("Grid order must be 3..21")
        axis=np.linspace(-cutoff,cutoff,order)
        delta=axis[1]-axis[0]
        wa=np.exp(-axis**2/2)/np.sqrt(2*np.pi)
        wa[[0,-1]]*=.5
        wa=wa/wa.sum()
        tuples=list(itertools.product(range(order),repeat=dim))
        q=np.array([[axis[j] for j in k] for k in tuples])
        weights=np.array([np.prod([wa[j] for j in k]) for k in tuples])
        kind="truncated_N0I_trapezoidal_grid"
    else:
        raise ValueError("Unknown Gaussian quadrature method")
    weights=np.asarray(weights,np.float64)
    weights/=weights.sum()
    if not np.isfinite(q).all() or not np.isfinite(weights).all():
        raise RuntimeError("Quadrature produced nonfinite Gaussian nodes")
    return q.astype(np.float64),weights,dict(
        distribution=kind,method=method,dim=dim,
        points=len(q),cutoff=cutoff,order=order,seed=seed,
        weight_sum=float(weights.sum()),
        q_mean=np.average(q,axis=0,weights=weights).tolist(),
        q_second=np.average(q*q,axis=0,weights=weights).tolist())
