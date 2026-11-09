"""Additional independent-factor control, NOT a pure Hessian metric."""
import numpy as np
from scipy.optimize import minimize

def three_skew_directions(z,starts=32,seed=913):
    """Symmetric rank-3 third-cumulant fit; global scalar scaling only.

    Assumes three skewed independent factors plus one nearly symmetric factor,
    and approximately symmetric independent representation noise. These are
    extra assumptions, not consequences of accurate decoder predictions.
    """
    z=np.asarray(z,float);z=z-z.mean(0)
    scale=np.sqrt(np.mean(z*z));z=z/scale;d=z.shape[1]
    tensor=np.einsum('na,nb,nc->abc',z,z,z)/len(z)
    norm=max(np.sum(tensor*tensor),1e-20);ridge=1e-5
    def objective(flat):
        b=flat.reshape(d,3)
        residual=np.einsum('ar,br,cr->abc',b,b,b)-tensor
        value=np.sum(residual*residual)/norm+ridge*np.sum(b*b)
        gradient=6*np.einsum('abc,br,cr->ar',residual,b,b)/norm+2*ridge*b
        return value,gradient.ravel()
    rng=np.random.default_rng(seed);solutions=[]
    u,_,_=np.linalg.svd(tensor.reshape(d,-1),full_matrices=False)
    for k in range(starts):
        if k==0:
            b=u[:,:3].copy()
            for j in range(3):b[:,j]*=np.cbrt(np.einsum('abc,a,b,c',tensor,b[:,j],b[:,j],b[:,j]))
        else:b=rng.normal(size=(d,3))*.5
        fit=minimize(objective,b.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-13,gtol=1e-9))
        solutions.append(dict(loss=float(fit.fun),converged=bool(fit.success),factors=fit.x.reshape(d,3)))
    best=min(solutions,key=lambda item:item['loss']);factors=best['factors']
    directions=factors/np.linalg.norm(factors,axis=0)
    info=dict(tensor=tensor,global_input_scale=scale,rank=3,ridge=ridge,
              loss=best['loss'],factors=factors,directions=directions,
              starts=starts,converged_starts=sum(x['converged'] for x in solutions),
              assumption='Three skewed independent factors, one near-symmetric factor and approximately symmetric independent representation noise; additional prior control')
    return directions,info
