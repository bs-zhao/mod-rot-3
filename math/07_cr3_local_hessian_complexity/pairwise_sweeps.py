"""Counterfactual group/axis additivity gap for ZZ without neural gradients.

For each of six U-axis pairs, hold all other U coordinates and every
X stimulus fixed, and evaluate a Cartesian 2D grid. Remove the two
additive main effects; the remaining residual measures NON-ADDITIVITY
in the chosen Z pair. A separable f_i(U_i,X)+f_j(U_j,X) has gap zero.
This is inspired by functional ANOVA / Friedman-Popescu H statistics,
but uses a controlled Cartesian grid and is NOT a population Sobol index.

All three sources use precisely the same candidate A, fixed anchors,
X bank, and coordinate grid; no arbitrary context splits.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np


PAIRS=tuple(combinations(range(4),2))


def _interaction_fraction(y):
    """y=(anchors, i-levels, j-levels, fixed-X)."""
    v=np.asarray(y,dtype=float)
    main_i=v.mean(axis=2,keepdims=True)
    main_j=v.mean(axis=1,keepdims=True)
    grand=v.mean(axis=(1,2),keepdims=True)
    residual=v-main_i-main_j+grand
    centered=v-grand
    mse=float(np.mean(residual**2))
    total=float(np.mean(centered**2))
    return mse/max(total,1e-15),mse,float(np.mean(np.abs(residual)))


def pairwise_additivity(poly,U_all,anchor_indices,b,R,oracle_module,
                        *,levels=5,quantiles=(.1,.9)):
    if levels<3:raise ValueError("Need 3 or more levels")
    U=np.asarray(U_all,dtype=float)
    b=np.asarray(b,dtype=float)
    anchors=U[np.asarray(anchor_indices,dtype=int)]
    N=len(anchors);M=len(poly.X)
    analytic={k:{} for k in range(3)}
    fitted={k:{} for k in range(6)}
    for i,j in PAIRS:
        lo_i,hi_i=np.quantile(U[:,i],quantiles)
        lo_j,hi_j=np.quantile(U[:,j],quantiles)
        ui=np.linspace(lo_i,hi_i,levels)
        uj=np.linspace(lo_j,hi_j,levels)
        grid=np.broadcast_to(anchors[:,None,None,:],
                              (N,levels,levels,4)).copy()
        grid[...,i]=ui[None,:,None]
        grid[...,j]=uj[None,None,:]
        flat=grid.reshape(-1,4)
        poly_values=poly.evaluate(flat).reshape(6,N,levels,levels,M)
        T=(flat-b)@R+b
        eta,p=oracle_module.oracle_outputs(T,poly.X)
        exact_values=np.stack((p,eta,p)).reshape(3,N,levels,levels,M)
        for t in range(3):
            analytic[t][(i,j)]=_interaction_fraction(exact_values[t])
        for t in range(6):
            fitted[t][(i,j)]=_interaction_fraction(poly_values[t])
    result={}
    for src in ("analytic","formula_y","nn_y"):
        for ti,target in enumerate(("choice","logit","probability")):
            item=analytic[ti] if src=="analytic" else fitted[ti+(3 if src=="nn_y" else 0)]
            ratios=[];raw=[];mae=[]
            res={}
            for (i,j),(ratio,mse,l1) in item.items():
                res[f"zz_pair_{i}{j}_additivity_gap_loss"]=float(ratio)
                res[f"zz_pair_{i}{j}_interaction_mse_loss"]=float(mse)
                res[f"zz_pair_{i}{j}_interaction_mae_loss"]=float(l1)
                ratios.append(ratio);raw.append(mse);mae.append(l1)
            res["zz_pairwise_additivity_mean_loss"]=float(np.mean(ratios))
            res["zz_pairwise_additivity_max_loss"]=float(np.max(ratios))
            res["zz_pairwise_interaction_mse_mean_loss"]=float(np.mean(raw))
            res["zz_pairwise_interaction_mae_mean_loss"]=float(np.mean(mae))
            result[(src,target)]=res
    return result
