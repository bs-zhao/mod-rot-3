"""Three-way LOCAL tensor errors, without conflating regression and NN effects.

For every derivative tensor K separately:
 K_nnfit - K_exact = (K_formulaYfit - K_exact)
                     + (K_nnfit - K_formulaYfit).

Only the first difference measures numeric-Y -> Hessian regression error
under an analytically controlled generator. The second is a difference
between *two fitted functions*, not proof about the NN's true derivatives.
"""
from __future__ import annotations
import numpy as np


def compare_three_arrays(truth,formula,nn,*,field,target):
    a=np.asarray(truth,dtype=float)
    f=np.asarray(formula,dtype=float)
    n=np.asarray(nn,dtype=float)
    if a.shape!=f.shape or a.shape!=n.shape:
        raise ValueError(f"Shape mismatch in {field}")
    valid=np.isfinite(a)&np.isfinite(f)&np.isfinite(n)
    a=a[valid];f=f[valid];n=n[valid]
    if not len(a):return dict(field=field,target=target,n=0)
    estimation=f-a
    nn_extra=n-f
    total=n-a
    if not np.allclose(total,estimation+nn_extra,atol=1e-12,rtol=1e-10):
        raise AssertionError("Error decomposition fails")
    denominator=max(float(np.linalg.norm(a)),1e-15)
    def norm(x):return float(np.linalg.norm(x))
    def corr(x,y):
        xx=x-x.mean();yy=y-y.mean()
        d=norm(xx)*norm(yy)
        return float(xx@yy/d) if d>1e-15 else None
    return dict(
        field=field,target=target,n=int(len(a)),
        formula_fit_relative_error=norm(estimation)/denominator,
        nn_fit_relative_error=norm(total)/denominator,
        nn_fit_minus_formula_fit_relative_error=norm(nn_extra)/denominator,
        formula_to_true_pearson=corr(a,f),
        nn_to_true_pearson=corr(a,n),
        nn_to_formula_pearson=corr(f,n),
        squared_error_cross_term=2*float(estimation@nn_extra),
        binary_choice_sampling_caveat=target=="choice",
    )


def compare_all_sources(exact,fit6,target_idx,*,include_fourth=True):
    """exact is 3-channel; fit6 is 6-channel, same local grid."""
    keys=("zz","zx","zzz","zzx","zxx")
    if include_fourth:keys+=("zzzz","zzxx","zxxx")
    records=[]
    for t,target in enumerate(("choice","logit","probability")):
        for key in keys:
            records.append(compare_three_arrays(
                exact[key][t],fit6[key][t],fit6[key][t+3],
                field=key,target=target))
    return records
