"""One self-contained effect-fingerprint rotation suite; not a global method."""
import numpy as np
from .surrogate import make_proxy
from .optimizer import optimize

_LAST_FIT=None


def fit(q,x,y,mask,config):
    global _LAST_FIT
    proxy=make_proxy(q,x,y,mask,config)
    R,optim,obj=optimize(proxy,config)
    _LAST_FIT=(proxy,obj,(np.asarray(q),np.asarray(x),np.asarray(y),np.asarray(mask)))
    report=dict(surrogate_validation=proxy.fidelity,fingerprint_optimization=optim,
                warning="Gradient ranks refer to a fitted ZXY response-profile polynomial, not ground-truth CPT derivatives")
    additional={}
    for degree,tensor in proxy.coef.items():
        additional["surrogate_tensor_degree_"+str(degree)]=tensor
    return R,report,additional


def postfit(folder,config):
    if _LAST_FIT is None:
        raise RuntimeError("No freshly fitted fingerprint proxy available")
    from .stability import save_postfit
    proxy,obj,inputs=_LAST_FIT
    save_postfit(folder,proxy,obj,inputs,config)
