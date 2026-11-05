"""A complete, independent rotation suite studying compositional functional complexity."""
from __future__ import annotations
import numpy as np

VERSION='1.0.0'
_LAST_FIT=None


def fit(q,x,y,mask,config):
    """Run a single joint multi-metric unsupervised rotation experiment.

    q,x,y,mask ONLY; theta and the known synthetic simulator are unavailable.
    """
    from .surrogate import make_proxy
    from .optimizer import optimize
    from .ablation import fit_ablations
    global _LAST_FIT
    proxy=make_proxy(q,x,y,mask,config)
    R,optimization,objective=optimize(proxy,config)
    ablations=fit_ablations(proxy,config,R)
    if not np.allclose(R@R.T,np.eye(len(R)),atol=1e-7):
        raise RuntimeError('Optimizer produced nonorthogonal matrix')
    _LAST_FIT=(proxy,objective,ablations)
    info={'surrogate_validation':proxy.fidelity,'structural_optimization':optimization,
          'ablation_objectives':list(ablations['optimization']),
          'mdl20_candidate_winner':ablations['mdl_winner'],
          'caution':'Rotation is fitted to a finite-degree response-profile surrogate, NOT the true conditional logit or oracle formula'}
    arrays={f'polynomial_tensor_degree{d}':arr for d,arr in proxy.coef.items()}
    arrays['fit_profile_coefficients']=proxy.h_train
    return R,info,arrays


def postfit(folder,config):
    if _LAST_FIT is None:
        raise RuntimeError('No fitted response-profile proxy in this process')
    from .diagnostics import save_postfit
    from .ablation import save_ablations
    from pathlib import Path
    with np.load(Path(folder)/'rotation__matrix.npz',allow_pickle=False) as file:
        R=np.asarray(file['R'])
    proxy,objective,ablations=_LAST_FIT
    save_postfit(folder,proxy,objective,R,config)
    save_ablations(folder,proxy,ablations,R,config)