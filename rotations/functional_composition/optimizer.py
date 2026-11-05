"""One multi-component SO(k) optimizer for the entire functional_composition suite."""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from .metrics import combined_scores


def as_rotation(a,k):
    a=np.asarray(a,float)
    if len(a)!=k*(k-1)//2:raise ValueError('wrong number of rotation angles')
    S=np.zeros((k,k))
    i,j=np.triu_indices(k,1)
    S[i,j]=a;S[j,i]=-a
    return expm(S)


class RotationObjective:
    """Normalize the five genuine axis-dependent structural scores at identity."""
    def __init__(self,proxy,config):
        self.proxy=proxy
        self.config=config
        q=proxy.q_train[proxy.fit_ids]
        self.anchor=q[np.random.default_rng(int(config['seed'])+918).choice(
            len(q),size=min(len(q),int(config['anchor_subjects'])),replace=False)]
        self.base=combined_scores(proxy,np.eye(proxy.k),config,self.anchor)
        self.keys=('operator','fingerprint','jointdiag','separation','hypergraph')
        self.weights={key:float(config['weight_'+key]) for key in self.keys}
        if any(not np.isfinite(v) or v<0 for v in self.weights.values()):
            raise ValueError('All structural loss weights must be nonnegative')
        if sum(self.weights.values())<=0:raise ValueError('At least one weight must be positive')
        self.calls=0

    def components(self,R):
        return combined_scores(self.proxy,R,self.config,self.anchor)

    def value(self,R):
        values=self.components(R)
        self.calls+=1
        cost=0.
        for k in self.keys:
            if self.weights[k]==0:continue
            # Fixed identity scale, no angle-dependent renormalization.
            denom=max(abs(self.base[k]),0.25 if k!='jointdiag' else 0.05)
            cost+=self.weights[k]*values[k]/denom
        return float(cost)


def optimize(proxy,config):
    objective=RotationObjective(proxy,config)
    k=proxy.k
    n_angles=k*(k-1)//2
    starts=int(config['starts']);maxiter=int(config['maxiter'])
    if starts<1 or maxiter<1:raise ValueError('starts, maxiter must be positive')
    algorithm=str(config['optimizer'])
    if algorithm not in ('Powell','BFGS','Nelder-Mead'):
        raise ValueError('Unsupported optimizer')
    rng=np.random.default_rng(int(config['seed']))
    candidates=[]
    for start in range(starts):
        guess=np.zeros(n_angles) if start==0 else rng.normal(0,.4,n_angles)
        opt=minimize(lambda a:objective.value(as_rotation(a,k)),guess,
                     method=algorithm,options={'maxiter':maxiter})
        if np.isfinite(opt.fun):candidates.append((float(opt.fun),as_rotation(opt.x,k),opt))
    if not candidates:raise RuntimeError('All optimization starts had nonfinite scores')
    best=min(candidates,key=lambda x:x[0])
    return best[1],dict(loss=best[0],loss_identity=objective.value(np.eye(k)),
                        identity_components=objective.base,
                        best_components=objective.components(best[1]),
                        optimizer=algorithm,starts=starts,maxiter=maxiter,
                        evaluations=objective.calls,converged=bool(best[2].success),
                        message=str(best[2].message),
                        nominal_num_angle_parameters=n_angles),objective