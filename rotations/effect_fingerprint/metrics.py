"""Independent effect-fingerprint rank functional; only frozen-ZXY surrogate derivatives.

The SAME stimulus-profile proxy is evaluated at each R: its derivatives are
rotated analytically. No theta or CR3-specific response logic is available.
Eigenspectrum effective rank is computed PER coordinate axis, so the sum
is NOT the invariant rank of the full multi-axis gradient operator.
"""
from __future__ import annotations
import numpy as np


def _renyi_rank(eigenvalues, mode):
    eig=np.maximum(np.asarray(eigenvalues,dtype=float),0.)
    total=float(eig.sum())
    if total<=1e-14:
        return 0., 0.
    p=eig/total
    if mode=="entropy":
        nonzero=p[p>1e-12]
        rank=float(np.exp(-np.sum(nonzero*np.log(nonzero))))
    elif mode=="participation":
        rank=float(1./np.sum(p*p))
    elif mode=="tail":
        # 1 - largest-mode energy, scaled by total channel dimension
        rank=float(1.+len(eig)*(1.-float(np.max(p))))
    else:
        raise ValueError("rank_metric must be entropy, participation or tail")
    return rank,total


def channel_transform(grad,mode="raw",floor=.025):
    """Derive one R-INDEPENDENT channel metric from all anchor gradients.

    'unitvar' changes the weighting of the original stimulus channels;
    'whiten' changes their metric through a fixed covariance whitening.
    This must never be recomputed for a proposed rotation R.
    """
    n,k,l=grad.shape
    flat=grad.reshape(n*k,l)
    if mode=="raw":
        return np.eye(l)
    if floor<=0:
        raise ValueError("channel_floor must be positive")
    cov=flat.T@flat/max(len(flat),1)
    trace=float(np.trace(cov))/max(l,1)
    ridge=max(trace*float(floor),1e-10)
    if mode=="unitvar":
        return np.diag(1./np.sqrt(np.maximum(np.diag(cov),ridge)))
    if mode=="whiten":
        vals,vec=np.linalg.eigh(cov)
        return (vec*(1./np.sqrt(np.maximum(vals,ridge))))@vec.T
    raise ValueError("channel_mode must be raw, unitvar or whiten")


class FingerprintObjective:
    def __init__(self,proxy,config,anchor=None):
        self.proxy=proxy
        self.config=config
        self.k=proxy.k
        self.metric=str(config.get("rank_metric","entropy"))
        self.channel_mode=str(config.get("channel_mode","raw"))
        self.min_energy=float(config.get("min_axis_energy_fraction",0.))
        self.weak_weight=float(config.get("weak_axis_weight",0.))
        if self.min_energy<0 or self.weak_weight<0:
            raise ValueError("Weak-axis settings must be nonnegative")
        if anchor is None:
            q=proxy.q_train[proxy.fit_ids]
            size=min(len(q),int(config.get("anchor_subjects",160)))
            rng=np.random.default_rng(int(config["seed"])+918)
            anchor=q[rng.choice(len(q),size,replace=False)]
        self.anchor=np.asarray(anchor,float)
        self.grad=np.asarray(proxy.gradients(self.anchor),float)
        if self.grad.ndim!=3 or self.grad.shape[1]!=self.k:
            raise ValueError("Invalid surrogate gradients")
        self.channel_map=channel_transform(self.grad,self.channel_mode,
                                           float(config.get("channel_floor",.025)))
        self.features=np.einsum("nil,lk->nik",self.grad,self.channel_map,optimize=True)
        self.evals=0

    def components(self,R):
        R=np.asarray(R,float)
        if R.shape!=(self.k,self.k):
            raise ValueError("Wrong R dimension")
        rotated=np.einsum("ia,nal->nil",R,self.features,optimize=True)
        ranks=[];energies=[];entropy=[]
        for axis in range(self.k):
            g=rotated[:,axis,:]
            eig=np.linalg.eigvalsh(g.T@g/max(len(g),1))
            rank,energy=_renyi_rank(eig,self.metric)
            ranks.append(rank);energies.append(energy)
            entropy.append(float(max(rank-1.,0.)))
        total=float(sum(energies))
        fractions=np.asarray(energies)/max(total,1e-12)
        # Extra penalty is OPTIONAL: it is off by default. A truly inactive
        # coordinate must not be automatically regarded as a wrong mechanism.
        penalty=float(np.sum(np.maximum(self.min_energy-fractions,0)**2))
        loss=float(np.sum(entropy)+self.weak_weight*penalty)
        return dict(loss=loss,rank_excess=float(sum(entropy)),
                    weak_axis_penalty=penalty,ranks=ranks,
                    axis_gradient_energies=energies,
                    axis_energy_fractions=fractions.tolist())

    def value(self,R):
        self.evals+=1
        return self.components(R)["loss"]
