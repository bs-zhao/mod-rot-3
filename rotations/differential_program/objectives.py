"""Full differential modules and axial differential recurrence; no theta inputs.

Blocks T[r,s] have flattened anchor dimension then r symmetric latent indices
and s symmetric standardized-stimulus indices. All pure and mixed entries are
retained. Symmetric tensor packing preserves the full Frobenius metric.
"""
from itertools import combinations_with_replacement, permutations
import math
import numpy as np
import torch
from scipy.optimize import minimize


def sym_basis(k,degree):
    if degree==0:return [()],np.ones((1,1))
    groups=list(combinations_with_replacement(range(k),degree))
    basis=np.zeros((k**degree,len(groups)))
    for col,group in enumerate(groups):
        variants=set(permutations(group))
        for term in variants:basis[np.ravel_multi_index(term,(k,)*degree),col]=1/np.sqrt(len(variants))
    return groups,basis


def rotation(v,k):
    skew=torch.zeros((k,k),dtype=v.dtype)
    i,j=torch.triu_indices(k,k,1)
    skew[i,j]=v;skew[j,i]=-v
    return torch.matrix_exp(skew)


def sym_power(a,basis,degree):
    power=a
    for _ in range(1,degree):power=torch.kron(power.contiguous(),a.contiguous())
    return basis.T@power@basis


class ProgramObjective:
    def __init__(self,blocks,k,d,h=.5,code_floor=.02,hx=None):
        self.k,self.d,self.h=k,d,float(h)
        self.qbasis={r:torch.tensor(sym_basis(k,r)[1],dtype=torch.float64) for r in [1,2,3]}
        self.cov={};self.scales={};self.energy=0.;self.group_count=0
        for (r,s),array in blocks.items():
            _,bs=sym_basis(d,s)
            bq=self.qbasis[r].numpy()
            raw=np.asarray(array,float).reshape(-1,k**r,d**s)
            packed=np.einsum('ap,nab,bt->npt',bq,raw,bs,optimize=True)
            factor=h**r*(h if hx is None else hx)**s/(math.factorial(r)*math.factorial(s))
            packed*=factor
            cov=np.einsum('npx,nqx->xpq',packed,packed,optimize=True)/len(packed)
            self.cov[r,s]=torch.tensor(cov,dtype=torch.float64)
            self.energy+=float(np.einsum('xii->',cov))
            self.group_count+=cov.shape[0]*cov.shape[1]
        self.norm=max(np.sqrt(self.energy),1e-15)
        self.floor=float(code_floor)/np.sqrt(max(self.group_count,1))

    def module_loss(self,a,kind='group_l1'):
        powers={r:sym_power(a,self.qbasis[r],r) for r in self.qbasis}
        strengths=[]
        for (r,s),cov in self.cov.items():
            p=powers[r]
            squared=torch.einsum('ia,xab,ib->xi',p,cov,p)/(self.norm**2)
            # RMS via second moments has ~sqrt(eps) cancellation at exact zero.
            # Smooth below 1e-6 normalized energy, well below the code floor.
            strengths.append((torch.sqrt(torch.clamp(squared,min=0)+1e-12)-1e-6).reshape(-1))
        g=torch.cat(strengths)
        if kind=='group_l1':return g.sum()
        if kind=='module_code':return torch.log1p(g/self.floor).sum()
        raise KeyError(kind)


class RecurrenceObjective:
    def __init__(self,blocks,k,n_subjects,n_x,h=.5):
        self.k=k;self.bases={r:torch.tensor(sym_basis(k,r)[1],dtype=torch.float64) for r in [1,2,3]}
        packed=[];self.slices={};start=0
        self.energy=0.
        for r in [1,2,3]:
            raw=np.asarray(blocks[r,0],float).reshape(n_subjects,n_x,k**r)
            value=(raw@self.bases[r].numpy())*h**r/math.factorial(r)
            packed.append(value);self.slices[r]=slice(start,start+value.shape[-1]);start+=value.shape[-1]
            self.energy+=float(np.sum(value*value)/n_subjects)
        all_values=np.concatenate(packed,axis=-1)
        self.cov=torch.tensor(np.einsum('nxa,nxb->xab',all_values,all_values)/n_subjects,dtype=torch.float64)
        self.total_dim=start;self.energy=max(self.energy,1e-30)
        self.pure={r:[sym_basis(k,r)[0].index((i,)*r) for i in range(k)] for r in [1,2,3]}

    def loss(self,a):
        weights=[]
        for r in [1,2,3]:
            power=sym_power(a,self.bases[r],r)
            pure=power[self.pure[r]]
            sl=self.slices[r]
            weights.append(torch.nn.functional.pad(pure,(sl.start,self.total_dim-sl.stop)))
        w=torch.stack(weights,dim=1)
        gram=torch.einsum('ira,xab,isb->ixrs',w,self.cov,w)
        eig=torch.linalg.eigvalsh(gram)
        # Residual to a rank-one derivative-order recurrence, retaining pure curvature.
        return torch.clamp(eig[...,:-1],min=0).sum()/self.energy


def optimize(score,k,seed=31,starts=4,maxiter=180):
    torch.set_num_threads(1)
    rng=np.random.default_rng(seed);records=[];rotations=[]
    for start in range(starts):
        initial=np.zeros(k*(k-1)//2) if start==0 else rng.normal(0,.6,k*(k-1)//2)
        def objective(v):
            t=torch.tensor(v,dtype=torch.float64,requires_grad=True)
            value=score(rotation(t,k));value.backward()
            return float(value.detach()),t.grad.detach().numpy().copy()
        result=minimize(objective,initial,method='BFGS',jac=True,options={'maxiter':maxiter,'gtol':1e-7})
        a=rotation(torch.tensor(result.x),k).detach().numpy()
        rotations.append(a);records.append(dict(start=start,loss=float(result.fun),success=bool(result.success),nfev=int(result.nfev),message=str(result.message)))
    best=int(np.argmin([r['loss'] for r in records]))
    return rotations[best],records,np.stack(rotations)
