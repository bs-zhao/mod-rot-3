"""Conditional coalition response profiles and efficient Shapley function shares.

Conditional means are numerical polynomial/radial regressions, not supplied
response equations. The full coalition is the observed centered response;
every coalition's prediction is shared across all Shapley increments.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

import conditional_ux as conditional
import conditional_ui_witness as ui


def shares_from_coalitions(values,d):
    weights=np.array([1/(d*math.comb(d-1,k)) for k in range(d)])
    first=next(iter(values.values()));shares=np.zeros((d,*first.shape),dtype=float)
    for mask,value in values.items():
        k=int(mask).bit_count()
        for i in range(d):
            coefficient=weights[k-1] if mask&(1<<i) else -weights[k]
            shares[i]+=coefficient*value
    return shares


def compress_targets(targets,tolerance=1e-10):
    combined=np.column_stack(targets);left,singular,right=np.linalg.svd(combined,full_matrices=False)
    total=float(np.sum(singular**2))
    rank=min(int(np.searchsorted(np.cumsum(singular**2)/total,1-tolerance))+1,len(singular))
    latent=left[:,:rank]*singular[:rank];projection=right[:rank];slices=[];start=0;errors=[]
    for target in targets:
        end=start+target.shape[1];slices.append((start,end))
        error=float(np.sum((latent@projection[:,start:end]-target)**2)/max(np.sum(target*target),1e-30))
        errors.append(error);start=end
    if max(errors)>1e-8:raise ArithmeticError('Shared response compression exceeded its per-X energy budget')
    return latent,projection,slices,dict(latent_rank=rank,original_modes=combined.shape[1],
        global_relative_squared_tolerance=tolerance,per_task_relative_squared_errors=errors,
        maximum_task_relative_squared_error=max(errors))


class CoalitionModels:
    def __init__(self,z,target,output,degree=3,centers=32,seed=471,device='cuda'):
        self.z,self.target,self.output=z,target,Path(output)
        self.degree,self.centers,self.seed,self.device=degree,centers,seed,device
        self.models={};self.audits=[];self.output.mkdir(parents=True,exist_ok=True)

    def design(self,basis,lengthscale):
        return conditional.subspace_design(self.z,basis,self.degree,self.centers,lengthscale,self.seed)

    @torch.no_grad()
    def fit(self,basis):
        projected=self.z@basis
        _,kernel_audit=ui.nystrom(projected,self.centers,self.seed)
        design,audit=self.design(basis,kernel_audit['lengthscale'])
        x=torch.tensor(design,dtype=torch.float64,device=self.device)
        k=audit['linear_count'];affine,nonlinear=x[:,:k],x[:,k:]
        inverse=torch.linalg.pinv(affine,rtol=1e-10)
        correction=inverse@nonlinear;residual=nonlinear-affine@correction
        affine_coef=inverse@self.target;response=self.target-affine@affine_coef
        gram=residual.T@residual;cross=residual.T@response
        eigen,axes=torch.linalg.eigh((gram+gram.T)/2);eigen=torch.clamp(eigen,min=0.)
        scale=max(float(torch.trace(gram))/max(len(gram),1),1e-12)
        rhs=axes.T@cross;normalizer=float(response.square().sum());n=len(self.z)
        options=[];coefficients=[]
        for ridge in (1e-8,1e-6,1e-4,.01,.1):
            coefficient=axes@(rhs/(eigen[:,None]+ridge*scale))
            error=normalizer-2*float((cross*coefficient).sum())+float((coefficient*(gram@coefficient)).sum())
            if error < -1e-7*max(normalizer,1.):raise ArithmeticError('Negative coalition regression risk beyond roundoff')
            error=max(error,0.);df=k+float(torch.sum(eigen/(eigen+ridge*scale)))
            options.append(dict(ridge=ridge,effective_degrees_of_freedom=df,training_function_mse=error/n,
                GCV=error/n/max(1-df/n,1e-8)**2))
            coefficients.append(coefficient)
        selected=int(np.argmin([o['GCV'] for o in options]));nonlinear_coef=coefficients[selected]
        coefficient=torch.cat((affine_coef-correction@nonlinear_coef,nonlinear_coef),dim=0)
        index=len(self.models)
        result=dict(basis=basis.copy(),lengthscale=kernel_audit['lengthscale'],coefficient=coefficient.cpu().numpy(),
            index=index,selected=selected,options=options,design_audit=audit)
        np.savez_compressed(self.output/f'model{index:05d}.npz',basis=basis,lengthscale=result['lengthscale'],
            coefficient=result['coefficient'],selected_ridge=options[selected]['ridge'])
        self.audits.append(dict(index=index,input_dimension=basis.shape[1],selected_index=selected,
            options=options,kernel_audit=kernel_audit,design_audit=audit))
        return result

    @torch.no_grad()
    def predict(self,axes):
        basis=np.linalg.qr(np.asarray(axes))[0][:,:axes.shape[1]]
        projector=basis@basis.T
        canonical=np.round(projector,6)
        canonical[canonical==0.]=0.  # canonicalize signed zero before hashing
        key=(basis.shape[1],hashlib.sha256(canonical.tobytes()).hexdigest())
        if key in self.models and np.max(np.abs(self.models[key]['basis']@self.models[key]['basis'].T-projector))>1e-6:
            key=(basis.shape[1],hashlib.sha256(projector.tobytes()).hexdigest())
        if key not in self.models:self.models[key]=self.fit(basis)
        model=self.models[key]
        design,_=self.design(model['basis'],model['lengthscale'])
        return torch.tensor(design,dtype=torch.float64,device=self.device)@torch.tensor(model['coefficient'],dtype=torch.float64,device=self.device)

    @torch.no_grad()
    def frame_shares(self,q):
        d=q.shape[1];weights=[1/(d*math.comb(d-1,k)) for k in range(d)]
        shares=self.target[None].expand(d,-1,-1).clone()/d
        for mask in range(1,2**d-1):
            selected=[i for i in range(d) if mask&(1<<i)];prediction=self.predict(q[:,selected]);k=len(selected)
            for i in range(d):shares[i]+=prediction*(weights[k-1] if mask&(1<<i) else -weights[k])
        error=float((shares.sum(0)-self.target).square().sum()/self.target.square().sum().clamp_min(1e-30))
        if error>1e-18:raise ArithmeticError('Shapley function shares failed their observed-response efficiency identity')
        return shares,error
