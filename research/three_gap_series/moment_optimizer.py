"""Differentiable scalar Hessian objectives; no gradients of response functions.

Only gradients of the rotation loss with respect to rotation angles are used.
This accelerates optimization without changing which field defines simplicity.
"""
import itertools
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment


def rotation_torch(angles,d):
    rows=list(torch.eye(d,dtype=angles.dtype,device=angles.device).unbind())
    for angle,(i,j) in zip(angles,itertools.combinations(range(d),2)):
        c,s=torch.cos(angle),torch.sin(angle);ri,rj=rows[i],rows[j]
        rows[i]=c*ri-s*rj;rows[j]=s*ri+c*rj
    return torch.stack(rows)


class MomentObjective:
    def __init__(self,loss):
        self.loss=loss;self.s=loss.spec;self.d=loss.d
        for name in ['cov','ux_moment','xm','common_uu_moment','common_uu_mean','functional_operators','operator_matrices',
                     'subject_ux_moment','subject_uu_moment','subject_scale','uu_moment','activation_second','activation_fourth']:
            value=getattr(loss,name)
            setattr(self,name,None if value is None else torch.tensor(value,dtype=torch.float64))
        self.taylor_tensors=None if loss.taylor_tensors is None else tuple(torch.tensor(t,dtype=torch.float64) for t in loss.taylor_tensors)
        self.rank_tensors=None if loss.rank_tensors is None else tuple(torch.tensor(t,dtype=torch.float64) for t in loss.rank_tensors)
        self.activation_raw=torch.tensor(loss.ux,dtype=torch.float64) if self.s['family']=='activation' and self.s['kind']=='hoyer' else None
        self.orthomax_raw=torch.tensor(loss.ux,dtype=torch.float64) if self.s['family']=='orthomax' and self.s['aggregation']=='local' else None

    def root(self,square):
        stable=torch.sqrt(square+1e-24)
        return stable+(torch.sqrt(square)-stable).detach()

    def score(self,values,kind,axis):
        v=values.abs()
        if kind=='l1':return v.mean()
        if kind=='l2sq':return (v*v).mean()
        if kind=='lp05':return torch.sqrt(v+1e-16).mean()
        if kind.startswith('lp@'):return ((v+1e-16)**float(kind.split('@')[1])).mean()
        if kind=='log01':return torch.log1p(v/.1).mean()
        if kind=='cap01':return torch.clamp(v/.1,max=1).mean()
        if kind=='entropy':
            prob=v/torch.clamp(v.sum(dim=axis,keepdim=True),min=1e-30)
            return (-(prob*torch.log(torch.clamp(prob,min=1e-30))).sum(dim=axis)).mean()
        if kind in ['soft003','soft01'] or kind.startswith('soft@'):
            threshold=.03 if kind=='soft003' else .1 if kind=='soft01' else float(kind.split('@')[1])
            return (v*v/(v*v+threshold**2)).mean()
        count=v.shape[axis];total=v.sum(dim=axis);energy=(v*v).sum(dim=axis)
        if kind=='hoyer':return (total/torch.sqrt(torch.clamp(count*energy,min=1e-30))).mean()
        if kind=='kurtosis':return (-(v**4).sum(dim=axis)/torch.clamp(energy**2,min=1e-30)).mean()
        if kind=='gini':
            ordered=torch.sort(v,dim=axis).values;shape=[1]*v.ndim;shape[axis]=count
            weights=torch.arange(count,0,-1,dtype=v.dtype).reshape(shape)
            result=2*(weights*ordered).sum(dim=axis)/torch.clamp(count*total,min=1e-30)-1/count
            return torch.where(total>1e-15,result,torch.zeros_like(result)).mean()
        raise ValueError(kind)

    def __call__(self,angles):
        r=rotation_torch(angles,self.d);cov=r@self.cov@r.T
        kind=self.s.get('axis_scale','marginal');sd=torch.sqrt(torch.diag(cov))
        if kind=='isotropic':sd=torch.ones(self.d,dtype=r.dtype)*torch.sqrt(torch.trace(self.cov)/self.d)
        elif kind in ['conditional','geometric']:
            conditional=torch.rsqrt(torch.diag(torch.linalg.inv(cov)))
            sd=conditional if kind=='conditional' else torch.sqrt(sd*conditional)
        a=sd[:,None]*r
        s=self.s;family=s['family']
        if family=='orthomax':
            if s['block']=='ux' and s['aggregation']=='local':square=torch.einsum('ia,nap->nip',a,self.orthomax_raw)**2/self.loss.scale**2
            elif s['block']=='ux':square=torch.clamp(torch.einsum('ia,pab,ib->ip',a,self.ux_moment,a),min=0)/self.loss.scale**2
            elif s['block']=='full':
                ux=torch.clamp(torch.einsum('ia,pab,ib->ip',a,self.ux_moment,a),min=0)
                transform=torch.einsum('ia,jb->ijab',a,a).reshape(self.d*self.d,-1)
                uu=torch.clamp((transform@self.common_uu_moment*transform).sum(dim=1),min=0).reshape(self.d,self.d)
                square=torch.cat((uu,ux),dim=-1)/self.loss.common_scale**2
            else:square=torch.clamp(torch.einsum('ia,qab,ib->iq',a,self.functional_operators,a),min=0)/self.loss.functional_scale**2
            if s['kaiser']:square=square/torch.clamp(square.sum(dim=-2,keepdim=True),min=1e-8)
            criterion=(square*square).sum(dim=-1)-s['gamma']/square.shape[-1]*square.sum(dim=-1)**2
            return -criterion.sum(dim=-1).mean()
        if family=='activation':
            second=torch.clamp(torch.einsum('ia,gpab,ib->gip',a,self.activation_second,a),min=0)/self.loss.activation_scale**2
            if s['kind']=='kurtosis':
                squared=torch.einsum('ia,ib->iab',a,a).reshape(self.d,-1)
                fourth=torch.clamp(torch.einsum('iq,gpqr,ir->gip',squared,self.activation_fourth,squared),min=0)/self.loss.activation_scale**4
                return (second**2/(fourth+s['threshold']**4)).mean()
            value=torch.einsum('ia,nap->nip',a,self.activation_raw).abs()/self.loss.activation_scale
            if s['conditioning']=='pooled':mean=value.mean(dim=0,keepdim=True)
            else:
                blocks=value.reshape(*self.loss.activation_shape,self.d,-1)
                mean=blocks.mean(dim=1 if s['conditioning']=='subject' else 0)
            return (mean**2/(second+s['threshold']**2)).mean()
        if family=='nonlinear_axes':
            transform=torch.einsum('ia,jb->ijab',a,a).reshape(self.d*self.d,-1)
            square=torch.clamp((transform@self.uu_moment*transform).sum(dim=1),min=0).reshape(self.d,self.d)
            own=self.root(torch.diag(square))
            if s['normalization']=='global':denominator=max(np.sqrt(self.loss.uu_energy),1e-12)
            elif s['normalization']=='row':denominator=torch.clamp(self.root(square.sum(dim=1)),min=1e-12)
            else:denominator=torch.clamp(self.root(torch.clamp(torch.einsum('ia,pab,ib->i',a,self.ux_moment,a),min=0)),min=1e-12)
            ratio=own/denominator/s['threshold']
            return (ratio*ratio/(1+ratio*ratio)).mean() if s['score']=='soft' else torch.log1p(ratio).mean()
        if family=='rank':
            hx,hu=self.rank_tensors;ux=torch.einsum('ia,apuv->ipuv',a,hx)
            matrices=ux.reshape(-1,*ux.shape[-2:])
            if s['part']=='edges':
                uu=torch.einsum('ia,abuv,jb->ijuv',a,hu,a);ix=torch.triu_indices(self.d,self.d)
                matrices=torch.cat((uu[ix[0],ix[1]],matrices),dim=0)
            matrices=matrices/self.loss.rank_scale
            gram=matrices.transpose(-2,-1)@matrices if matrices.shape[-1]<=matrices.shape[-2] else matrices@matrices.transpose(-2,-1)
            values=torch.clamp(torch.linalg.eigvalsh(gram),min=0)
            if s['score'].startswith('soft@'):
                threshold=float(s['score'].split('@')[1]);values=values/(values+threshold**2)
            elif s['score']=='log01':values=torch.log1p(values/.1**2)
            else:
                power=1. if s['score']=='schatten1' else .5;eps=1e-4
                values=(values+eps**2)**(power/2)-eps**power
            return values.sum(dim=-1).mean()
        if family=='taylor':
            from taylor_hessian import unique_indices
            first,second,third=self.taylor_tensors
            linear=torch.einsum('ia,xap->xip',a,first)
            quadratic=torch.einsum('ia,xab,jb->xij',a,second,a)
            indices,factors=unique_indices(self.d,2);ix=torch.tensor(indices);factors=torch.tensor(factors)
            values=[self.root((linear*linear).mean(dim=0)).reshape(-1),self.root((quadratic[:,ix[0],ix[1]]**2).mean(dim=0))/factors]
            if s['response_order']==3:
                cubic=torch.einsum('ia,jb,kc,xabc->xijk',a,a,a,third)
                indices,factors=unique_indices(self.d,3);ix=torch.tensor(indices);factors=torch.tensor(factors)
                values.append(self.root((cubic[:,ix[0],ix[1],ix[2]]**2).mean(dim=0))/factors)
            return self.score(torch.cat(values)/self.loss.taylor_scale,s['score'],-1)
        if family in ['subject_ux','subject_edges']:
            ux=self.root(torch.clamp(torch.einsum('ia,spab,ib->sip',a,self.subject_ux_moment,a),min=0))
            if family=='subject_ux':values=ux
            else:
                transform=torch.einsum('ia,jb->ijab',a,a).reshape(self.d*self.d,-1)
                square=torch.clamp(torch.einsum('ia,sab,ib->si',transform,self.subject_uu_moment,transform),min=0).reshape(-1,self.d,self.d)
                indices=torch.triu_indices(self.d,self.d)
                values=torch.cat((self.root(square)[:,indices[0],indices[1]],ux.reshape(len(ux),-1)),dim=1)
            if s['normalization']=='global':values=values/self.loss.common_scale
            else:values=values/self.subject_scale.reshape((-1,)+(1,)*(values.ndim-1))
            return self.score(values,s['score'],-1)
        if family=='operator':
            matrices=torch.einsum('ia,kab,jb->kij',a,self.operator_matrices,a)
            off=matrices[:,~torch.eye(self.d,dtype=torch.bool)]
            if s['score']=='relative':return (off*off).sum()/torch.clamp((matrices*matrices).sum(),min=1e-30)
            return self.score(off,s['score'],-1)
        ux_square=torch.clamp(torch.einsum('ia,pab,ib->ip',a,self.ux_moment,a),min=0)
        ux=self.root(ux_square)
        if family=='ux':
            normalization=s.get('normalization','reference')
            if normalization=='column':
                sd=torch.sqrt(torch.diag(self.cov))
                norm=torch.sqrt(torch.clamp(torch.einsum('a,paa,a->p',sd,self.ux_moment,sd),min=0))
                values=ux/torch.clamp(norm,min=1e-12)
            else:values=ux/(max(np.sqrt(self.loss.ux_energy),1e-12) if normalization=='ux' else self.loss.scale)
        elif family=='functional':
            values=self.root(torch.clamp(torch.einsum('ia,qab,ib->iq',a,self.functional_operators,a),min=0))/self.loss.functional_scale
        elif family in ['full','edge']:
            transform=torch.einsum('ia,jb->ijab',a,a).reshape(self.d*self.d,-1)
            square=torch.clamp((transform@self.common_uu_moment*transform).sum(dim=1),min=0).reshape(self.d,self.d)
            values=torch.cat((self.root(square),ux),dim=-1)/self.loss.common_scale
            if family=='edge':
                indices=torch.triu_indices(self.d,self.d)
                values=torch.cat((values[indices[0],indices[1]],values[:,self.d:].reshape(-1)))
        elif family=='ux_gram':
            if s['stat']=='signed':vectors=a@self.xm;gram=vectors@vectors.T
            else:gram=torch.einsum('ia,pab,jb->ij',a,self.ux_moment,a)
            if s['normalized']:
                norms=torch.sqrt(torch.clamp(torch.diag(gram),min=0)+1e-24)
                gram=gram/torch.clamp(norms[:,None]*norms[None,:],min=1e-20)
            else:gram=gram/self.loss.scale**2
            return (gram[~torch.eye(self.d,dtype=torch.bool)]**2).mean()
        else:raise ValueError(family)
        if s['axis']=='global':values=values.reshape(-1)
        return self.score(values,s['score'],-2 if s['axis']=='column' else -1)

    def scipy(self,angles):
        angles=torch.tensor(angles,dtype=torch.float64,requires_grad=True)
        value=self(angles);gradient=torch.autograd.grad(value,angles)[0]
        if not torch.isfinite(gradient).all():raise ValueError('Nonfinite rotation-objective gradient')
        return float(value.detach()),gradient.detach().numpy()


def optimize(loss,starts,seed,maxiter=180):
    objective=MomentObjective(loss);d=loss.d;n=d*(d-1)//2;rng=np.random.default_rng(seed)
    solutions=[]
    for index in range(starts):
        x0=np.zeros(n) if index==0 else rng.uniform(-np.pi,np.pi,n)
        fitted=minimize(objective.scipy,x0,jac=True,method='L-BFGS-B',
            options=dict(maxiter=maxiter,ftol=1e-12,gtol=1e-7))
        r=rotation_torch(torch.tensor(fitted.x),d).numpy()
        # Evaluate the original NumPy objective, not the smoothed computational
        # square-root implementation, to rank and serialize final rotations.
        value=loss(r);ii,jj=linear_sum_assignment(-abs(r))
        solutions.append(dict(start=index,loss=value,r=r,axis_error=float(np.rad2deg(np.arccos(np.clip(abs(r[ii,jj]).min(),0,1)))),
            converged=bool(fitted.success),iterations=fitted.nit,message=str(fitted.message)))
    return min(solutions,key=lambda row:row['loss']),solutions
