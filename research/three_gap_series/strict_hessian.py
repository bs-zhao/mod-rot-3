"""Model-blind M1/M3 catalogue: Hessian blocks and their variation only.

No model names, labels, cognitive templates, gradients or response values.
Every candidate ID has the same formula and constants in every dimension.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

CATALOG={}


def register(name,**spec):
    CATALOG['hs_'+name]=dict(order=2,**spec)


for stat in ['rms','sd','signed','local']:
    for axis in ['row','column']:
        for k in [1,2]:
            for score in ['tail1','tail2sq','relative_tail2']:
                register(f'ux_{stat}_{axis}_{score}_k{k}',family='ux',stat=stat,axis=axis,score=score,k=k)
    for score in ['l1','lp05','soft003','soft01','cap01','log01','entropy']:
        register(f'ux_{stat}_{score}',family='ux',stat=stat,axis='all',score=score)
for part in ['self','off','all']:
    for stat in ['rms','sd','signed','local']:
        for score in ['l1','lp05','soft01','log01']:
            register(f'uu_{part}_{stat}_{score}',family='uu',part=part,stat=stat,score=score)
for degree in [1,2]:
    for score in ['l1','lp05','soft003','soft01','log01','tail1']:
        register(f'ux_profile{degree}_{score}',family='profile',degree=degree,score=score,axis='column',k=1)
for ux in ['rms','signed','profile1','profile2']:
    for part in ['self','off']:
        for weight in [.01,.1,1.,10.,100.]:
            register(f'joint_{ux}_{part}_w{weight:g}',family='joint',ux=ux,part=part,weight=weight,score='l1')
for ux in ['rms','profile1','profile2']:
    for kind in ['self','mixed']:
        for weight in [.01,.1,1.,10.]:
            register(f'joint_{ux}_third{kind}_w{weight:g}',family='third_joint',ux=ux,kind=kind,weight=weight,score='l1')
            CATALOG['hs_'+f'joint_{ux}_third{kind}_w{weight:g}']['order']=3


def penalty(values,score,axis=-1,k=1):
    v=np.abs(values)
    if score=='l1':return float(v.mean())
    if score=='lp05':return float(np.sqrt(v+1e-16).mean())
    if score=='soft003':return float(np.mean(v*v/(v*v+.03**2)))
    if score=='soft01':return float(np.mean(v*v/(v*v+.1**2)))
    if score=='cap01':return float(np.mean(np.minimum(v/.1,1)))
    if score=='log01':return float(np.mean(np.log1p(v/.1)))
    if score=='entropy':
        prob=v/np.maximum(v.sum(axis=axis,keepdims=True),1e-30)
        return float(np.mean(-np.sum(prob*np.log(np.maximum(prob,1e-30)),axis=axis)))
    sorted_values=np.sort(v,axis=axis)
    size=v.shape[axis]
    if k>=size:raise ValueError('Tail k must leave at least one residual in every model')
    tail=np.take(sorted_values,np.arange(size-k),axis=axis)
    if score=='tail1':return float(np.sum(tail,axis=axis).mean())
    energy=np.sum(tail*tail,axis=axis)
    if score=='tail2sq':return float(energy.mean())
    if score=='relative_tail2':return float(np.sqrt(energy/np.maximum(np.sum(v*v,axis=axis),1e-30)).mean())
    raise ValueError(score)


class HessianLoss:
    def __init__(self,fields,covariance,d,metric):
        if 'h' not in fields:raise ValueError('Hessian required, missing F2 cannot be treated as zero')
        self.spec=CATALOG[metric];self.metric=metric;self.d=d;self.cov=covariance
        h=fields['h'];self.n=len(h);self.uu=h[:,:d,:d];self.ux=h[:,:d,d:]
        self.um=self.uu.mean(0);self.xm=self.ux.mean(0)
        flat=self.uu.reshape(self.n,-1);self.uu_moment=flat.T@flat/self.n
        self.ux_moment=np.einsum('nap,nbp->pab',self.ux,self.ux,optimize=True)/self.n
        sd=np.sqrt(np.diag(covariance))
        u=self.uu*sd[None,:,None]*sd[None,None,:];x=self.ux*sd[None,:,None]
        self.scale=max(float(np.sqrt(np.mean(np.sum(u*u,axis=(1,2))+np.sum(x*x,axis=(1,2))))),1e-12)
        self.third=fields.get('third')
        if self.spec['order']==3 and self.third is None:raise ValueError('Third derivative of Hessian required')
        self.profiles={}
        for degree in [1,2]:
            phi=PolynomialFeatures(degree).fit_transform(fields['strict_x'])
            ridge=np.eye(phi.shape[1])*1e-8;ridge[0,0]=0
            coeff=np.linalg.solve(phi.T@phi/self.n+ridge,phi.T@self.ux.reshape(self.n,-1)/self.n)
            self.profiles[degree]=coeff.reshape(-1,d,self.ux.shape[2]).transpose(1,0,2).reshape(d,-1)

    def _ux(self,a,stat):
        if stat.startswith('profile'):return (a@self.profiles[int(stat[-1])])/self.scale
        if stat=='local':return np.einsum('ia,nap->nip',a,self.ux,optimize=True)/self.scale
        mean=a@self.xm
        if stat=='signed':return mean/self.scale
        square=np.maximum(np.einsum('ia,pab,ib->ip',a,self.ux_moment,a,optimize=True),0)
        if stat=='sd':square=np.maximum(square-mean*mean,0)
        return np.sqrt(square)/self.scale

    def _uu(self,a,stat,part):
        if stat=='local':mat=np.einsum('ia,nab,jb->nij',a,self.uu,a,optimize=True)/self.scale
        else:
            mean=a@self.um@a.T
            if stat=='signed':mat=mean/self.scale
            else:
                transform=np.einsum('ia,jb->ijab',a,a).reshape(self.d*self.d,-1)
                square=np.maximum(np.einsum('ia,ab,ib->i',transform,self.uu_moment,transform,optimize=True),0).reshape(self.d,self.d)
                if stat=='sd':square=np.maximum(square-mean*mean,0)
                mat=np.sqrt(square)/self.scale
        if part=='self':return np.diagonal(mat,axis1=-2,axis2=-1)
        if part=='off':return mat[...,~np.eye(self.d,dtype=bool)]
        return mat

    def __call__(self,r):
        sd=np.sqrt(np.diag(r@self.cov@r.T))
        a=sd[:,None]*np.linalg.inv(r).T
        s=self.spec;family=s['family']
        if family=='ux':
            v=self._ux(a,s['stat']);axis=-1 if s['axis']!='column' else -2
            return penalty(v,s['score'],axis,s.get('k',1))
        if family=='profile':return penalty(self._ux(a,f'profile{s["degree"]}'),s['score'],-2,s.get('k',1))
        if family=='uu':return penalty(self._uu(a,s['stat'],s['part']),s['score'])
        ux=penalty(self._ux(a,s['ux']),'l1')
        if family=='joint':return ux+s['weight']*penalty(self._uu(a,'rms',s['part']),'l1')
        if family=='third_joint':
            t=self.third[:,:self.d,:self.d,:self.d]
            transformed=np.einsum('ia,jb,kc,nabc->nijk',a,a,a,t,optimize=True)/self.scale
            if s['kind']=='self':v=transformed[:,np.arange(self.d),np.arange(self.d),np.arange(self.d)]
            else:
                mask=np.ones((self.d,)*3,dtype=bool);mask[np.arange(self.d),np.arange(self.d),np.arange(self.d)]=False
                v=transformed[:,mask]
            return ux+s['weight']*penalty(v,'l1')
        raise ValueError(family)

    def describe(self,r):
        sd=np.sqrt(np.diag(r@self.cov@r.T));a=sd[:,None]*np.linalg.inv(r).T
        return dict(category='Hessian-only, model-blind M1/M3',specification=self.spec,
                    fixed_block_scale=self.scale,uu_diag_rms=self._uu(a,'rms','self'),
                    uu_off_rms=self._uu(a,'rms','off'),ux_rms=self._ux(a,'rms'),
                    ux_signed=self._ux(a,'signed'),loss=self(r))
