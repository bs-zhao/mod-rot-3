"""Non-neural UXY regressions and explicitly labeled quadratic functional proxies.

The complete U-polynomial proxy is fit ONCE per base coordinate system.
For scans its coefficients/derivatives are transformed exactly, not refitted
at every angle. These derivatives belong to this fitted low-order proxy,
not to the NP or the generating function; grouped-CV fidelity is reported.
"""
from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import cho_factor,cho_solve,block_diag
from sklearn.preprocessing import SplineTransformer
from .dependence import cdist,correlation


def pairs(k):return list(combinations_with_replacement(range(k),2))


def symmetric_basis(k):
    b=np.zeros((k*k,k*(k+1)//2))
    for col,(i,j) in enumerate(pairs(k)):
        b[i*k+j,col]=1. if i==j else 1/np.sqrt(2)
        if i!=j:b[j*k+i,col]=1/np.sqrt(2)
    return b


def poly(v):
    columns=[np.ones(len(v))]+[v[:,i] for i in range(v.shape[1])]
    columns.extend(v[:,i]*v[:,j]*(1 if i==j else np.sqrt(2)) for i,j in pairs(v.shape[1]))
    return np.column_stack(columns)


def poly_derivatives(v):
    n,k=v.shape;p=1+k+k*(k+1)//2
    d=np.zeros((n,k,p));h=np.zeros((k,k,p))
    for i in range(k):d[:,i,1+i]=1
    for index,(i,j) in enumerate(pairs(k),1+k):
        if i==j:d[:,i,index]=2*v[:,i];h[i,i,index]=2
        else:
            d[:,i,index]=np.sqrt(2)*v[:,j];d[:,j,index]=np.sqrt(2)*v[:,i]
            h[i,j,index]=h[j,i,index]=np.sqrt(2)
    return d,h


def ridge_fit(design,y,alpha):
    gram=design.T@design
    gram.flat[::len(gram)+1]+=alpha
    # Penalizing every coordinate, including constant, makes the complete
    # polynomial Euclidean penalty exactly invariant under orthogonal U rotations.
    return cho_solve(cho_factor(gram,lower=True,check_finite=False),design.T@y,check_finite=False)


def cv_predict(design,y,split,alpha):
    prediction=np.zeros_like(y)
    for tr,te in split:prediction[te]=design[te]@ridge_fit(design[tr],y[tr],alpha)
    return prediction


def r2(y,pred):return 1-np.sum((y-pred)**2,axis=0)/np.maximum(np.sum((y-y.mean(0))**2,axis=0),1e-20)


class Regressions:
    def __init__(self,u,x,y,split,cfg):
        self.u,self.x,self.split,self.cfg=u,x,split,cfg
        self.y=(y-y.mean(0))/np.maximum(y.std(0),1e-10)
        self.n,self.k=u.shape;self.ny=y.shape[1]
        self.px=poly(x);self.pu=poly(u)
        design=(self.pu[:,:,None]*self.px[:,None,:]).reshape(len(u),-1)
        self.coefficient=ridge_fit(design,self.y,cfg['ridge']).reshape(self.pu.shape[1],self.px.shape[1],-1)
        self.quad_score=r2(self.y,cv_predict(design,self.y,split,cfg['ridge']))
        linear=(self.pu[:,:1+self.k,None]*self.px[:,None,:]).reshape(len(u),-1)
        self.linear_score=r2(self.y,cv_predict(linear,self.y,split,cfg['ridge']))
        self.linear_coefficient=ridge_fit(linear,self.y,cfg['ridge']).reshape(1+self.k,self.px.shape[1],-1)
        du,hu=poly_derivatives(u);dx,_=poly_derivatives(x)
        self.gradient=np.einsum('nip,pqy,nq->niy',du,self.coefficient,self.px,optimize=True)
        self.hessian=np.einsum('ijp,pqy,nq->nijy',hu,self.coefficient,self.px,optimize=True)
        self.mixed=np.einsum('nip,pqy,njq->nijy',du,self.coefficient,dx,optimize=True)
        self.xgram=self.px.T@self.px/len(x)
        # Common observed X for a diagnostic, whole-proxy effect fingerprint.
        ids=np.random.default_rng(cfg['seed']+773).choice(len(x),min(len(x),cfg['profile_x_points']),replace=False)
        px=poly(x[ids]);unique_u=np.unique(u,axis=0)
        du0,_=poly_derivatives(unique_u)
        self.profile=np.einsum('nip,pqy,tq->nity',du0,self.coefficient,px,optimize=True)
        self.spline_x,self.x_blocks=self.splines(x)
        self.x_only_pred=cv_predict(self.spline_x,self.y,split,cfg['ridge'])
        # Full Euclidean RBF prediction control. One fixed orthogonal U rotation
        # leaves the kernel and all predictions unchanged.
        d2=cdist(np.column_stack([u,x]),np.column_stack([u,x]))**2
        bandwidth=max(np.median(d2[d2>0]),1e-12)
        kernel=np.exp(-d2/bandwidth);pred=np.zeros_like(self.y)
        for tr,te in split:
            kk=kernel[np.ix_(tr,tr)]+cfg['ridge']*np.eye(len(tr))
            pred[te]=kernel[np.ix_(te,tr)]@cho_solve(cho_factor(kk,check_finite=False),self.y[tr],check_finite=False)
        self.rbf_score=r2(self.y,pred)

    def splines(self,v):
        blocks=[];indices=[];start=1
        for j in range(v.shape[1]):
            block=SplineTransformer(n_knots=self.cfg['spline_knots'],degree=3,include_bias=False).fit_transform(v[:,j,None])
            blocks.append(block);indices.append(np.arange(start,start+block.shape[1]));start+=block.shape[1]
        return np.column_stack([np.ones(len(v))]+blocks),indices

    def gam(self,u):
        # Knot fitting is unsupervised in the descriptive sample; all supervised
        # ridge prediction and variable deletion use held-out subject folds.
        su,blocks=self.splines(u)
        design=np.column_stack([self.spline_x,su[:,1:]])
        offset=self.spline_x.shape[1]-1
        pred=cv_predict(design,self.y,self.split,self.cfg['ridge'])
        sf=np.sum((self.y-pred)**2,axis=0)
        total=np.sum((self.y-self.y.mean(0))**2,axis=0)
        partial=[];semipartial=[]
        for block in blocks:
            keep=np.ones(design.shape[1],bool);keep[block+offset]=False
            reduced=cv_predict(design[:,keep],self.y,self.split,self.cfg['ridge'])
            sr=np.sum((self.y-reduced)**2,axis=0)
            partial.append((sr-sf)/np.maximum(sr,1e-20))
            semipartial.append((sr-sf)/np.maximum(total,1e-20))
        # Squared-error prediction, also for binary Y; not a logistic GAM.
        result={'gam_cv_r2':1-sf/total,
                'gam_increment_over_X_cv_r2':(np.sum((self.y-self.x_only_pred)**2,axis=0)-sf)/total}
        axis={'gam_partial_cv_r2':np.asarray(partial),'gam_semipartial_cv_r2':np.asarray(semipartial)}
        for name,value in axis.items():
            result[name+'__axis_sum']=value.sum(0)
            result[name+'__axis_min']=value.min(0)
        return result,axis

    def compute(self,a,u):
        k=self.k;b=symmetric_basis(k)
        power=b.T@np.kron(a,a)@b
        transform=block_diag(np.ones((1,1)),a,power)
        c=np.einsum('ip,pqy->iqy',transform,self.coefficient,optimize=True)
        energy=np.einsum('iqy,qr,iry->iy',c,self.xgram,c,optimize=True)
        energy=np.maximum(energy,0.)
        strength=np.sqrt(energy[1:]+1e-20)
        denominator=np.sqrt(energy[1:].sum(0)).clip(1e-15)
        g=strength/denominator
        floor=self.cfg['module_floor']/np.sqrt(len(g))
        derivative=np.einsum('ij,njy->niy',a,self.gradient,optimize=True)
        hessian=np.einsum('ia,naby,jb->nijy',a,self.hessian,a,optimize=True)
        mixed=np.einsum('ia,nady->nidy',a,self.mixed,optimize=True)
        he=np.mean(hessian*hessian,axis=0);me=np.mean(mixed*mixed,axis=0)
        diag=np.einsum('iiy->y',he);total=he.sum((0,1)).clip(1e-20)
        sensitivity=np.mean(derivative*derivative,axis=0)
        normalized=sensitivity/np.maximum(sensitivity.sum(0),1e-20)
        xm=me.transpose(1,0,2);p=xm/np.maximum(xm.sum(1,keepdims=True),1e-20)
        result={'functional_linear_cv_r2_control':self.linear_score,
                'functional_quadratic_cv_r2_control':self.quad_score,
                'full_rbf_cv_r2_control':self.rbf_score,
                'proxy_module_code':np.log1p(g/floor).sum(0),
                'proxy_module_group_l1':g.sum(0),
                'proxy_linear_U_module_code':np.log1p(g[:k]/floor).sum(0),
                'proxy_quadratic_U_module_code':np.log1p(g[k:]/floor).sum(0),
                'proxy_hessian_offdiag_fraction':(total-diag)/total,
                'proxy_hessian_diagonal_fraction':diag/total,
                'proxy_hessian_group_l1':np.sqrt(he+1e-20).sum((0,1))/np.sqrt(total),
                'proxy_hessian_energy_control':total,
                'proxy_gradient_energy_control':sensitivity.sum(0),
                'proxy_axis_sensitivity_entropy':-np.sum(normalized*np.log(normalized+1e-30),axis=0)/np.log(k),
                'proxy_XU_energy_control':me.sum((0,1)),
                'proxy_XU_group_l1':np.sqrt(me+1e-20).sum((0,1))/np.sqrt(me.sum((0,1))).clip(1e-15),
                'proxy_XU_channel_entropy':np.mean(-np.sum(p*np.log(p+1e-30),axis=1)/np.log(k),axis=0),
                'proxy_XU_channel_geomin':np.mean(np.exp(np.log(p+1e-6).mean(1)),axis=0)}
        profile=np.einsum('ij,njty->nity',a,self.profile,optimize=True)
        ranks=[]
        for axis in range(k):
            values=profile[:,axis]
            gram=np.einsum('nty,nsy->yts',values,values,optimize=True)
            eigen=np.linalg.eigvalsh(gram).clip(0)
            probabilities=eigen/eigen.sum(1,keepdims=True).clip(1e-20)
            ranks.append(np.exp(-np.sum(probabilities*np.log(probabilities+1e-30),axis=1)))
        ranks=np.array(ranks)
        result['proxy_effect_profile_effective_rank_mean']=ranks.mean(0)
        gam,axis=self.gam(u)
        axis['proxy_effect_profile_effective_rank']=ranks
        axis['proxy_axis_sensitivity_energy']=sensitivity
        result.update(gam)
        return result,axis
