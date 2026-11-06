"""Function-supported output translation/gain axes; never parameter labels.

An additive-output axis has stimulus-constant derivative. A multiplicative
output axis has derivative proportional to the current whole logit profile.
Both are global directions across states, unlike per-subject radial scales.
"""
import numpy as np
from scipy.linalg import null_space


def fields(eta,g):
    # g is subject x stimulus x latent coordinate.
    n,x,k=g.shape
    active=np.einsum('nxi,nxj->nij',g,g)/x
    add=g-g.mean(axis=1,keepdims=True)
    add_cov=np.einsum('nxi,nxj->nij',add,add)/x
    energy=np.mean(eta*eta,axis=1)
    energy=np.maximum(energy,max(float(np.median(energy))*1e-6,1e-12))
    beta=np.einsum('nx,nxi->ni',eta,g)/x/energy[:,None]
    shape=(g-beta[:,None,:]*eta[:,:,None])/np.sqrt(energy[:,None,None])
    gain_cov=np.einsum('nxi,nxj->nij',shape,shape)/x
    return active,add_cov,gain_cov


def detect(eta,g,relative_cap=.2,gap_multiplier=2.,activity_floor=.05):
    active,add,gain=fields(np.asarray(eta,float),np.asarray(g,float));k=g.shape[-1]
    energy=np.mean(np.asarray(eta,float)**2,axis=1)
    energy=np.maximum(energy,max(float(np.median(energy))*1e-6,1e-12))
    normalized_active=active/energy[:,None,None]
    accepted=[];reports=[]
    for name,cov,denominator in [('output_translation',add,active),('output_gain',gain,normalized_active)]:
        basis=np.eye(k) if not accepted else null_space(np.stack(accepted))
        if basis.shape[1]<2:break
        c=np.einsum('ai,nab,bj->nij',basis,cov,basis)
        metric=basis.T@denominator.mean(0)@basis
        me,mv=np.linalg.eigh(metric)
        inverse_root=(mv/np.sqrt(np.maximum(me,max(float(me.max())*1e-8,1e-15))))@mv.T
        c=np.einsum('ia,nab,bj->nij',inverse_root,c,inverse_root)
        eigen,vectors=np.linalg.eigh(c.mean(0));candidate=basis@inverse_root@vectors[:,0]
        candidate/=np.linalg.norm(candidate)
        half=len(c)//2
        uncertainty=float(np.linalg.norm(c[:half].mean(0)-c[half:].mean(0),2)/2)
        ratio=float(max(eigen[0],0)/max(eigen[1],1e-15))
        gap=float(eigen[1]-eigen[0]);activity=float(candidate@active.mean(0)@candidate/(np.trace(active.mean(0))/k))
        supported=bool(eigen[1]>1e-6 and ratio<=relative_cap and gap>gap_multiplier*uncertainty and activity>=activity_floor)
        reports.append(dict(property=name,eigenvalues=eigen.tolist(),residual_ratio=ratio,eigengap=gap,
                            split_half_uncertainty=uncertainty,activity_ratio=activity,accepted=supported,direction=candidate.tolist()))
        if supported:accepted.append(candidate)
    fixed=np.stack(accepted) if accepted else np.zeros((0,k))
    remainder=null_space(fixed).T if accepted else np.eye(k)
    frame=np.vstack((fixed,remainder))
    if np.linalg.det(frame)<0:frame[-1]*=-1
    return frame,len(accepted),reports
