"""Group empirical-Bayes GP: complete polynomial dictionary plus radial remainder.

The polynomial degree is an approximation resolution, not a supplied data
equation. Groups are anonymous U-coordinate subsets. No baseline labels enter.
"""
import numpy as np
import torch

import run as wave
import conditional_ui_witness as ui


def dictionary(z,degree,radial,reference=None):
    powers=wave.sur.powers_for(z.shape[1],degree)[1:]
    poly=wave.sur.monomials(z,powers,wave.sur.weights_for(powers))
    bits=np.sum((powers>0)*(2**np.arange(z.shape[1]))[None],axis=1)
    raw=np.column_stack((poly,radial));bits=np.r_[bits,np.full(radial.shape[1],2**z.shape[1]-1)]
    masks=np.unique(bits);groups=np.searchsorted(masks,bits)
    fit=raw if reference is None else raw[reference]
    mean=fit.mean(0);phi=raw-mean;scales=[]
    for group in range(len(masks)):
        which=groups==group
        scale=max(float(np.mean(np.sum((fit[:,which]-mean[which])**2,axis=1)))**.5,1e-30)
        phi[:,which]/=scale;scales.append(scale)
    return phi,groups,masks,dict(degree=degree,radial_features=radial.shape[1],features=phi.shape[1],
        group_masks=masks.tolist(),group_scales=scales,feature_mean=mean.tolist(),
        normalization='one unit empirical feature trace per U-subset group')


@torch.no_grad()
def system(gram,cross,yy,groups,variances,noise2,n):
    # cross[task,feature,output mode]; one expected function-ensemble density
    # per task, not independent copies for each retained output mode.
    prior=variances[:,groups];root=torch.sqrt(prior)
    matrix=torch.eye(len(gram),dtype=gram.dtype,device=gram.device)[None]+root[:,:,None]*gram[None]*root[:,None,:]/noise2[:,None,None]
    chol=torch.linalg.cholesky((matrix+matrix.transpose(1,2))/2)
    covariance=root[:,:,None]*torch.cholesky_inverse(chol)*root[:,None,:]
    mean=covariance @ cross/noise2[:,None,None]
    logdet=n*torch.log(noise2)+2*torch.log(torch.diagonal(chol,dim1=1,dim2=2)).sum(1)
    quadratic=(yy-(cross*mean).sum((1,2)))/noise2
    loss=.5*(logdet+quadratic)
    return loss,mean,covariance


@torch.no_grad()
def fit(gram,cross,yy,groups,n,steps=100,initial=None,noise_initial=.1):
    tasks=len(cross);count=int(groups.max())+1
    if initial is None:
        variances=torch.full((tasks,count),1./count,dtype=gram.dtype,device=gram.device)
        noise2=torch.full((tasks,),noise_initial**2,dtype=gram.dtype,device=gram.device)
    else:variances,noise2=(v.clone() for v in initial)
    best=torch.full((tasks,),torch.inf,dtype=gram.dtype,device=gram.device)
    state=(variances.clone(),noise2.clone());history=[]
    for step in range(steps+1):
        loss,mean,covariance=system(gram,cross,yy,groups,variances,noise2,n)
        if not torch.isfinite(loss).all():raise ArithmeticError('Nonfinite group GP evidence')
        improve=loss<best;best[improve]=loss[improve]
        state[0][improve]=variances[improve];state[1][improve]=noise2[improve]
        if step%10==0 or step==steps:
            history.append(dict(step=step,mean_nll_per_subject=float(loss.mean()/n),
                minimum_noise=float(torch.sqrt(noise2).min()),minimum_group_variance=float(variances.min())))
        if step==steps:break
        diagonal=torch.diagonal(covariance,dim1=1,dim2=2)+mean.square().sum(2)
        for group in range(count):variances[:,group]=diagonal[:,groups==group].mean(1)
        expected_error=yy-2*(cross*mean).sum((1,2))+(mean*(gram @ mean)).sum((1,2))
        expected_error+=(covariance*gram[None]).sum((1,2))
        noise2=torch.clamp(expected_error/n,min=1e-8,max=10.)
        variances.clamp_(1e-12,100.)
    return state,best.cpu().numpy(),history


@torch.no_grad()
def component_energies(gram,mean,covariance,groups,masks,n,d):
    point=torch.zeros((len(mean),d),dtype=mean.dtype,device=mean.device)
    posterior=point.clone();by_group=[]
    for g,mask in enumerate(masks):
        ids=torch.nonzero(groups==g,as_tuple=True)[0];block=gram[ids][:,ids]
        beta=mean[:,ids];sigma=covariance[:,ids][:,:,ids]
        energy=(beta*(block @ beta)).sum((1,2))/n
        uncertainty=(sigma*block[None]).sum((1,2))/n
        by_group.append(energy)
        for i in range(d):
            if int(mask)&(1<<i):point[:,i]+=energy;posterior[:,i]+=energy+uncertainty
    return point.cpu().numpy(),posterior.cpu().numpy(),torch.stack(by_group,1).cpu().numpy()
