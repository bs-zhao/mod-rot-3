"""Model-specific input equivariance profiles from F0/F1 only.

No CR3 expression or parameter labels are provided to the derivative estimator.
M1 assumes monetary power/scale and probability-odds symmetries and the known
100-unit reference. This is an explicit structural prior, not a generic loss.
"""
import numpy as np
import torch
from scipy.special import expit,logit

def operator_targets(x,gx,eta):
    amount=[0,1,3,4];loss=[1,4];prob=[2,5]
    op_alpha=np.sum(gx[:,:,amount]*(x[:,amount]*np.log(x[:,amount]/100))[None],axis=2)
    op_lambda=np.sum(gx[:,:,loss]*x[:,loss][None],axis=2)
    op_gamma=np.sum(gx[:,:,prob]*(x[:,prob]*(1-x[:,prob])*logit(x[:,prob]))[None],axis=2)
    return np.stack((op_alpha,op_lambda,op_gamma,eta),axis=-1)

def transport_fields(args,c,fn,v,protocol):
    if args.model!='CR3':raise ValueError('Current transport operator bank is CR3-specific')
    if args.estimator not in ['direct','finite_difference']:
        raise ValueError('Current full-gradient transport supports direct or query-only finite differences')
    data=c['data'];sub=data['subject_index'];trial=data['trial_index'];xx=data['x'];d=c['d']
    ix=np.flatnonzero(sub==0);ix=ix[np.argsort(trial[ix])];bank=xx[ix]
    valid=(bank[:,[0,1,3,4]]>1).all(1)&(bank[:,[2,5]]>.03).all(1)&(bank[:,[2,5]]<.97).all(1)
    rng=np.random.default_rng(args.seed+3171)
    stim=rng.choice(np.flatnonzero(valid),min(args.mean_stimuli,int(valid.sum())),replace=False)
    ids=rng.choice(c['splits']['train'],min(args.mean_subjects,len(c['splits']['train'])),replace=False)
    x=bank[stim];xs=(x-protocol['x_mean'])/protocol['x_sd'];n=len(ids);k=len(stim)
    w=np.concatenate((np.repeat(v[ids],k,axis=0),np.tile(xs,(n,1))),axis=1)
    def query(q):
        with torch.no_grad():return np.concatenate([torch.vmap(fn)(b).numpy() for b in torch.tensor(q).split(2048)])
    values=query(w)
    if args.estimator=='direct':
        grad=torch.func.jacrev(fn)
        gradient=np.concatenate([torch.vmap(grad)(b).detach().numpy() for b in torch.tensor(w).split(128)])
    else:
        gradient=np.empty_like(w)
        for j in range(w.shape[1]):
            delta=np.eye(w.shape[1])[j]*args.step
            gradient[:,j]=(query(w+delta)-query(w-delta))/(2*args.step)
    gz=gradient[:,:d].reshape(n,k,d)
    gx=(gradient[:,d:]/protocol['x_sd']).reshape(n,k,6)
    eta=values.reshape(n,k);targets=operator_targets(x,gx,eta)
    probability=expit(eta);weights=(probability*(1-probability)).clip(1e-12)**args.transport_power
    weights/=weights.sum(1,keepdims=True)
    gram=np.einsum('ns,nsa,nsb->nab',weights,gz,gz)
    cross=np.einsum('ns,nsa,nsk->nak',weights,gz,targets)
    energy=np.einsum('ns,nsk,nsk->nk',weights,targets,targets)
    reg=args.transport_ridge*np.trace(gram,axis1=1,axis2=2)/d
    vectors=np.linalg.solve(gram+reg[:,None,None]*np.eye(d),cross)
    vector_grams=np.einsum('nak,nbk->kab',vectors,vectors)/n
    explained=np.einsum('nak,nak->nk',cross,vectors)/np.maximum(energy,1e-30)
    fields=dict(g=gz.reshape(-1,d),full_g=gradient,value=values,
                transport_gram=gram,transport_cross=cross,transport_energy=energy,
                transport_vector_grams=vector_grams)
    info=dict(protocol='Model-specific Lie/equivariance operator profiles; generic F0/F1 derivative inference',
              operators=['sum amount*log(amount/100)*d/damount','sum loss*d/dloss',
                         'sum p*(1-p)*logit(p)*d/dp','output log odds'],
              assumption='Single-factor equivariance under amount-power, loss-scale, probability-odds and output-scale transformations; fixed 100 monetary units',
              mean_subject_ids=ids,bank_trial_ids=stim,domain_fraction=float(valid.mean()),
              capabilities=['F0','F1_Z','F1_X'],hessians_available=False,
              weighted_transport_residual_mean=np.mean(np.maximum(1-explained,0),axis=0),
              gram_eigenvalues=np.linalg.eigvalsh(gram),transport_power=args.transport_power,
              transport_ridge=args.transport_ridge)
    return fields,info
