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

def flow_input(w,epsilon,kind,d,x_mean,x_sd):
    x=w[:,d:]*x_sd+x_mean
    x=torch.where(abs(x)<1e-10,torch.zeros_like(x),x)
    scale=torch.exp(epsilon);columns=[]
    for j in range(6):
        value=x[:,j]
        if kind==0 and j in [0,1,3,4]:
            value=torch.where(value>0,100*torch.exp(scale*torch.log((value/100).clamp_min(1e-30))),torch.zeros_like(value))
        elif kind==1 and j in [1,4]:value=value*scale
        elif kind==2 and j in [2,5]:value=torch.sigmoid(scale*torch.logit(value))
        columns.append(value)
    new_x=torch.stack(columns,dim=1)
    return torch.cat((w[:,:d],(new_x-x_mean)/x_sd),dim=1)

def transport_fields(args,c,fn,v,protocol):
    if args.estimator not in ['direct','finite_difference']:
        raise ValueError('Current full-gradient transport supports direct or query-only finite differences')
    data=c['data'];sub=data['subject_index'];trial=data['trial_index'];xx=data['x'];d=c['d']
    rng=np.random.default_rng(args.seed+3171);cr3=args.model=='CR3'
    if cr3:
        ix=np.flatnonzero(sub==0);ix=ix[np.argsort(trial[ix])];bank=xx[ix]
        valid=(bank[:,[2,5]]>.03).all(1)&(bank[:,[2,5]]<.97).all(1)
    else:
        pool=np.flatnonzero(np.isin(sub,c['splits']['train']))
        ix=rng.choice(pool,min(args.mean_stimuli,len(pool)),replace=False);bank=xx[ix]
        valid=np.ones(len(bank),bool)
    # The same support rule is applied to direct/FD, so changing h does not
    # confound their comparison with illegal negative amounts/probabilities.
    margin=1.05*args.step*protocol['x_sd']
    if cr3 and args.transport_access=='partial':
        valid&=(bank[:,[0,1,3,4]]>1).all(1)
        valid&=(bank[:,[0,1,3,4]]>margin[[0,1,3,4]]).all(1)
        valid&=(bank[:,[2,5]]>margin[[2,5]]).all(1)&(bank[:,[2,5]]<1-margin[[2,5]]).all(1)
    if valid.sum()<d+3:raise ValueError('Too few trials support symmetric finite-scale queries')
    stim=rng.choice(np.flatnonzero(valid),min(args.mean_stimuli,int(valid.sum())),replace=False) if cr3 else np.arange(len(bank))
    ids=rng.choice(c['splits']['train'],min(args.mean_subjects,len(c['splits']['train'])),replace=False)
    x=bank[stim];xs=(x-protocol['x_mean'])/protocol['x_sd'];n=len(ids);k=len(stim)
    w=np.concatenate((np.repeat(v[ids],k,axis=0),np.tile(xs,(n,1))),axis=1)
    def query(q):
        with torch.no_grad():return np.concatenate([torch.vmap(fn)(b).numpy() for b in torch.tensor(q).split(2048)])
    values=query(w)
    if args.estimator=='direct':
        if cr3 and args.transport_access=='partial':
            grad=torch.func.jacrev(fn)
            gradient=np.concatenate([torch.vmap(grad)(b).detach().numpy() for b in torch.tensor(w).split(128)])
        else:
            def latent_fn(z,x):return fn(torch.cat((z,x)))
            grad=torch.func.jacrev(latent_fn,argnums=0)
            gradient=np.concatenate([torch.vmap(grad)(b[:,:d],b[:,d:]).detach().numpy() for b in torch.tensor(w).split(128)])
    else:
        width=w.shape[1] if cr3 and args.transport_access=='partial' else d
        gradient=np.empty((len(w),width))
        for j in range(width):
            delta=np.eye(w.shape[1])[j]*args.step
            gradient[:,j]=(query(w+delta)-query(w-delta))/(2*args.step)
    gz=gradient[:,:d].reshape(n,k,d)
    eta=values.reshape(n,k)
    if not cr3:
        from profile_tensor import stimulus_library_numpy
        library=stimulus_library_numpy(args.model,x)
        targets=np.broadcast_to(library,(n,*library.shape)).copy()
    elif args.transport_access=='partial':
        gx=(gradient[:,d:]/protocol['x_sd']).reshape(n,k,6)
        targets=operator_targets(x,gx,eta)
    else:
        xm=torch.tensor(protocol['x_mean']);sx=torch.tensor(protocol['x_sd']);wt=torch.tensor(w)
        effects=[]
        for kind in range(3):
            if args.estimator=='direct':
                parts=[]
                for batch in wt.split(128):
                    def along(e):return torch.vmap(fn)(flow_input(batch,e,kind,d,xm,sx))
                    _,tangent=torch.func.jvp(along,(torch.tensor(0.,dtype=torch.float64),),(torch.tensor(1.,dtype=torch.float64),))
                    parts.append(tangent.detach().numpy())
                effects.append(np.concatenate(parts))
            else:
                plus=flow_input(wt,torch.tensor(args.step,dtype=torch.float64),kind,d,xm,sx).numpy()
                minus=flow_input(wt,torch.tensor(-args.step,dtype=torch.float64),kind,d,xm,sx).numpy()
                effects.append((query(plus)-query(minus))/(2*args.step))
        targets=np.stack([*effects,values],axis=-1).reshape(n,k,4)
    probability=expit(eta);weights=(probability*(1-probability)).clip(1e-12)**args.transport_power
    weights/=weights.sum(1,keepdims=True)
    gram=np.einsum('ns,nsa,nsb->nab',weights,gz,gz)
    cross=np.einsum('ns,nsa,nsk->nak',weights,gz,targets)
    energy=np.einsum('ns,nsk,nsk->nk',weights,targets,targets)
    reg=args.transport_ridge*np.trace(gram,axis1=1,axis2=2)/d
    vectors=np.linalg.solve(gram+reg[:,None,None]*np.eye(d),cross)
    vector_grams=np.einsum('nak,nbk->kab',vectors,vectors)/n
    explained=np.einsum('nak,nak->nk',cross,vectors)/np.maximum(energy,1e-30)
    residual=np.mean(np.maximum(1-explained,0),axis=0)
    selected=np.argsort(residual)[:d] if targets.shape[2]>d else np.arange(targets.shape[2])
    fields=dict(g=gz.reshape(-1,d),full_g=gradient,value=values,
                transport_gram=gram,transport_cross=cross[:,:,selected],transport_energy=energy[:,selected],
                transport_vector_grams=vector_grams[selected])
    info=dict(protocol='Model-specific Lie/equivariance operator profiles; generic F0/F1 derivative inference',
              operators=(['sum amount*log(amount/100)*d/damount','sum loss*d/dloss',
                         'sum p*(1-p)*logit(p)*d/dp','output log odds'] if cr3 else ['Output translations by all raw degree<=2 stimulus monomials' if args.model=='C1' else 'Output translations by option-antisymmetric degree<=2 attribute monomials plus intercept']),
              assumption=('Single-factor equivariance under amount-power, loss-scale, probability-odds and output-scale transformations; fixed 100 monetary units' if cr3 else 'Low-degree stimulus effects and one-axis functional transport; active templates selected without parameter labels'),
              mean_subject_ids=ids,bank_trial_ids=stim,domain_fraction=float(valid.mean()),
              capabilities=(['F0','F1_Z','F1_X'] if cr3 and args.transport_access=='partial' else ['F0','F1_Z','flow_effects' if cr3 else 'stimulus_translation_effects']),hessians_available=False,
              weighted_transport_residual_mean=residual,selected_operator_indices=selected,
              selection_rule='Smallest TRAIN response-template projection residuals; no T labels',
              gram_eigenvalues=np.linalg.eigvalsh(gram),transport_power=args.transport_power,
              transport_ridge=args.transport_ridge)
    info['finite_scale_step']=args.step
    info['physical_x_query_margin']=margin
    info['transport_access']=args.transport_access
    info['stimulus_dataset_rows']=ix[stim]
    return fields,info
