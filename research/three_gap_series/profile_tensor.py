"""Unknown-WY stimulus-profile x latent-polynomial tensor sieve for C1-C3."""
import math
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures

def stimulus_library_numpy(model,x):
    if model=='C1':
        a,b,c=x.T
        return np.column_stack((np.ones(len(x)),a,b,c,a*a,a*b,a*c,b*b,b*c,c*c))
    a,b,c,d=x.T
    return np.column_stack((np.ones(len(x)),a-b,c-d,a*a-b*b,a*c-b*d,c*c-d*d))

def stimulus_library_torch(model,x):
    one=x.new_tensor(1.)
    if model=='C1':
        a,b,c=x.unbind();return torch.stack((one,a,b,c,a*a,a*b,a*c,b*b,b*c,c*c))
    a,b,c,d=x.unbind();return torch.stack((one,a-b,c-d,a*a-b*b,a*c-b*d,c*c-d*d))

def fit_profile_tensor(args,c,fn,v,protocol):
    if args.model not in ['C1','C2','C3']:raise ValueError('Polynomial stimulus profile supports C1-C3')
    data=c['data'];sub=data['subject_index'];xx=data['x'];d=c['d']
    if np.any(np.diff(sub)<0):raise ValueError('Expected subject-grouped trial table')
    boundaries=np.r_[0,np.cumsum(np.bincount(sub))]
    rng=np.random.default_rng(args.seed+1901)
    train=rng.choice(c['splits']['train'],min(args.train_subjects,len(c['splits']['train'])),replace=False)
    validation=rng.choice(c['splits']['validation'],min(args.validation_subjects,len(c['splits']['validation'])),replace=False)
    pooled=np.flatnonzero(np.isin(sub,c['splits']['train']))
    sample=rng.choice(pooled,min(4096,len(pooled)),replace=False)
    scale=np.maximum(np.sqrt(np.mean(stimulus_library_numpy(args.model,xx[sample])**2,axis=0)),1e-10)
    channels=len(scale)
    def profile(ids,retain=False):
        answers=[];records=[]
        for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
            counts=boundaries[chunk+1]-boundaries[chunk]
            x=np.concatenate([xx[boundaries[s]:boundaries[s+1]] for s in chunk])
            w=np.concatenate((np.repeat(v[chunk],counts,axis=0),(x-protocol['x_mean'])/protocol['x_sd']),axis=1)
            with torch.no_grad():y=np.concatenate([torch.vmap(fn)(b).numpy() for b in torch.tensor(w).split(2048)])
            design=stimulus_library_numpy(args.model,x)/scale;start=0
            for count in counts:
                ds=design[start:start+count];ys=y[start:start+count]
                coefficient=np.linalg.solve(ds.T@ds/count+args.utility_ridge*np.eye(channels),ds.T@ys/count)
                answers.append(coefficient)
                if retain:records.append((ds,ys))
                start+=count
        return np.asarray(answers),records
    target,_=profile(train);val_target,records=profile(validation,True)
    hidden,_=np.linalg.qr(rng.normal(size=(d,d)))
    basis=PolynomialFeatures(args.degree).fit(v[:1]);powers=basis.powers_
    weight=np.array([math.sqrt(math.factorial(int(p.sum()))/math.prod(math.factorial(int(j)) for j in p)) for p in powers])
    phi=basis.transform(v[train]@hidden.T)*weight;valphi=basis.transform(v[validation]@hidden.T)*weight
    penalty=np.ones(len(weight))*args.ridge;penalty[0]=0
    coefficient=np.linalg.solve(phi.T@phi/len(phi)+np.diag(penalty),phi.T@target/len(phi))
    val_pred=valphi@coefficient
    sse=sum(float(np.sum((ds@cp-y)**2)) for cp,(ds,y) in zip(val_pred,records));total=sum(len(y) for _,y in records)
    tm=torch.tensor(protocol['x_mean']);ts=torch.tensor(protocol['x_sd']);hm=torch.tensor(hidden)
    pm=torch.tensor(powers,dtype=torch.long);wm=torch.tensor(weight);bm=torch.tensor(coefficient);sm=torch.tensor(scale)
    def estimated(w):
        z=w[:d]@hm.T;features=torch.ones(len(weight),dtype=w.dtype,device=w.device)
        for j in range(d):
            pp=[torch.ones_like(z[j])]
            for _ in range(args.degree):pp.append(pp[-1]*z[j])
            features=features*torch.stack(pp)[pm[:,j]]
        return ((features*wm)@bm)@(stimulus_library_torch(args.model,tm+ts*w[d:])/sm)
    info=dict(protocol='Original-row WY polynomial stimulus-profile x complete latent polynomial sieve',
              assumptions=('All stimulus monomials through degree 2' if args.model=='C1' else 'All option-antisymmetric attribute monomials through degree 2, plus choice intercept'),
              train_ids=train,validation_ids=validation,latent_degree=args.degree,latent_ridge=args.ridge,
              stimulus_ridge=args.utility_ridge,stimulus_scale=scale,hidden_rotation=hidden,coefficient=coefficient,
              validation_logit_rmse=np.sqrt(sse/total),validation_profile_rmse=np.sqrt(np.mean((val_pred-val_target)**2)),
              capabilities=['F0','F1','F2','F3'],warning='No true parameters or generating coefficients are used; operator selection uses response-template projection residuals only')
    return estimated,info
