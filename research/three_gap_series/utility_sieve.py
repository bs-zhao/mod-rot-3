"""E02 structured tensor sieve on numerical WY, without a cognitive formula.

Assumes additive option/gain/loss surfaces. Separate arbitrary gain/loss
surfaces are expanded in a tensor dictionary; no common alpha/gamma, scalar
lambda or temperature is fitted or supplied. This is a strong explicit
stimulus-structure prior, not a generic unrestricted function estimator.
"""
import math
import numpy as np
import torch
from scipy.special import expit,logit
from sklearn.preprocessing import PolynomialFeatures

MONEY=np.linspace(.2,1.4,7)
ODDS=np.linspace(.2,1.4,7)

def stimulus_basis_numpy(x):
    def surface(amount,p):
        money=np.where(amount[:,None]>0,np.maximum(amount[:,None]/100,1e-30)**MONEY[None],0.)
        prob=expit(logit(np.clip(p,1e-12,1-1e-12))[:,None]*ODDS[None])
        return (money[:,:,None]*prob[:,None,:]).reshape(len(x),-1)
    gain=surface(x[:,0],x[:,2])-surface(x[:,3],x[:,5])
    loss=surface(x[:,1],1-x[:,2])-surface(x[:,4],1-x[:,5])
    return np.concatenate((gain,-loss),axis=1)

def stimulus_basis_torch(x):
    x=torch.where(abs(x)<1e-10,torch.zeros_like(x),x)
    b=x.new_tensor(MONEY);c=x.new_tensor(ODDS)
    def surface(amount,p):
        money=torch.where(amount>0,(amount.clamp_min(1e-30)/100)**b,torch.zeros_like(b))
        prob=torch.sigmoid(c*torch.logit(p.clamp(1e-12,1-1e-12)))
        return (money[:,None]*prob[None,:]).flatten()
    return torch.cat((surface(x[0],x[2])-surface(x[3],x[5]),
                      -surface(x[1],1-x[2])+surface(x[4],1-x[5])))

def fit_utility_sieve(args,c,fn,v,protocol):
    if args.model!='CR3':raise ValueError('Current structured utility sieve requires CR3 input layout')
    data=c['data'];sub=data['subject_index'];ti=data['trial_index'];xx=data['x'];d=c['d']
    ix=np.flatnonzero(sub==0);ix=ix[np.argsort(ti[ix])];bank=xx[ix]
    rng=np.random.default_rng(args.seed+791)
    fit_ids=rng.choice(c['splits']['train'],min(args.train_subjects,len(c['splits']['train'])),replace=False)
    val_ids=rng.choice(c['splits']['validation'],min(args.validation_subjects,len(c['splits']['validation'])),replace=False)
    xs=(bank-protocol['x_mean'])/protocol['x_sd']
    def response(ids):
        answers=[]
        for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
            w=np.concatenate((np.repeat(v[chunk],500,axis=0),np.tile(xs,(len(chunk),1))),axis=1)
            with torch.no_grad():values=np.concatenate([torch.vmap(fn)(b).numpy() for b in torch.tensor(w).split(2048)])
            answers.append(values.reshape(len(chunk),500))
        return np.concatenate(answers)
    y=response(fit_ids);vy=response(val_ids)
    design=stimulus_basis_numpy(bank);scale=np.maximum(np.sqrt(np.mean(design*design,axis=0)),1e-10)
    design=design/scale;channels=design.shape[1]
    hidden,_=np.linalg.qr(rng.normal(size=(d,d)))
    basis=PolynomialFeatures(args.degree).fit(v[:1]);powers=basis.powers_
    weights=np.array([math.sqrt(math.factorial(int(p.sum()))/math.prod(math.factorial(int(k)) for k in p)) for p in powers])
    phi=basis.transform(v[fit_ids]@hidden.T)*weights
    valphi=basis.transform(v[val_ids]@hidden.T)*weights
    penalty=np.ones(len(weights))*args.ridge;penalty[0]=0
    def fit_coefficient(trials):
        dd=design[trials];target=y[:,trials]
        profiles=np.linalg.solve(dd.T@dd/len(trials)+args.utility_ridge*np.eye(channels),dd.T@target.T/len(trials)).T
        return np.linalg.solve(phi.T@phi/len(phi)+np.diag(penalty),phi.T@profiles/len(phi))
    subset=rng.permutation(500);train_trials=subset[:400];test_trials=subset[400:]
    side=fit_coefficient(train_trials)
    heldout=valphi@side@design[test_trials].T
    heldout_rmse=float(np.sqrt(np.mean((heldout-vy[:,test_trials])**2)))
    coefficient=fit_coefficient(np.arange(500));prediction=valphi@coefficient@design.T
    validation_rmse=float(np.sqrt(np.mean((prediction-vy)**2)))
    tm=torch.tensor(protocol['x_mean']);ts=torch.tensor(protocol['x_sd']);hm=torch.tensor(hidden)
    pm=torch.tensor(powers,dtype=torch.long);wm=torch.tensor(weights);bm=torch.tensor(coefficient);sm=torch.tensor(scale)
    def estimated(w):
        z=w[:d]@hm.T;features=torch.ones(len(weights),dtype=w.dtype,device=w.device)
        for j in range(d):
            pp=[torch.ones_like(z[j])]
            for _ in range(args.degree):pp.append(pp[-1]*z[j])
            features=features*torch.stack(pp)[pm[:,j]]
        features=features*wm
        sx=stimulus_basis_torch(tm+ts*w[d:])/sm
        return (features@bm)@sx
    info=dict(protocol='Original-bank WY structured tensor sieve; separate arbitrary gain/loss surfaces, no cognitive parameter equation',
              assumptions=['Additive option differences','Separate gain/loss utility surfaces','Smooth monetary-power x probability-odds dictionary'],
              latent_degree=args.degree,latent_ridge=args.ridge,stimulus_ridge=args.utility_ridge,
              money_dictionary=MONEY,odds_dictionary=ODDS,stimulus_channels=channels,latent_features=len(weights),
              hidden_rotation=hidden,stimulus_scale=scale,coefficient=coefficient,
              train_ids=fit_ids,validation_ids=val_ids,heldout_trial_ids=test_trials,
              validation_logit_rmse=validation_rmse,heldout_subject_and_trial_logit_rmse=heldout_rmse,
              capabilities=['F0','F1','F2','F3 within fitted class'],
              warning='Structured projection may intentionally differ from exact decoder derivatives; no complete CR3 formula or T labels are used')
    return estimated,info
