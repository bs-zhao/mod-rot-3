"""F0 algebraic fingerprints: power homogeneity and gain/loss proportionality.

These are explicit CR3 structural hypotheses, not complete formula inversion.
The estimator receives only responses to input interventions. No T labels.
"""
import numpy as np
import torch
from scipy.special import expit

def loss_mirror(gain):
    mirror=gain.copy();mirror[:,[1,4]]=gain[:,[0,3]]
    mirror[:,[0,3]]=0;mirror[:,[2,5]]=1-gain[:,[2,5]]
    return mirror

def fit_proportional_fingerprint(g,l,power=1.):
    pg,pl=expit(g),expit(l)
    weights=(pg*(1-pg)*pl*(1-pl)).clip(1e-24)**(.5*power)
    num=np.sum(weights*g*l,axis=1);den=np.sum(weights*g*g,axis=1)
    estimate=-num/np.maximum(den,1e-20)
    error=np.sum(weights*(l+estimate[:,None]*g)**2,axis=1)/np.maximum(np.sum(weights*l*l,axis=1),1e-20)
    return estimate,error

def fit_power_fingerprint(values,scales,power=1.):
    """Profile out one arbitrary response amplitude per stimulus; fit exponent."""
    baseline=int(np.argmin(abs(scales-1)))
    p=expit(values[:,baseline]);weights=(p*(1-p)).clip(1e-12)**power
    weights/=weights.sum(1,keepdims=True)
    gram=np.einsum('nt,nst,nrt->nsr',weights,values,values)
    grid=np.linspace(-.5,1.5,2001);h=scales[None,:]**grid[:,None]
    scores=np.einsum('as,nsr,ar->na',h,gram,h,optimize=True)/(h*h).sum(1)[None]
    index=scores.argmax(1);inner=index.clip(1,len(grid)-2);row=np.arange(len(values))
    left,middle,right=scores[row,inner-1],scores[row,inner],scores[row,inner+1]
    den=left-2*middle+right
    offset=np.divide(.5*(left-right),den,out=np.zeros_like(den),where=abs(den)>1e-20).clip(-1,1)
    estimate=grid[inner]+offset*(grid[1]-grid[0])
    estimate=np.where((index==0)|(index==len(grid)-1),grid[index],estimate)
    hh=scales[None,:]**estimate[:,None]
    explained=np.einsum('ns,nsr,nr->n',hh,gram,hh)/(hh*hh).sum(1)
    total=np.trace(gram,axis1=1,axis2=2)
    residual=np.maximum(1-explained/np.maximum(total,1e-30),0)
    return estimate,residual,float(np.mean((index==0)|(index==len(grid)-1)))

def fingerprint_fields(args,c,fn,v,protocol):
    data=c['data'];sub=data['subject_index'];ti=data['trial_index'];xx=data['x']
    ix=np.flatnonzero(sub==0);ix=ix[np.argsort(ti[ix])];bank=xx[ix]
    rng=np.random.default_rng(args.seed+741);ids=c['splits']['train'];d=c['d']
    scales=np.array([.6,.8,1.,1.2])
    eligible=np.flatnonzero(np.max(bank[:,[0,1,3,4]],axis=1)<=150)
    selected=rng.choice(eligible,min(args.mean_stimuli,len(eligible)),replace=False)
    template=bank[selected];scaled=[]
    for scale in scales:
        x=template.copy();x[:,[0,1,3,4]]*=scale;scaled.append(x)
    def table(stimuli):
        xs=(stimuli-protocol['x_mean'])/protocol['x_sd'];answers=[]
        for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
            w=np.concatenate((np.repeat(v[chunk],len(xs),axis=0),np.tile(xs,(len(chunk),1))),axis=1)
            with torch.no_grad():
                value=np.concatenate([torch.vmap(fn)(b).numpy() for b in torch.tensor(w).split(2048)])
            answers.append(value.reshape(len(chunk),len(xs)))
        return np.concatenate(answers)
    responses=table(np.concatenate(scaled)).reshape(len(ids),len(scales),len(selected))
    alpha,homogeneity_error,bound=fit_power_fingerprint(responses,scales,args.fingerprint_power)
    gains=np.flatnonzero((bank[:,[1,4]]==0).all(1)&(bank[:,[0,3]]>0).all(1))
    if len(gains)<5:raise ValueError('Gain-only bank unavailable')
    gain=bank[gains];mirror=loss_mirror(gain)
    paired=table(np.concatenate((gain,mirror)));g,l=np.split(paired,2,axis=1)
    lam,error=fit_proportional_fingerprint(g,l,args.fingerprint_power)
    features=np.column_stack((alpha,lam));center=features-features.mean(0)
    cross=v[ids].T@center/len(ids);variance=np.mean(center*center,axis=0)
    if np.any(variance<1e-12) or not np.isfinite(features).all():raise ValueError('Algebraic fingerprint collapsed')
    fields=dict(fingerprint_cross=cross,fingerprint_variance=variance)
    info=dict(protocol='F0 power homogeneity and sign-exchanged gain/loss proportionality; no complete CR3 formula fitted',
        assumption='Common money scaling has one power exponent; gain/loss mirror responses are proportional',
        scales=scales,monetary_max_after_scaling=float(max(scales)*np.max(template[:,[0,1,3,4]])),
        power_template_trial_ids=selected,gain_trial_ids=gains,train_ids=ids,
        feature_mean=features.mean(0),feature_sd=features.std(0),
        homogeneity_relative_error_quantiles=np.quantile(homogeneity_error,[.1,.5,.9,.99]),
        proportionality_relative_error_quantiles=np.quantile(error,[.1,.5,.9,.99]),
        exponent_search_boundary_fraction=bound,fingerprint_power=args.fingerprint_power,
        feature_quantiles=np.quantile(features,[.01,.1,.5,.9,.99],axis=0))
    return fields,info
