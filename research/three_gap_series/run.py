"""One candidate per invocation; independent G1/G2/G3 experiment entry point."""
import argparse
import csv
import itertools
from pathlib import Path
import time
import hashlib
import json
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from core import DIMS, load, dump, formula, decoder, derivatives, correspondence, sha
from metrics import METRICS, Loss
from strict_hessian import CATALOG as HESSIAN_CATALOG, HessianLoss,is_combination

def rotation(angles, d):
    r=np.eye(d)
    for (i,j), a in zip(itertools.combinations(range(d),2),angles):
        c,s=np.cos(a),np.sin(a)
        rowi=r[i].copy(); rowj=r[j].copy()
        r[i]=c*rowi-s*rowj; r[j]=s*rowi+c*rowj
    return r

def numerical_fields(fn, w, d, step, third=False,latent_only=False):
    """E09 query-access central differences. Receives only point->value oracle.

    Does not access any expression or derivative. Not an original-table estimator.
    """
    if third: raise ValueError('F3 finite differences not implemented')
    n,width=w.shape
    p=d if latent_only else width
    def query(v):
        with torch.no_grad():
            return torch.vmap(fn)(torch.tensor(v,dtype=torch.float64)).numpy()
    f=query(w); g=np.zeros((n,p)); h=np.zeros((n,p,p))
    for i in range(p):
        vi=np.eye(width)[i]*step
        plus,minus=query(w+vi),query(w-vi)
        g[:,i]=(plus-minus)/(2*step)
        h[:,i,i]=(plus-2*f+minus)/step**2
        for j in range(i):
            vj=np.eye(width)[j]*step
            mixed=(query(w+vi+vj)-query(w+vi-vj)-query(w-vi+vj)+query(w-vi-vj))/(4*step**2)
            h[:,i,j]=h[:,j,i]=mixed
    return dict(g=g,h=h)


def hessian_study_fields(fn,w,d,third=False):
    """Full F2, but only the latent F3 needed by UU-curvature variation.

    Does not compute unused X third derivatives. Same estimand as slicing the
    full third tensor, with much smaller memory and forward-mode cost.
    """
    fields=derivatives(fn,w,d,False,False)
    if third:
        def fixed(z,x):return fn(torch.cat((z,x)))
        op=torch.func.jacfwd(torch.func.jacfwd(torch.func.jacrev(fixed,argnums=0),argnums=0),argnums=0)
        fields['third']=np.concatenate([torch.vmap(op)(batch[:,:d],batch[:,d:]).detach().numpy() for batch in torch.tensor(w).split(128)])
    return fields

def prepare(args, c):
    rng=np.random.default_rng(args.seed)
    data=c['data']; d=c['d']; train=c['splits']['train']
    subject=data['subject_index']
    candidates=np.flatnonzero(np.isin(subject,train))
    x=data['x']
    # G1/G2 derivative protocol: CR3 positive-money interior domain only.
    if args.model=='CR3' and not args.metric.startswith('conditional_'):
        selected=x[candidates]
        mask=(selected[:,[0,1,3,4]]>1.).all(1)&(selected[:,[2,5]]>.03).all(1)&(selected[:,[2,5]]<.97).all(1)
        candidates=candidates[mask]
    rows=rng.choice(candidates,min(args.samples,len(candidates)),replace=False)
    sx=x[candidates].std(0); xm=x[candidates].mean(0)
    xs=(x[rows]-xm)/sx
    # TRUE parameter access is confined to analytic benchmark construction.
    if args.source=='formula':
        t=data['parameters']; tm=t[train].mean(0); ts=t[train].std(0)
        v=(t-tm)/ts
        tm_t,ts_t,xm_t,sx_t=map(torch.tensor,(tm,ts,xm,sx))
        def base(w): return formula(args.model,tm_t+ts_t*w[:d],xm_t+sx_t*w[d:])
    else:
        v=c['v']; base_raw=decoder(c)
        xm_t,sx_t=map(torch.tensor,(xm,sx))
        def base(w): return base_raw(torch.cat((w[:d],xm_t+sx_t*w[d:])))
    if args.channel=='logit': fn=base
    else:
        def fn(w): return torch.sigmoid(base(w))
    w=np.concatenate((v[subject[rows]],xs),axis=1)
    groups=None
    if args.metric.startswith('conditional_'):
        if args.model!='CR3': raise ValueError('This conditional candidate is CR3-design-specific')
        xx=x[rows]
        groups=np.full(len(rows),3)
        groups[(xx[:,[0,3]]==0).all(1)]=2
        groups[(xx[:,[2,5]]==.5).all(1)]=1
        groups[(xx[:,[1,4]]==0).all(1)]=0
        if len(np.unique(groups))<4: raise ValueError('All four CR3 design groups must be sampled')
    return fn,w,np.cov(v[train],rowvar=False,bias=True),v,rows,dict(x_mean=xm,x_sd=sx,domain_fraction=len(candidates)/np.isin(subject,train).sum(),groups=groups)

def polynomial_fields(args,c,fn,w,v,protocol,third):
    from estimators import PolynomialSieve
    rng=np.random.default_rng(args.seed+317)
    data=c['data']; d=c['d']; sub=data['subject_index']
    tables=[]; selected_rows=[]
    for split,n in [('train',args.train_rows),('validation',args.validation_rows)]:
        rows=np.flatnonzero(np.isin(sub,c['splits'][split]))
        rows=rng.choice(rows,min(n,len(rows)),replace=False); selected_rows.append(rows)
        x=(data['x'][rows]-protocol['x_mean'])/protocol['x_sd']
        inputs=np.concatenate((v[sub[rows]],x),axis=1)
        values=[]
        with torch.no_grad():
            for batch in torch.tensor(inputs).split(1024): values.append(torch.vmap(fn)(batch).numpy())
        y=np.concatenate(values)
        if args.channel=='binary': y=rng.binomial(1,y).astype(float)
        tables.append([inputs,y])
    # Hide the analytic/input coordinate orientation from the estimator.
    hidden=rotation(rng.uniform(-np.pi,np.pi,d*(d-1)//2),d)
    b=np.eye(w.shape[1]);b[:d,:d]=hidden
    train,y=tables[0]; val,vy=tables[1]
    sieve=PolynomialSieve(args.degree,args.ridge,args.channel).fit(train@b.T,y,val@b.T,vy)
    f=sieve.fields(w@b.T,third,args.structure_link)
    fields=dict(g=f['g']@b,h=np.einsum('ai,nab,bj->nij',b,f['h'],b,optimize=True))
    if third: fields['third']=np.einsum('ai,bj,ck,nabc->nijk',b,b,b,f['third'],optimize=True)
    if args.metric.startswith('conditional_'):
        fields=dict(g=fields['g'][:,:d],h=fields['h'][:,:d,:d])
    reference=derivatives(structure_function(fn,args),w,d,third,args.metric.startswith('conditional_'))
    errors={k:float(np.linalg.norm(fields[k]-reference[k])/max(np.linalg.norm(reference[k]),1e-20)) for k in fields}
    info=dict(**sieve.info,relative_errors=errors,hidden_rotation=hidden,train_row_hash=__import__('hashlib').sha256(selected_rows[0].tobytes()).hexdigest(),
              validation_row_hash=__import__('hashlib').sha256(selected_rows[1].tobytes()).hexdigest(),
              binary_protocol='one fixed-seed Bernoulli draw per selected original row' if args.channel=='binary' else None)
    return fields,info


def local_fields(args,c,fn,w,v,protocol,third):
    from local_wy import LocalPolynomial
    rng=np.random.default_rng(args.seed+317);data=c['data'];d=c['d'];sub=data['subject_index']
    tables=[];selected=[]
    for split,count in [('train',args.train_rows),('validation',args.validation_rows)]:
        ids=np.flatnonzero(np.isin(sub,c['splits'][split]));ids=rng.choice(ids,min(count,len(ids)),replace=False)
        x=(data['x'][ids]-protocol['x_mean'])/protocol['x_sd']
        inputs=np.concatenate((v[sub[ids]],x),axis=1)
        with torch.no_grad():y=np.concatenate([torch.vmap(fn)(b).numpy() for b in torch.tensor(inputs).split(2048)])
        if args.channel=='binary':y=rng.binomial(1,y).astype(float)
        tables.append((inputs,y));selected.append(ids)
    hidden=rotation(rng.uniform(-np.pi,np.pi,d*(d-1)//2),d);b=np.eye(w.shape[1]);b[:d,:d]=hidden
    fit=LocalPolynomial(args.degree,args.neighbors,args.bandwidth,args.ridge,args.channel).fit(tables[0][0]@b.T,tables[0][1])
    f=fit.fields(w@b.T,third,args.structure_link)
    fields=dict(g=f['g']@b,h=np.einsum('ai,nab,bj->nij',b,f['h'],b,optimize=True))
    if third:fields['third']=np.einsum('ai,bj,ck,nabc->nijk',b,b,b,f['third'],optimize=True)
    # Validation is grouped by subject. Limit costly local queries by a fixed rule.
    valw,valy=tables[1];vout=fit.fields((valw@b.T)[:min(128,len(valw))],False,args.structure_link)
    pred=vout['value'];target=valy[:len(pred)]
    if args.channel=='probability' and args.structure_link=='log_odds':target=np.log(np.clip(target,1e-12,1-1e-12)/np.clip(1-target,1e-12,1-1e-12))
    if args.channel=='binary' and args.structure_link=='log_odds':pred=expit(pred)
    reference=derivatives(structure_function(fn,args),w,d,third,False)
    errors={key:float(np.linalg.norm(fields[key]-reference[key])/max(np.linalg.norm(reference[key]),1e-20)) for key in fields}
    return fields,dict(**fit.info,relative_errors=errors,validation_rmse=float(np.sqrt(np.mean((pred-target)**2))),
        hidden_rotation=hidden,train_row_ids=selected[0],validation_row_ids=selected[1],
        validation_queried_rows=len(pred),binary_protocol='one seeded Bernoulli draw per original row' if args.channel=='binary' else None)


def tensor_fields(args,c,fn,w,v,protocol,third,extra_w=None):
    from tensor_wy import TensorWY
    if args.channel=='binary':raise ValueError('Generic TensorWY currently supports deterministic logit/probability; binary uses a separate Bernoulli inference experiment')
    rng=np.random.default_rng(args.seed+791);data=c['data'];sub=data['subject_index'];xx=data['x'];d=c['d']
    if np.any(np.diff(sub)<0):raise ValueError('Tensor adapter expects subject-grouped numerical WY')
    boundaries=np.r_[0,np.cumsum(np.bincount(sub))]
    ids_used=[];tables=[]
    for split,count in [('train',args.train_subjects),('validation',args.validation_subjects)]:
        ids=rng.choice(c['splits'][split],min(count,len(c['splits'][split])),replace=False);ids_used.append(ids)
        xt=[];yt=[]
        for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
            counts=boundaries[chunk+1]-boundaries[chunk]
            xs=[(xx[boundaries[s]:boundaries[s+1]]-protocol['x_mean'])/protocol['x_sd'] for s in chunk]
            inputs=np.concatenate((np.repeat(v[chunk],counts,axis=0),np.concatenate(xs)),axis=1)
            with torch.no_grad():y=np.concatenate([torch.vmap(fn)(b).numpy() for b in torch.tensor(inputs).split(2048)])
            if args.channel=='probability':y=np.log(np.clip(y,1e-12,1-1e-12)/np.clip(1-y,1e-12,1-1e-12))
            start=0
            for x,count_ in zip(xs,counts):xt.append(x);yt.append(y[start:start+count_]);start+=count_
        tables.append((xt,yt))
    hidden=rotation(rng.uniform(-np.pi,np.pi,d*(d-1)//2),d)
    fit=TensorWY(args.degree,args.ridge,args.utility_ridge,args.rbf_centers,args.rbf_length,args.seed,args.stimulus_kernel,
        args.latent_kernel,args.latent_centers,args.latent_length).fit(v[ids_used[0]]@hidden.T,*tables[0])
    def estimated(q):
        eta=fit.value(torch.cat((q[:d]@torch.tensor(hidden).T,q[d:])))
        return torch.sigmoid(eta) if args.channel=='probability' and args.structure_link=='native' else eta
    fields=hessian_study_fields(estimated,w,d,third and extra_w is None) if args.metric in HESSIAN_CATALOG else derivatives(estimated,w,d,third,False)
    if extra_w is not None:
        latent=fit.latent_fields(extra_w[:,:d]@hidden.T,extra_w[:,d:],third)
        lg=latent['g']@hidden
        lh=np.einsum('ai,nab,bj->nij',hidden,latent['h'],hidden,optimize=True)
        lt=np.einsum('ai,bj,ck,nabc->nijk',hidden,hidden,hidden,latent['third'],optimize=True) if third else None
        if args.channel=='probability' and args.structure_link=='native':
            pr=expit(latent['value']);s1=pr*(1-pr);s2=s1*(1-2*pr);s3=s1*(1-6*pr+6*pr*pr)
            if third:lt=s1[:,None,None,None]*lt+s2[:,None,None,None]*(lh[:,:,:,None]*lg[:,None,None,:]+lh[:,:,None,:]*lg[:,None,:,None]+lh[:,None,:,:]*lg[:,:,None,None])+s3[:,None,None,None]*lg[:,:,None,None]*lg[:,None,:,None]*lg[:,None,None,:]
            lh=s1[:,None,None]*lh+s2[:,None,None]*lg[:,:,None]*lg[:,None,:]
        fields['uu_h']=lh
        if third:fields['uu_t']=lt
    sse=0.;nval=0
    for sid,x,y in zip(ids_used[1],*tables[1]):
        prediction=fit.predict_subject(v[sid]@hidden.T,x)
        if args.channel=='probability' and args.structure_link=='native':y=expit(y);prediction=expit(prediction)
        sse+=float(np.sum((prediction-y)**2));nval+=len(y)
    reference=hessian_study_fields(structure_function(fn,args),w,d,third and extra_w is None) if args.metric in HESSIAN_CATALOG else derivatives(structure_function(fn,args),w,d,third,False)
    errors={key:float(np.linalg.norm(fields[key]-reference[key])/max(np.linalg.norm(reference[key]),1e-20)) for key in ['g','h']+(['third'] if third and extra_w is None else [])}
    block_errors={}
    for name,sl in [('UU',(slice(None),slice(None,d),slice(None,d))),('UX',(slice(None),slice(None,d),slice(d,None))),('XX',(slice(None),slice(d,None),slice(d,None)))]:
        diff=fields['h'][sl]-reference['h'][sl]
        block_errors[name]=dict(relative=float(np.linalg.norm(diff)/max(np.linalg.norm(reference['h'][sl]),1e-20)),absolute_rmse=float(np.sqrt(np.mean(diff*diff))))
    if extra_w is not None:
        truth_uu=derivatives(structure_function(fn,args),extra_w,d,False,True)['h'];diff=fields['uu_h']-truth_uu
        block_errors['UU_full_domain']=dict(relative=float(np.linalg.norm(diff)/max(np.linalg.norm(truth_uu),1e-20)),absolute_rmse=float(np.sqrt(np.mean(diff*diff))))
    return fields,dict(**fit.info,hidden_rotation=hidden,train_ids=ids_used[0],validation_ids=ids_used[1],
        validation_rmse=np.sqrt(sse/nval),relative_errors=errors,block_errors=block_errors)

def structure_function(fn,args):
    if args.channel!='logit' and args.structure_link=='log_odds':
        def linked(w): return torch.logit(fn(w).clamp(1e-12,1-1e-12))
        return linked
    return fn


def neural_fields(args,c,fn,w,v,protocol,third,extra_w,artifact_path):
    from neural_wy import NeuralWY
    if args.channel!='logit':raise ValueError('Neural WY branch currently benchmarks continuous logit only')
    rng=np.random.default_rng(args.seed+317);data=c['data'];sub=data['subject_index'];xx=data['x'];tables=[];selected=[];d=c['d']
    for split,count in [('train',args.train_rows),('validation',args.validation_rows)]:
        pool=np.flatnonzero(np.isin(sub,c['splits'][split]));rows=rng.choice(pool,min(count,len(pool)),replace=False)
        inputs=np.concatenate((v[sub[rows]],(xx[rows]-protocol['x_mean'])/protocol['x_sd']),axis=1)
        with torch.no_grad():y=np.concatenate([torch.vmap(fn)(batch).numpy() for batch in torch.tensor(inputs).split(2048)])
        tables.append((inputs,y));selected.append(rows)
    hidden=rotation(rng.uniform(-np.pi,np.pi,d*(d-1)//2),d)
    transform=np.eye(w.shape[1]);transform[:d,:d]=hidden
    train,y=tables[0];validation,vy=tables[1]
    fit=NeuralWY(d,args.neural_width,args.neural_epochs,args.seed).fit(train@transform.T,y,validation@transform.T,vy)
    fit.save(artifact_path);matrix=torch.tensor(transform)
    def estimated(q):return fit.value(q@matrix.T)
    fields=hessian_study_fields(estimated,w,d,third and extra_w is None)
    if extra_w is not None:
        fields['uu_h']=derivatives(estimated,extra_w,d,False,True)['h']
        if third:
            def fixed(z,x):return estimated(torch.cat((z,x)))
            op=torch.func.jacfwd(torch.func.jacfwd(torch.func.jacrev(fixed,argnums=0),argnums=0),argnums=0)
            fields['uu_t']=np.concatenate([torch.vmap(op)(batch[:,:d],batch[:,d:]).detach().numpy() for batch in torch.tensor(extra_w).split(32)])
    reference=hessian_study_fields(fn,w,d,False)
    errors={}
    for name,sl in [('UU',(slice(None),slice(None,d),slice(None,d))),('UX',(slice(None),slice(None,d),slice(d,None))),('XX',(slice(None),slice(d,None),slice(d,None)))]:
        diff=fields['h'][sl]-reference['h'][sl]
        errors[name]=dict(relative=float(np.linalg.norm(diff)/max(np.linalg.norm(reference['h'][sl]),1e-20)),absolute_rmse=float(np.sqrt(np.mean(diff*diff))))
    if extra_w is not None:
        truth=derivatives(fn,extra_w,d,False,True)['h'];diff=fields['uu_h']-truth
        errors['UU_full_domain']=dict(relative=float(np.linalg.norm(diff)/max(np.linalg.norm(truth),1e-20)),absolute_rmse=float(np.sqrt(np.mean(diff*diff))))
    return fields,dict(**fit.info,block_errors=errors,hidden_rotation=hidden,surrogate_path=str(artifact_path),
        surrogate_sha256=sha(artifact_path),train_row_hash=__import__('hashlib').sha256(selected[0].tobytes()).hexdigest(),
        validation_row_hash=__import__('hashlib').sha256(selected[1].tobytes()).hexdigest())

def bank_fields(args,c,fn,w,v,protocol,rows):
    from estimators import BankSieve
    if args.model!='CR3' or not args.metric.startswith('conditional_'):
        raise ValueError('Fixed-bank sieve currently supports CR3 conditional loss only')
    rng=np.random.default_rng(args.seed+411);d=c['d'];data=c['data']
    # NpzFile re-decompresses a member on EVERY subscription. Cache once.
    subject_index=data['subject_index'];all_trials=data['trial_index'];all_x=data['x']
    indices=np.flatnonzero(subject_index==0)
    order=np.argsort(all_trials[indices]);indices=indices[order]
    bank=all_x[indices];trial=all_trials[indices]
    if len(bank)!=500 or not np.array_equal(trial,np.arange(500)): raise ValueError('Wrong bank indexing')
    xs=(bank-protocol['x_mean'])/protocol['x_sd'];tables=[];ids_used=[]
    grouped=bool(np.all(np.diff(subject_index)>=0))
    boundaries=np.r_[0,np.cumsum(np.bincount(subject_index))] if grouped else None
    for split,n in [('train',args.train_subjects),('validation',args.validation_subjects)]:
        ids=rng.choice(c['splits'][split],min(n,len(c['splits'][split])),replace=False);ids_used.append(ids)
        # Verify fixed X bank for the subjects used, not just subject zero.
        for sid in ids:
            ix=np.arange(boundaries[sid],boundaries[sid+1]) if grouped else np.flatnonzero(subject_index==sid)
            ix=ix[np.argsort(all_trials[ix])]
            if not np.array_equal(all_x[ix],bank): raise ValueError('Subject does not share CR3 bank')
        query=np.concatenate((np.repeat(v[ids],500,axis=0),np.tile(xs,(len(ids),1))),axis=1)
        values=[]
        with torch.no_grad():
            for batch in torch.tensor(query).split(2048): values.append(torch.vmap(fn)(batch).numpy())
        y=np.concatenate(values).reshape(len(ids),500)
        if args.channel=='binary': y=rng.binomial(1,y).astype(float)
        tables.append((v[ids],y))
    hidden=rotation(rng.uniform(-np.pi,np.pi,d*(d-1)//2),d)
    train,y=tables[0];val,vy=tables[1]
    sieve=BankSieve(args.degree,args.ridge,args.channel).fit(train@hidden.T,y,val@hidden.T,vy)
    sieve.eval_stimulus=all_trials[rows].astype(int)
    if args.center_point:
        sieve.eval_stimulus=np.arange(500)
        f=sieve.fields(np.zeros((500,d)),False,args.structure_link)
    elif args.subject_mean:
        f=sieve.mean_fields(v[c['splits']['train']]@hidden.T)
    else: f=sieve.fields(w[:,:d]@hidden.T,False,args.structure_link)
    fields=dict(g=f['g']@hidden,h=np.einsum('ai,nab,bj->nij',hidden,f['h'],hidden,optimize=True))
    if args.center_point:
        values=sieve.raw_derivative(np.zeros((500,d)),())
        fields['value']=values if args.channel=='logit' or args.structure_link=='log_odds' else expit(values)
    elif not args.subject_mean:
        values=sieve.raw_derivative(w[:,:d]@hidden.T,())
        fields['value']=values if args.channel=='logit' or args.structure_link=='log_odds' else expit(values)
    if args.subject_mean or args.center_point:
        fields['groups']=group_labels(bank)
        errors=None
    else:
        reference=derivatives(structure_function(fn,args),w,d,False,True)
        errors={k:float(np.linalg.norm(fields[k]-reference[k])/max(np.linalg.norm(reference[k]),1e-20)) for k in ['g','h']}
    return fields,dict(**sieve.info,relative_errors=errors,hidden_rotation=hidden,train_ids=ids_used[0],validation_ids=ids_used[1],
                       structure_link=args.structure_link)

def center_fields(args,c,fn,protocol):
    data=c['data'];sub=data['subject_index'];ti=data['trial_index'];xx=data['x'];d=c['d']
    ix=np.flatnonzero(sub==0);ix=ix[np.argsort(ti[ix])];bank=xx[ix]
    query=np.concatenate((np.zeros((500,d)),(bank-protocol['x_mean'])/protocol['x_sd']),axis=1)
    target=structure_function(fn,args)
    f=derivatives(target,query,d,False,True) if args.estimator=='direct' else numerical_fields(target,query,d,args.step,False,True)
    f['groups']=group_labels(bank)
    with torch.no_grad():f['value']=torch.vmap(target)(torch.tensor(query)).numpy()
    return f,dict(protocol='Q11: TRAIN latent center, full original CR3 bank; latent derivatives only',latent_center=np.zeros(d),bank_trials=500)

def group_labels(xx):
    labels=np.full(len(xx),3)
    labels[(xx[:,[0,3]]==0).all(1)]=2
    labels[(xx[:,[2,5]]==.5).all(1)]=1
    labels[(xx[:,[1,4]]==0).all(1)]=0
    return labels

def direct_mean_fields(args,c,fn,v,protocol):
    data=c['data'];d=c['d'];sub=data['subject_index']
    ix=np.flatnonzero(sub==0);ix=ix[np.argsort(data['trial_index'][ix])]
    bank=data['x'][ix];x=(bank-protocol['x_mean'])/protocol['x_sd']
    rng=np.random.default_rng(args.seed+912)
    ids=rng.choice(c['splits']['train'],min(args.mean_subjects,len(c['splits']['train'])),replace=False)
    query=np.concatenate((np.repeat(v[ids],500,axis=0),np.tile(x,(len(ids),1))),axis=1)
    fields=derivatives(structure_function(fn,args),query,d,False,True)
    fields={k:value.reshape(len(ids),500,*value.shape[1:]).mean(0) for k,value in fields.items()}
    fields['groups']=group_labels(bank)
    return fields,dict(aggregation='subject mean of signed fields first, then stimulus energy',mean_subject_ids=ids)

def anchor_fields(args,c,fn,v,protocol):
    """Additional MODEL-INFORMED query experiment, outside original X bank.

    These anchors use neutral amounts/probabilities. Some are dominated;
    decoder extrapolation is explicitly part of this additional experiment.
    No claim of observational-only inference or generic model-free discovery.
    """
    anchors=[[100,0,.5,0,0,.5],[100,100,.5,100,0,.5]]
    groups=[0,1]
    for gain in [20,50,80,125,150]:anchors.append([gain,0,.5,0,0,.5]);groups.append(2)
    for prob in [.1,.2,.35,.65,.8,.9]:anchors.append([100,0,prob,0,0,.5]);groups.append(3)
    anchors=np.asarray(anchors,float);groups=np.asarray(groups)
    xs=(anchors-protocol['x_mean'])/protocol['x_sd'];k=len(xs);d=c['d']
    rng=np.random.default_rng(args.seed+461);ids=rng.choice(c['splits']['train'],min(args.mean_subjects,len(c['splits']['train'])),replace=False)
    query=np.concatenate((np.repeat(v[ids],k,axis=0),np.tile(xs,(len(ids),1))),axis=1)
    target=structure_function(fn,args)
    if args.estimator in ['direct','finite_difference']:
        f=derivatives(target,query,d,False,True) if args.estimator=='direct' else numerical_fields(target,query,d,args.step,False,True)
        fields={key:value.reshape(len(ids),k,*value.shape[1:]).mean(0) for key,value in f.items()}
        info={}
    elif args.estimator=='stimulus':
        from estimators import BankSieve
        tables=[]
        for split,num in [('train',args.train_subjects),('validation',args.validation_subjects)]:
            use=rng.choice(c['splits'][split],min(num,len(c['splits'][split])),replace=False)
            wq=np.concatenate((np.repeat(v[use],k,axis=0),np.tile(xs,(len(use),1))),axis=1)
            with torch.no_grad(): yy=torch.vmap(fn)(torch.tensor(wq)).numpy().reshape(len(use),k)
            if args.channel=='binary': yy=rng.binomial(1,yy).astype(float)
            tables.append((v[use],yy))
        sieve=BankSieve(args.degree,args.ridge,args.channel).fit(*tables[0],*tables[1])
        fields=sieve.mean_fields(v[ids]);info=sieve.info
    else: raise ValueError('Anchor candidate supports direct, finite_difference or stimulus query sieve')
    fields['groups']=groups
    info.update(anchors=anchors,mean_subject_ids=ids,protocol='MODEL-INFORMED counterfactual query anchors; outside original design; signed subject means first',
                warning='Some anchors are dominated/degenerate lotteries absent from training; extrapolation, not original-table inference')
    return fields,info

def cross_mean_fields(args,c,fn,v,protocol):
    """Q06 on actual CR3 interior bank: average signed full fields over Z first."""
    if args.estimator not in ['direct','finite_difference']:
        raise ValueError('Full-field subject mean currently supports direct or E09 query inference')
    data=c['data'];sub=data['subject_index'];xx=data['x'];ti=data['trial_index']
    ix=np.flatnonzero(sub==0);ix=ix[np.argsort(ti[ix])]
    bank=xx[ix];valid=(bank[:,[0,1,3,4]]>1).all(1)&(bank[:,[2,5]]>.03).all(1)&(bank[:,[2,5]]<.97).all(1)
    rng=np.random.default_rng(args.seed+819)
    stim=rng.choice(np.flatnonzero(valid),min(args.mean_stimuli,int(valid.sum())),replace=False)
    ids=rng.choice(c['splits']['train'],min(args.mean_subjects,len(c['splits']['train'])),replace=False)
    x=(bank[stim]-protocol['x_mean'])/protocol['x_sd'];k=len(x);d=c['d']
    query=np.concatenate((np.repeat(v[ids],k,axis=0),np.tile(x,(len(ids),1))),axis=1)
    target=structure_function(fn,args)
    f=derivatives(target,query,d,False) if args.estimator=='direct' else numerical_fields(target,query,d,args.step)
    fields={key:value.reshape(len(ids),k,*value.shape[1:]).mean(0) for key,value in f.items()}
    return fields,dict(protocol='Q06 TRAIN subjects x actual interior bank; signed full derivatives averaged over subjects first',
                       mean_subject_ids=ids,bank_trial_ids=stim,domain_fraction=float(valid.mean()),query_points=len(query),
                       final_field_points=k,capabilities=['mean_F1','mean_F2_ZZ','mean_F2_ZX','mean_F2_XX'])

def relative_fields(args,c,fn,v,protocol):
    """Scale separability: Var_X[grad log|eta| | Z] vanishes for pure temperature.

    Requires canonical eta, not log probability. Zero-crossing neighborhoods
    are excluded by an output-relative mask, identical rule for every source.
    """
    data=c['data'];sub=data['subject_index'];xx=data['x'];ti=data['trial_index'];d=c['d']
    ix=np.flatnonzero(sub==0);ix=ix[np.argsort(ti[ix])];bank=xx[ix]
    xs=(bank-protocol['x_mean'])/protocol['x_sd'];rng=np.random.default_rng(args.seed+614)
    ids=rng.choice(c['splits']['train'],min(args.mean_subjects,len(c['splits']['train'])),replace=False)
    query=np.concatenate((np.repeat(v[ids],500,axis=0),np.tile(xs,(len(ids),1))),axis=1)
    target=structure_function(fn,args);info={}
    if args.estimator in ['direct','finite_difference']:
        f=derivatives(target,query,d,False,True) if args.estimator=='direct' else numerical_fields(target,query,d,args.step,False,True)
        g=f['g'].reshape(len(ids),500,d);h=f['h'].reshape(len(ids),500,d,d)
        with torch.no_grad():
            eta=np.concatenate([torch.vmap(target)(batch).numpy() for batch in torch.tensor(query).split(2048)]).reshape(len(ids),500)
    elif args.estimator=='stimulus':
        from estimators import BankSieve
        tables=[]
        if not np.array_equal(np.bincount(sub),np.full(len(v),500)) or np.any(np.diff(sub)<0):
            raise ValueError('Relative fixed-bank estimator requires grouped 500-row subjects')
        xgrid=xx.reshape(len(v),500,6)
        for split,num in [('train',args.train_subjects),('validation',args.validation_subjects)]:
            use=rng.choice(c['splits'][split],min(num,len(c['splits'][split])),replace=False)
            if not np.array_equal(xgrid[use],np.broadcast_to(bank,(len(use),500,6))):raise ValueError('Fixed bank mismatch')
            wq=np.concatenate((np.repeat(v[use],500,axis=0),np.tile(xs,(len(use),1))),axis=1)
            with torch.no_grad():
                yy=np.concatenate([torch.vmap(fn)(batch).numpy() for batch in torch.tensor(wq).split(2048)]).reshape(len(use),500)
            if args.channel=='binary':yy=rng.binomial(1,yy).astype(float)
            tables.append((v[use],yy))
        hidden=rotation(rng.uniform(-np.pi,np.pi,d*(d-1)//2),d)
        sieve=BankSieve(args.degree,args.ridge,args.channel).fit(tables[0][0]@hidden.T,tables[0][1],tables[1][0]@hidden.T,tables[1][1])
        grid=sieve.grid_fields(v[ids]@hidden.T)
        eta=grid['value'];g=grid['g']@hidden;h=np.einsum('ai,nsab,bj->nsij',hidden,grid['h'],hidden,optimize=True)
        info=sieve.info
    else:raise ValueError('Relative candidate supports direct, E09 or fixed-bank WY')
    covariance=[];gram=[];retained=[]
    for values,gradient in zip(eta,g):
        mask=abs(values)>args.relative_cutoff*max(np.sqrt(np.mean(values*values)),1e-20)
        if mask.sum()<d+2:raise ValueError('Too few nonzero-logit stimuli')
        rel=gradient[mask]/values[mask,None]
        probability=expit(values[mask])
        qw=(probability*(1-probability)).clip(1e-12)**args.fisher_power if args.q_weight=='fisher' else np.ones(len(rel))
        qw/=qw.sum();center=rel-qw@rel
        covariance.append(center.T@(center*qw[:,None]));gram.append(rel.T@(rel*qw[:,None]));retained.append(mask.mean())
    fields=dict(g=g.reshape(-1,d),h=h.reshape(-1,d,d),groups=np.tile(group_labels(bank),len(ids)),
                relative_cov=np.mean(covariance,axis=0),relative_gram=np.mean(gram,axis=0))
    fields['value']=eta.reshape(-1)
    fields['latent_train']=v[c['splits']['train']]
    info.update(protocol='Conditional raw-gradient absence plus within-subject stimulus variance of gradient log|eta|',
                mean_subject_ids=ids,relative_cutoff=args.relative_cutoff,retained_fraction=np.mean(retained),
                derivative_target='canonical log odds; relative gradients use log absolute canonical log odds')
    return fields,info

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model',choices=DIMS,required=True)
    p.add_argument('--beta',type=float,required=True); p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--source',choices=['formula','decoder'],default='formula')
    p.add_argument('--estimator',choices=['direct','finite_difference','polynomial','local','tensor','neural','stimulus','utility','profile'],default='direct')
    p.add_argument('--channel',choices=['logit','probability','binary'],default='logit')
    p.add_argument('--metric',choices=METRICS+tuple(HESSIAN_CATALOG),required=True)
    p.add_argument('--samples',type=int,default=384); p.add_argument('--starts',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261108); p.add_argument('--step',type=float,default=.01)
    p.add_argument('--maxiter',type=int,default=250); p.add_argument('--label',default='round01')
    p.add_argument('--degree',type=int,default=3);p.add_argument('--ridge',type=float,default=.001)
    p.add_argument('--train-rows',type=int,default=32000);p.add_argument('--validation-rows',type=int,default=8000)
    p.add_argument('--train-subjects',type=int,default=1024);p.add_argument('--validation-subjects',type=int,default=256)
    p.add_argument('--structure-link',choices=['native','log_odds'],default='native')
    p.add_argument('--subject-mean',type=int,choices=[0,1],default=0)
    p.add_argument('--mean-subjects',type=int,default=128)
    p.add_argument('--field-mean',type=int,choices=[0,1],default=0)
    p.add_argument('--mean-stimuli',type=int,default=64)
    p.add_argument('--center-point',type=int,choices=[0,1],default=0)
    p.add_argument('--relative-cutoff',type=float,default=.1)
    p.add_argument('--q-weight',choices=['uniform','fisher'],default='uniform')
    p.add_argument('--fisher-power',type=float,default=1.)
    p.add_argument('--transport-power',type=float,default=1.)
    p.add_argument('--transport-ridge',type=float,default=1e-6)
    p.add_argument('--transport-access',choices=['partial','flow'],default='partial')
    p.add_argument('--fingerprint-power',type=float,default=1.)
    p.add_argument('--utility-ridge',type=float,default=1e-5)
    p.add_argument('--gate-prescreen',action='store_true',help='Reject a G1 candidate as soon as a plane counterexample is found; never promote on planes alone')
    p.add_argument('--neighbors',type=int,default=2048)
    p.add_argument('--bandwidth',type=float,default=1.)
    p.add_argument('--uu-full-domain',action='store_true',help='Separate block domains: UU on all original TRAIN rows, UX on the differentiable X domain')
    p.add_argument('--hessian-grid',type=int,default=0,help='Subjects in a TRAIN product evaluation grid; generic Hessian profile JD only')
    p.add_argument('--grid-stimuli',type=int,default=64)
    p.add_argument('--g1-angle-tolerance',type=float,default=5.,help='User-accepted maximum axis error in degrees, after sign/permutation matching')
    p.add_argument('--combination-rationale',help='JSON evidence file with single-loss runs, complementarity and a common weight rule; required for combined candidates')
    p.add_argument('--rbf-centers',type=int,default=64)
    p.add_argument('--rbf-length',type=float,default=1.)
    p.add_argument('--stimulus-kernel',choices=['gaussian','anova2','anova3'],default='gaussian')
    p.add_argument('--latent-kernel',choices=['polynomial','gaussian'],default='polynomial')
    p.add_argument('--latent-centers',type=int,default=256)
    p.add_argument('--latent-length',type=float,default=1.)
    p.add_argument('--neural-width',type=int,default=128)
    p.add_argument('--neural-epochs',type=int,default=200)
    p.add_argument('--field-reference',help='Reuse fixed numerical fields from a compatible Hessian-only summary; no old R or T evaluation is consumed')
    args=p.parse_args(); torch.set_num_threads(2); started=time.time()
    if args.metric in HESSIAN_CATALOG and is_combination(args.metric):
        if not args.combination_rationale:raise ValueError('User requires observed complementarity before combining losses; supply --combination-rationale')
        evidence_path=Path(args.combination_rationale).resolve()
        if not evidence_path.is_relative_to(Path(__file__).resolve().parent):raise ValueError('Combination evidence must stay within this experiment directory')
        evidence=json.loads(evidence_path.read_text(encoding='utf8'))
        if not evidence.get('component_runs') or not evidence.get('complementarity') or not evidence.get('weight_rule'):
            raise ValueError('Combination evidence needs actual component runs, complementarity and a common weight rule')
        for relative_path in evidence['component_runs']:
            component_path=(Path(__file__).resolve().parent/relative_path).resolve()
            if not component_path.is_relative_to(Path(__file__).resolve().parent):raise ValueError('Component evidence outside experiment directory')
            component=json.loads(component_path.read_text(encoding='utf8'))
            ca=component['args']
            if ca['source']!='formula' or ca['estimator']!='direct' or ca['metric'] not in HESSIAN_CATALOG or is_combination(ca['metric']):
                raise ValueError('Combination rationale must cite actual single-loss analytic experiments')
    if args.channel=='binary' and args.estimator not in ['polynomial','stimulus','local']:
        raise ValueError('Binary observations require a Bernoulli sieve; use --estimator polynomial')
    if args.subject_mean and (args.model!='CR3' or not args.metric.startswith('conditional_') or args.estimator not in ['direct','stimulus'] or (args.channel!='logit' and args.structure_link!='log_odds')):
        raise ValueError('Subject-mean currently requires CR3 conditional canonical-logit direct or bank fields')
    if args.metric=='conditional_anchor' and args.channel!='logit' and args.structure_link!='log_odds':
        raise ValueError('Additional anchor experiment currently targets canonical log odds only')
    if args.metric.startswith('conditional_relative') and args.channel!='logit' and args.structure_link!='log_odds':
        raise ValueError('Relative-gradient structure requires canonical log odds')
    if args.metric.startswith('conditional_relative') and (args.center_point or args.subject_mean or args.field_mean or not 0<=args.relative_cutoff<1):
        raise ValueError('Relative-gradient has its own conditional averaging protocol; use a cutoff in [0,1)')
    if args.field_mean and (args.model!='CR3' or args.metric.startswith(('conditional_','signature_')) or args.metric=='zzx_mixed_rms'):
        raise ValueError('Full-field subject mean currently supports CR3 F2 metrics only')
    if args.center_point and (args.model!='CR3' or not args.metric.startswith('conditional_') or args.metric=='conditional_anchor' or args.estimator not in ['direct','finite_difference','stimulus','utility']):
        raise ValueError('Center-point control requires CR3 original-bank conditional F2 metrics')
    if args.q_weight!='uniform' and (not args.metric.startswith('conditional_') or args.metric=='conditional_anchor' or args.subject_mean or args.field_mean or args.estimator not in ['direct','finite_difference','stimulus']):
        raise ValueError('Fisher-Q currently supports point/local original-bank conditional fields only')
    if not np.isfinite(args.fisher_power) or args.fisher_power<=0:raise ValueError('Fisher power must be positive')
    if args.metric.startswith('transport_') and args.channel!='logit' and args.structure_link!='log_odds':
        raise ValueError('Current transport dictionaries require canonical log odds')
    if args.metric=='conditional_fingerprint' and (not args.center_point or args.channel=='binary' or args.channel!='logit' and args.structure_link!='log_odds'):
        raise ValueError('Algebraic fingerprint currently uses centered Gamma-null plus deterministic canonical-logit queries; binary not yet supported')
    c=load(args); d=c['d']; nangle=d*(d-1)//2
    out=Path(__file__).resolve().parent/'results'/args.label/f'{args.model}_beta{args.beta:g}_zdim{args.zdim}_pca{args.pca}'/args.source/args.channel/args.estimator/args.metric/f'seed{args.seed}'
    config={k:value for k,value in vars(args).items() if k not in ['repo','label','metric']}
    config['field_protocol']=('fingerprint' if args.metric=='conditional_fingerprint' else 'transport' if args.metric.startswith('transport_') else 'anchors' if args.metric=='conditional_anchor' else 'relative' if args.metric.startswith('conditional_relative') else 'original')
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    # Compact directories avoid Windows MAX_PATH problems. Full flags in summary.
    out=out.with_name(f's{args.seed}_n{args.samples}_cfg{config_hash}')
    out.mkdir(parents=True,exist_ok=True)
    # One field cache shared among candidate losses, no fitted R or T labels.
    third=args.metric.startswith('signature') or args.metric=='zzx_mixed_rms' or HESSIAN_CATALOG.get(args.metric,{}).get('order')==3
    fn,w,cov,v,rows,protocol=prepare(args,c)
    grid_groups=None
    if args.hessian_grid:
        if args.metric not in HESSIAN_CATALOG:raise ValueError('Hessian grids are reserved for the model-blind Hessian study')
        gridrng=np.random.default_rng(args.seed+673)
        subject_ids=gridrng.choice(c['splits']['train'],min(args.hessian_grid,len(c['splits']['train'])),replace=False)
        stimulus_rows=gridrng.choice(rows,min(args.grid_stimuli,len(rows)),replace=False)
        # Generic product Q. No cognitive condition groups or model-specific X templates.
        xgrid=(c['data']['x'][stimulus_rows]-protocol['x_mean'])/protocol['x_sd']
        w=np.concatenate((np.repeat(v[subject_ids],len(xgrid),axis=0),np.tile(xgrid,(len(subject_ids),1))),axis=1)
        grid_groups=np.repeat(np.arange(len(subject_ids)),len(xgrid))
        protocol['hessian_grid']=dict(subject_ids=subject_ids,stimulus_rows=stimulus_rows,
            evaluation_points=len(w),access='TRAIN product evaluation Q; G2/G3 must fit original WY rows, not query teacher at this grid for fitting')
    full_w=None;full_rows=None
    if args.uu_full_domain:
        if args.metric not in HESSIAN_CATALOG or args.estimator not in ['direct','tensor','neural']:
            raise ValueError('Separate all-row UU supports direct fields and generic TensorWY inference')
        all_sub=c['data']['subject_index'];all_x=c['data']['x']
        pool=np.flatnonzero(np.isin(all_sub,c['splits']['train']))
        full_rows=np.random.default_rng(args.seed+511).choice(pool,min(args.samples,len(pool)),replace=False)
        full_w=np.concatenate((v[all_sub[full_rows]],(all_x[full_rows]-protocol['x_mean'])/protocol['x_sd']),axis=1)
    latent_only=args.metric.startswith('conditional_')
    cache=out.parents[1]/f'fields_n{args.samples}_s{args.seed}_h{args.step:g}_third{int(third)}_degree{args.degree}_ridge{args.ridge:g}_nt{args.train_rows}_nv{args.validation_rows}_latent{int(latent_only)}_link{args.structure_link}_ns{args.train_subjects}_vs{args.validation_subjects}.npz'
    cache=cache.with_name(f'fields_{config_hash}_third{int(third)}_latent{int(latent_only)}.npz')
    field_provenance=None
    if args.field_reference:
        if args.metric not in HESSIAN_CATALOG:raise ValueError('Field-reference reuse is restricted to the Hessian-only study')
        from field_reference import validate_reference
        cache,field_provenance=validate_reference(args,c,protocol,w.shape,full_rows,third)
    estimator_info=None
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['checkpoint_hash']) != c['hashes']['checkpoint']: raise ValueError('Stale field cache')
            fields={k:saved[k] for k in ['g','h','third','uu_h','uu_t','uu_row_ids','groups','relative_cov','relative_gram','value','weights','latent_train','full_g','transport_gram','transport_cross','transport_energy','transport_vector_grams','fingerprint_cross','fingerprint_variance'] if k in saved.files}
        if cache.with_suffix('.json').exists():
            estimator_info=json.loads(cache.with_suffix('.json').read_text())
    else:
        if args.estimator in ['utility','profile']:
            from types import SimpleNamespace
            if args.estimator=='utility':
                from utility_sieve import fit_utility_sieve
                estimated,estimator_info=fit_utility_sieve(args,c,structure_function(fn,args),v,protocol)
            else:
                from profile_tensor import fit_profile_tensor
                estimated,estimator_info=fit_profile_tensor(args,c,structure_function(fn,args),v,protocol)
            calc=SimpleNamespace(**vars(args));calc.estimator='direct';calc.channel='logit';calc.structure_link='native'
            if args.metric=='conditional_fingerprint':
                fields,field_info=center_fields(calc,c,estimated,protocol)
                from fingerprints import fingerprint_fields
                fp,fp_info=fingerprint_fields(calc,c,estimated,v,protocol)
                fields.update(fp);field_info['algebraic_fingerprints']=fp_info
            elif args.metric.startswith('transport_'):
                from transport import transport_fields
                fields,field_info=transport_fields(calc,c,estimated,v,protocol)
            elif args.center_point:fields,field_info=center_fields(calc,c,estimated,protocol)
            else:
                fields=derivatives(estimated,w,d,third,latent_only);field_info={}
            estimator_info['field_aggregation']=field_info
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.metric=='conditional_fingerprint':
            if args.estimator=='stimulus':fields,estimator_info=bank_fields(args,c,fn,w,v,protocol,rows)
            elif args.estimator in ['direct','finite_difference']:fields,estimator_info=center_fields(args,c,fn,protocol)
            else:raise ValueError('Fingerprint supports direct, query or fixed-bank estimator')
            from fingerprints import fingerprint_fields
            fp,fp_info=fingerprint_fields(args,c,structure_function(fn,args),v,protocol)
            fields.update(fp);estimator_info['algebraic_fingerprints']=fp_info
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.metric.startswith('transport_'):
            from transport import transport_fields
            fields,estimator_info=transport_fields(args,c,structure_function(fn,args),v,protocol)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.metric.startswith('conditional_relative'):
            fields,estimator_info=relative_fields(args,c,fn,v,protocol)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.center_point and args.estimator!='stimulus':
            fields,estimator_info=center_fields(args,c,fn,protocol)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.field_mean:
            fields,estimator_info=cross_mean_fields(args,c,fn,v,protocol)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.metric=='conditional_anchor':
            fields,estimator_info=anchor_fields(args,c,fn,v,protocol)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.estimator=='tensor':
            fields,estimator_info=tensor_fields(args,c,fn,w,v,protocol,third,full_w)
            if full_rows is not None:fields['uu_row_ids']=full_rows
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.estimator=='neural':
            fields,estimator_info=neural_fields(args,c,fn,w,v,protocol,third,full_w,cache.with_suffix('.pt'))
            if full_rows is not None:fields['uu_row_ids']=full_rows
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.estimator=='local':
            fields,estimator_info=local_fields(args,c,fn,w,v,protocol,third)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.estimator=='polynomial':
            fields,estimator_info=polynomial_fields(args,c,fn,w,v,protocol,third)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.estimator=='stimulus':
            fields,estimator_info=bank_fields(args,c,fn,w,v,protocol,rows)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.subject_mean:
            fields,estimator_info=direct_mean_fields(args,c,fn,v,protocol)
            dump(cache.with_suffix('.json'),estimator_info)
        else:
            target=structure_function(fn,args)
            fields=((hessian_study_fields(target,w,d,third and not args.uu_full_domain) if args.metric in HESSIAN_CATALOG else derivatives(target,w,d,third,latent_only)) if args.estimator=='direct' else numerical_fields(target,w,d,args.step,third,latent_only))
        np.savez_compressed(cache,**fields,checkpoint_hash=c['hashes']['checkpoint'])
    if args.uu_full_domain:
        if 'uu_h' not in fields:
            target=structure_function(fn,args)
            extra=derivatives(target,full_w,d,False,True);fields['uu_h']=extra['h'];fields['uu_row_ids']=full_rows
            if third:
                def zfn(z,x):return target(torch.cat((z,x)))
                op=torch.func.jacfwd(torch.func.jacfwd(torch.func.jacrev(zfn,argnums=0),argnums=0),argnums=0)
                fields['uu_t']=np.concatenate([torch.vmap(op)(batch[:,:d],batch[:,d:]).detach().numpy() for batch in torch.tensor(full_w).split(128)])
            np.savez_compressed(cache,**fields,checkpoint_hash=c['hashes']['checkpoint'])
        protocol['block_domains']=dict(uu='all original TRAIN rows; only latent derivatives, valid even at zero X amounts',ux='original rows in the differentiable X domain',uu_rows=fields['uu_row_ids'])
    if not all(np.isfinite(f).all() for f in fields.values()): raise ValueError('Nonfinite derivative')
    if latent_only and 'groups' not in fields: fields['groups']=protocol['groups']
    if args.metric=='conditional_relative_cumulant' and 'latent_train' not in fields:fields['latent_train']=v[c['splits']['train']]
    if args.q_weight=='fisher':
        if 'value' not in fields:
            target=structure_function(fn,args)
            with torch.no_grad():fields['value']=torch.vmap(target)(torch.tensor(w)).numpy()
        probability=expit(fields['value']) if args.channel=='logit' or args.structure_link=='log_odds' else fields['value']
        fields['weights']=(probability*(1-probability)).clip(1e-12)**args.fisher_power
        if len(fields['weights'])!=len(fields['h']):raise ValueError('Weight/field domain mismatch')
        protocol['q_weight']=dict(name='predictive Fisher p(1-p)',power=args.fisher_power,
            field_effective_points=float(fields['weights'].sum()**2/np.sum(fields['weights']**2)),
            note='Weighted field points are not independent latent subjects')
    if args.metric in HESSIAN_CATALOG:
        if args.estimator in ['utility','profile','stimulus'] or args.subject_mean or args.field_mean or args.center_point or args.q_weight!='uniform':
            raise ValueError('Hessian-only study disallows model-specific inference/grouping and weights')
        fields['strict_x']=w[:,d:]
        if grid_groups is not None:fields['strict_subject_groups']=grid_groups
        loss=HessianLoss(fields,cov,d,args.metric)
    else:loss=Loss(fields,cov,d,args.metric)
    rng=np.random.default_rng(args.seed+1)
    # Mathematical invariance audit over arbitrary signed scales, permutations and rotations.
    audits=[]
    for _ in range(5):
        r=rotation(rng.uniform(-np.pi,np.pi,nangle),d)
        scales=rng.choice([-1,1],d)*10.**rng.uniform(-3,3,d)
        perm=np.eye(d)[rng.permutation(d)]*rng.choice([-1,1],d)[:,None]
        val=loss(r); err=max(abs(loss(scales[:,None]*r)-val),abs(loss(perm@r)-val))/(1+abs(val))
        audits.append(err)
    if max(audits)>1e-8: raise ValueError(f'Invariance audit failed: {audits}')
    truth=loss(np.eye(d)); solutions=[]
    plane_counterexample=None;prescreen_sweep=[]
    if args.gate_prescreen:
        if args.source!='formula' or args.estimator!='direct':raise ValueError('Gate prescreen applies to G1 only')
        for i,j in itertools.combinations(range(d),2):
            plane=[]
            for angle in np.linspace(0,90,37):
                rr=np.eye(d);aa=np.deg2rad(angle);rr[i,i]=rr[j,j]=np.cos(aa);rr[i,j]=-np.sin(aa);rr[j,i]=np.sin(aa)
                value=loss(rr)
                plane.append(dict(i=i,j=j,angle=angle,loss=value))
            prescreen_sweep.extend(plane)
            near=min(p['loss'] for p in plane if min(p['angle'],90-p['angle'])<=args.g1_angle_tolerance)
            far=[p for p in plane if min(p['angle'],90-p['angle'])>args.g1_angle_tolerance]
            if far:
                wrong=min(far,key=lambda p:p['loss'])
                if wrong['loss']<near-1e-6*(1+abs(near)) and (plane_counterexample is None or wrong['loss']<plane_counterexample['loss']):
                    aa=np.deg2rad(wrong['angle']);rr=np.eye(d);rr[i,i]=rr[j,j]=np.cos(aa);rr[i,j]=-np.sin(aa);rr[j,i]=np.sin(aa)
                    plane_counterexample=dict(**wrong,r=rr,near_axis_minimum=near,angle_tolerance=args.g1_angle_tolerance)
        if plane_counterexample is not None:
            rr=plane_counterexample['r'];ii,jj=__import__('scipy').optimize.linear_sum_assignment(-abs(rr))
            solutions.append(dict(start=-1,loss=plane_counterexample['loss'],initial_loss=truth,
                axis_min=float(abs(rr[ii,jj]).min()),axis_mean=float(abs(rr[ii,jj]).mean()),success=True,
                message='G1 rejected: plane minimum outside the user-accepted angle neighborhood; no optimizer needed',iterations=0,r=rr))
    # Include identity only as analytic diagnostic; decoder starts NEVER use T/oracle directions.
    for start in range(0 if plane_counterexample is not None else args.starts):
        x0=np.zeros(nangle) if start==0 else rng.uniform(-np.pi,np.pi,nangle)
        fit=minimize(lambda a:loss(rotation(a,d)),x0,method='L-BFGS-B', options=dict(maxiter=args.maxiter,ftol=1e-12,gtol=1e-7))
        r=rotation(fit.x,d)
        from scipy.optimize import linear_sum_assignment
        ii,jj=linear_sum_assignment(-abs(r))
        solutions.append(dict(start=start,loss=float(fit.fun),initial_loss=float(loss(rotation(x0,d))),axis_min=float(abs(r[ii,jj]).min()),axis_mean=float(abs(r[ii,jj]).mean()),
                              success=bool(fit.success),message=str(fit.message),iterations=fit.nit,r=r))
        if args.metric in HESSIAN_CATALOG and (start+1)%8==0:
            current=min(solutions,key=lambda s:s['loss'])
            dump(out/'progress.json',dict(completed_starts=start+1,total_starts=args.starts,
                best_loss=current['loss'],elapsed_seconds=time.time()-started,solutions=solutions))
            print(args.model,args.source,args.metric,'starts',start+1,'of',args.starts,'best_loss',round(current['loss'],7),flush=True)
    best=min(solutions,key=lambda x:x['loss']); rhat=best['r']
    # Lock R before any TEST labels are read. No labels are used to select candidates.
    np.save(out/'rotation.npy',rhat)
    sweep=prescreen_sweep
    if not sweep:
        for i,j in itertools.combinations(range(d),2):
            for angle in np.linspace(0,90,37):
                r=np.eye(d); a=np.deg2rad(angle); r[i,i]=r[j,j]=np.cos(a);r[i,j]=-np.sin(a);r[j,i]=np.sin(a)
                sweep.append(dict(i=i,j=j,angle=angle,loss=loss(r)))
    with (out/'planes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(sweep[0]));writer.writeheader();writer.writerows(sweep)
    ranges=[np.ptp([x['loss'] for x in sweep if x['i']==i and x['j']==j]) for i,j in itertools.combinations(range(d),2)]
    strict_pass=(best['axis_min']>.995 and best['loss']>=truth-1e-6*(1+abs(truth)) and min(ranges)>1e-5*(1+abs(truth)))
    plane_axes=[]
    for i,j in itertools.combinations(range(d),2):
        plane=[p for p in sweep if p['i']==i and p['j']==j];minimum=min(plane,key=lambda p:p['loss'])
        plane_axes.append(dict(i=i,j=j,minimum_angle=minimum['angle'],axis_error=min(minimum['angle'],90-minimum['angle'])))
    worst_axis_error=float(np.rad2deg(np.arccos(np.clip(best['axis_min'],0,1))))
    angle_pass=bool(worst_axis_error<=args.g1_angle_tolerance+1e-7 and
        max(p['axis_error'] for p in plane_axes)<=args.g1_angle_tolerance and
        min(ranges)>1e-5*(1+abs(truth)) and plane_counterexample is None)
    evaluation=None
    if args.source=='decoder':
        test=c['splits']['test']
        # First access to true T in decoder branch happens AFTER R is locked.
        with np.load(c['latent_path'],allow_pickle=False) as latent: t=latent['theta_true']
        evaluation=correspondence(v[test]@rhat.T,t[test])
    result=dict(args=vars(args),hashes=c['hashes'],protocol=protocol,rows=rows,train_ids=c['splits']['train'],
                invariance_max=max(audits),truth_loss=truth,best=best,solutions=solutions,plane_ranges=ranges,
                gap1_strict_pass=bool(strict_pass) if args.source=='formula' else None,
                gap1_angle_pass=angle_pass if args.source=='formula' else None,
                worst_axis_error_degrees=worst_axis_error,plane_minima=plane_axes,
                evaluation=evaluation,elapsed_seconds=time.time()-started)
    result['estimator_info']=estimator_info
    result['stage']='G1' if args.source=='formula' and args.estimator=='direct' else 'G2' if args.source=='formula' else 'DECODER_DIRECT_DIAGNOSTIC' if args.estimator=='direct' else 'G3'
    result['angle_diagnostic_coordinate']='Tstd true axes' if args.source=='formula' else 'Raw latent identity only; NOT a parameter recovery error'
    if args.metric in HESSIAN_CATALOG and args.source=='formula' and args.estimator!='direct':
        reference=hessian_study_fields(structure_function(fn,args),w,d,third and full_w is None)
        reference['strict_x']=w[:,d:]
        if grid_groups is not None:reference['strict_subject_groups']=grid_groups
        if full_w is not None:
            target=structure_function(fn,args)
            reference['uu_h']=derivatives(target,full_w,d,False,True)['h']
            if third:
                def fixed(z,x):return target(torch.cat((z,x)))
                op=torch.func.jacfwd(torch.func.jacfwd(torch.func.jacrev(fixed,argnums=0),argnums=0),argnums=0)
                reference['uu_t']=np.concatenate([torch.vmap(op)(batch[:,:d],batch[:,d:]).detach().numpy() for batch in torch.tensor(full_w).split(128)])
                result['estimator_info']['UU_Zvariation_full_relative_error']=float(np.linalg.norm(fields['uu_t']-reference['uu_t'])/max(np.linalg.norm(reference['uu_t']),1e-20))
        reference_loss=HessianLoss(reference,cov,d,args.metric)
        actual=[];estimate=[];per_plane=[]
        for i,j in itertools.combinations(range(d),2):
            exact_plane=[];estimated_plane=[]
            for row in sweep:
                if (row['i'],row['j'])!=(i,j):continue
                rr=np.eye(d);aa=np.deg2rad(row['angle']);rr[i,i]=rr[j,j]=np.cos(aa);rr[i,j]=-np.sin(aa);rr[j,i]=np.sin(aa)
                exact_plane.append(reference_loss(rr));estimated_plane.append(row['loss'])
            actual.extend(exact_plane);estimate.extend(estimated_plane)
            per_plane.append(float(np.corrcoef(exact_plane,estimated_plane)[0,1]) if np.ptp(exact_plane)>1e-14 and np.ptp(estimated_plane)>1e-14 else 0.)
        relative=float(np.linalg.norm(np.asarray(actual)-estimate)/max(np.linalg.norm(actual),1e-20))
        result['gap2_audit']=dict(curve_relative_error=relative,minimum_plane_correlation=min(per_plane),plane_correlations=per_plane,
            estimated_axis_pass=angle_pass,pass_gate=bool(angle_pass and relative<=.1 and min(per_plane)>=.9),
            thresholds=dict(max_axis_error_degrees=args.g1_angle_tolerance,max_curve_relative_error=.1,min_plane_correlation=.9),
            explanation='Same W, same metric and block domains; exact derivatives are post-fit diagnostics only')
    result['plane_counterexample']=plane_counterexample
    result['scope']='hessian_only' if args.metric in HESSIAN_CATALOG else 'historical_catalogue'
    result['field_cache']=str(cache)
    result['field_provenance']=field_provenance
    result['code_hashes']={f.name:sha(f) for f in Path(__file__).resolve().parent.glob('*.py')}
    result['near10_diagnostic']=bool(best['axis_min']>np.cos(np.deg2rad(10)) and min(ranges)>1e-5*(1+abs(truth)))
    result['preprocessing']=dict(train_mean=c['mean'],components=c['components'],
                                 explained_variance_ratio=c['explained_variance_ratio'],covariance=cov,
                                 pca_whitened=False,source_latent_sha256=c['hashes']['latent'])
    result['matrix_metrics_best']=loss.describe(rhat)
    result['matrix_metrics_identity']=loss.describe(np.eye(d))
    dump(out/'summary.json',result)
    print(args.model,args.source,args.channel,args.metric,'truth',round(truth,6),'best',round(best['loss'],6),
          'axis_min',round(best['axis_min'],5),'G1_exact',strict_pass if args.source=='formula' else None,'G1_angle',angle_pass if args.source=='formula' else None,'max_true_axis_angle',round(worst_axis_error,3) if args.source=='formula' else None,'seconds',round(time.time()-started,1),flush=True)

if __name__=='__main__': main()
