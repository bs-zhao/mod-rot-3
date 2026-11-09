"""One candidate per invocation; independent G1/G2/G3 experiment entry point."""
import argparse
import csv
import itertools
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import minimize
from core import DIMS, load, dump, formula, decoder, derivatives, correspondence, sha
from metrics import METRICS, Loss

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

def structure_function(fn,args):
    if args.channel!='logit' and args.structure_link=='log_odds':
        def linked(w): return torch.logit(fn(w).clamp(1e-12,1-1e-12))
        return linked
    return fn

def bank_fields(args,c,fn,w,v,protocol,rows):
    from estimators import BankSieve
    if args.model!='CR3' or not args.metric.startswith('conditional_'):
        raise ValueError('Fixed-bank sieve currently supports CR3 conditional loss only')
    rng=np.random.default_rng(args.seed+411);d=c['d'];data=c['data']
    indices=np.flatnonzero(data['subject_index']==0)
    order=np.argsort(data['trial_index'][indices]);indices=indices[order]
    bank=data['x'][indices];trial=data['trial_index'][indices]
    if len(bank)!=500 or not np.array_equal(trial,np.arange(500)): raise ValueError('Wrong bank indexing')
    xs=(bank-protocol['x_mean'])/protocol['x_sd'];tables=[];ids_used=[]
    for split,n in [('train',args.train_subjects),('validation',args.validation_subjects)]:
        ids=rng.choice(c['splits'][split],min(n,len(c['splits'][split])),replace=False);ids_used.append(ids)
        # Verify fixed X bank for the subjects used, not just subject zero.
        for sid in ids:
            ix=np.flatnonzero(data['subject_index']==sid)
            ix=ix[np.argsort(data['trial_index'][ix])]
            if not np.array_equal(data['x'][ix],bank): raise ValueError('Subject does not share CR3 bank')
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
    sieve.eval_stimulus=data['trial_index'][rows].astype(int)
    f=sieve.fields(w[:,:d]@hidden.T,False,args.structure_link)
    fields=dict(g=f['g']@hidden,h=np.einsum('ai,nab,bj->nij',hidden,f['h'],hidden,optimize=True))
    reference=derivatives(structure_function(fn,args),w,d,False,True)
    errors={k:float(np.linalg.norm(fields[k]-reference[k])/max(np.linalg.norm(reference[k]),1e-20)) for k in fields}
    return fields,dict(**sieve.info,relative_errors=errors,hidden_rotation=hidden,train_ids=ids_used[0],validation_ids=ids_used[1],
                       structure_link=args.structure_link)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model',choices=DIMS,required=True)
    p.add_argument('--beta',type=float,required=True); p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--source',choices=['formula','decoder'],default='formula')
    p.add_argument('--estimator',choices=['direct','finite_difference','polynomial','stimulus'],default='direct')
    p.add_argument('--channel',choices=['logit','probability','binary'],default='logit')
    p.add_argument('--metric',choices=METRICS,required=True)
    p.add_argument('--samples',type=int,default=384); p.add_argument('--starts',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261108); p.add_argument('--step',type=float,default=.01)
    p.add_argument('--maxiter',type=int,default=250); p.add_argument('--label',default='round01')
    p.add_argument('--degree',type=int,default=3);p.add_argument('--ridge',type=float,default=.001)
    p.add_argument('--train-rows',type=int,default=32000);p.add_argument('--validation-rows',type=int,default=8000)
    p.add_argument('--train-subjects',type=int,default=1024);p.add_argument('--validation-subjects',type=int,default=256)
    p.add_argument('--structure-link',choices=['native','log_odds'],default='native')
    args=p.parse_args(); torch.set_num_threads(2); started=time.time()
    if args.channel=='binary' and args.estimator not in ['polynomial','stimulus']:
        raise ValueError('Binary observations require a Bernoulli sieve; use --estimator polynomial')
    c=load(args); d=c['d']; nangle=d*(d-1)//2
    out=Path(__file__).resolve().parent/'results'/args.label/f'{args.model}_beta{args.beta:g}_zdim{args.zdim}_pca{args.pca}'/args.source/args.channel/args.estimator/args.metric/f'seed{args.seed}'
    out=out.with_name(out.name+f'_n{args.samples}_starts{args.starts}_degree{args.degree}_ridge{args.ridge:g}_link{args.structure_link}_nt{args.train_rows}_nv{args.validation_rows}_ns{args.train_subjects}_vs{args.validation_subjects}_h{args.step:g}')
    out.mkdir(parents=True,exist_ok=True)
    # One field cache shared among candidate losses, no fitted R or T labels.
    third=args.metric.startswith('signature') or args.metric=='zzx_mixed_rms'
    fn,w,cov,v,rows,protocol=prepare(args,c)
    latent_only=args.metric.startswith('conditional_')
    cache=out.parents[1]/f'fields_n{args.samples}_s{args.seed}_h{args.step:g}_third{int(third)}_degree{args.degree}_ridge{args.ridge:g}_nt{args.train_rows}_nv{args.validation_rows}_latent{int(latent_only)}_link{args.structure_link}_ns{args.train_subjects}_vs{args.validation_subjects}.npz'
    estimator_info=None
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['checkpoint_hash']) != c['hashes']['checkpoint']: raise ValueError('Stale field cache')
            fields={k:saved[k] for k in ['g','h','third'] if k in saved.files}
        if cache.with_suffix('.json').exists():
            import json
            estimator_info=json.loads(cache.with_suffix('.json').read_text())
    else:
        if args.estimator=='polynomial':
            fields,estimator_info=polynomial_fields(args,c,fn,w,v,protocol,third)
            dump(cache.with_suffix('.json'),estimator_info)
        elif args.estimator=='stimulus':
            fields,estimator_info=bank_fields(args,c,fn,w,v,protocol,rows)
            dump(cache.with_suffix('.json'),estimator_info)
        else:
            target=structure_function(fn,args)
            fields=(derivatives(target,w,d,third,latent_only) if args.estimator=='direct' else numerical_fields(target,w,d,args.step,third,latent_only))
        np.savez_compressed(cache,**fields,checkpoint_hash=c['hashes']['checkpoint'])
    if not all(np.isfinite(f).all() for f in fields.values()): raise ValueError('Nonfinite derivative')
    if latent_only: fields['groups']=protocol['groups']
    loss=Loss(fields,cov,d,args.metric)
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
    # Include identity only as analytic diagnostic; decoder starts NEVER use T/oracle directions.
    for start in range(args.starts):
        x0=np.zeros(nangle) if start==0 else rng.uniform(-np.pi,np.pi,nangle)
        fit=minimize(lambda a:loss(rotation(a,d)),x0,method='L-BFGS-B', options=dict(maxiter=args.maxiter,ftol=1e-12,gtol=1e-7))
        r=rotation(fit.x,d)
        from scipy.optimize import linear_sum_assignment
        ii,jj=linear_sum_assignment(-abs(r))
        solutions.append(dict(start=start,loss=float(fit.fun),initial_loss=float(loss(rotation(x0,d))),axis_min=float(abs(r[ii,jj]).min()),axis_mean=float(abs(r[ii,jj]).mean()),
                              success=bool(fit.success),message=str(fit.message),iterations=fit.nit,r=r))
    best=min(solutions,key=lambda x:x['loss']); rhat=best['r']
    # Lock R before any TEST labels are read. No labels are used to select candidates.
    np.save(out/'rotation.npy',rhat)
    sweep=[]
    for i,j in itertools.combinations(range(d),2):
        for angle in np.linspace(0,90,37):
            r=np.eye(d); a=np.deg2rad(angle); r[i,i]=r[j,j]=np.cos(a);r[i,j]=-np.sin(a);r[j,i]=np.sin(a)
            sweep.append(dict(i=i,j=j,angle=angle,loss=loss(r)))
    with (out/'planes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(sweep[0]));writer.writeheader();writer.writerows(sweep)
    ranges=[np.ptp([x['loss'] for x in sweep if x['i']==i and x['j']==j]) for i,j in itertools.combinations(range(d),2)]
    strict_pass=(best['axis_min']>.995 and best['loss']>=truth-1e-6*(1+abs(truth)) and min(ranges)>1e-5*(1+abs(truth)))
    evaluation=None
    if args.source=='decoder':
        test=c['splits']['test']
        # First access to true T in decoder branch happens AFTER R is locked.
        with np.load(c['latent_path'],allow_pickle=False) as latent: t=latent['theta_true']
        evaluation=correspondence(v[test]@rhat.T,t[test])
    result=dict(args=vars(args),hashes=c['hashes'],protocol=protocol,rows=rows,train_ids=c['splits']['train'],
                invariance_max=max(audits),truth_loss=truth,best=best,solutions=solutions,plane_ranges=ranges,
                gap1_strict_pass=bool(strict_pass) if args.source=='formula' else None,
                evaluation=evaluation,elapsed_seconds=time.time()-started)
    result['estimator_info']=estimator_info
    result['code_hashes']={f.name:sha(f) for f in Path(__file__).resolve().parent.glob('*.py')}
    result['near10_diagnostic']=bool(best['axis_min']>np.cos(np.deg2rad(10)) and min(ranges)>1e-5*(1+abs(truth)))
    dump(out/'summary.json',result)
    print(args.model,args.source,args.channel,args.metric,'truth',round(truth,6),'best',round(best['loss'],6),
          'axis_min',round(best['axis_min'],5),'G1',strict_pass,'seconds',round(time.time()-started,1),flush=True)

if __name__=='__main__': main()
