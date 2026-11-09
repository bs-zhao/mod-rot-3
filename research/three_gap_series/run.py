"""One candidate per invocation; independent G1/G2/G3 experiment entry point."""
import argparse
import csv
import itertools
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import minimize
from core import DIMS, load, dump, formula, decoder, derivatives, correspondence
from metrics import METRICS, Loss

def rotation(angles, d):
    r=np.eye(d)
    for (i,j), a in zip(itertools.combinations(range(d),2),angles):
        c,s=np.cos(a),np.sin(a)
        rowi=r[i].copy(); rowj=r[j].copy()
        r[i]=c*rowi-s*rowj; r[j]=s*rowi+c*rowj
    return r

def numerical_fields(fn, w, d, step, third=False):
    """E09 query-access central differences. Receives only point->value oracle.

    Does not access any expression or derivative. Not an original-table estimator.
    """
    if third: raise ValueError('F3 finite differences not implemented')
    n,p=w.shape
    def query(v):
        with torch.no_grad():
            return torch.vmap(fn)(torch.tensor(v,dtype=torch.float64)).numpy()
    f=query(w); g=np.zeros((n,p)); h=np.zeros((n,p,p))
    for i in range(p):
        vi=np.eye(p)[i]*step
        plus,minus=query(w+vi),query(w-vi)
        g[:,i]=(plus-minus)/(2*step)
        h[:,i,i]=(plus-2*f+minus)/step**2
        for j in range(i):
            vj=np.eye(p)[j]*step
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
    if args.model=='CR3':
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
    if args.channel=='binary':
        raise ValueError('Binary requires a Bernoulli estimator; differentiating draws is prohibited')
    w=np.concatenate((v[subject[rows]],xs),axis=1)
    return fn,w,np.cov(v[train],rowvar=False,bias=True),v,rows,dict(x_mean=xm,x_sd=sx,domain_fraction=len(candidates)/np.isin(subject,train).sum())

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model',choices=DIMS,required=True)
    p.add_argument('--beta',type=float,required=True); p.add_argument('--zdim',type=int,required=True)
    p.add_argument('--pca',type=int,choices=[0,1],required=True)
    p.add_argument('--source',choices=['formula','decoder'],default='formula')
    p.add_argument('--estimator',choices=['direct','finite_difference'],default='direct')
    p.add_argument('--channel',choices=['logit','probability','binary'],default='logit')
    p.add_argument('--metric',choices=METRICS,required=True)
    p.add_argument('--samples',type=int,default=384); p.add_argument('--starts',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261108); p.add_argument('--step',type=float,default=.01)
    p.add_argument('--maxiter',type=int,default=250); p.add_argument('--label',default='round01')
    args=p.parse_args(); torch.set_num_threads(2); started=time.time()
    c=load(args); d=c['d']; nangle=d*(d-1)//2
    out=Path(__file__).resolve().parent/'results'/args.label/f'{args.model}_beta{args.beta:g}_zdim{args.zdim}_pca{args.pca}'/args.source/args.channel/args.estimator/args.metric/f'seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    # One field cache shared among candidate losses, no fitted R or T labels.
    third=args.metric.startswith('signature') or args.metric=='zzx_mixed_rms'
    fn,w,cov,v,rows,protocol=prepare(args,c)
    cache=out.parents[1]/f'fields_n{args.samples}_s{args.seed}_h{args.step:g}_third{int(third)}.npz'
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['checkpoint_hash']) != c['hashes']['checkpoint']: raise ValueError('Stale field cache')
            fields={k:saved[k] for k in ['g','h','third'] if k in saved.files}
    else:
        fields=(derivatives(fn,w,d,third) if args.estimator=='direct' else numerical_fields(fn,w,d,args.step,third))
        np.savez_compressed(cache,**fields,checkpoint_hash=c['hashes']['checkpoint'])
    if not all(np.isfinite(f).all() for f in fields.values()): raise ValueError('Nonfinite derivative')
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
        solutions.append(dict(start=start,loss=float(fit.fun),axis_min=float(abs(r[ii,jj]).min()),axis_mean=float(abs(r[ii,jj]).mean()),
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
    dump(out/'summary.json',result)
    print(args.model,args.source,args.channel,args.metric,'truth',round(truth,6),'best',round(best['loss'],6),
          'axis_min',round(best['axis_min'],5),'G1',strict_pass,'seconds',round(time.time()-started,1),flush=True)

if __name__=='__main__': main()
