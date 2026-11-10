"""Black-box interaction estimands. NO access to generators or oracle data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def _rms_mixed_fd(predict, W, step, valid):
    """Mean-square mixed partial via 4 corner evaluations; invalid pairs=NaN."""
    n,p=W.shape; out=np.full((p,p),np.nan)
    base=np.asarray(predict(W),dtype=float).reshape(-1)
    for i in range(p):
        if not valid[i]:continue
        a=W.copy();a[:,i]+=step[i]
        b=W.copy();b[:,i]-=step[i]
        fa=np.asarray(predict(a));fb=np.asarray(predict(b))
        out[i,i]=np.sqrt(np.mean(((fa-2*base+fb)/step[i]**2)**2))
        for j in range(i+1,p):
            if not valid[j]:continue
            pp=W.copy();pm=W.copy();mp=W.copy();mm=W.copy()
            pp[:,[i,j]] += step[[i,j]]
            pm[:,i]+=step[i];pm[:,j]-=step[j]
            mp[:,i]-=step[i];mp[:,j]+=step[j]
            mm[:,[i,j]] -= step[[i,j]]
            val=(predict(pp)-predict(pm)-predict(mp)+predict(mm))/(4*step[i]*step[j])
            out[i,j]=out[j,i]=np.sqrt(np.mean(np.asarray(val)**2))
    return out


def hessian_rms(model,W,valid,batch=12):
    """Autodiff of smooth trained neural predictor on physical W coordinates."""
    import torch
    torch.set_num_threads(min(torch.get_num_threads(),4))
    fn=lambda x:model.tensor_forward(x[None,:]).sum()
    h=torch.func.hessian(fn)
    result=[]
    for sub in np.array_split(W,max(1,int(np.ceil(len(W)/batch)))):
        xx=torch.as_tensor(sub,dtype=torch.float32,device=model.device)
        result.append(torch.vmap(h)(xx).detach().cpu().numpy())
    hs=np.concatenate(result)
    a=np.sqrt(np.mean(hs.astype('float64')**2,axis=0))
    for j,ok in enumerate(valid):
        if not ok:a[j,:]=np.nan;a[:,j]=np.nan
    return a


def friedman_pd(predict,W,valid,background=48,grid=8,seed=19):
    """Centered partial-dependence H interaction; pair scores not Hessian RMS.
    All PD interventions may leave the observed joint support; CR3 X excluded.
    """
    rng=np.random.default_rng(seed)
    n,p=W.shape
    base=W[rng.choice(n,min(background,n),replace=False)]
    gridvals=[np.unique(np.quantile(W[:,i],np.linspace(.08,.92,grid))) for i in range(p)]
    score=np.full((p,p),np.nan)
    for i in range(p):
        if not valid[i]:continue
        score[i,i]=0.
        for j in range(i+1,p):
            if not valid[j]:continue
            gj=gridvals[j];gi=gridvals[i]
            stack=np.tile(base,(len(gi)*len(gj),1))
            ii=np.repeat(gi,len(gj));jj=np.tile(gj,len(gi))
            stack[:,i]=np.repeat(ii,len(base))
            stack[:,j]=np.repeat(jj,len(base))
            pd=predict(stack).reshape(len(gi),len(gj),len(base)).mean(axis=2)
            pd=pd-pd.mean()
            pair=pd-pd.mean(axis=1,keepdims=True)-pd.mean(axis=0,keepdims=True)
            denom=max(float(np.mean(pd**2)),1e-12)
            score[i,j]=score[j,i]=float(np.sqrt(np.mean(pair**2)/denom))
    return score


def pure_anova(predict,W,valid,order=3,anchors=64,background=48,seed=19):
    """Monte Carlo intervention-product functional ANOVA (approximate).
    Requires interpretation under a product reference measure; empirical
    columns are sampled independently. Returns weights by subset and order.
    """
    from itertools import combinations
    rng=np.random.default_rng(seed)
    p=W.shape[1];use=np.flatnonzero(valid)
    a=W[rng.choice(len(W),min(anchors,len(W)),replace=False)]
    bg=W[rng.choice(len(W),min(background,len(W)),replace=False)].copy()
    # Construct synthetic product-distribution backgrounds via column-wise shuffle.
    for j in use:bg[:,j]=bg[rng.permutation(len(bg)),j]
    f0=float(np.mean(predict(bg)))
    funcs={():np.full(len(a),f0)}
    scores={}; pairs=np.zeros((p,p),dtype=float)
    for k in range(1,min(order,len(use))+1):
        for subset in combinations(use,k):
            w=np.repeat(bg[None,:,:],len(a),axis=0)
            for j in subset:w[:,:,j]=a[:,j,None]
            pd=np.asarray(predict(w.reshape(-1,p))).reshape(len(a),len(bg)).mean(axis=1)
            component=pd.copy()
            for subset2, val in funcs.items():
                if set(subset2).issubset(subset):component-=val
            funcs[subset]=component
            var=float(np.mean(component**2))
            scores[','.join(map(str,subset))]=var
            if len(subset)>=2:
                for i,j in combinations(subset,2):pairs[i,j]+=var;pairs[j,i]+=var
    by_order={str(k):float(sum(v for name,v in scores.items() if len(name.split(','))==k))
              for k in range(1,min(order,len(use))+1)}
    # Normalization is not forced to sum to 1: finite MC does not enforce orthogonality.
    return pairs,{'component_variance':scores,'by_order_variance':by_order,'mean_reference':f0,
                  'warning':'MC product-intervention ANOVA; not valid as observational dependent-input Sobol.'}


def stimulus_modulation(predict,W,zdim,seed=19,n_pairs=128,n_points=64):
    """CR3 valid-bank estimand: Z effect on differences between two REAL stimuli.
    Not a per-X-coordinate interaction matrix.
    """
    rng=np.random.default_rng(seed)
    if zdim>=W.shape[1]:return np.full(zdim,np.nan)
    ids=rng.choice(len(W),min(n_points,len(W)),replace=False)
    z=W[ids,:zdim]
    xs=np.unique(W[:,zdim:],axis=0)
    if len(xs)<2:return np.full(zdim,np.nan)
    v=np.maximum(W[:,:zdim].std(axis=0)*.04,1e-4)
    i1=rng.integers(len(xs),size=n_pairs);i2=rng.integers(len(xs),size=n_pairs)
    valid=i1!=i2;i1=i1[valid];i2=i2[valid]
    # n_z * n_pairs, with real, non-synthetic stimuli; pair may differ in all features.
    zz=np.repeat(z,len(i1),axis=0)
    aa=np.tile(xs[i1],(len(z),1)); bb=np.tile(xs[i2],(len(z),1))
    a=np.c_[zz,aa];b=np.c_[zz,bb]
    dif=predict(a)-predict(b)
    result=[]
    for j in range(zdim):
        shifted=zz.copy();shifted[:,j]+=v[j]
        da=predict(np.c_[shifted,aa])-predict(np.c_[shifted,bb])
        result.append(float(np.sqrt(np.mean(((da-dif)/v[j])**2))))
    return np.asarray(result)


def score_matrix(matrix,zdim):
    zx=np.asarray(matrix[:zdim,zdim:],dtype=float)
    if zx.size==0 or not np.isfinite(zx).any():return {'zx_l1':None,'zx_effective_degree':None}
    weights=np.nan_to_num(zx,nan=0.)**2
    sums=weights.sum(axis=0)
    probs=np.divide(weights,sums[None,:],out=np.zeros_like(weights),where=sums[None,:]>0)
    entropy=-(np.where(probs>0,probs*np.log(np.maximum(probs,1e-12)),0.)).sum(axis=0)
    degree=np.where(sums>0,np.exp(entropy),0.)
    return {'zx_l1':float(np.nansum(zx)),'zx_effective_degree':float(degree.sum()),
            'zx_effective_per_x':degree.tolist()}


def detect(predict,W,zdim,model_name,method='fd',n_eval=96,seed=31,order=3,mlp_obj=None):
    rng=np.random.default_rng(seed)
    sample=W[rng.choice(len(W),min(n_eval,len(W)),replace=False)].astype(float)
    # CR3 stimulus coordinates are discrete with zeros and probability boundaries.
    valid=np.array([True]*zdim+[model_name!='CR3']*(W.shape[1]-zdim))
    step=np.maximum(.025*W.std(axis=0),1e-4)
    details={'detector':method,'n_eval':len(sample),'valid_coords':valid.tolist()}
    if method=='hessian':
        if mlp_obj is None:raise ValueError('Autodiff Hessian requires --method mlp')
        matrix=hessian_rms(mlp_obj,sample,valid)
    elif method=='fd':matrix=_rms_mixed_fd(predict,sample,step,valid)
    elif method=='friedman':matrix=friedman_pd(predict,sample,valid,seed=seed)
    elif method=='anova':
        matrix, extra=pure_anova(predict,sample,valid,order=order,seed=seed)
        details.update(extra)
        for i in range(W.shape[1]):
            if not valid[i]:matrix[i,:]=np.nan;matrix[:,i]=np.nan
    else:raise ValueError(method)
    details.update(score_matrix(matrix,zdim))
    if model_name=='CR3':
        details['valid_stimulus_z_modulation']=stimulus_modulation(predict,W,zdim,seed=seed).tolist()
        details['cr3_note']='ZX/XX derivative-based scores intentionally undefined for discrete bank; report valid lottery contrasts.'
    return matrix,details


def save_result(matrix,details,out,meta):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savetxt(out/'interaction_matrix.csv',matrix,delimiter=',',fmt='%.9g')
    (out/'interaction.json').write_text(json.dumps(details,indent=2),encoding='utf8')
    fig,ax=plt.subplots(figsize=(7,6))
    image=ax.imshow(matrix,interpolation='nearest',aspect='equal')
    z=meta['z_dim'];ax.axvline(z-.5,color='white',lw=1.5);ax.axhline(z-.5,color='white',lw=1.5)
    ax.set_title(f"{meta['model']} | {details['detector']} | {meta['label']}")
    ax.set_xlabel('Input j (Z then X)');ax.set_ylabel('Input i (Z then X)')
    fig.colorbar(image,ax=ax,label='interaction score');fig.tight_layout()
    fig.savefig(out/'interaction_matrix.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    from models import load_fit
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--fit',required=True);p.add_argument('--output',required=True)
    p.add_argument('--detector',choices=['hessian','fd','friedman','anova'],default='fd')
    p.add_argument('--n-eval',type=int,default=96);p.add_argument('--order',type=int,default=3)
    p.add_argument('--seed',type=int,default=31)
    args=p.parse_args();data=Path(args.data)
    meta=json.loads((data/'meta.json').read_text())
    with np.load(data/'test.npz') as d:W=d['W']
    m, info=load_fit(args.fit)
    predict=(lambda x:m.predict_proba(x)[:,1]) if info['method']=='ebm' and meta['label']=='bernoulli' else m.predict
    matrix,details=detect(predict,W,meta['z_dim'],meta['model'],args.detector,args.n_eval,args.seed,
                          args.order,m if info['method']=='mlp' and args.detector=='hessian' else None)
    details.update({'model':meta['model'],'target_scale':meta['label']})
    save_result(matrix,details,args.output,meta)
    print(json.dumps(details,indent=2))
