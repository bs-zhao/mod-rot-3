"""WY-only semiparametric unknown-link single-index interaction estimator.

The rank-SVM uses only the ordering of continuous conditional responses, so
any common strictly increasing output transform leaves its training labels
unchanged. This is a finite-dimensional RANK SURROGATE, not the exact maximum
rank correlation estimator, ACE, or GSAMUL.

CR3 binary choices: first learn E[Y|W] with the existing held-out MLP training
script. Pass --source-fit to learn ranks of its predicted probabilities. There
is no access to generator probabilities or oracle derivatives in this script.
"""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score,mean_squared_error
from basis import Basis

ROOT=Path(__file__).resolve().parents[2]


def rank_pairs(y,n_pairs,rng,min_rank_gap=.02):
    y=np.asarray(y,dtype=float).reshape(-1)
    if len(y)<20 or np.ptp(y)<=1e-9:
        raise ValueError('Continuous, nonconstant responses required for rank regression')
    ranks=rankdata(y,method='average')
    take_i=[];take_j=[];take_s=[];limit=min_rank_gap*len(y)
    for _ in range(40):
        need=n_pairs-len(take_i)
        if need<=0:break
        k=min(50000,max(1000,need*2))
        i=rng.integers(len(y),size=k)
        j=rng.integers(len(y)-1,size=k);j=j+(j>=i)
        good=np.flatnonzero(abs(ranks[i]-ranks[j])>limit)[:need]
        take_i.extend(i[good]);take_j.extend(j[good]);
        take_s.extend(np.sign(y[i[good]]-y[j[good]]).astype(int))
    if len(take_i)<min(100,n_pairs):
        raise RuntimeError('Not enough distinct rank pairs')
    return np.array(take_i,dtype=int),np.array(take_j,dtype=int),np.array(take_s,dtype=int)


def rank_accuracy(X,y,coef,n_pairs,seed):
    i,j,t=rank_pairs(y,n_pairs,np.random.default_rng(seed))
    return float(np.mean(np.sign((X[i]-X[j])@coef)==t))


def estimate(Wtr,ytr,Wv,yv,Wt,yt,zdim,method='rank_svm',z_degree=1,x_degree=2,
             pairs=12000,seed=31,C=.5):
    basis=Basis(z_degree,x_degree).fit(Wtr,zdim)
    tr=basis.transform(Wtr);va=basis.transform(Wv);te=basis.transform(Wt)
    start=time.perf_counter()
    if method=='rank_svm':
        i,j,sign=rank_pairs(ytr,pairs,np.random.default_rng(seed))
        design=(tr[i]-tr[j]);model=LinearSVC(C=C,fit_intercept=False,dual=False,
                 max_iter=2000,tol=1e-4,random_state=seed)
        model.fit(design,sign)
        coef=np.asarray(model.coef_[0],dtype=float)
        nit=int(np.max(np.atleast_1d(model.n_iter_)))
    elif method=='ridge_y':
        model=Ridge(alpha=1.,solver='lsqr').fit(tr,ytr)
        coef=np.asarray(model.coef_,dtype=float);nit=1
    else:raise ValueError(method)
    vtr=tr@coef; vv=va@coef; vt=te@coef
    # An unconstrained monotone calibration estimates the outer link from WY.
    cal=IsotonicRegression(out_of_bounds='clip').fit(vtr,ytr)
    yp=cal.predict(vt)
    details={'method':method,'z_degree':z_degree,'x_degree':x_degree,
             'train_rows':int(len(tr)),'val_rows':int(len(va)),'test_rows':int(len(te)),
             'rank_pairs_used':int(pairs) if method=='rank_svm' else 0,
             'iterations':nit,'elapsed_seconds':float(time.perf_counter()-start),
             'test_rank_accuracy':rank_accuracy(te,yt,coef,min(3500,len(te)*4),seed+55),
             'val_rank_accuracy':rank_accuracy(va,yv,coef,min(3500,len(va)*4),seed+53),
             'test_calibrated_r2':float(r2_score(yt,yp)),
             'test_calibrated_rmse':float(np.sqrt(mean_squared_error(yt,yp))),
             'warning':'Rank-SVM is not exact maximum-rank-correlation; polynomial basis may miss nonlinear CR3 structure.'}
    return basis,coef,basis.coefficient_matrix(coef),details


def load_data(data,max_train,seed,source_fit=None):
    d=Path(data);meta=json.loads((d/'meta.json').read_text(encoding='utf8'))
    blocks=[]
    for name in ('train','val','test'):
        with np.load(d/(name+'.npz'),allow_pickle=False) as f:
            blocks.append((np.asarray(f['W'],dtype=float),np.asarray(f['y'],dtype=float).reshape(-1)))
    y_source='observed_continuous_Y'
    if meta['label']=='bernoulli':
        if source_fit is None:
            raise ValueError('Bernoulli 0/1 data cannot be ranked as conditional probabilities. '
                             'Supply --source-fit trained from choices to estimate E[Y|W].')
        old=ROOT/'math'/'11_wy_nonlinear_interactions'
        sys.path.insert(0,str(old))
        from models import load_fit
        model,info=load_fit(source_fit)
        if info['model']!=meta['model'] or info['label']!='bernoulli':
            raise ValueError('Source fitted model must match dataset and Bernoulli label')
        blocks=[(w,np.asarray(model.predict(w),dtype=float).reshape(-1)) for w,_ in blocks]
        y_source='wy_trained_probability_surrogate'
    elif source_fit is not None:
        raise ValueError('--source-fit only allowed with Bernoulli labels')
    if meta['label'] not in ('bernoulli','prob','logit','softplus'):
        raise ValueError('Unsupported label')
    if len(np.unique(blocks[0][1])) <= 2:
        raise ValueError('Continuous response with more than 2 unique values needed')
    w,y=blocks[0]
    if max_train>0 and len(w)>max_train:
        idx=np.random.default_rng(seed+800).choice(len(w),max_train,replace=False)
        blocks[0]=(w[idx],y[idx])
    return meta,blocks,y_source


def train(data,output,method='rank_svm',seed=31,max_train=9000,
          n_pairs=12000,C=.5,z_degree=None,x_degree=2,source_fit=None):
    meta,blocks,y_source=load_data(data,max_train,seed,source_fit)
    if z_degree is None:
        z_degree=2 if meta['model']=='CR3' else 1
    basis,coef,B,stats=estimate(*(item for pair in blocks for item in pair),
                 meta['z_dim'],method,z_degree,x_degree,n_pairs,seed,C)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # SAME deterministic test anchors for oracle and WY, no labels revealed to scan.
    wt=blocks[2][0]
    anchor_idx=np.random.default_rng(4004).choice(len(wt),size=min(200,len(wt)),replace=False)
    anchors=wt[anchor_idx,:meta['z_dim']]
    np.savez_compressed(out/'coefficients.npz',coef_scaled=coef,B=B,anchors=anchors)
    np.savetxt(out/'zx_coefficients.csv',B,delimiter=',',fmt='%.9g')
    result={**stats,'source':'wy_only','response_source':y_source,
            'model':meta['model'],'label':meta['label'],'zdim':int(meta['z_dim']),
            'xdim':int(meta['x_dim']),'seed':seed,'n_pairs_requested':n_pairs,
            'basis':basis.serialize(),'source_fit':source_fit,
            'strict_WY_only':True}
    (out/'fit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='basis'},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['rank_svm','ridge_y'],default='rank_svm')
    p.add_argument('--max-train',type=int,default=9000)
    p.add_argument('--n-pairs',type=int,default=12000)
    p.add_argument('--seed',type=int,default=31)
    p.add_argument('--C',type=float,default=.5)
    p.add_argument('--z-degree',type=int,choices=[1,2,3])
    p.add_argument('--x-degree',type=int,choices=[1,2],default=2)
    p.add_argument('--source-fit',help='WY-trained choice probability predictor for Bernoulli condition')
    args=p.parse_args()
    train(args.data,args.output,args.method,args.seed,args.max_train,args.n_pairs,
          args.C,args.z_degree,args.x_degree,args.source_fit)
