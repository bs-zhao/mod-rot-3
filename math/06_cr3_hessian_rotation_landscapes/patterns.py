"""Hessian recovery diagnostics: numeric accuracy AND structural pattern.

For symmetric blocks, count each off-diagonal pair exactly ONCE.
Do not report a near-perfect cosine as success if the relative scale
or the important sign pattern is very wrong.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

PARTS=("ZZ","ZZ_diag","ZZ_offdiag","ZX","XX","XX_diag","XX_offdiag","FULL")


def _indices(name):
    if name.startswith("ZZ"):
        rows,cols=np.indices((4,4))
        condition=np.ones((4,4),dtype=bool)
        if name=="ZZ_diag":condition=rows==cols
        else:condition=rows<=cols
        if name=="ZZ_offdiag":condition=rows<cols
        return rows[condition],cols[condition]
    if name.startswith("XX"):
        rows,cols=np.indices((6,6))
        condition=np.ones((6,6),dtype=bool)
        if name=="XX_diag":condition=rows==cols
        else:condition=rows<=cols
        if name=="XX_offdiag":condition=rows<cols
        return 4+rows[condition],4+cols[condition]
    if name=="ZX":
        r,c=np.indices((4,6))
        return r.ravel(),(4+c).ravel()
    if name=="FULL":
        r,c=np.triu_indices(10)
        return r,c
    raise KeyError(name)


def score(true,estimated,validity,name,strong_fraction=.1):
    a=np.asarray(true,dtype=float)
    b=np.asarray(estimated,dtype=float)
    support=np.asarray(validity,dtype=float)
    if a.shape!=b.shape or a.shape!=(10,10):raise ValueError("Expected two 10x10 Hessians")
    i,j=_indices(name)
    ok=(np.isfinite(a[i,j])&np.isfinite(b[i,j])&
        (support[i,j]>0))
    x=a[i[ok],j[ok]];y=b[i[ok],j[ok]]
    if len(x)==0:return {"n":0}
    diff=y-x
    na=float(np.linalg.norm(x));nb=float(np.linalg.norm(y))
    cosine=float(x@y/(na*nb)) if na>1e-16 and nb>1e-16 else None
    xx=x-x.mean();yy=y-y.mean()
    denom=float(np.linalg.norm(xx)*np.linalg.norm(yy))
    pearson=float(xx@yy/denom) if denom>1e-16 else None
    ar=np.abs(x);br=np.abs(y)
    spearman=float(spearmanr(ar,br).statistic) if len(x)>=3 and (
        np.ptp(ar)>1e-15 and np.ptp(br)>1e-15) else None
    significant=ar>=max(np.max(ar)*strong_fraction,1e-12)
    strong_n=int(significant.sum())
    sign_match=(np.sign(y[significant])==np.sign(x[significant]))
    q=np.square(diff)
    qsum=float(q.sum())
    top=np.sort(q)[::-1]
    return dict(
        n=int(len(x)),cosine=cosine,pearson=pearson,
        abs_rank_spearman=spearman,mae=float(np.mean(abs(diff))),
        rmse=float(np.sqrt(np.mean(diff*diff))),
        relative_frobenius=float(np.linalg.norm(diff)/max(na,1e-15)),
        norm_ratio=float(nb/max(na,1e-15)),
        strong_threshold_fraction=float(strong_fraction),
        important_count=strong_n,
        important_sign_agreement=float(sign_match.mean()) if strong_n else None,
        important_sign_correct=int(sign_match.sum()) if strong_n else 0,
        top1_squared_error_fraction=float(top[0]/qsum) if qsum>0 else 0.,
        top2_squared_error_fraction=float(top[:2].sum()/qsum) if qsum>0 else 0.,
        true_frobenius=na, estimated_frobenius=nb,
    )


def all_blocks(true,estimated,validity):
    return {key:score(true,estimated,validity,key) for key in PARTS}


def flatten_rows(source,method,parameters,channel,metric,blocks):
    rows=[]
    for block,fields in blocks.items():
        rows.append({"source":source,"method":method,
                     "parameters":parameters,"channel":channel,
                     "metric":metric,"block":block,**fields})
    return rows
