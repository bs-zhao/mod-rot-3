"""Hessian-only local pattern normalization, using the numerical TRAIN metric."""
import numpy as np


def normalize_hessian_field(fields,cov,d,kind):
    result=dict(fields);h=fields['h'];hu=h[:,:d,:d];hx=h[:,:d,d:]
    energy=np.maximum(np.einsum('nab,bc,ncd,da->n',hu,cov,hu,cov,optimize=True)+
        np.einsum('nap,ab,nbp->n',hx,cov,hx,optimize=True),0)
    if kind=='common':scale=np.ones(len(h))
    elif kind=='point':scale=np.sqrt(energy)
    else:
        groups=fields['strict_subject_groups'];ids=np.unique(groups);blocks=np.stack([energy[groups==sid] for sid in ids])
        if kind=='subject':scale=np.concatenate([np.full(len(row),np.sqrt(row.mean())) for row in blocks])
        elif kind=='stimulus':scale=np.tile(np.sqrt(blocks.mean(axis=0)),len(ids))
        else:raise ValueError(kind)
    if kind!='common':scale=np.maximum(scale,max(float(.03*np.sqrt(energy.mean())),1e-12))
    result['h']=h/scale[:,None,None]
    # All graph components now use the same differentiable product-grid domain.
    result['uu_h']=result['h'][:,:d,:d]
    result.pop('uu_rank',None);result.pop('uu_row_ids',None)
    result['_field_scaled']=kind
    return result
