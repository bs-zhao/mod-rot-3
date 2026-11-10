"""M1 local integral of Hessian fields over numerical latent neighborhoods."""
import numpy as np


def integrate_latent_hessian(fields,cov,d,bandwidth):
    result=dict(fields);groups=fields['strict_subject_groups'];ids=np.unique(groups)
    z=np.stack([fields['strict_z'][groups==sid][0] for sid in ids])
    blocks=np.stack([fields['h'][groups==sid] for sid in ids])
    if bandwidth=='mean':weights=np.full((len(ids),len(ids)),1/len(ids))
    elif bandwidth==0:weights=np.eye(len(ids))
    else:
        difference=z[:,None,:]-z[None,:,:]
        distance=np.maximum(np.einsum('ija,ab,ijb->ij',difference,np.linalg.pinv(cov),difference)/d,0)
        weights=np.exp(-distance/(2*float(bandwidth)**2));weights/=weights.sum(axis=1,keepdims=True)
    integrated=(weights@blocks.reshape(len(ids),-1)).reshape(blocks.shape)
    result['h']=integrated.reshape(fields['h'].shape);result['uu_h']=result['h'][:,:d,:d]
    result.pop('uu_rank',None);result.pop('uu_row_ids',None)
    result['_latent_smoothed']=bandwidth
    return result
