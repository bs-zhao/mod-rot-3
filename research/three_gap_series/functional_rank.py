"""Separation complexity of each Hessian interaction as a subject x X function."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def project_hessian_functions(fields,d,degree):
    groups=fields['strict_subject_groups'];ids=np.unique(groups);z=[];hu=[];hx=[]
    for sid in ids:
        mask=groups==sid;z.append(fields['strict_z'][mask][0]);hu.append(fields['h'][mask,:d,:d]);hx.append(fields['h'][mask,:d,d:])
    x=fields['strict_x'][groups==ids[0]];z=np.asarray(z);hu=np.asarray(hu);hx=np.asarray(hx)
    qz=np.linalg.qr(PolynomialFeatures(degree).fit_transform(z),mode='reduced')[0]
    qx=np.linalg.qr(PolynomialFeatures(degree).fit_transform(x),mode='reduced')[0]
    scale=np.sqrt(len(z)*len(x))
    cu=np.einsum('su,sxab,xv->abuv',qz,hu,qx,optimize=True)/scale
    cx=np.einsum('su,sxap,xv->apuv',qz,hx,qx,optimize=True)/scale
    return cx,cu


def transformed_rank_matrices(tensors,a,part):
    hx,hu=tensors;d=len(a)
    ux=np.einsum('ia,apuv->ipuv',a,hx,optimize=True)
    matrices=ux.reshape(-1,*ux.shape[-2:])
    if part=='edges':
        uu=np.einsum('ia,abuv,jb->ijuv',a,hu,a,optimize=True)
        ix=np.triu_indices(d)
        matrices=np.concatenate((uu[ix[0],ix[1]],matrices),axis=0)
    return matrices


def rank_spectrum(tensors,a,part,scale):
    matrices=transformed_rank_matrices(tensors,a,part)/scale
    if matrices.shape[-1]<=matrices.shape[-2]:gram=matrices.transpose(0,2,1)@matrices
    else:gram=matrices@matrices.transpose(0,2,1)
    return np.maximum(np.linalg.eigvalsh(gram),0)


def rank_score(eigenvalues,score):
    if score.startswith('soft@'):
        threshold=float(score.split('@')[1]);values=eigenvalues/(eigenvalues+threshold**2)
    elif score=='log01':values=np.log1p(eigenvalues/.1**2)
    else:
        power=1. if score=='schatten1' else .5;eps=1e-4
        values=(eigenvalues+eps**2)**(power/2)-eps**power
    return float(np.mean(np.sum(values,axis=-1)))
