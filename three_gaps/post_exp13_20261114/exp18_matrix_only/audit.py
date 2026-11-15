"""Read-only post-fit teacher diagnostics. Never called by a fitter or selector."""
import numpy as np
import torch
from torch.func import jacfwd,jacrev,vmap
import core as C
from common.decoder import decoder_h
from common.analytic import make_h

def evaluate_fields(c,points,derivatives,source,channel):
    d=c['Vtrain'].shape[1]; dev=C.io.pick_device('auto')
    data,truth=C.io.load_all(c['model'],5e-5 if c['model']=='CR3' else .0005,d,0,c['repeat'])
    if source=='analytic':
        teacher=make_h(c['model'],channel,truth.mu,truth.sd); inverse=C.tensor(C.oracle(c)).to(dev)
        h=lambda v,x:teacher(inverse@v,x)
    else:
        net,_=C.io.load_decoder(data['paths']['checkpoint'],dev)
        h=decoder_h(net,data['mu_z'],data['Minv'],channel)
    vv=C.tensor(points[:,:d]).to(dev); xx=C.tensor(points[:,d:]).to(dev)
    g=jacrev(h,0)
    refg=vmap(g)(vv,xx).cpu().numpy()
    reftt=vmap(jacfwd(g,0))(vv,xx).cpu().numpy()
    reftx=vmap(jacfwd(g,1))(vv,xx).cpu().numpy()
    R=C.rotations(d,1000,91); A=C.effective(R,C.cov(c['Vtrain']))
    reference=C.terms(A,C.moments(C.tensor(reftx),C.tensor(reftt))).numpy()
    result={'reference':'analytic true function' if source=='analytic' else 'original frozen NP decoder',
            'choice_caveat':'NP derivative is a proxy reference, not ground truth conditional-choice derivative' if source=='choice' else None,
            'timing':'post-fit/post-search; no feedback to selection or training','methods':{}}
    for name,arrays in derivatives.items():
        errors={}
        for block,pred,true in zip(('gradient','UU','UX'),arrays,(refg,reftt,reftx)):
            norm=float(np.linalg.norm(true)); abs_error=float(np.linalg.norm(pred-true)/np.sqrt(true.size))
            errors[block]=dict(rmse=abs_error,relative_frobenius=float(np.linalg.norm(pred-true)/norm) if norm>1e-9 else None,
                               reference_rms=float(norm/np.sqrt(true.size)))
        vals=C.terms(A,C.moments(C.tensor(arrays[2]),C.tensor(arrays[1]))).numpy()
        errors['landscape_correlation']=[float(np.corrcoef(vals[:,k],reference[:,k])[0,1])
                                         if vals[:,k].std()>1e-9 and reference[:,k].std()>1e-9 else None for k in range(2)]
        result['methods'][name]=errors
    return result
