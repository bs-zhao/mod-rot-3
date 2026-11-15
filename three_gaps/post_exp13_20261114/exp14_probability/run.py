"""EXP14: probability Hessian decomposition and fixed ablations (doc 05)."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
import fields as F

def main(a):
    if C.selftest_stage(a): return
    c=C.load(a.model,a.repeat,a.channel,a.subjects,a.trials); c,b=F.get(c,a.source)
    out=C.result_dir('exp14_probability',a.tag,a.model,a.repeat,a.source)
    cs=F.components(b); ms={k:C.moments(*v) for k,v in cs.items()}
    methods={}
    def add(name,ux,uu,weights=None):
        pair=(ms[ux][0],ms[uu][1]); methods[name]=C.Objective(c['Vtrain'],lambda A,pair=pair:C.terms(A,pair),weights)
    for name in ('eta','A','B','p'): add(name+'__combo',name,name)
    for name in ('eta','A','B','p'):
        add(name+'__UX',name,name,[1,0]); add(name+'__UU',name,name,[0,1])
    for ux in ('eta','A','B','p'):
        for uu in ('eta','A','B','p'):
            if ux!=uu: add(f'{ux}UX_{uu}UU',ux,uu)
    for name in ('gg','p1gg'): add(name+'__combo',name,name)
    for k in (.25,.5,2.,4.):
        key=f'A_plus_{k:g}B'; ms[key]=C.moments(*(aa+k*bb for aa,bb in zip(cs['A'],cs['B'])))
        add(key,key,key)
    for name in ('tanh','softplus','asinh'):
        ms[name]=C.moments(*F.transformed(b,name)); add(name,name,name)
    # First-order energy and X-context gradient covariance: fixed unsupervised X partition.
    from sklearn.cluster import KMeans
    x=c['X'].numpy(); xs=(x-x.mean(0))/(x.std(0)+1e-9)
    labels=KMeans(4,random_state=11,n_init=10).fit_predict(xs)
    s2=torch.einsum('ska,skb->ab',b['g'],b['g'])/(b['g'].shape[0]*b['g'].shape[1])
    means=torch.stack([b['g'][:,labels==k].mean(1) for k in range(4)],1)
    centered=means-means.mean(1,keepdim=True)
    ctx=torch.einsum('sca,scb->cab',centered,centered)/len(means)
    def gradterms(A):
        e=torch.einsum('...ia,ab,...ib->...i',A,s2,A)
        ec=torch.einsum('...ia,cab,...ib->...ic',A,ctx,A)
        return torch.stack([C.neff(e[...,None]),C.neff(ec)],-1)
    methods['gradient_and_context']=C.Objective(c['Vtrain'],gradterms)
    # Signed energy cross term is diagnostic only, never a sparse probability mass.
    decomp={}
    for label,r in [('identity',np.eye(c['Vtrain'].shape[1])),('oracle',C.oracle(c))]:
        A=C.effective(C.tensor(r),C.cov(c['Vtrain']))
        for block,index in [('UX',0),('UU',1)]:
            es={k:C.energies(A,ms[k])[index] for k in ('A','B','p')}
            cross=es['p']-es['A']-es['B']
            decomp[label+'_'+block]={**{k:v.numpy() for k,v in es.items()},'2EAB':cross.numpy()}
    C.dump(out/'decomposition.json',decomp)
    if a.source=='decoder':
        p=C.load(a.model,a.repeat,'prob',a.subjects,a.trials)
        residual={}
        for name,actual in [('UX',p['Htx']),('UU',p['Htt'])]:
            expected=cs['p'][0 if name=='UX' else 1]
            residual[name]=float((expected-C.tensor(actual)).norm()/(C.tensor(actual).norm()+1e-12))
        C.dump(out/'identity_check.json',residual)
        if max(residual.values())>1e-4: raise AssertionError('p Hessian identity mismatch')
    if a.only:
        requested=a.only.split(','); methods={k:methods[k] for k in requested}
    C.run_methods(c,out,methods,a,dict(source=a.source,evidence='Gap1 analytic' if a.source=='analytic' else 'decoder-autograd',
                   component_definitions='A=sigmoid_prime(eta) H_eta; B=sigmoid_second(eta) gg^T',
                   context_labels=labels))

if __name__=='__main__':
    p=C.parser(__doc__); p.add_argument('--source',choices=['decoder','analytic'],default='decoder')
    p.add_argument('--only',default=''); main(p.parse_args())
