"""Fresh Gaussian integration. Budgets increase by moment/landscape error, no T."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import nodes
import numpy as np
import torch
from scipy.stats import spearmanr,chi2,skew,kurtosis
from scipy.spatial import cKDTree

def support_diagnostics(st,V,sx,t4,w):
    train=st.V[st.tr];Ci=np.linalg.inv(st.C)
    rt=np.einsum('na,ab,nb->n',train,Ci,train)
    radius=np.einsum('na,ab,nb->n',V,Ci,V)
    edges=np.r_[0,np.quantile(rt,[.5,.9,.95,.99]),np.inf]
    ux=np.trace(sx,axis1=-2,axis2=-1).sum(-1)
    uu=np.trace(t4.reshape(len(V),st.d**2,st.d**2),axis1=-2,axis2=-1)
    whiten=np.linalg.cholesky(st.C)
    A=np.linalg.solve(whiten,train.T).T;B=np.linalg.solve(whiten,V.T).T
    tree=cKDTree(A);train_nn=tree.query(A,k=2)[0][:,1];dist=tree.query(B)[0]
    region=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(radius>=lo)&(radius<hi)
        region.append(dict(lo=lo,hi=hi,node_mass=float(w[mask].sum()),
            UX_energy_share=float((w[mask]*ux[mask]).sum()/(w@ux)),UU_energy_share=float((w[mask]*uu[mask]).sum()/(w@uu))))
    return dict(regions=region,mahalanobis_quantiles=np.quantile(radius,[.5,.9,.95,.99]),
        knn_quantiles=np.quantile(dist,[.5,.9,.95,.99]),
        outside_train_nn99_mass=float(w[dist>np.quantile(train_nn,.99)].sum()),weights_sum=float(w.sum()))

def main(a):
    st=c.Setting(a);out=c.OUT/'exp20B'/st.key/a.tag
    # Physical prior mapping is implemented for pca=0 only; avoid adding mu in PC space.
    if a.pca:raise ValueError('P3 physical-prior protocol requires pca=0 in this runner')
    H=torch.tensor(c.haar_set(st.d,1000,11),dtype=torch.float32)
    summaries=[]; refs={}
    # At least five MC seeds and four independent QMC scrambles; nested prefixes.
    for method in ['qmc','gh','mc']:
        maxn=8192 if a.model=='CR3' else 4096
        seeds=range(4) if method=='qmc' else range(5) if method=='mc' else [0]
        orders=[3,5,7,9] if st.d==4 else [3,5]  # no huge C3 tensor grid
        for seed in seeds:
            if method=='gh':budgets=orders
            else:budgets=[256]+([300] if method=='mc' else [])+[512,1024,2048,4096]+([8192] if maxn==8192 else [])
            if method!='gh':
                e,w,meta=nodes.gaussian(maxn,st.d,seed+20261115,method)
                V=nodes.physical(e,st.C,st.mu)
                c.save_npz(out/f'P1_{method}_{seed}'/'nodes.npz',epsilon=e,V=V,weights=w)
                c.dump(out/f'P1_{method}_{seed}'/'integrator.json',meta)
                sx,t4=c.stream(st,V,out/f'P1_{method}_{seed}')
                c.dump(out/f'P1_{method}_{seed}'/'support.json',support_diagnostics(st,V,sx,t4,w))
            previous=None
            for budget in budgets:
                if method=='gh':
                    e,w,meta=nodes.gaussian(0,st.d,0,'gh',order=budget);V=nodes.physical(e,st.C,st.mu)
                    c.save_npz(out/f'P1_gh_m{budget}'/'nodes.npz',epsilon=e,V=V,weights=w)
                    c.dump(out/f'P1_gh_m{budget}'/'integrator.json',meta)
                    sx,t4=c.stream(st,V,out/f'P1_gh_m{budget}')
                    c.dump(out/f'P1_gh_m{budget}'/'support.json',support_diagnostics(st,V,sx,t4,w))
                    ss=np.einsum('n,n...->...',w,sx);tt=np.einsum('n,n...->...',w,t4)
                else:ss=sx[:budget].mean(0);tt=t4[:budget].mean(0)
                ml=c.Loss(ss,tt,st.C)
                with torch.no_grad():L=ml.total(H).numpy()
                err=None if previous is None else [float(np.linalg.norm(ss-previous[0])/np.linalg.norm(ss)),float(np.linalg.norm(tt-previous[1])/np.linalg.norm(tt)),float(spearmanr(L,previous[2]).statistic)]
                name=f'P1_{method}_seed{seed}_'+(f'm{budget}' if method=='gh' else f'N{budget}')
                # Optimize each budget; common starts, no MCC convergence decisions.
                c.run_loss(st,name,ss,tt,experiment='exp20B',extra=dict(integrator=method,seed=seed,budget=budget,nodes=budget**st.d if method=='gh' else budget,convergence_vs_previous=err))
                summaries.append(dict(name=name,relative_SX=None if err is None else err[0],relative_T4=None if err is None else err[1],spearman=None if err is None else err[2],converged=bool(err and max(err[:2])<.01 and err[2]>.99)))
                previous=(ss,tt,L)
            if method=='qmc':refs[seed]=(sx.mean(0),t4.mean(0))
    # Compare P0 / P1 / P3 with fixed full-TRAIN covariance and identical X.
    sx,t4=c.stream(st,st.V[st.tr],out/'P0_fresh')
    c.run_loss(st,'P0_full_train',sx.mean(0),t4.mean(0),experiment='exp20B')
    ss=np.mean([v[0] for v in refs.values()],0);tt=np.mean([v[1] for v in refs.values()],0)
    c.run_loss(st,'P1_QMC_4scramble_mean',ss,tt,experiment='exp20B')
    p3=[]
    for seed in range(4):
        e,w,_=nodes.gaussian(maxn,st.d,seed+20261115,'qmc');V=nodes.physical(e,st.C,st.mu,'P3')
        c.save_npz(out/f'P3_qmc_{seed}'/'nodes.npz',epsilon=e,V=V,weights=w)
        s,t=c.stream(st,V,out/f'P3_qmc_{seed}');p3.append((s.mean(0),t.mean(0)))
    c.run_loss(st,'P3_QMC_4scramble_mean',np.mean([v[0] for v in p3],0),np.mean([v[1] for v in p3],0),experiment='exp20B')
    c.run_loss(st,'P3_QMC_native_cov',np.mean([v[0] for v in p3],0),np.mean([v[1] for v in p3],0),C=np.eye(st.d),experiment='exp20B',extra=dict(C_mode='measure_native'))
    hp=c.HERE/'exp13_reproduce/results'/st.key/a.tag/'A0/diagnostics.json'
    import json
    hs=json.loads(hp.read_text())['haar_scales']
    c.run_loss(st,'P1_QMC_histSD',ss,tt,scales=hs,experiment='exp20B',extra=dict(fixed_historical_scales=True))
    import pandas as pd
    small=c.HERE/'exp20B/results'/st.key/a.tag;small.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(small/'convergence.csv',index=False)
    V=st.V[st.tr];r2=np.einsum('na,ab,nb->n',V,np.linalg.inv(st.C),V)
    c.dump(small/'distribution.json',dict(mean_Z=st.mu,sd_Z=np.sqrt(np.diag(st.C)),correlation=np.corrcoef(V.T),eigenvalues=np.linalg.eigvalsh(st.C),
        skew=skew(V),excess_kurtosis=kurtosis(V),mahalanobis_quantiles=np.quantile(r2,[.5,.9,.95,.99]),chi2_quantiles=chi2.ppf([.5,.9,.95,.99],st.d)))
    from scipy.stats import norm
    quant=np.linspace(.005,.995,199);qq=[]
    rng=np.random.default_rng(20);directions=rng.normal(size=(16,st.d));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    for k,direction in enumerate(np.r_[np.eye(st.d),directions]):
        proj=V@direction;proj=(proj-proj.mean())/proj.std()
        qq.extend(dict(direction=k,quantile=q,normal=t,empirical=v) for q,t,v in zip(quant,norm.ppf(quant),np.quantile(proj,quant)))
    pd.DataFrame(qq).to_csv(small/'marginal_and_projection_QQ.csv',index=False)

if __name__=='__main__':main(c.args(__doc__).parse_args())
