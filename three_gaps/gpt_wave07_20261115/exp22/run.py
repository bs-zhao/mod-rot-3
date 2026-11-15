"""64x64 convergence then independent 300xfull-bank signed smoothing.

The decision to increase inner nodes uses moments and common Haar only, never T.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
import torch
from scipy.stats import spearmanr
from kernels import smooth

def main(a):
    st=c.Setting(a);rng=np.random.default_rng(21)  # same common centres as EXP21
    ids=st.ids0[:64];xi=np.sort(rng.choice(len(st.X),64,False))
    sigmas=[float(s) for s in a.sigmas.split(',')]
    if a.inner_cap<8 or a.inner_cap&(a.inner_cap-1):raise ValueError('inner cap must be a power of two >=8')
    inner=[2**m for m in range(3,int(np.log2(a.inner_cap))+1)]
    if a.stage=='pilot':V=st.V[ids];X=st.X[xi]
    elif a.stage=='full':V=st.V[st.ids0];X=st.X;inner=[a.inner_nodes]
    elif a.stage=='joint_train':V=st.V[st.tr];X=st.X;inner=[a.inner_nodes];sigmas=[0,.25]
    elif a.stage=='joint_gaussian':
        import nodes
        e,_,_=nodes.gaussian(1024,st.d,20261115);V=nodes.physical(e,st.C,st.mu);X=st.X;inner=[a.inner_nodes];sigmas=[0,.25]
    dest=c.OUT/'exp22'/st.key/a.tag/a.stage
    if (dest/'centres.npz').exists():
        with np.load(dest/'centres.npz') as z:
            if not np.array_equal(z['V'],V) or not np.array_equal(z['X'],X):raise ValueError('saved smoothing centres mismatch')
    else:c.save_npz(dest/'centres.npz',V=V,X=X)
    res=smooth(st,V,X,sigmas,inner,dest,kernel=a.kernel)
    common=torch.tensor(c.haar_set(st.d,1000,11),dtype=torch.float32)
    rows=[];old={};selected={}
    for sigma in sigmas:
        budgets=[1] if sigma==0 else inner
        for n in budgets:
            pre=f's{sigma:g}_n{n}'
            for order in ['s','e']:
                sx=res[pre+'_SX_'+order].mean(0);t4=res[pre+'_T4_'+order].mean(0)
                ml=c.Loss(sx,t4,st.C)
                with torch.no_grad():L=ml.total(common).numpy()
                previous=old.get((sigma,order));err=None
                if previous:
                    err=[float(np.linalg.norm(sx-previous[0])/np.linalg.norm(sx)),float(np.linalg.norm(t4-previous[1])/np.linalg.norm(t4)),float(spearmanr(L,previous[2]).statistic)]
                dp=res[pre+'_p']-res['p0']
                extra=dict(stage=a.stage,sigma=sigma,inner=n,order='smooth_then_square' if order=='s' else 'square_then_average',
                    mean_abs_dp=float(abs(dp).mean()),rms_dp=float(np.sqrt(np.mean(dp**2))),convergence_vs_previous=err,
                    outer_subjects=len(V),X_points=len(X),centres_path=str(dest/'centres.npz'),centres_sha256=c.sha(dest/'centres.npz'),
                    outer_subject_ids=ids.tolist() if a.stage=='pilot' else st.ids0.tolist() if a.stage=='full' else st.tr.tolist() if a.stage=='joint_train' else None,
                    X_indices=xi.tolist() if a.stage=='pilot' else list(range(len(st.X))))
                # Optimize every pilot budget to show numerical sensitivity, not select by MCC.
                c.run_loss(st,f'{a.stage}_sigma{sigma:g}_n{n}_{order}',sx,t4,extra=extra,experiment='exp22')
                rows.append(extra);old[(sigma,order)]=(sx,t4,L)
                if err and max(err[:2])<.01 and err[2]>.99:selected.setdefault(f'{sigma:g}_{order}',n)
    c.dump(c.HERE/'exp22/results'/st.key/a.tag/f'{a.stage}_convergence.json',dict(rows=rows,first_passing_budget=selected,
        cap=max(inner),unresolved=[f'{s:g}_{o}' for s in sigmas if s>0 for o in ['s','e'] if f'{s:g}_{o}' not in selected]))
    from predict import audit
    c.dump(c.HERE/'exp22/results'/st.key/a.tag/f'{a.stage}_validation_prediction.json',audit(st,sigmas,max(inner)))

if __name__=='__main__':
    p=c.args(__doc__);p.add_argument('--stage',choices=['pilot','full','joint_train','joint_gaussian'],default='pilot')
    p.add_argument('--sigmas',default='0,.05,.1,.25,.5,1')
    p.add_argument('--inner-cap',type=int,default=128);p.add_argument('--kernel',choices=['cov','spherical'],default='cov')
    p.add_argument('--inner-nodes',type=int,default=64);main(p.parse_args())
