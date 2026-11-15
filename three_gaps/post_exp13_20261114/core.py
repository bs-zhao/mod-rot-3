"""Shared mechanics for the doc-05 campaign; historical experiments are read-only.

Search takes derivative fields / V only. Truth and supervised rotations are loaded
by `evaluate` / `diagnose` after the unsupervised search has been saved.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'three_gaps'))
from common import io
from common.rotlib import haar, givens, corr_matrix, match_report

torch.set_num_threads(4)
COND = {'C1': 'beta0p0005_lambda0_ctx0p25to0p8_L4',
        'C2': 'beta0p0005_lambda0_ctx0p25to0p8_L4',
        'C3': 'beta0p0005_lambda0_ctx0p25to0p8_L6',
        'CR3': 'beta5em05_lambda0_ctx0p25to0p8_L4'}
EPS = 1e-12

def condition(model, repeat):
    return COND[model] + (f'_rep{repeat:02d}' if repeat != 1 else '')

def load(model, repeat=1, channel='prob', subjects=None, trials=None):
    path = ROOT / 'outputs/three_gaps/exp13' / f'cache2_{model}_{condition(model,repeat)}_{channel}.pt'
    raw = torch.load(path, weights_only=False, map_location='cpu')
    # Physically exclude T and TEST IDs from the optimization object.
    c = {k: raw[k] for k in ('g', 'Htt', 'Htx', 'h', 'X', 'V', 'ids')}
    if subjects:
        si = np.sort(np.random.default_rng(2301).choice(len(c['ids']), min(subjects,len(c['ids'])), replace=False))
        c['ids'] = np.asarray(c['ids'])[si]
        for k in ('g','Htt','Htx','h'): c[k] = c[k][si]
    if trials:
        ki = np.sort(np.random.default_rng(2302).choice(len(c['X']), min(trials,len(c['X'])), replace=False))
        c['X'] = c['X'][ki]
        for k in ('g','Htt','Htx','h'): c[k] = c[k][:,ki]
    c['cache_path'] = str(path)
    c['Vtrain'] = torch.tensor(c['V'][c['ids']], dtype=torch.float64)
    c['model'], c['repeat'], c['channel'] = model, repeat, channel
    return c

def tensor(a): return torch.as_tensor(a, dtype=torch.float64)

def cov(v):
    z = v-v.mean(0)
    return z.T @ z / len(z)

def effective(R, covariance):
    sd = torch.einsum('...ia,ab,...ib->...i', R, covariance, R).clamp_min(EPS).sqrt()
    # All searches here are orthogonal. inverse(R).T == R.
    return sd[..., :, None]*R

def rotations(d, n=200, seed=1):
    rng = np.random.default_rng(seed)
    return tensor(np.stack([haar(d,rng) for _ in range(n)]))

def moments(tx, tt):
    tx,tt = tensor(tx).reshape(-1,*tx.shape[-2:]),tensor(tt).reshape(-1,*tt.shape[-2:])
    n,d = len(tt),tt.shape[-1]
    return torch.einsum('naj,nbj->jab',tx,tx)/n, (tt.reshape(n,-1).T@tt.reshape(n,-1)/n).reshape(d,d,d,d)

def energies(A, moment):
    sx,t4 = moment
    ux = torch.einsum('...ia,jab,...ib->...ij',A,sx,A)
    uu = torch.einsum('...ia,...kb,abcd,...ic,...kd->...ik',A,A,t4,A,A)
    return ux.clamp_min(0),uu.clamp_min(0)

def neff(e):
    q=e/(e.sum((-2,-1),keepdim=True)+EPS)
    return (-(q*(q+EPS).log()).sum((-2,-1))).exp()

def share(e): return e.diagonal(dim1=-2,dim2=-1).sum(-1)/(e.sum((-2,-1))+EPS)

def terms(A,m):
    ux,uu=energies(A,m)
    return torch.stack([neff(ux),share(uu)],-1)

class Objective:
    def __init__(self, v, fn, weights=None, haar_n=200, haar_seed=1, sd_floor=1e-9):
        self.cov=cov(tensor(v)); self.fn=fn
        self.haar=rotations(v.shape[1],haar_n,haar_seed)
        with torch.no_grad(): self.null=fn(effective(self.haar,self.cov))
        self.scales=self.null.std(0,unbiased=False)
        self.active=self.scales>sd_floor
        self.weights=torch.ones_like(self.scales) if weights is None else tensor(weights)
        self.weights=self.weights*self.active
        self.scales=self.scales.clamp_min(sd_floor)
    def raw(self,R): return self.fn(effective(R,self.cov))
    def __call__(self,R): return (self.raw(R)/self.scales*self.weights).sum(-1)

def optimize(obj, d, starts=24, steps=250, seed=7, lr=.05):
    """Independent Adam states, vectorized across starts; no T / Oracle inputs."""
    q=rotations(d,starts,seed)
    w=torch.zeros(starts,d,d,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([w],lr=lr)
    for _ in range(steps):
        r=torch.matrix_exp(w-w.transpose(-1,-2))@q
        loss=obj(r)
        if not torch.isfinite(loss).all(): raise FloatingPointError('non-finite loss')
        opt.zero_grad(); loss.sum().backward()
        if not torch.isfinite(w.grad).all(): raise FloatingPointError('non-finite SO gradient')
        opt.step()
    with torch.no_grad():
        r=torch.matrix_exp(w-w.transpose(-1,-2))@q; loss=obj(r)
    order=torch.argsort(loss)
    return loss[order].numpy(),r[order].numpy()

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def default(x):
        if isinstance(x,Path): return str(x)
        if torch.is_tensor(x): return x.detach().cpu().tolist()
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n',encoding='utf8')

def environment():
    return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
                platform=platform.platform(),cuda=torch.cuda.is_available(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())

def result_dir(exp,tag,model,repeat,channel):
    out=HERE/exp/'results'/tag/f'{model}_rep{repeat}_{channel}'
    out.mkdir(parents=True,exist_ok=True)
    return out

def evaluate(c,R):
    """Post-hoc only. Save signed, unpermuted correlation separately."""
    raw=torch.load(c['cache_path'],weights_only=False,map_location='cpu')
    te=raw['te']; v,t=c['V'][te],raw['Tst'][te]
    cr=corr_matrix(v@np.asarray(R).T,t)
    rep=match_report(v@np.asarray(R).T,t)
    return dict(mcc=rep['mcc'],leakage=rep['off_mean'],max_leakage=rep['off_max'],
                matched_correlations=np.abs(cr)[np.arange(len(R)),rep['perm']].tolist(),
                permutation=rep['perm'],correlation=cr.tolist(),T_correlation=corr_matrix(t,t).tolist())

def oracle(c):
    if 'diagnostic_oracle' in c: return c['diagnostic_oracle']
    if c.get('source')=='analytic': return np.eye(c['Vtrain'].shape[1])
    return np.load(ROOT/'outputs/three_gaps/exp00'/f"{c['model']}_{condition(c['model'],c['repeat'])}_pca0"/'R_oracle_raw.npy')

def heatmap(c, methods, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(methods),figsize=(3*len(methods),3.2),squeeze=False)
    for ax,(name,R) in zip(axes[0],methods.items()):
        met=evaluate(c,R); C=np.abs(np.asarray(met['correlation']))[:,met['permutation']]
        ax.imshow(C,vmin=0,vmax=1,cmap='viridis')
        ax.set_title(f"{name}\nMCC {met['mcc']:.3f}; leak {met['leakage']:.3f}",fontsize=9)
        for i in range(len(R)):
            for j in range(len(R)):
                ax.text(j,i,f'{C[i,j]:.2f}',ha='center',va='center',fontsize=7,color='w' if C[i,j]<.6 else 'k')
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)

def baseline_rotation(c):
    if c.get('source') in ('analytic','matrix_analytic'): return np.eye(c['Vtrain'].shape[1])
    p=HERE/'baseline/results'/f"{c['model']}_rep{c['repeat']}_prob"/'rotations.npz'
    if p.exists(): return np.load(p)['R'][0]
    p=ROOT/'outputs/three_gaps/exp13'/f"Q_{c['model']}_rep{c['repeat']}_prob_UX_neff+UUd_share.npy"
    return np.load(p) if p.exists() else np.eye(c['Vtrain'].shape[1])

def diagnose(obj,c,R,random_n=1000):
    ro=tensor(oracle(c)); found=tensor(R); random=rotations(len(R),random_n,91)
    with torch.no_grad():
        vals=obj(random).numpy(); lv=float(obj(ro)); lf=float(obj(found))
    rows=[]
    for i,j in itertools.combinations(range(len(R)),2):
        rs=tensor(np.stack([givens(len(R),i,j,np.deg2rad(a))@ro.numpy() for a in range(0,91,5)]))
        with torch.no_grad(): yy=obj(rs).numpy()
        rows.extend(dict(i=i,j=j,angle=a,loss=float(y)) for a,y in zip(range(0,91,5),yy))
    a=torch.zeros(len(R),len(R),dtype=torch.float64,requires_grad=True)
    z=obj(torch.matrix_exp(a-a.T)@ro); grad=torch.autograd.grad(z,a)[0]
    return dict(oracle_loss=lv,found_loss=lf,oracle_z=float((vals.mean()-lv)/(vals.std()+1e-12)),
                oracle_random_percentile=float((vals<=lv).mean()),random_min=float(vals.min()),
                oracle_gradient_norm=float(grad.norm()),random_n=random_n),rows,vals

def run_methods(c,out,methods,args,extra=None):
    """Methods are predeclared; never pick which ones to report using TEST T."""
    config={**vars(args),'environment':environment(),'cache':c['cache_path'],
            'suite_source_sha256':{str(p.relative_to(HERE)):io.sha256(p) for p in HERE.rglob('*.py')},
            'cache_sha256':io.sha256(c['cache_path']),'subjects':len(c['ids']),
            'trials':len(c['X']),'ids':c['ids'],'X':c['X'],
            'V_mean':c['Vtrain'].mean(0),'V_cov':cov(c['Vtrain']),
            'evidence':(extra or {}).get('evidence','decoder-autograd'),'selection':'lowest unsupervised final loss',
            'TEST_reuse':'Exploratory; these benchmark TEST sets were already inspected in EXP13.',
            'extra':extra or {}}
    dump(out/'config.json',config)
    if args.stage=='selftest':
        subprocess.run([sys.executable,str(HERE/'test_core.py')],check=True)
        return []
    if args.stage=='cache': return []
    previous=np.load(out/'rotations.npz') if args.stage in ('evaluate','report') else None
    rows=[]; plane=[]; diag={}; saved={}; allRs={}; losses={}
    for name,obj in methods.items():
        t=time.perf_counter()
        if args.stage in ('evaluate','report'):
            rr=previous[name+'__R']; ll=previous[name+'__loss']
        elif args.stage=='scan':
            rr=np.eye(c['Vtrain'].shape[1])[None]; ll=np.array([float(obj(tensor(rr[0])))])
        else:
            ll,rr=optimize(obj,c['Vtrain'].shape[1],args.starts,args.steps,args.seed)
        # Save before loading T.
        saved[name+'__R']=rr; saved[name+'__loss']=ll
        saved[name+'__sd']=np.std(c['V'][c['ids']]@rr[0].T,axis=0)
        saved[name+'__scales']=obj.scales.numpy(); saved[name+'__weights']=obj.weights.numpy()
        np.savez_compressed(out/'rotations.npz',**saved)
        met=evaluate(c,rr[0]); dd,pp,rv=diagnose(obj,c,rr[0],args.random_n)
        diag[name]={**dd,'all_endpoints':[evaluate(c,r) for r in rr],
                    'inactive_terms':np.flatnonzero(~obj.active.numpy()).tolist(),
                    'raw_best_terms':obj.raw(tensor(rr[0])).detach().numpy()}
        plane.extend({**p,'method':name} for p in pp)
        losses[name]=rv; allRs[name]=rr[0]
        row=dict(model=c['model'],repeat=c['repeat'],channel=c['channel'],beta=5e-5 if c['model']=='CR3' else .0005,
                 zdim=len(rr[0]),pca=0,method=name,seed=args.seed,starts=args.starts,steps=args.steps,
                 loss=float(ll[0]),seconds=time.perf_counter()-t,oracle_mcc=evaluate(c,oracle(c))['mcc'],
                 **{k:met[k] for k in ('mcc','leakage','max_leakage')})
        rows.append(row); pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        dump(out/'diagnostics.json',diag)
        print(json.dumps(row),flush=True)
    pd.DataFrame(plane).to_csv(out/'plane_curves.csv',index=False)
    np.savez_compressed(out/'random_landscapes.npz',**losses)
    # Keep one fixed main candidate in the overview; every candidate has raw C in diagnostics.
    first=next(iter(allRs))
    baseline_label='Gap2 identity' if c.get('source')=='matrix_analytic' else 'EXP13 p'
    heatmap(c,{'Identity':np.eye(c['Vtrain'].shape[1]),baseline_label:baseline_rotation(c),
               first:allRs[first],'Oracle':oracle(c)},out/'UT_heatmaps.png')
    dump(out/'COMPLETE.json',dict(methods=list(allRs),rows=len(rows)))
    return rows

def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--model',choices=list(COND),default='CR3'); p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--channel',choices=['prob','logit'],default='prob')
    p.add_argument('--stage',choices=['selftest','cache','scan','optimize','evaluate','report','all'],default='all')
    p.add_argument('--tag',default='round1'); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--starts',type=int,default=24); p.add_argument('--steps',type=int,default=250)
    p.add_argument('--random-n',type=int,default=1000); p.add_argument('--subjects',type=int,default=64)
    p.add_argument('--trials',type=int,default=64)
    return p
