"""Independent doc07 run. Search loads Z/X/checkpoint, never truth or Oracle.

All Hessians are recomputed from the original frozen decoder. Only historical
EXP13 caches may be opened by the separate audit program, never by this module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'three_gaps'))
from common import io
from common.decoder import decoder_h
from common.derivs import base_derivatives
from common.rotlib import haar, givens

torch.set_num_threads(4)
OUT = ROOT / 'outputs/three_gaps' / HERE.name
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def arrsha(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape, str(x.dtype))).encode() + x.tobytes()).hexdigest()

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)), encoding='utf-8')
    os.replace(tmp, path)

def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz'); np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def cov(V):
    V = np.asarray(V, np.float64); V = V - V.mean(0)
    return V.T @ V / len(V)

def args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--model', required=True, choices=['CR3','C1','C2','C3'])
    p.add_argument('--beta', type=float, required=True)
    p.add_argument('--zdim', type=int, required=True)
    p.add_argument('--pca', type=int, choices=[0,1], required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--channel', choices=['prob','logit'], default='prob')
    p.add_argument('--device', default='auto'); p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--tag', default='independent_v1'); p.add_argument('--resume', action='store_true')
    p.add_argument('--starts', type=int, default=24); p.add_argument('--steps', type=int, default=250)
    return p

class Setting:
    def __init__(self, a):
        self.a = a; self.dev = io.pick_device(a.device)
        self.paths = io.paths(a.model, a.beta, a.zdim, a.repeat)
        self.source = {k: dict(path=str(self.paths[k]), sha256=sha(self.paths[k]))
                       for k in ['latent','checkpoint']}
        with np.load(self.paths['latent'], allow_pickle=False) as z:
            # Explicit allowlist: theta_true / posterior labels are never accessed.
            self.z = np.asarray(z['z'], np.float64)
            self.tr, self.va, self.te = [np.asarray(z[k], int) for k in ['train_ids','validation_ids','test_ids']]
            if str(z['checkpoint_sha256']) != self.source['checkpoint']['sha256']: raise ValueError('checkpoint hash')
            if str(z['context_mode']) != 'all_observed_trials': raise ValueError('context mode')
        if self.z.shape[1] != a.zdim: raise ValueError('zdim mismatch')
        # Dimension is an interface constraint, never a role-dependent objective.
        if a.zdim != io.D_T[a.model]: raise ValueError('higher-dimensional PCA requires an explicit residual protocol')
        self.mu = self.z[self.tr].mean(0); self.V = self.z - self.mu
        self.Minv = np.eye(a.zdim)
        if a.pca:
            ev, Q = np.linalg.eigh(cov(self.V[self.tr])); Q = Q[:, ::-1]
            Q *= np.sign(Q[np.abs(Q).argmax(0), np.arange(a.zdim)])
            self.V = self.V @ Q; self.Minv = Q
        self.C = cov(self.V[self.tr]); self.d = self.V.shape[1]
        self.net, self.net_cfg = io.load_decoder(self.paths['checkpoint'], self.dev)
        self.h = decoder_h(self.net, self.mu, self.Minv, a.channel)
        self.cond = self.paths['condition'] + (f'_rep{a.repeat:02d}' if a.repeat != 1 else '')
        self.key = f'{a.model}_{self.cond}_pca{a.pca}_{a.channel}'
        # Recreate frozen EXP13 grid, using its documented RNG recipe, no reuse.
        rng = np.random.default_rng(20261108)
        self.ids0 = np.sort(rng.choice(self.tr, 300, replace=False))
        if a.model == 'CR3':
            with np.load(self.paths['dataset'], allow_pickle=False) as data:
                si = data['subject_index']; self.X = data['x'][si == 0].astype(np.float64)
        else:
            x = rng.uniform(-1, 1, (200, self.net_cfg.x_dim))
            self.X = np.concatenate([x,-x])
        self.p = self.X.shape[1]
        self.source['X'] = dict(sha256=arrsha(self.X), recipe='EXP13 seed20261108, shared fixed bank / antithetic grid')
        self.source['V'] = dict(sha256=arrsha(self.V))

    def config(self):
        return dict(args=vars(self.a), source=self.source, dataset_sha256=sha(self.paths['dataset']), train_ids=self.tr, validation_ids=self.va,
            test_ids=self.te, mu=self.mu, Minv=self.Minv, C_ref=self.C,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            code_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py')},
            decoder_dtype='float32', derivative_input_dtype='float64', field_storage='float32', moment_accum='float64',
            search_dtype='float32', device=str(self.dev), torch=torch.__version__, numpy=np.__version__,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            truth_access=False, prior_results_reused=False)

def moments(tt, tx):
    tt, tx = tt.double(), tx.double()
    n,k,d,_ = tt.shape
    sx = torch.einsum('nkaj,nkbj->njab',tx,tx)/k
    flat=tt.reshape(n,k,d*d)
    t4=(flat.transpose(1,2)@flat/k).reshape(n,d,d,d,d)
    return sx.cpu().numpy(),t4.cpu().numpy()

def fields(st, V, X=None, chunk=None):
    return base_derivatives(st.h, torch.as_tensor(V,dtype=torch.float64,device=st.dev),
        torch.as_tensor(st.X if X is None else X,dtype=torch.float64,device=st.dev),
        which=('Htt','Htx'), chunk=chunk or st.a.chunk)

def stream(st, V, directory, X=None, batch=16):
    """Resume per-batch arrays, hash configuration; never reuse other experiment caches."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    cfg=dict(source=st.source, V=arrsha(V), X=arrsha(st.X if X is None else X), batch=batch, chunk=st.a.chunk, channel=st.a.channel)
    cp=directory/'config.json'
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8')) != cfg: raise ValueError('cache configuration mismatch')
    dump(cp,cfg); sx=[];t4=[];t0=time.perf_counter()
    for i in range(0,len(V),batch):
        p=directory/f'batch_{i:06d}.npz'
        if not p.exists():
            b=fields(st,V[i:i+batch],X); s,t=moments(b['Htt'],b['Htx']);save_npz(p,SX=s,T4=t)
        with np.load(p) as b: sx.append(b['SX']);t4.append(b['T4'])
        if i % 512 == 0: print('stream',i,'/',len(V),flush=True)
    sx,t4=np.concatenate(sx),np.concatenate(t4)
    save_npz(directory/'moments.npz',SX=sx,T4=t4,V=V)
    dump(directory/'COMPLETE.json',dict(seconds=time.perf_counter()-t0,nodes=len(V),K=len(st.X if X is None else X),sha256=sha(directory/'moments.npz')))
    return sx,t4

def effective_A(R,C):
    scale=torch.diagonal(R@C@R.transpose(-1,-2),dim1=-2,dim2=-1).clamp_min(1e-30).sqrt()
    return scale[..., :,None]*torch.linalg.inv(R).transpose(-1,-2)

class Loss:
    def __init__(self,SX,T4,C,scales=None):
        self.SX=torch.as_tensor(SX,dtype=torch.float32); self.T4=torch.as_tensor(T4,dtype=torch.float32)
        self.C=torch.as_tensor(C,dtype=torch.float32);self.d=len(C)
        H=torch.as_tensor(haar_set(self.d,200,1),dtype=torch.float32)
        val=self.terms(H).detach().numpy()
        sd=val.std(0)
        if np.any(sd<1e-8): raise ValueError(f'no Haar signal: SD={sd}')
        self.scales=np.asarray(scales) if scales is not None else sd+1e-9
    def energies(self,R):
        A=effective_A(R,self.C.to(R));SX=self.SX.to(R);T=self.T4.to(R)
        ux=torch.einsum('...ia,jab,...ib->...ij',A,SX,A)
        B=torch.einsum('...ia,...kb->...ikab',A,A).flatten(-2)
        uu=((B@T.flatten(0,1).flatten(1)) * B).sum(-1)
        return ux.clamp_min(0),uu.clamp_min(0)
    def terms(self,R):
        ux,uu=self.energies(R)
        q=ux/(ux.sum((-2,-1),keepdim=True)+1e-12)+1e-12
        neff=(-(q*q.log()).sum((-2,-1))).exp()
        share=uu.diagonal(dim1=-2,dim2=-1).sum(-1)/(uu.sum((-2,-1))+1e-12)
        return torch.stack([neff,share],-1)
    def total(self,R): return (self.terms(R)/torch.as_tensor(self.scales).to(R)).sum(-1)

def haar_set(d,n,seed):
    r=np.random.default_rng(seed);return np.stack([haar(d,r) for _ in range(n)])

def search(loss,starts=24,steps=250,serial=False):
    q0=torch.as_tensor(haar_set(loss.d,starts,7),dtype=torch.float32)
    groups=[q0[i:i+1] for i in range(starts)] if serial else [q0]
    Rs=[];Ls=[];trace=[]
    for Q0 in groups:
        W=torch.zeros_like(Q0,requires_grad=True);opt=torch.optim.Adam([W],lr=.05)
        for j in range(steps):
            R=torch.matrix_exp(W-W.transpose(-1,-2))@Q0;v=loss.total(R)
            if not torch.isfinite(v).all(): raise FloatingPointError('nonfinite loss')
            opt.zero_grad();v.sum().backward()
            if not torch.isfinite(W.grad).all(): raise FloatingPointError('nonfinite gradient')
            opt.step()
            if j%25==0: trace.append([j,float(v.min().detach())])
        with torch.no_grad():
            R=torch.matrix_exp(W-W.transpose(-1,-2))@Q0
            Rs.append(R.detach().cpu().numpy());Ls.append(loss.total(R).detach().cpu().numpy())
    R=np.concatenate(Rs);L=np.concatenate(Ls);return R,L,q0.numpy(),np.asarray(trace)

def run_loss(st,name,sx,t4,C=None,scales=None,serial=False,extra=None,experiment='exp20A',objective=None):
    C=st.C if C is None else C
    local=OUT/experiment/st.key/st.a.tag/name
    small=HERE/experiment/'results'/st.key/st.a.tag/name
    if (small/'COMPLETE.json').exists():
        if not st.a.resume: raise FileExistsError(small)
        old=json.loads((small/'config.json').read_text(encoding='utf-8'))
        # A code improvement may add diagnostics, but scientific inputs/budgets
        # cannot silently reuse another run. Array hashes also verified below.
        if old['source'] != st.source or not np.array_equal(np.asarray(old['C_ref']),C) or old['starts']!=st.a.starts or old['steps']!=st.a.steps:
            raise ValueError('completed run configuration mismatch')
        with np.load(local/'moments.npz') as z:
            if not np.array_equal(z['SX'],sx) or not np.array_equal(z['T4'],t4): raise ValueError('completed moments differ')
        return json.loads((small/'diagnostics.json').read_text())
    local.mkdir(parents=True,exist_ok=True);small.mkdir(parents=True,exist_ok=True)
    cfg=st.config();cfg.update(name=name,C_ref=C,extra=extra or {},starts=st.a.starts,steps=st.a.steps)
    dump(small/'config.json',cfg);save_npz(local/'moments.npz',SX=sx,T4=t4,C=C)
    ml=Loss(sx,t4,C,scales) if objective is None else objective;t0=time.perf_counter()
    R,L,Q0,tr=search(ml,st.a.starts,st.a.steps,serial)
    best=int(np.argmin(L));A=effective_A(torch.tensor(R),torch.tensor(C,dtype=torch.float32)).numpy()
    save_npz(small/'rotations.npz',R=R,loss=L,initial_R=Q0,best_index=best,A=A,C_ref=C,trace=tr)
    H=haar_set(st.d,1000,11)
    with torch.no_grad(): landscape=ml.total(torch.tensor(H,dtype=torch.float32)).numpy()
    save_npz(local/'random_landscapes.npz',R=H,loss=landscape)
    # Blind planes about the minimum; Oracle curves belong to evaluate.py only.
    rows=[]
    for i in range(st.d):
        for j in range(i+1,st.d):
            for deg in range(0,91,5):
                G=givens(st.d,i,j,np.deg2rad(deg));v=float(ml.total(torch.tensor(G@R[best],dtype=torch.float32)))
                rows.append((i,j,deg,v))
    import pandas as pd
    pd.DataFrame(rows,columns=['i','j','degrees','loss']).to_csv(small/'plane_curves.csv',index=False)
    diag=dict(name=name,lowest_loss=float(L[best]),haar_scales=ml.scales.tolist(),
        seconds=time.perf_counter()-t0,selected_by='lowest_loss',truth_loaded=False,
        moments_sha256=sha(local/'moments.npz'),landscape_path=str(local/'random_landscapes.npz'),**(extra or {}))
    dump(small/'diagnostics.json',diag)
    dump(small/'COMPLETE.json',dict(rotations_sha256=sha(small/'rotations.npz'),**diag))
    print(experiment,st.key,name,'loss',L[best],f'{diag["seconds"]:.2f}s',flush=True)
    return diag
