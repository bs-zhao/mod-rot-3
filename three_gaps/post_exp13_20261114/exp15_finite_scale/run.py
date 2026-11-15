"""EXP15: recompute finite responses at EVERY R; fixed, supported X pairs."""
from __future__ import annotations
import itertools
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
from scipy.spatial.distance import cdist

class Finite:
    def __init__(self,c,scales=(.1,.25,.5,1.)):
        self.c=c; self.d=c['Vtrain'].shape[1]; self.scales=scales
        data,_=C.io.load_all(c['model'],5e-5 if c['model']=='CR3' else .0005,self.d,0,c['repeat'])
        self.dev=C.io.pick_device('auto'); self.net,_=C.io.load_decoder(data['paths']['checkpoint'],self.dev)
        self.net=self.net.double() # exact weights; reduces cancellation at h=.1.
        self.mu=C.tensor(data['mu_z']).to(self.dev); self.inv=C.tensor(data['Minv']).to(self.dev)
        self.cov=C.cov(c['Vtrain']); self.v=c['Vtrain'].to(self.dev)
        self.x=c['X'].to(self.dev).double(); self.channel=c['channel']
        xs=(self.x.cpu().numpy()-self.x.cpu().numpy().mean(0))/(self.x.cpu().numpy().std(0)+1e-9)
        dis=cdist(xs,xs); dis[dis<1e-10]=np.inf; self.pair=dis.argmin(1)
        self.distance=C.tensor(dis[np.arange(len(xs)),self.pair]).to(self.dev)
        self.direction=C.tensor((xs[self.pair]-xs)/(self.distance.cpu().numpy()[:,None])).to(self.dev)
        self.train_cov_inv=torch.linalg.pinv(C.cov(C.tensor(data['V'][data['train_ids']]))).to(self.dev)
        tv=C.tensor(data['V'][data['train_ids']]).to(self.dev)
        self.support_threshold=float(torch.einsum('na,ab,nb->n',tv,self.train_cov_inv,tv).quantile(.99))
        self.calls=0; self.support={}
        import copy
        net32=copy.deepcopy(self.net).float()
        with torch.no_grad():
            z=self.mu+self.v[:1]@self.inv.T
            xx=self.x[:8]; zz=z.expand(len(xx),-1)
            inp=torch.cat([xx,zz],-1)
            self.precision_max_logit_error=float((self.net.decoder(inp)-net32.decoder(inp.float()).double()).abs().max())
        del net32
    @torch.no_grad()
    def query(self,v,x):
        v,x=torch.broadcast_tensors(v,x) if v.shape[-1]==x.shape[-1] else (v,x)
        v,x=v.reshape(-1,self.d),x.reshape(-1,self.x.shape[-1])
        out=[]
        for i in range(0,len(v),32768):
            z=self.mu+v[i:i+32768]@self.inv.T
            lg=self.net.decoder(torch.cat([x[i:i+32768],z],-1)); eta=lg[:,1]-lg[:,0]
            out.append(eta if self.channel=='logit' else eta.sigmoid())
        self.calls+=len(v)
        return torch.cat(out)
    @torch.no_grad()
    def raw_one(self,R):
        R=C.tensor(R); A=C.effective(R,self.cov).to(self.dev)
        S,K=len(self.v),len(self.x); N=S*K
        v=self.v[:,None,:].expand(S,K,self.d).reshape(N,self.d)
        x=self.x[None].expand(S,K,-1).reshape(N,-1)
        xp=self.x[self.pair][None].expand(S,K,-1).reshape(N,-1)
        f0=self.query(v,x); values=[]
        for h in self.scales:
            offsets=[h*A[i] for i in range(self.d)]
            shifts=[torch.zeros_like(A[0])]+[s*o for o in offsets for s in (1,-1)]
            for i,j in itertools.combinations(range(self.d),2):
                shifts += [si*offsets[i]+sj*offsets[j] for si,sj in ((1,1),(1,-1),(-1,1),(-1,-1))]
            sh=torch.stack(shifts)
            vv=v[None]+sh[:,None]; xx=x[None].expand(len(sh),N,-1)
            vals=self.query(vv,xx).reshape(len(sh),N)
            diag=[]; grad=[]
            for i in range(self.d):
                fp,fm=vals[1+2*i:3+2*i]
                diag.append((fp-2*f0+fm)/h**2); grad.append((fp-fm)/(2*h))
            uu=torch.zeros(self.d,self.d,dtype=torch.float64,device=self.dev)
            for i in range(self.d): uu[i,i]=diag[i].square().mean()
            idx=1+2*self.d
            for i,j in itertools.combinations(range(self.d),2):
                y=(vals[idx]-vals[idx+1]-vals[idx+2]+vals[idx+3])/(4*h*h); idx+=4
                uu[i,j]=uu[j,i]=y.square().mean()
            # All X queries are actual bank points; use the same pair for every R.
            sh1=torch.stack([s*o for o in offsets for s in (1,-1)])
            paired=self.query(v[None]+sh1[:,None],xp[None].expand(len(sh1),N,-1)).reshape(self.d,2,S,K)
            dg=torch.stack(grad).reshape(self.d,S,K)
            pairgrad=(paired[:,0]-paired[:,1])/(2*h)
            chord=(pairgrad-dg)/self.distance[None,None,:]
            # Direction-projected chord energy, NOT an identified full ambient Hessian.
            ux=(chord[...,None]*self.direction[None,None]).square().mean((1,2))
            g=torch.stack(grad).square().mean(1)
            off=uu*(1-torch.eye(self.d,device=self.dev))
            values += [float(C.neff(ux)),float(C.share(uu)),float(C.neff(g[:,None])),float(C.neff(off))]
            mah=torch.einsum('qna,ab,qnb->qn',vv,self.train_cov_inv,vv)
            # Diagnostic only; never filter by R-dependent support.
            self.support[h]=float((mah<=self.support_threshold).double().mean())
        return C.tensor(values)
    def raw(self,rs):
        single=rs.ndim==2
        rs=rs[None] if single else rs
        with torch.no_grad():
            A=C.effective(C.tensor(rs),self.cov).to(self.dev)
            B,S,K=len(rs),len(self.v),len(self.x); N=S*K
            v=self.v[:,None,:].expand(S,K,self.d).reshape(N,self.d)
            x=self.x[None].expand(S,K,-1).reshape(N,-1)
            xp=self.x[self.pair][None].expand(S,K,-1).reshape(N,-1)
            f0=self.query(v,x); allvalues=[]
            for scale_index,h in enumerate(self.scales):
                active=getattr(self,'active_scales',None)
                if active is not None and scale_index not in active:
                    allvalues.extend([torch.zeros(B,dtype=torch.float64,device=self.dev)]*4)
                    continue
                offsets=h*A
                sh=[torch.zeros_like(A[:,0])]+[s*offsets[:,i] for i in range(self.d) for s in (1,-1)]
                for i,j in itertools.combinations(range(self.d),2):
                    sh += [si*offsets[:,i]+sj*offsets[:,j] for si,sj in ((1,1),(1,-1),(-1,1),(-1,-1))]
                sh=torch.stack(sh,1); Q=sh.shape[1]
                vv=v[None,None]+sh[:,:,None]; xx=x[None,None].expand(B,Q,N,-1)
                vals=self.query(vv,xx).reshape(B,Q,N)
                diag=[]; grad=[]
                for i in range(self.d):
                    fp,fm=vals[:,1+2*i],vals[:,2+2*i]
                    diag.append((fp-2*f0+fm)/h**2); grad.append((fp-fm)/(2*h))
                uu=torch.zeros(B,self.d,self.d,dtype=torch.float64,device=self.dev)
                for i in range(self.d): uu[:,i,i]=diag[i].square().mean(-1)
                idx=1+2*self.d
                for i,j in itertools.combinations(range(self.d),2):
                    z=(vals[:,idx]-vals[:,idx+1]-vals[:,idx+2]+vals[:,idx+3])/(4*h*h); idx+=4
                    uu[:,i,j]=uu[:,j,i]=z.square().mean(-1)
                sh1=torch.stack([s*offsets[:,i] for i in range(self.d) for s in (1,-1)],1)
                pair=self.query(v[None,None]+sh1[:,:,None],xp[None,None].expand(B,len(sh1[0]),N,-1))
                pair=pair.reshape(B,self.d,2,S,K)
                dg=torch.stack(grad,1).reshape(B,self.d,S,K)
                chord=((pair[:,:,0]-pair[:,:,1])/(2*h)-dg)/self.distance[None,None,None,:]
                ux=(chord[...,None]*self.direction[None,None,None]).square().mean((2,3))
                g=torch.stack(grad,1).square().mean(-1)
                off=uu*(1-torch.eye(self.d,device=self.dev))
                allvalues.extend([C.neff(ux),C.share(uu),C.neff(g[...,None]),C.neff(off)])
                mah=torch.einsum('...i,ij,...j->...',vv,self.train_cov_inv,vv)
                self.support[h]=float((mah<=self.support_threshold).double().mean())
            result=torch.stack(allvalues,-1).cpu()
            return result[0] if single else result

def search(fd,scales,weights,starts,steps,seed):
    rng=np.random.default_rng(seed); rs=C.rotations(fd.d,starts,seed)
    fd.active_scales=sorted(set((torch.nonzero(weights).flatten()//4).tolist()))
    def fun(rr): return ((fd.raw(rr)/scales)*weights).sum(-1)
    vals=fun(rs); history=[]
    pairs=list(itertools.combinations(range(fd.d),2))
    # Same deterministic Givens proposals across methods, no autograd approximation.
    for step in range(steps):
        i,j=pairs[step%len(pairs)]
        angle=rng.normal(0,.35*(.05**(step/max(steps-1,1))))
        proposal=C.tensor(C.givens(fd.d,i,j,angle))[None]@rs
        vv=fun(proposal); take=vv<vals
        rs[take]=proposal[take]; vals[take]=vv[take]
        history.append(float(vals.min()))
    order=torch.argsort(vals)
    fd.active_scales=None
    return vals[order].numpy(),rs[order].numpy(),history

def main(a):
    if C.selftest_stage(a): return
    c=C.load(a.model,a.repeat,a.channel,a.subjects,a.trials)
    out=C.result_dir('exp15_finite_scale',a.tag,a.model,a.repeat,a.channel)
    fd=Finite(c); t=time.perf_counter()
    nullrs=C.rotations(fd.d,a.haar_n,1); null=fd.raw(nullrs)
    scales=null.std(0,unbiased=False).clamp_min(1e-9)
    definitions={}
    for j,h in enumerate(fd.scales):
        for label,ix in [('UU',1),('UX',0),('combo',None),('gradient',2),('UUoff',3)]:
            w=torch.zeros(16,dtype=torch.float64)
            if ix is None: w[4*j:4*j+2]=1
            else: w[4*j+ix]=1
            definitions[f'h{h}_{label}']=w
    w=torch.zeros(16,dtype=torch.float64); w[0::4]=1; w[1::4]=1
    definitions['multiscale_equal']=w/4
    if a.only: definitions={k:definitions[k] for k in a.only.split(',')}
    C.dump(out/'config.json',dict(**vars(a),environment=C.environment(),scales=fd.scales,ids=c['ids'],X=c['X'],
                                train_cov=C.cov(c['Vtrain']),haar_seed=1,haar_scales=scales,X_pair=fd.pair,
                                chord='unit-SD X nearest-bank chord projected on its direction; not ambient UX Hessian',
                                evidence='decoder-query',support='TRAIN Mahalanobis 99% ellipsoid diagnostic, NO points removed',
                                float64_checkpoint_cast_max_logit_error=fd.precision_max_logit_error,
                                optimizer='common-seed greedy Givens proposals; equal objective-query budget per method'))
    # Evaluate once for all methods at all plane / random candidates.
    oracle=C.oracle(c); planes=[]; plane_rs=[]
    for i,j in itertools.combinations(range(fd.d),2):
        for angle in range(0,91,5):
            planes.append(dict(i=i,j=j,angle=angle)); plane_rs.append(C.givens(fd.d,i,j,np.deg2rad(angle))@oracle)
    pv=fd.raw(C.tensor(np.stack(plane_rs)))
    random=fd.raw(C.rotations(fd.d,a.random_n,91)); ov=fd.raw(C.tensor(oracle))
    rows=[]; saved={}; ds={}; pcs=[]; rrmain={}
    for name,w in definitions.items():
        before=fd.calls; tt=time.perf_counter()
        ll,rr,hist=search(fd,scales,w,a.starts,a.steps,a.seed)
        saved[name+'__R']=rr; saved[name+'__loss']=ll
        saved[name+'__axis_sd']=np.std(c['V'][c['ids']]@rr[0].T,axis=0)
        np.savez_compressed(out/'rotations.npz',**saved)
        met=C.evaluate(c,rr[0]); rloss=((random/scales)*w).sum(-1).numpy(); ol=float((ov/scales*w).sum())
        fd.raw(C.tensor(rr[0])) # Record support at the selected R, not the last rejected proposal.
        ds[name]=dict(oracle_z=float((rloss.mean()-ol)/(rloss.std()+1e-12)),oracle_loss=ol,
                      support_by_scale=fd.support,all_endpoints=[C.evaluate(c,r) for r in rr],history=hist)
        pcs += [{**p,'method':name,'loss':float(y)} for p,y in zip(planes,(pv/scales*w).sum(-1))]
        rows.append(dict(model=a.model,repeat=a.repeat,channel=a.channel,method=name,starts=a.starts,steps=a.steps,
                         centers=len(c['ids'])*len(c['X']),loss=ll[0],seconds=time.perf_counter()-tt,
                         query_rows=fd.calls-before,**{k:met[k] for k in ('mcc','leakage','max_leakage')}))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False); C.dump(out/'diagnostics.json',ds)
        rrmain[name]=rr[0]; print(rows[-1],flush=True)
    pd.DataFrame(pcs).to_csv(out/'plane_curves.csv',index=False)
    C.heatmap(c,{'Identity':np.eye(fd.d),'EXP13':C.baseline_rotation(c),next(iter(rrmain)):next(iter(rrmain.values())),
                 'Oracle':oracle},out/'UT_heatmaps.png')
    C.dump(out/'COMPLETE.json',dict(methods=list(definitions),seconds=time.perf_counter()-t,query_rows=fd.calls))

if __name__=='__main__':
    p=C.parser(__doc__); p.set_defaults(subjects=8,trials=8,starts=8,steps=100,random_n=100)
    p.add_argument('--haar-n',type=int,default=200); p.add_argument('--only',default='h0.1_combo,h0.25_combo,h0.5_combo,h1.0_combo,multiscale_equal')
    main(p.parse_args())
