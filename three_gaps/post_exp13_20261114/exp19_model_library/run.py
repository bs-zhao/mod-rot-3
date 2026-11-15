"""EXP19: disjoint formula families, explicit negative controls, learned score.

The learned score receives permutation-invariant rotated structure summaries of
a polynomial fitted only to synthetic W,Y. Synthetic T/Q supply training labels.
"""
from __future__ import annotations
import argparse
import itertools
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
sys.path.insert(0,str(C.HERE/'exp18_matrix_only'))
from surrogates import Polynomial

FAMILIES={'train':['additive','products','shared','sigmoid'],
          'validation':['cubic'],
          'test':['composite','units','difference_square','radial']}

def formula(family,t,x,coeff):
    a,b,c,d=t.T; x0,x1,x2,x3=x.T; k=coeff
    if family=='additive': y=k[0]*a*x0+k[1]*b*x1+k[2]*c*x2+k[3]*d*x3
    elif family=='products': y=a*x0+b*x1+c*x2+d*x3+k[0]*a*b*x2+k[1]*c*d*x0
    elif family=='shared': y=a*(x0+x1**2)+b*(x0*x2+x3)+c*x2+d*x3**2+k[0]*a*d*x1
    elif family=='sigmoid': y=2/(1+np.exp(-np.clip(a*x0+b*x1+c*x2+d*x3,-100,100)))-1
    elif family=='cubic': y=a*x0+b*x1+c*x2+d*x3+k[0]*a*b*c*x0+k[1]*a*d**2*x1
    elif family=='composite': y=np.tanh(a*np.sin(x0)+b*x1*x2)+c*np.exp(.2*d*x3)+.2*d*x0
    elif family=='units': y=a*(100*x0/100)+b*(.01*x1/.01)+c*x2+d*x3+.2*a*c*x1
    elif family=='difference_square': y=(a*a-b*b)*x0+2*a*b*x1+c*x2+d*x3+.1*(a*x2+b*x3)
    elif family=='radial': y=(a*a+b*b+c*c+d*d)*x0
    else: raise ValueError(family)
    return y

def features(A,mm,s2):
    ux,uu=C.energies(A,mm); d=uu.shape[-1]; p=ux.shape[-1]
    q=ux/(ux.sum((-2,-1),keepdim=True)+C.EPS)
    row=q.sum(-1); col=q.sum(-2)
    entrow=(-(row*(row+C.EPS).log()).sum(-1)).exp()/d
    entcol=(-(col*(col+C.EPS).log()).sum(-1)).exp()/p
    g=torch.einsum('...ia,ab,...ib->...i',A,s2,A)
    off=uu*(1-torch.eye(d,dtype=uu.dtype))
    return torch.stack([C.neff(ux)/(d*p),C.share(uu),C.neff(off)/(d*(d-1)),
                        entrow,entcol,C.neff(g[...,None])/d],-1)

def target(R,Q):
    product=np.abs(R@Q); i,j=linear_sum_assignment(-product)
    return float(1-product[i,j].mean())

def build(family,index,split,a):
    rng=np.random.default_rng(1900+index+100*list(sum(FAMILIES.values(),[])).index(family))
    q=C.rotations(4,1,1900+index)[0].numpy(); coeff=rng.uniform(.2,.8,4)
    units=np.array([100.,.01,1.,1.]) if family=='units' else np.ones(4)
    def matrix(n):
        t=rng.normal(size=(n,4)); x=rng.uniform(-1,1,(n,4)); v=t@q.T
        return np.concatenate([v,x*units],-1),formula(family,t,x,coeff)
    w,y=matrix(a.fit_rows); v,vy=matrix(1024)
    model=Polynomial().fit(w,y,v,vy)
    t=rng.normal(size=(32,4)); x=rng.uniform(-1,1,(32,4)); latent=t@q.T
    points=np.concatenate([np.repeat(latent,32,axis=0),np.tile(x*units,(32,1))],-1)
    g,tt,tx=model.derivatives(points,4)
    archive=C.ROOT/'outputs/three_gaps/post_exp13_20261114/model_library'/a.tag
    archive.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(archive/f'{split}_{family}_{index}.npz',W=w,Y=y,validation_W=v,validation_Y=vy,
                        points=points,g=g,HUU=tt,HUX=tx,powers=model.powers,coef=model.coef,mu=model.mu,sd=model.sd)
    mm=C.moments(C.tensor(tx),C.tensor(tt)); s2=C.tensor(g).T@C.tensor(g)/len(g)
    covariance=C.cov(C.tensor(latent))
    rng2=np.random.default_rng(1910+index)
    # Training includes random candidates and perturbations near known axes; labels are synthetic supervision.
    rs=C.rotations(4,200,1901+index).numpy()
    near=[]
    for _ in range(80):
        skew=rng2.normal(size=(4,4))*.3; near.append(torch.matrix_exp(C.tensor(skew-skew.T)).numpy()@q.T)
    rs=np.concatenate([rs,np.array(near),q.T[None]],axis=0)
    ff=features(C.effective(C.tensor(rs),covariance),mm,s2).numpy()
    yy=np.array([target(r,q) for r in rs])
    # Response Jacobian rank, separate from coordinate-axis uniqueness.
    h=1e-4; pt=np.array([.4,.7,-.2,.9]); xx=rng.uniform(-1,1,(128,4))
    gg=np.stack([(formula(family,np.tile(pt+h*np.eye(4)[i],(128,1)),xx,coeff)-
                  formula(family,np.tile(pt-h*np.eye(4)[i],(128,1)),xx,coeff))/(2*h) for i in range(4)],1)
    sv=np.linalg.svd(gg,compute_uv=False)
    gs=np.linalg.svd(g.reshape(32,32,4),compute_uv=False)
    rank_ratio=float(np.median(gs[:,-1]/np.maximum(gs[:,0],1e-12)))
    return dict(family=family,index=index,split=split,Q=q,coef=coeff,features=ff,target=yy,mm=mm,s2=s2,
                observed_X_units=units,estimated_local_jacobian_rank_ratio=rank_ratio,
                V=latent,cov=covariance,validation_mse=float(np.mean((model.predict(v)-vy)**2)),
                response_singular_values=sv,local_rank=int((sv>sv[0]*1e-5).sum()) if sv[0]>1e-12 else 0,
                negative_control=family in ('radial','difference_square'))

class Learned:
    def __init__(self,item,net,mu,sd):
        self.item=item; self.net=net; self.mu=mu; self.sd=sd; self.cov=item['cov']
    def __call__(self,R):
        f=features(C.effective(R,self.cov),self.item['mm'],self.item['s2'])
        return self.net((f-self.mu)/self.sd).squeeze(-1)

def train(items,a):
    torch.manual_seed(1901)
    train=[x for x in items if x['split']=='train']; val=[x for x in items if x['split']=='validation']
    x=C.tensor(np.concatenate([z['features'] for z in train])); y=C.tensor(np.concatenate([z['target'] for z in train]))
    vx=C.tensor(np.concatenate([z['features'] for z in val])); vy=C.tensor(np.concatenate([z['target'] for z in val]))
    mu=x.mean(0); sd=x.std(0).clamp_min(1e-6)
    net=torch.nn.Sequential(torch.nn.Linear(6,24),torch.nn.Tanh(),torch.nn.Linear(24,24),torch.nn.Tanh(),torch.nn.Linear(24,1)).double()
    opt=torch.optim.Adam(net.parameters(),lr=.002); best=float('inf'); state=None; hist=[]
    for step in range(a.fit_steps):
        idx=torch.randint(len(x),(256,)); loss=(net((x[idx]-mu)/sd).squeeze()-y[idx]).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1)%50==0:
            with torch.no_grad(): error=float((net((vx-mu)/sd).squeeze()-vy).square().mean())
            hist.append(dict(step=step+1,validation_mse=error))
            if error<best: best=error; state={k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state); net.eval(); net.requires_grad_(False)
    return net,mu,sd,hist

def invariant_test(item):
    P=C.rotations(4,1,99)[0]; R=C.rotations(4,3,98)
    # v'=P v -> g'=P g, H'=P H P.T; candidate R'=R P.T.
    sx,t4=item['mm']; sx2=torch.einsum('ia,jab,kb->jik',P,sx,P)
    t42=torch.einsum('ia,kb,abcd,jc,ld->ikjl',P,P,t4,P,P)
    s22=P@item['s2']@P.T
    f=features(C.effective(R,item['cov']),item['mm'],item['s2'])
    f2=features(C.effective(R@P.T,P@item['cov']@P.T),(sx2,t42),s22)
    torch.testing.assert_close(f,f2,rtol=1e-7,atol=1e-8)
    return float((f-f2).abs().max())

def _main(a):
    out=Path(__file__).parent/'results'/a.tag; out.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    items=[]
    for split,families in FAMILIES.items():
        for family in families:
            for i in range(a.formulas_per_family):
                items.append(build(family,i,split,a)); print('BUILD',split,family,i,flush=True)
    eq=invariant_test(items[0]); net,mu,sd,hist=train(items,a)
    big=C.ROOT/'outputs/three_gaps/post_exp13_20261114/learned_score'; big.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=net.state_dict(),mu=mu,sd=sd,families=FAMILIES),big/(a.tag+'.pt'))
    C.dump(out/'config.json',dict(**vars(a),environment=C.environment(),family_split=FAMILIES,
                                grammar='fixed composition grammar: +, *, powers <=3, sigmoid/tanh, sin, bounded exp',
                                synthetic_supervision=True,structure_estimator='degree2/3/4 ridge selected by held-out Y',
                                feature_equivariance_max_error=eq,
                                label='1 - mean maximum signed-permutation diagonal of R Q',
                                warning='Learned score encodes these training formula priors; no universal guarantee'))
    C.dump(out/'training.json',hist); rows=[]; saved={}; planes=[]
    for item in items:
        learned=Learned(item,net,mu,sd)
        baseline=C.Objective(C.tensor(item['V']),lambda A,item=item:C.terms(A,item['mm']))
        # Unrotated axis baseline; all searches share the same start seeds/budget.
        methods={'EXP13':baseline,'learned':learned}
        if item['split']=='train': continue
        for name,obj in methods.items():
            ll,rr=C.optimize(obj,4,a.starts,a.steps,7)
            label=f"{item['family']}_{item['index']}_{name}"
            saved[label+'__R']=rr; saved[label+'__loss']=ll
            rs=np.random.default_rng(1950).normal(size=(5000,4)); vv=rs@item['Q'].T
            met=C.match_report(vv@rr[0].T,rs)
            near=rr[ll<=ll[0]+max(.01,.01*abs(ll[0]))]
            disagreement=max([target(r@rr[0].T,np.eye(4)) for r in near],default=0.)
            # Diagnostic abstention uses fitted response rank and low-loss disagreement, not T labels.
            ambiguous=bool(item['estimated_local_jacobian_rank_ratio']<1e-3 or disagreement>.15)
            rows.append(dict(split=item['split'],family=item['family'],formula=item['index'],method=name,
                             mcc=met['mcc'],leakage=met['off_mean'],max_leakage=met['off_max'],
                             axis_error=target(rr[0],item['Q']),loss=ll[0],validation_prediction_mse=item['validation_mse'],
                             response_rank=item['local_rank'],negative_control=item['negative_control'],
                             abstain=ambiguous,low_loss_disagreement=disagreement))
            for i,j in itertools.combinations(range(4),2):
                prs=C.tensor(np.stack([C.givens(4,i,j,np.deg2rad(ang))@item['Q'].T for ang in range(0,91,5)]))
                with torch.no_grad(): yy=obj(prs).numpy()
                planes.extend(dict(family=item['family'],formula=item['index'],method=name,i=i,j=j,angle=ang,loss=float(l))
                              for ang,l in zip(range(0,91,5),yy))
            print(rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        np.savez_compressed(out/'rotations.npz',**saved)
    pd.DataFrame(planes).to_csv(out/'plane_curves.csv',index=False)
    C.dump(out/'model_manifest.json',[{k:v for k,v in item.items() if k not in ('mm','s2','V','features','target')}
                                      for item in items])
    C.dump(out/'COMPLETE.json',dict(rows=len(rows),seconds=time.perf_counter()-start))

def main(a):
    out=Path(__file__).parent/'results'/a.tag; out.mkdir(parents=True,exist_ok=True)
    with C.exclusive_run(out):
        if (out/'COMPLETE.json').exists():
            print('Already completed:',out,flush=True); return
        _main(a)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--tag',default='round1')
    p.add_argument('--formulas-per-family',type=int,default=3); p.add_argument('--fit-rows',type=int,default=4096)
    p.add_argument('--fit-steps',type=int,default=2000); p.add_argument('--starts',type=int,default=24)
    p.add_argument('--steps',type=int,default=250); main(p.parse_args())
