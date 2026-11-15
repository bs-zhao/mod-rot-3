"""Strict fitting boundary: only arrays W,Y and validation arrays are accepted.

No original decoder, oracle, generator or T can be accessed by this module.
"""
from __future__ import annotations
import itertools
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from torch.func import jacfwd,jacrev,vmap

class Polynomial:
    def fit(self,W,y,V,vy):
        self.mu=W.mean(0); self.sd=W.std(0).clip(1e-6)
        w=(W-self.mu)/self.sd; v=(V-self.mu)/self.sd
        best=None; self.candidates=[]
        for degree,alpha in itertools.product((2,3,4),(.001,.1,10.)):
            pf=PolynomialFeatures(degree,include_bias=True); design=pf.fit_transform(w)
            fit=Ridge(alpha=alpha,fit_intercept=False).fit(design,y)
            error=float(np.mean((fit.predict(pf.transform(v))-vy)**2))
            self.candidates.append(dict(degree=degree,alpha=alpha,validation_mse=error))
            if best is None or error<best[0]: best=(error,pf.powers_,fit.coef_)
        self.error,self.powers,self.coef=best
        return self
    def predict(self,W):
        w=(W-self.mu)/self.sd
        return np.prod(w[:,None,:]**self.powers[None],axis=-1)@self.coef
    def derivatives(self,W,d):
        w=(W-self.mu)/self.sd; n,p=w.shape; H=np.zeros((n,d,p))
        def deriv(indices):
            powers=self.powers.copy(); co=self.coef.copy()
            for i in indices:
                co=co*powers[:,i]/self.sd[i]; powers[:,i]=np.maximum(powers[:,i]-1,0)
            return np.prod(w[:,None,:]**powers[None],axis=-1)@co
        g=np.stack([deriv([i]) for i in range(d)],-1)
        for i in range(d):
            for j in range(p): H[:,i,j]=deriv([i,j])
        return g,H[:,:,:d],H[:,:,d:]

class Kernel:
    def fit(self,W,y,V,vy):
        self.mu=W.mean(0); self.sd=W.std(0).clip(1e-6)
        rng=np.random.default_rng(1801); idx=rng.choice(len(W),min(384,len(W)),replace=False)
        self.centers=(W[idx]-self.mu)/self.sd
        w=(W-self.mu)/self.sd; v=(V-self.mu)/self.sd
        self.candidates=[]; best=None
        for ell,alpha in itertools.product((1.,2.,4.),(.001,.1,10.)):
            design=np.exp(-((w[:,None]-self.centers[None])**2).sum(-1)/(2*ell**2))
            fit=Ridge(alpha=alpha).fit(design,y)
            pv=np.exp(-((v[:,None]-self.centers[None])**2).sum(-1)/(2*ell**2))
            err=float(np.mean((fit.predict(pv)-vy)**2))
            self.candidates.append(dict(lengthscale=ell,alpha=alpha,validation_mse=err))
            if best is None or err<best[0]: best=(err,ell,fit.coef_,fit.intercept_)
        self.error,self.ell,self.coef,self.intercept=best; return self
    def predict(self,W):
        x=(W-self.mu)/self.sd
        return np.exp(-((x[:,None]-self.centers[None])**2).sum(-1)/(2*self.ell**2))@self.coef+self.intercept
    def derivatives(self,W,d):
        x=(W-self.mu)/self.sd; dx=x[:,None]-self.centers[None]; e=self.ell**2
        k=np.exp(-(dx**2).sum(-1)/(2*e))*self.coef
        g=(-(k[...,None]*dx/e)).sum(1)/self.sd
        H=np.einsum('nm,nma,nmb->nab',k,dx[:,:,:d],dx)/e**2
        for i in range(d): H[:,i,i]-=k.sum(1)/e
        H=H/(self.sd[None,:d,None]*self.sd[None,None,:])
        return g[:,:d],H[:,:,:d],H[:,:,d:]

class GP(Kernel):
    """Explicit capped exact GP (384 observations), not a million-row GP claim."""
    def fit(self,W,y,V,vy):
        self.mu=W.mean(0); self.sd=W.std(0).clip(1e-6)
        idx=np.random.default_rng(1801).choice(len(W),min(384,len(W)),replace=False)
        self.centers=(W[idx]-self.mu)/self.sd; v=(V-self.mu)/self.sd
        self.candidates=[]; best=None
        for ell,alpha in itertools.product((1.,2.,4.),(.001,.1)):
            fit=GaussianProcessRegressor(RBF(ell),alpha=alpha,optimizer=None,normalize_y=False).fit(self.centers,y[idx])
            err=float(np.mean((fit.predict(v)-vy)**2))
            self.candidates.append(dict(lengthscale=ell,alpha=alpha,validation_mse=err,training_rows=len(idx)))
            if best is None or err<best[0]: best=(err,ell,fit.alpha_)
        self.error,self.ell,self.coef=best; self.intercept=0.; return self

class MLP:
    def __init__(self,seed=31,penalty=0.,steps=1000,latent_dim=4):
        self.seed=seed; self.penalty=penalty; self.steps=steps; self.latent_dim=latent_dim
    def fit(self,W,y,V,vy):
        torch.manual_seed(self.seed); self.mu=W.mean(0); self.sd=W.std(0).clip(1e-6)
        self.dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net=torch.nn.Sequential(torch.nn.Linear(W.shape[1],64),torch.nn.Tanh(),
                    torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1)).to(self.dev)
        w=torch.tensor((W-self.mu)/self.sd,dtype=torch.float32,device=self.dev)
        yy=torch.tensor(y,dtype=torch.float32,device=self.dev)
        v=torch.tensor((V-self.mu)/self.sd,dtype=torch.float32,device=self.dev)
        target=torch.tensor(vy,dtype=torch.float32,device=self.dev)
        opt=torch.optim.Adam(self.net.parameters(),lr=.002)
        best=float('inf'); self.candidates=[]; state=None
        for step in range(self.steps):
            ids=torch.randint(len(w),(min(256,len(w)),),device=self.dev); x=w[ids]
            pred=self.net(x).flatten(); loss=(pred-yy[ids]).square().mean()
            if self.penalty:
                # Isotropic directions in original frozen V (UU only), then map into standardized network input.
                direction=torch.randn_like(x); direction[:,self.latent_dim:]=0
                direction=direction/direction.norm(dim=1,keepdim=True)
                direction=direction/torch.tensor(self.sd,dtype=x.dtype,device=self.dev)
                hh=.25; f2=(self.net(x+hh*direction).flatten()-2*pred+self.net(x-hh*direction).flatten())/hh**2
                loss=loss+self.penalty*f2.square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (step+1)%50==0:
                with torch.no_grad(): error=float((self.net(v).flatten()-target).square().mean())
                self.candidates.append(dict(step=step+1,validation_mse=error,penalty=self.penalty,seed=self.seed))
                if error<best:
                    best=error; state={k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()}
        self.net.load_state_dict(state); self.net=self.net.cpu().double().eval(); self.net.requires_grad_(False)
        self.error=best; return self
    def predict(self,W):
        with torch.no_grad(): return self.net(torch.tensor((W-self.mu)/self.sd,dtype=torch.float64)).flatten().numpy()
    def derivatives(self,W,d):
        mu=torch.tensor(self.mu); sd=torch.tensor(self.sd)
        h=lambda x:self.net((x-mu)/sd).squeeze()
        g=vmap(jacrev(h))(torch.tensor(W,dtype=torch.float64))
        hh=vmap(jacfwd(jacrev(h)))(torch.tensor(W,dtype=torch.float64))
        return g[:,:d].numpy(),hh[:,:d,:d].numpy(),hh[:,:d,d:].numpy()
