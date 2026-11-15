"""R-dependent finite differences, independent fresh implementation."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
import torch
from scales import chord_design

class Field:
    def __init__(self,st,V,X):
        self.st=st;self.d=st.d;self.p=st.p;self.C=torch.tensor(st.C,dtype=torch.float64,device=st.dev)
        self.decoder=copy.deepcopy(st.net.decoder).double()  # independent copy; original checkpoint untouched
        self.mu=torch.tensor(st.mu,dtype=torch.float64,device=st.dev);self.Minv=torch.tensor(st.Minv,dtype=torch.float64,device=st.dev)
        self.V=torch.tensor(V,dtype=torch.float64,device=st.dev).repeat_interleave(len(X),0)
        self.X=torch.tensor(X,dtype=torch.float64,device=st.dev).repeat(len(V),1)
        self.Xbank=torch.tensor(st.X,dtype=torch.float64,device=st.dev)
        self.Xcentres=X;self.query_rows=0
        self.chord=None;self.last_chord_residual=None
    def f(self,V,X):
        self.query_rows+=len(V)
        lg=self.decoder(torch.cat([X,self.mu+V@self.Minv.T],-1));return torch.sigmoid(lg[:,1]-lg[:,0])
    def scalar(self,v,x):
        lg=self.decoder(torch.cat([x,self.mu+self.Minv@v],-1));return torch.sigmoid(lg[1]-lg[0])
    def energies(self,R,h,mode,hX=None):
        R=R.to(self.C);A=c.effective_A(R,self.C);v,x=self.V,self.X
        if mode=='autograd':
            from torch.func import jacrev,jacfwd,vmap
            g=jacrev(self.scalar,0)
            tt=vmap(jacfwd(g,0))(v,x);tx=vmap(jacfwd(g,1))(v,x)
            uu=torch.einsum('ia,nab,kb->nik',A,tt,A)
            ux=torch.einsum('ia,naj->nij',A,tx)
            return ux.square().mean(0),uu.square().mean(0)
        if h<=0:raise ValueError('halfwidth must be positive')
        step=h*A
        f0=self.f(v,x);plus=[self.f(v+step[i],x) for i in range(self.d)];minus=[self.f(v-step[i],x) for i in range(self.d)]
        uu=[]
        for i in range(self.d):
            row=[]
            for j in range(self.d):
                if i==j:D=(plus[i]-2*f0+minus[i])/h**2
                else:D=(self.f(v+step[i]+step[j],x)-self.f(v+step[i]-step[j],x)-self.f(v-step[i]+step[j],x)+self.f(v-step[i]-step[j],x))/(4*h*h)
                row.append(D.square().mean())
            uu.append(torch.stack(row))
        uu=torch.stack(uu);ux=[]
        for i in range(self.d):
            if mode=='hybrid':
                from torch.func import jacrev,vmap
                gx=vmap(jacrev(self.scalar,1))
                D=(gx(v+step[i],x)-gx(v-step[i],x))/(2*h)
            elif mode=='ambient':
                rows=[]
                for j in range(self.p):
                    dx=torch.zeros(self.p,dtype=torch.float64,device=x.device);dx[j]=float(hX[j])
                    rows.append((self.f(v+step[i],x+dx)-self.f(v+step[i],x-dx)-self.f(v-step[i],x+dx)+self.f(v-step[i],x-dx))/(4*h*hX[j]))
                D=torch.stack(rows,-1)
            elif mode in ['chord','chord_limit']:
                # Each chosen centre uses its actual-bank neighbours, reconstructed
                # in SD-X units and converted back to raw-X derivative units.
                if self.chord is None:self.chord=chord_design(self.st.X)
                nb,length,pinv,rank,cond=self.chord
                ci=np.argmin(((self.Xcentres[:,None,:]-self.st.X[None,:,:])**2).sum(-1),axis=1)
                K=len(ci);S=len(v)//K
                idx=torch.tensor(nb[ci],device=x.device)
                xb=self.Xbank[idx].repeat(S,1,1);q=xb.shape[1]
                vs=v[:,None,:].expand(-1,q,-1).reshape(-1,self.d);xb=xb.reshape(-1,self.p)
                if mode=='chord':
                    ga=(plus[i]-minus[i])/(2*h)
                    gb=(self.f(vs+step[i],xb)-self.f(vs-step[i],xb)).reshape(len(v),q)/(2*h)
                else:
                    from torch.func import jacrev,vmap
                    gv=vmap(jacrev(self.scalar,0));ga=(gv(v,x)*A[i]).sum(-1)
                    gb=(gv(vs,xb)*A[i]).sum(-1).reshape(len(v),q)
                ell=torch.tensor(length[ci],device=x.device).repeat(S,1)
                directional=(gb-ga[:,None])/ell
                P=torch.tensor(pinv[ci],device=x.device).repeat(S,1,1)
                Dscaled=(P@directional[:,:,None]).squeeze(-1)
                unit=(xb.reshape(len(v),q,self.p)-x[:,None,:])/torch.tensor(self.st.X.std(0),device=x.device)/ell[:,:,None]
                resid=(torch.einsum('nqp,np->nq',unit,Dscaled)-directional)
                self.last_chord_residual=float((resid.square().mean()/(directional.square().mean()+1e-30)).detach())
                D=Dscaled/torch.tensor(self.st.X.std(0),device=x.device)
            else:raise ValueError(mode)
            ux.append(D.square().mean(0))
        return torch.stack(ux),uu

class FDLoss:
    def __init__(self,field,h,mode,hX=None,scales=None):
        self.field=field;self.d=field.d;self.h=h;self.mode=mode;self.hX=hX
        with torch.no_grad():
            vals=self.terms(torch.tensor(c.haar_set(self.d,200,1),dtype=torch.float32)).cpu().numpy()
        sd=vals.std(0)
        if (sd<1e-8).any():raise ValueError('no Haar signal')
        self.scales=sd+1e-9 if scales is None else scales
    def terms(self,R):
        if R.ndim==3:return torch.stack([self.terms(q) for q in R])
        ux,uu=self.field.energies(R,self.h,self.mode,self.hX)
        q=ux/(ux.sum()+1e-12)+1e-12
        return torch.stack([(-(q*q.log()).sum()).exp(),uu.diag().sum()/(uu.sum()+1e-12)])
    def total(self,R):return (self.terms(R)/torch.as_tensor(self.scales).to(R)).sum(-1)
