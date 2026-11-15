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
    def __init__(self,st,V,X,decoder_dtype=torch.float32,engine='scalar'):
        self.st=st;self.d=st.d;self.p=st.p;self.C=torch.tensor(st.C,dtype=torch.float64,device=st.dev)
        self.decoder_dtype=decoder_dtype
        self.decoder=copy.deepcopy(st.net.decoder).to(decoder_dtype)  # original checkpoint untouched
        self.mu=torch.tensor(st.mu,dtype=torch.float64,device=st.dev);self.Minv=torch.tensor(st.Minv,dtype=torch.float64,device=st.dev)
        self.V=torch.tensor(V,dtype=torch.float64,device=st.dev).repeat_interleave(len(X),0)
        self.X=torch.tensor(X,dtype=torch.float64,device=st.dev).repeat(len(V),1)
        self.Xbank=torch.tensor(st.X,dtype=torch.float64,device=st.dev)
        self.Xcentres=X;self.query_rows=0;self.forward_calls=0;self.gradX_rows=0;self.gradX_evaluations=0
        self.chord=None;self.last_chord_residual=None
        self.engine=engine
    def f(self,V,X):
        self.query_rows+=len(V)
        self.forward_calls+=1
        parts=[]
        for begin in range(0,len(V),65536):
            z=self.mu+V[begin:begin+65536]@self.Minv.T
            lg=self.decoder(torch.cat([X[begin:begin+65536],z],-1).to(self.decoder_dtype)).double()
            parts.append(torch.sigmoid(lg[:,1]-lg[:,0]))
        return torch.cat(parts)
    def scalar(self,v,x):
        lg=self.decoder(torch.cat([x,self.mu+self.Minv@v],-1).to(self.decoder_dtype)).double();return torch.sigmoid(lg[1]-lg[0])
    def energies(self,R,h,mode,hX=None):
        if getattr(self,'engine','scalar')=='vectorized' and mode in ['ambient','hybrid']:
            return self.energies_vectorized(R,h,mode,hX)
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
                self.gradX_rows+=2*len(v);self.gradX_evaluations+=2
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

    def energies_vectorized(self,R,h,mode,hX=None):
        """Same coordinate stencils in large row batches; symmetric UU reused.

        R still changes EVERY physical query. No finite tensor is rotated.
        The 65536-row MLP chunks bound temporary memory, with original dtype.
        """
        if h<=0:raise ValueError('positive finite halfwidth')
        R=R.to(self.C);A=c.effective_A(R,self.C);v,x=self.V,self.X
        d,p,N=self.d,self.p,len(v)
        eye=torch.eye(d,dtype=v.dtype,device=v.device)
        coef=[torch.zeros(d,dtype=v.dtype,device=v.device)]
        for i in range(d):coef.extend([eye[i],-eye[i]])
        pairs=[]
        for i in range(d):
            for k in range(i+1,d):
                pairs.append((i,k,len(coef)))
                coef.extend([eye[i]+eye[k],eye[i]-eye[k],-eye[i]+eye[k],-eye[i]-eye[k]])
        shifts=h*(torch.stack(coef)@A)
        Vq=(v[None,:,:]+shifts[:,None,:]).reshape(-1,d)
        values=self.f(Vq,x.repeat(len(coef),1)).reshape(len(coef),N)
        uu=[[None for _ in range(d)] for _ in range(d)]
        for i in range(d):
            D=(values[1+2*i]-2*values[0]+values[2+2*i])/h**2
            uu[i][i]=D.square().mean()
        for i,k,start in pairs:
            D=(values[start]-values[start+1]-values[start+2]+values[start+3])/(4*h*h)
            e=D.square().mean();uu[i][k]=e;uu[k][i]=e
        uu=torch.stack([torch.stack(row) for row in uu])
        if mode=='hybrid':
            from torch.func import jacrev,vmap
            axcoef=torch.stack(coef[1:1+2*d]);shift=h*(axcoef@A)
            Vax=(v[None,:,:]+shift[:,None,:]).reshape(-1,d)
            self.gradX_rows+=2*d*N;self.gradX_evaluations+=1
            gx=vmap(jacrev(self.scalar,1))(Vax,x.repeat(2*d,1)).reshape(d,2,N,p)
            ux=((gx[:,0]-gx[:,1])/(2*h)).square().mean(1)
        else:
            hu=torch.as_tensor(hX,dtype=v.dtype,device=v.device)
            ex=torch.eye(p,dtype=v.dtype,device=v.device);vc=[];xc=[]
            for i in range(d):
                for j in range(p):
                    vc.extend([eye[i],eye[i],-eye[i],-eye[i]])
                    xc.extend([ex[j],-ex[j],ex[j],-ex[j]])
            vs=h*(torch.stack(vc)@A);xs=torch.stack(xc)*hu
            Vq=(v[None,:,:]+vs[:,None,:]).reshape(-1,d)
            Xq=(x[None,:,:]+xs[:,None,:]).reshape(-1,p)
            f=self.f(Vq,Xq).reshape(d,p,4,N)
            D=(f[:,:,0]-f[:,:,1]-f[:,:,2]+f[:,:,3])/(4*h*hu[None,:,None])
            ux=D.square().mean(-1)
        return ux,uu

class FDLoss:
    def __init__(self,field,h,mode,hX=None,scales=None):
        self.field=field;self.d=field.d;self.h=h;self.mode=mode;self.hX=hX
        if scales is not None:
            self.scales=np.asarray(scales)
            if np.any(self.scales<=0):raise ValueError('positive fixed Haar scales required')
            return
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
    def total(self,R):
        term=self.terms(R)
        return (term/torch.as_tensor(self.scales).to(term)).sum(-1)
