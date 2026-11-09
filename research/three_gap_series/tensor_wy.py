"""Generic E02 smooth tensor regression from subject-grouped numerical WY.

Same dictionary in every dataset: complete degree-2 X polynomial plus
isotropic Gaussian RBFs, and a complete latent polynomial. No model names,
option structure, generating expressions, parameter labels or derivative data.
"""
import math
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures


class TensorWY:
    def __init__(self,degree=3,ridge=1e-6,stimulus_ridge=1e-5,centers=64,length=1.,seed=20261127):
        self.degree=degree;self.ridge=ridge;self.stimulus_ridge=stimulus_ridge
        self.ncenters=centers;self.length=length;self.seed=seed

    def xfeatures(self,x):
        poly=self.xbasis.transform(x/np.sqrt(x.shape[1]))*self.xweights
        sq=np.maximum(np.sum(x*x,1)[:,None]+np.sum(self.centers*self.centers,1)[None]-2*x@self.centers.T,0)
        rbf=np.exp(-sq/(2*self.sigma**2))
        return np.column_stack((poly,rbf))

    def fit(self,z,x_tables,y_tables):
        # Inputs are numeric arrays only. Shared X detection is a data property,
        # not a model-specific estimator branch.
        rng=np.random.default_rng(self.seed);pool=np.concatenate(x_tables)
        pick=rng.choice(len(pool),min(8192,len(pool)),replace=False);pool=pool[pick]
        self.centers=pool[rng.choice(len(pool),min(self.ncenters,len(pool)),replace=False)].copy()
        self.sigma=self.length*np.sqrt(pool.shape[1])
        self.xbasis=PolynomialFeatures(2).fit(pool[:1]);xp=self.xbasis.powers_
        self.xweights=np.array([math.sqrt(math.factorial(int(a.sum()))/math.prod(math.factorial(int(j)) for j in a)) for a in xp])
        self.xscale=np.maximum(np.sqrt(np.mean(self.xfeatures(pool)**2,axis=0)),1e-8)
        q=len(self.xscale);polyq=len(xp)
        penalty=np.r_[np.full(polyq,1e-10),np.full(len(self.centers),self.stimulus_ridge)]
        penalty[0]=0
        shared=all(np.array_equal(x_tables[0],x) for x in x_tables[1:])
        if shared:
            design=self.xfeatures(x_tables[0])/self.xscale
            projection=np.linalg.solve(design.T@design/len(design)+np.diag(penalty),design.T/len(design))
            profiles=np.asarray(y_tables)@projection.T
        else:
            profiles=[]
            for x,y in zip(x_tables,y_tables):
                design=self.xfeatures(x)/self.xscale
                profiles.append(np.linalg.solve(design.T@design/len(design)+np.diag(penalty),design.T@y/len(design)))
            profiles=np.asarray(profiles)
        self.zbasis=PolynomialFeatures(self.degree).fit(z[:1]);self.powers=self.zbasis.powers_
        self.zweights=np.array([math.sqrt(math.factorial(int(a.sum()))/math.prod(math.factorial(int(j)) for j in a)) for a in self.powers])
        phi=self.zbasis.transform(z)*self.zweights
        latent_penalty=np.full(phi.shape[1],self.ridge);latent_penalty[0]=0
        self.coefficient=np.linalg.solve(phi.T@phi/len(phi)+np.diag(latent_penalty),phi.T@profiles/len(phi))
        self.d=z.shape[1]
        self.info=dict(protocol='Generic E02 full latent polynomial x complete quadratic-plus-Gaussian stimulus dictionary',
            assumptions=['Smooth response surface','Complete latent polynomial truncation','Generic isotropic RBF interpolation in original standardized X'],
            model_identity_used=False,latent_degree=self.degree,latent_ridge=self.ridge,stimulus_ridge=self.stimulus_ridge,
            stimulus_features=q,latent_features=len(self.powers),centers=self.centers,length_scale=self.sigma,
            shared_stimulus_design_detected=shared,train_subjects=len(z),train_rows=sum(len(y) for y in y_tables),
            capabilities=['F0','F1','F2','F3'],warning='Prediction accuracy does not imply Hessian accuracy; fixed X designs require interpolation assumptions')
        self._tensors=tuple(torch.tensor(a) for a in [self.powers,self.zweights,self.xbasis.powers_,self.xweights,self.centers,self.xscale,self.coefficient])
        return self

    def predict_subject(self,z,x):
        phi=self.zbasis.transform(np.asarray(z)[None])*self.zweights
        return ((phi@self.coefficient)@(self.xfeatures(x)/self.xscale).T).ravel()

    def value(self,w):
        powers,zw,xpowers,xw,centers,xscale,coef=self._tensors
        z=w[:self.d];x=w[self.d:];zf=torch.ones(len(zw),dtype=w.dtype,device=w.device)
        for j in range(self.d):
            pp=[torch.ones_like(z[j])]
            for _ in range(self.degree):pp.append(pp[-1]*z[j])
            zf=zf*torch.stack(pp)[powers[:,j].long()]
        scaled=x/np.sqrt(len(x));xf=torch.ones(len(xw),dtype=w.dtype,device=w.device)
        for j in range(len(x)):
            pp=torch.stack((torch.ones_like(scaled[j]),scaled[j],scaled[j]*scaled[j]))
            xf=xf*pp[xpowers[:,j].long()]
        radial=torch.exp(-torch.sum((x[None]-centers)**2,dim=1)/(2*self.sigma**2))
        features=torch.cat((xf*xw,radial))/xscale
        return ((zf*zw)@coef)@features
