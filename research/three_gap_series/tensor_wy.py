"""Generic E02 smooth tensor regression from subject-grouped numerical WY.

Same dictionary in every dataset: complete degree-2 X polynomial plus
isotropic Gaussian RBFs, and a complete latent polynomial. No model names,
option structure, generating expressions, parameter labels or derivative data.
"""
import math
import itertools
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures


class TensorWY:
    def __init__(self,degree=3,ridge=1e-6,stimulus_ridge=1e-5,centers=64,length=1.,seed=20261127,kernel='gaussian',
                 latent_kernel='polynomial',latent_centers=256,latent_length=1.):
        self.degree=degree;self.ridge=ridge;self.stimulus_ridge=stimulus_ridge
        self.ncenters=centers;self.length=length;self.seed=seed;self.kernel=kernel
        if kernel not in ['gaussian','anova2','anova3']:raise ValueError(kernel)
        self.latent_kernel=latent_kernel;self.latent_centers=latent_centers;self.latent_length=latent_length

    def xfeatures(self,x):
        poly=self.xbasis.transform(x/np.sqrt(x.shape[1]))*self.xweights
        if self.kernel=='gaussian':
            sq=np.maximum(np.sum(x*x,1)[:,None]+np.sum(self.centers*self.centers,1)[None]-2*x@self.centers.T,0)
            rbf=np.exp(-sq/(2*self.sigma**2))
        else:
            radial=np.exp(-(x[:,None,:]-self.centers[None,:,:])**2/(2*self.sigma**2))
            terms=[]
            for order in range(1,min(int(self.kernel[-1]),x.shape[1])+1):
                terms.append(sum(np.prod(radial[:,:,list(ix)],axis=2) for ix in itertools.combinations(range(x.shape[1]),order))/math.comb(x.shape[1],order))
            rbf=sum(terms)/len(terms)
        return np.column_stack((poly,rbf))

    def fit(self,z,x_tables,y_tables):
        # Inputs are numeric arrays only. Shared X detection is a data property,
        # not a model-specific estimator branch.
        rng=np.random.default_rng(self.seed);pool=np.concatenate(x_tables)
        pick=rng.choice(len(pool),min(8192,len(pool)),replace=False);pool=pool[pick]
        unique_pool=np.unique(pool,axis=0)
        self.centers=unique_pool[rng.choice(len(unique_pool),min(self.ncenters,len(unique_pool)),replace=False)].copy()
        self.sigma=self.length*(np.sqrt(pool.shape[1]) if self.kernel=='gaussian' else 1.)
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
        self.zcenters=z[rng.choice(len(z),min(self.latent_centers,len(z)),replace=False)].copy() if self.latent_kernel=='gaussian' else np.empty((0,z.shape[1]))
        self.zsigma=self.latent_length*max(float(np.sqrt(np.trace(np.cov(z,rowvar=False,bias=True)))),1e-8)
        phi=self.zfeatures(z)
        latent_penalty=np.full(phi.shape[1],self.ridge)
        if self.latent_kernel=='gaussian':latent_penalty[:len(self.powers)]=1e-10
        latent_penalty[0]=0
        self.coefficient=np.linalg.solve(phi.T@phi/len(phi)+np.diag(latent_penalty),phi.T@profiles/len(phi))
        self.d=z.shape[1]
        self.info=dict(protocol='Generic E02 full latent polynomial x complete quadratic-plus-Gaussian stimulus dictionary',
            assumptions=['Smooth response surface','Complete latent polynomial truncation','Generic isotropic RBF interpolation in original standardized X'],
            model_identity_used=False,latent_degree=self.degree,latent_ridge=self.ridge,stimulus_ridge=self.stimulus_ridge,
            stimulus_features=q,latent_features=phi.shape[1],centers=self.centers,length_scale=self.sigma,
            center_rule='Uniform sample without replacement from distinct TRAIN X rows; no model condition groups',
            shared_stimulus_design_detected=shared,train_subjects=len(z),train_rows=sum(len(y) for y in y_tables),
            capabilities=['F0','F1','F2','F3'],warning='Prediction accuracy does not imply Hessian accuracy; fixed X designs require interpolation assumptions')
        self.info['stimulus_kernel']=self.kernel
        self.info['latent_kernel']=self.latent_kernel;self.info['latent_rbf_centers']=len(self.zcenters);self.info['latent_rbf_length']=self.zsigma
        self.info['kernel_interaction_rule']='All coordinate subsets up to order k, equal averaged weights; no named X groups' if self.kernel!='gaussian' else 'Isotropic Gaussian'
        self._tensors=tuple(torch.tensor(a) for a in [self.powers,self.zweights,self.xbasis.powers_,self.xweights,self.centers,self.xscale,self.coefficient,self.zcenters])
        return self

    def zfeatures(self,z):
        polynomial=self.zbasis.transform(z)*self.zweights
        if not len(self.zcenters):return polynomial
        radial=np.exp(-np.sum((z[:,None,:]-self.zcenters[None,:,:])**2,axis=2)/(2*self.zsigma**2))
        return np.column_stack((polynomial,radial))

    def predict_subject(self,z,x):
        phi=self.zfeatures(np.asarray(z)[None])
        return ((phi@self.coefficient)@(self.xfeatures(x)/self.xscale).T).ravel()

    def latent_fields(self,z,x,third=False):
        """Exact latent derivatives of the fitted numerical surrogate.

        Uses polynomial algebra, not teacher derivatives. This only changes
        computation, and is tested against autodiff of the same fitted value.
        """
        profiles=(self.xfeatures(x)/self.xscale)@self.coefficient.T
        n,d=z.shape
        def derivative(indices):
            powers=self.powers.copy();factor=self.zweights.copy()
            for j in indices:factor*=powers[:,j];powers[:,j]-=1
            valid=np.all(powers>=0,axis=1)&(factor!=0)
            value=np.ones((n,int(valid.sum())))
            for j in range(d):value*=z[:,j,None]**powers[None,valid,j]
            result=np.sum(value*factor[None,valid]*profiles[:,:len(self.powers)][:,valid],axis=1)
            if len(self.zcenters):
                delta=z[:,None,:]-self.zcenters[None,:,:];s2=self.zsigma**2
                radial=np.exp(-np.sum(delta*delta,axis=2)/(2*s2))
                if len(indices)==1:radial=radial*(-delta[:,:,indices[0]]/s2)
                elif len(indices)==2:
                    i,j=indices;radial=radial*(delta[:,:,i]*delta[:,:,j]/s2**2-(i==j)/s2)
                elif len(indices)==3:
                    i,j,k=indices;radial=radial*(-delta[:,:,i]*delta[:,:,j]*delta[:,:,k]/s2**3+
                        ((i==j)*delta[:,:,k]+(i==k)*delta[:,:,j]+(j==k)*delta[:,:,i])/s2**2)
                result+=np.sum(radial*profiles[:,len(self.powers):],axis=1)
            return result
        result=dict(value=derivative(()),g=np.empty((n,d)),h=np.empty((n,d,d)))
        for j in range(d):result['g'][:,j]=derivative((j,))
        for j,k in itertools.combinations_with_replacement(range(d),2):result['h'][:,j,k]=result['h'][:,k,j]=derivative((j,k))
        if third:
            result['third']=np.empty((n,d,d,d))
            for ijk in itertools.combinations_with_replacement(range(d),3):
                value=derivative(ijk)
                for perm in set(itertools.permutations(ijk)):result['third'][(slice(None),*perm)]=value
        return result

    def value(self,w):
        powers,zw,xpowers,xw,centers,xscale,coef,zcenters=self._tensors
        z=w[:self.d];x=w[self.d:];zf=torch.ones(len(zw),dtype=w.dtype,device=w.device)
        for j in range(self.d):
            pp=[torch.ones_like(z[j])]
            for _ in range(self.degree):pp.append(pp[-1]*z[j])
            zf=zf*torch.stack(pp)[powers[:,j].long()]
        scaled=x/np.sqrt(len(x));xf=torch.ones(len(xw),dtype=w.dtype,device=w.device)
        for j in range(len(x)):
            pp=torch.stack((torch.ones_like(scaled[j]),scaled[j],scaled[j]*scaled[j]))
            xf=xf*pp[xpowers[:,j].long()]
        if self.kernel=='gaussian':radial=torch.exp(-torch.sum((x[None]-centers)**2,dim=1)/(2*self.sigma**2))
        else:
            one=torch.exp(-(x[None]-centers)**2/(2*self.sigma**2));terms=[]
            for order in range(1,min(int(self.kernel[-1]),len(x))+1):
                terms.append(sum(torch.prod(one[:,list(ix)],dim=1) for ix in itertools.combinations(range(len(x)),order))/math.comb(len(x),order))
            radial=sum(terms)/len(terms)
        features=torch.cat((xf*xw,radial))/xscale
        zfeatures=zf*zw
        if len(zcenters):zfeatures=torch.cat((zfeatures,torch.exp(-torch.sum((z[None]-zcenters)**2,dim=1)/(2*self.zsigma**2))))
        return (zfeatures@coef)@features
