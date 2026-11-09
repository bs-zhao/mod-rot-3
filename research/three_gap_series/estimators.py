"""M2: finite original-table WY polynomial / Bernoulli sieve, no formula access."""
import itertools
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures

class PolynomialSieve:
    """Total-degree polynomial with rotationally invariant coefficient norm.

    Degree-k monomial weight sqrt(k! / product(alpha_i!)) makes its kernel
    (w dot w')**k. Therefore latent rotation does not change the fitted model
    class or penalty. No per-coordinate latent scaling is applied.
    """
    def __init__(self, degree=3, ridge=.001, channel='logit'):
        self.degree=degree; self.ridge=ridge; self.channel=channel

    def fit(self,w,y,valw,valy):
        self.basis=PolynomialFeatures(self.degree).fit(w[:1])
        self.powers=self.basis.powers_
        self.weights=np.array([math.sqrt(math.factorial(int(a.sum()))/math.prod(math.factorial(int(k)) for k in a)) for a in self.powers])
        design=self.basis.transform(w)*self.weights
        vd=self.basis.transform(valw)*self.weights
        n=len(w); penalty=np.ones(design.shape[1])*self.ridge;penalty[0]=0
        if self.channel=='binary':
            def objective(b):
                eta=design@b
                loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty*b*b)
                grad=design.T@(expit(eta)-y)/n+penalty*b
                return loss,grad
            init=np.zeros(design.shape[1])
            fitted=minimize(objective,init,method='L-BFGS-B',jac=True,options=dict(maxiter=500,gtol=1e-7,ftol=1e-11))
            b=fitted.x
            self.info=dict(converged=bool(fitted.success),message=str(fitted.message),iterations=fitted.nit,
                           validation_nll=float(np.mean(np.logaddexp(0,vd@b)-valy*(vd@b))))
        else:
            # For p, logit is a declared link; derivatives below target p, not eta.
            target=y if self.channel=='logit' else np.log(np.clip(y,1e-8,1-1e-8)/np.clip(1-y,1e-8,1-1e-8))
            b=np.linalg.solve(design.T@design/n+np.diag(penalty),design.T@target/n)
            pred=vd@b if self.channel=='logit' else expit(vd@b)
            self.info=dict(converged=True,validation_rmse=float(np.sqrt(np.mean((pred-valy)**2))))
        self.coef=b*self.weights
        self.info.update(degree=self.degree,ridge=self.ridge,features=len(b),train_rows=n,validation_rows=len(valw),
                         capabilities=['F0','F1','F2','F3'],protocol='original-table WY only; isotropic total-degree basis')
        return self

    def raw_derivative(self,w,indices):
        powers=self.powers.copy(); factor=np.ones(len(powers))
        for i in indices:
            factor*=powers[:,i];powers[:,i]-=1
        valid=np.all(powers>=0,axis=1)&(factor!=0)
        a=powers[valid]; coeff=self.coef[valid]*factor[valid].reshape((-1,)+(1,)*(self.coef.ndim-1))
        values=np.ones((len(w),len(a)))
        for i in range(w.shape[1]): values*=w[:,i,None]**a[None,:,i]
        return values@coeff

    def fields(self,w,third=False,structure_link='native'):
        n,p=w.shape
        eta=self.raw_derivative(w,());g=np.empty((n,p));h=np.empty((n,p,p))
        for i in range(p): g[:,i]=self.raw_derivative(w,(i,))
        for i,j in itertools.combinations_with_replacement(range(p),2):
            h[:,i,j]=h[:,j,i]=self.raw_derivative(w,(i,j))
        out=dict(g=g,h=h)
        if third:
            t=np.empty((n,p,p,p))
            for ijk in itertools.combinations_with_replacement(range(p),3):
                v=self.raw_derivative(w,ijk)
                for idx in set(itertools.permutations(ijk)): t[(slice(None),*idx)]=v
            out['third']=t
        if self.channel!='logit' and structure_link=='native':
            pr=expit(eta);s1=pr*(1-pr);s2=s1*(1-2*pr);s3=s1*(1-6*pr+6*pr*pr)
            if third:
                out['third']=(s1[:,None,None,None]*out['third']+
                    s2[:,None,None,None]*(h[:,:,:,None]*g[:,None,None,:]+h[:,:,None,:]*g[:,None,:,None]+h[:,None,:,:]*g[:,:,None,None])+
                    s3[:,None,None,None]*g[:,:,None,None]*g[:,None,:,None]*g[:,None,None,:])
            out['h']=s1[:,None,None]*h+s2[:,None,None]*g[:,:,None]*g[:,None,:]
            out['g']=s1[:,None]*g
        return out

class BankSieve(PolynomialSieve):
    """E02: latent polynomial x discrete-stimulus one-hot. No X derivatives."""
    def fit(self,w,y,valw,valy):
        self.basis=PolynomialFeatures(self.degree).fit(w[:1]);self.powers=self.basis.powers_
        self.weights=np.array([math.sqrt(math.factorial(int(a.sum()))/math.prod(math.factorial(int(k)) for k in a)) for a in self.powers])
        design=self.basis.transform(w)*self.weights;vd=self.basis.transform(valw)*self.weights
        n=len(w);penalty=np.ones(design.shape[1])*self.ridge;penalty[0]=0
        if self.channel=='binary':
            shape=(design.shape[1],y.shape[1])
            def objective(flat):
                b=flat.reshape(shape);eta=design@b
                loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty[:,None]*b*b)/y.shape[1]
                grad=design.T@(expit(eta)-y)/(n*y.shape[1])+penalty[:,None]*b/y.shape[1]
                return loss,grad.ravel()
            fit=minimize(objective,np.zeros(np.prod(shape)),method='L-BFGS-B',jac=True,options=dict(maxiter=500,gtol=1e-7,ftol=1e-10))
            b=fit.x.reshape(shape)
            self.info=dict(converged=bool(fit.success),iterations=fit.nit,validation_nll=float(np.mean(np.logaddexp(0,vd@b)-valy*(vd@b))))
        else:
            target=y if self.channel=='logit' else np.log(np.clip(y,1e-8,1-1e-8)/np.clip(1-y,1e-8,1-1e-8))
            b=np.linalg.solve(design.T@design/n+np.diag(penalty),design.T@target/n)
            pred=vd@b if self.channel=='logit' else expit(vd@b)
            self.info=dict(converged=True,validation_rmse=float(np.sqrt(np.mean((pred-valy)**2))))
        self.coef=b*self.weights[:,None]
        self.info.update(degree=self.degree,ridge=self.ridge,features=len(b),train_subjects=n,
                         validation_subjects=len(valw),stimuli=y.shape[1],capabilities=['F0','F1_z','F2_zz','F3_zzz'],
                         protocol='original-table latent polynomial x fixed-stimulus indicators; no continuous X derivatives')
        return self

    def raw_derivative(self,w,indices):
        values=super().raw_derivative(w,indices)
        return values[np.arange(len(w)),self.eval_stimulus]

    def mean_fields(self,w):
        """A01 over subjects FIRST, then graph/curvature energies over stimuli.

        A different M1 estimand from E_subject,stimulus[g g.T].
        Canonical logit structure only; the outer sigmoid is not averaged away.
        """
        p=w.shape[1];stimuli=self.coef.shape[1]
        g=np.empty((stimuli,p));h=np.empty((stimuli,p,p))
        for i in range(p): g[:,i]=super().raw_derivative(w,(i,)).mean(0)
        for i,j in itertools.combinations_with_replacement(range(p),2):
            values=super().raw_derivative(w,(i,j)).mean(0)
            h[:,i,j]=h[:,j,i]=values
        return dict(g=g,h=h)

    def grid_fields(self,w):
        p=w.shape[1];values=super().raw_derivative(w,())
        g=np.stack([super(BankSieve,self).raw_derivative(w,(i,)) for i in range(p)],axis=-1)
        h=np.empty((*values.shape,p,p))
        for i,j in itertools.combinations_with_replacement(range(p),2):
            component=super().raw_derivative(w,(i,j));h[...,i,j]=h[...,j,i]=component
        return dict(value=values,g=g,h=h)
