"""Three independent T,Y-only Hessian estimators.

No CR3 oracle, original formula, or reference data imports in this file.
1. Nyström-approximated Gaussian Kernel Ridge Regression (KRR).
2. Penalized tensor-product cubic B-splines.
3. Second-order integration-by-parts (weighted Stein identity).

All input columns are the four original true-T coordinates. For methods 1
and 2, Y is the average of the 500 fixed stimulus responses and the
estimated object is the Hessian of E[average_Y|T].
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import BSpline
from scipy.special import expit

BOUNDS = np.array([[0.,1.],[.3,5.],[0.,1.],[.5,4.]], dtype=float)


def _rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def split_subjects(n, seed=20261106):
    if n < 40: raise ValueError('Need at least 40 independent subjects')
    order=np.random.default_rng(seed).permutation(n)
    n_train=int(n*.7); n_valid=int(n*.15)
    return dict(train=order[:n_train], val=order[n_train:n_train+n_valid],
                test=order[n_train+n_valid:])


@dataclass
class KRRModel:
    landmarks: np.ndarray
    center: np.ndarray
    sd: np.ndarray
    lengthscale: float
    alpha: np.ndarray  # coefficients against the unwhitened RBF basis

    def _kernel_and_deriv(self, T):
        z=(np.asarray(T,dtype=float)-self.center)/self.sd
        l=(self.landmarks-self.center)/self.sd
        d=z[:,None,:]-l[None,:,:]
        k=np.exp(-np.sum(d**2,axis=-1)/(2*self.lengthscale**2))
        deriv=-d/(self.lengthscale**2*self.sd[None,None,:])
        return k,deriv

    def predict(self,T):
        return self._kernel_and_deriv(T)[0]@self.alpha

    def hessian(self,T,batch=300):
        T=np.asarray(T,float)
        output=np.empty((len(T),4,4))
        diag_correction=1./(self.lengthscale**2*self.sd**2)
        for start in range(0,len(T),batch):
            stop=min(start+batch,len(T))
            k,d=self._kernel_and_deriv(T[start:stop])
            weighted=k*self.alpha[None,:]
            H=np.einsum('nm,nmi,nmj->nij',weighted,d,d,optimize=True)
            diagonal_weight=weighted.sum(axis=1)
            for j in range(4):H[:,j,j]-=diagonal_weight*diag_correction[j]
            output[start:stop]=(H+H.transpose(0,2,1))*.5
        return output


def fit_krr(T,channels,split,rank=256,lengthscales=(.5,1.,2.),lambdas=(1e-6,1e-4,1e-2),seed=13):
    """Fit Nyström KRR with subject-level validation; never sees analytic truth."""
    T=np.asarray(T,float)
    train,val=split['train'],split['val']
    Y=np.column_stack([channels[k] for k in channels])
    names=list(channels)
    ctr=T[train].mean(0);sd=T[train].std(0,ddof=1)
    sd=np.maximum(sd,1e-8)
    rng=np.random.default_rng(seed)
    landmarks=T[rng.choice(train,min(rank,len(train)),replace=False)].copy()
    lt=(landmarks-ctr)/sd
    scores={key:float('inf') for key in names}
    models={}
    selected={}
    for ell in lengthscales:
        def rbf(A,B):
            a=(A-ctr)/sd;b=(B-ctr)/sd
            r=np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
            return np.exp(-r/(2*ell**2))
        Kmm=rbf(landmarks,landmarks)
        eigvals, vectors=np.linalg.eigh(Kmm)
        eigvals=np.maximum(eigvals,0.)
        # Nyström inverse square root; tiny eigenvalues truncated to
        # prevent artificial high-derivative modes.
        floor=max(1e-9,float(eigvals.max())*1e-8)
        invroot=(vectors*(1/np.sqrt(np.maximum(eigvals,floor)))[None,:])@vectors.T
        Phi_train=rbf(T[train],landmarks)@invroot
        Phi_val=rbf(T[val],landmarks)@invroot
        gram=Phi_train.T@Phi_train/len(train)
        rhs=Phi_train.T@Y[train]/len(train)
        ev,vec=np.linalg.eigh(gram)
        ev=np.maximum(ev,0.)
        rhs_proj=vec.T@rhs
        for reg in lambdas:
            b=vec@(rhs_proj/(ev[:,None]+reg))
            prediction=Phi_val@b
            for j,name in enumerate(names):
                err=_rmse(prediction[:,j],Y[val,j])
                if err < scores[name]:
                    scores[name]=err
                    models[name]=KRRModel(landmarks,ctr,sd,ell,invroot@b[:,j])
                    selected[name]=dict(lengthscale=ell,ridge=reg,
                                        nystrom_landmarks=len(landmarks),validation_rmse=err)
    return models,selected


class SplineDesign:
    def __init__(self,n_basis=6,degree=3,bounds=BOUNDS):
        if n_basis < degree+1:raise ValueError('n_basis too small')
        self.n_basis=n_basis
        self.degree=degree
        self.bounds=np.asarray(bounds,float)
        self.splines=[]
        for lo,hi in self.bounds:
            grid=np.linspace(lo,hi,n_basis-degree+1)
            knots=np.r_[np.full(degree+1,lo),grid[1:-1],np.full(degree+1,hi)]
            funcs=[]
            for col in np.eye(n_basis):
                fun=BSpline(knots,col,degree,extrapolate=True)
                funcs.append((fun,fun.derivative(1),fun.derivative(2)))
            self.splines.append(funcs)

    def features(self,T,derivative=()):
        T=np.asarray(T,float)
        counters=np.bincount(derivative,minlength=4)
        factors=[]
        for dim in range(4):
            factors.append(np.column_stack([f[int(counters[dim])](T[:,dim])
                                            for f in self.splines[dim]]))
        # Cartesian tensor basis ordering: last coordinate fastest.
        out=np.einsum('na,nb,nc,nd->nabcd',*factors,optimize=True)
        return out.reshape(len(T),-1)

    def roughness(self):
        n=self.n_basis
        D=np.diff(np.eye(n),n=2,axis=0)
        R=D.T@D
        P=np.zeros((n**4,n**4))
        for dim in range(4):
            pieces=[np.eye(n) for _ in range(4)]
            pieces[dim]=R
            term=pieces[0]
            for piece in pieces[1:]:term=np.kron(term,piece)
            P+=term
        return P


@dataclass
class SplineModel:
    design: SplineDesign
    coefficients: np.ndarray

    def predict(self,T):
        return self.design.features(T)@self.coefficients

    def hessian(self,T,batch=96):
        T=np.asarray(T,float);H=np.empty((len(T),4,4))
        for start in range(0,len(T),batch):
            stop=min(start+batch,len(T));x=T[start:stop]
            for i in range(4):
                for j in range(i,4):
                    v=self.design.features(x,(i,j))@self.coefficients
                    H[start:stop,i,j]=v
                    H[start:stop,j,i]=v
        return H


def fit_spline(T,channels,split,n_basis=6,lambdas=(1e-7,1e-5,1e-3)):
    T=np.asarray(T,float)
    design=SplineDesign(n_basis=n_basis)
    train,val=split['train'],split['val']
    Y=np.column_stack([channels[k] for k in channels]);names=list(channels)
    Phi_train=design.features(T[train]);Phi_val=design.features(T[val])
    gram=Phi_train.T@Phi_train/len(train)
    rhs=Phi_train.T@Y[train]/len(train)
    penalty=design.roughness()
    best={k:float('inf') for k in names};models={};selected={}
    for lam in lambdas:
        M=gram+lam*penalty
        M.flat[::len(M)+1]+=1e-9
        factor=cho_factor(M,lower=True,check_finite=False)
        coeff=cho_solve(factor,rhs,check_finite=False)
        pred=Phi_val@coeff
        for j,name in enumerate(names):
            err=_rmse(pred[:,j],Y[val,j])
            if err < best[name]:
                best[name]=err
                models[name]=SplineModel(design,coeff[:,j].copy())
                selected[name]=dict(n_basis_per_dimension=n_basis,
                                    number_of_features=n_basis**4,
                                    smoothing_penalty=lam,validation_rmse=err)
    return models,selected


def second_order_score(T):
    """Weighted Stein II estimator for known independent bounded priors.

    Given f(T)=E[Y|T], for w=prod_k 16z_k²(1-z_k)²,
    E[w(T)∂ij f(T)]=E[f(T)*∂ij(w(T)π(T))/π(T)].
    Our w and first derivatives vanish at every boundary, so both
    integration-by-parts boundary terms vanish. The target is WEIGHTED,
    not the original unweighted E[H].
    """
    T=np.asarray(T,float)
    if T.ndim!=2 or T.shape[1]!=4:raise ValueError('T must be (N,4)')
    lo=BOUNDS[:,0];width=BOUNDS[:,1]-lo
    z=(T-lo)/width
    if np.any((z<=0)|(z>=1)):
        raise ValueError('All T values must lie strictly inside prior supports')
    v=16*z**2*(1-z)**2
    vp=32*z*(1-z)*(1-2*z)/width
    vpp=32*(1-6*z+6*z**2)/width**2
    score=np.empty_like(T);scoreprime=np.empty_like(T)
    for dim, (a,b) in ((0,(12.,3.)),(2,(9.,4.))):
        score[:,dim]=(a-1)/T[:,dim]-(b-1)/(1-T[:,dim])
        scoreprime[:,dim]=-(a-1)/T[:,dim]**2-(b-1)/(1-T[:,dim])**2
    score[:,1]=8/T[:,1]-1/.22
    scoreprime[:,1]=-8/T[:,1]**2
    score[:,3]=-(T[:,3]-2.2)/(.55**2)
    scoreprime[:,3]=-1/(.55**2)
    W=np.prod(v,axis=1)
    W1=np.empty((len(T),4));W2=np.empty((len(T),4,4))
    for i in range(4):
        other=np.prod(v[:,[k for k in range(4) if k!=i]],axis=1)
        W1[:,i]=vp[:,i]*other
        W2[:,i,i]=vpp[:,i]*other
        for j in range(i):
            others=np.prod(v[:,[k for k in range(4) if k!=i and k!=j]],axis=1)
            W2[:,i,j]=W2[:,j,i]=vp[:,i]*vp[:,j]*others
    psi=W2+W1[:,:,None]*score[:,None,:]+score[:,:,None]*W1[:,None,:]
    psi+=W[:,None,None]*score[:,:,None]*score[:,None,:]
    for i in range(4):psi[:,i,i]+=W*scoreprime[:,i]
    return W,psi


def fit_stein_mean(T,channel,weights=None):
    """Direct weighted average Hessian, with delta-method subject-cluster SE."""
    T=np.asarray(T,float)
    y=np.asarray(channel,float)
    if y.shape!=(len(T),):raise ValueError('Y must be one value per subject')
    W,psi=second_order_score(T)
    if weights is None:weights=np.full(len(T),1/len(T))
    weights=np.asarray(weights,float); weights=weights/weights.sum()
    mean_w=float(weights@W)
    numerator=np.einsum('n,n,nij->ij',weights,y,psi)
    mean=numerator/mean_w
    # SE valid for IID subjects and equal weights only; quadrature weights
    # are deterministic and do not have sampling SE.
    se=None
    if np.allclose(weights,np.full(len(T),1/len(T))):
        IF=(y[:,None,None]*psi-W[:,None,None]*mean[None,:,:])/mean_w
        se=IF.std(axis=0,ddof=1)/np.sqrt(len(T))
    return mean,se,mean_w
