"""Fit a supervised binary-choice response-profile proxy from only frozen FIT ZXY.

The subject-specific response-profile is a ridge least-squares projection of
binary Y onto data-derived stimulus features. A cubic polynomial in whitened
latent q predicts those coefficients. No real theta or oracle formula enters.
This is NOT an exact estimate of logit P(Y|Z,X); fidelity is measured on held-out
FIT subjects. Derivatives refer to the fitted proxy only.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations_with_replacement
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _safe_ridge(A, B, ridge, penalize_intercept=False):
    A=np.asarray(A,dtype=float)
    B=np.asarray(B,dtype=float)
    gram=A.T@A
    reg=np.full(gram.shape[0],float(ridge))
    if not penalize_intercept: reg[0]=0.0
    gram.flat[::len(reg)+1]+=reg
    rhs=A.T@B
    try:
        return cho_solve(cho_factor(gram,check_finite=False),rhs,check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return np.linalg.lstsq(gram,rhs,rcond=1e-9)[0]


@dataclass
class StimulusMap:
    mean: np.ndarray
    std: np.ndarray
    features: list[tuple[int,...]]
    feat_mean: np.ndarray
    feat_std: np.ndarray

    def transform(self,x):
        x=np.asarray(x,float)
        v=np.clip((x-self.mean)/self.std,-8.0,8.0)
        out=[np.ones(v.shape[:-1])]
        for group in self.features:
            out.append(np.prod(v[...,list(group)],axis=-1))
        z=np.stack(out,axis=-1)
        z[...,1:]=(z[...,1:]-self.feat_mean)/self.feat_std
        return np.asarray(z,dtype=np.float32)


def fit_stimulus_map(x,mask,n_features=18):
    samples=np.asarray(x[mask],dtype=float)
    mean=samples.mean(axis=0)
    std=np.maximum(samples.std(axis=0),1e-6)
    d=x.shape[-1]
    singles=[(i,) for i in range(d)]
    pairs=list(combinations_with_replacement(range(d),2))
    all_terms=(singles+pairs)[:max(1,int(n_features)-1)]
    z=np.clip((samples-mean)/std,-8,8)
    matrix=np.stack([np.prod(z[:,list(g)],axis=-1) for g in all_terms],axis=1)
    fmean=matrix.mean(0)
    fstd=np.maximum(matrix.std(0),1e-6)
    return StimulusMap(mean,std,all_terms,fmean,fstd)


def subject_profiles(phi,y,mask,ridge=1.0):
    """Estimate Y~phi(X) separately for subjects, with input-only design regularization."""
    phi=np.asarray(phi,dtype=float)
    y=np.asarray(y,dtype=float)
    mask=np.asarray(mask,dtype=bool)
    N,T,L=phi.shape
    # Use batch matrix products, with padding excluded. Each subject has ≥1 trial.
    pm=phi*mask[...,None]
    gram=np.einsum('ntl,ntm->nlm',pm,pm,optimize=True)
    gram+=np.eye(L)[None,:,:]*float(ridge)
    # Regularize intercept only very weakly so choice-rate level remains measurable.
    gram[:,0,0]-=float(ridge)-1e-6
    rhs=np.einsum('ntl,nt->nl',pm,y,optimize=True)
    try:
        beta=np.linalg.solve(gram,rhs[...,None])[...,0]
    except np.linalg.LinAlgError:
        beta=np.stack([np.linalg.lstsq(gram[i],rhs[i],rcond=1e-8)[0] for i in range(N)])
    return beta


def monomials(k,degree):
    return [()] + [inds for n in range(1,int(degree)+1)
                   for inds in combinations_with_replacement(range(k),n)]


def poly_design(q,terms):
    q=np.asarray(q,float)
    cols=[np.ones(q.shape[0])]
    for inds in terms[1:]:
        cols.append(np.prod(q[:,list(inds)],axis=1))
    return np.stack(cols,axis=1)


@dataclass
class PolynomialProxy:
    coef: dict[int,np.ndarray]
    terms: list[tuple[int,...]]
    stimulus: StimulusMap
    h_train: np.ndarray
    q_train: np.ndarray
    phi_train: np.ndarray
    y_train: np.ndarray
    mask_train: np.ndarray
    fit_ids: np.ndarray
    validation_ids: np.ndarray
    fidelity: dict
    stimulus_ridge: float
    train_x: np.ndarray
    # Independent half-FIT tensor estimates for noise-calibrated module selection.
    split_coefficients: tuple[dict[int,np.ndarray],dict[int,np.ndarray]] | None = None

    @property
    def k(self): return self.coef[1].shape[0]
    @property
    def channels(self): return self.coef[1].shape[-1]

    def predict_profiles(self,q):
        q=np.asarray(q,dtype=float)
        out=np.broadcast_to(self.coef[0],(len(q),self.channels)).copy()
        out+=q@self.coef[1]
        if 2 in self.coef:out+=np.einsum('ni,nj,ijl->nl',q,q,self.coef[2],optimize=True)
        if 3 in self.coef:out+=np.einsum('ni,nj,nk,ijkl->nl',q,q,q,self.coef[3],optimize=True)
        return out

    def gradients(self,q):
        q=np.asarray(q,float)
        n=len(q)
        g=np.broadcast_to(self.coef[1],(n,*self.coef[1].shape)).copy()
        if 2 in self.coef:g+=2*np.einsum('nj,ijl->nil',q,self.coef[2],optimize=True)
        if 3 in self.coef:g+=3*np.einsum('nj,nk,ijkl->nil',q,q,self.coef[3],optimize=True)
        return g

    def hessians(self,q):
        q=np.asarray(q,float)
        out=np.zeros((len(q),self.k,self.k,self.channels))
        if 2 in self.coef:out+=2*self.coef[2][None,:,:,:]
        if 3 in self.coef:out+=6*np.einsum('nk,ijkl->nijl',q,self.coef[3],optimize=True)
        return out

    def cubic(self):return self.coef.get(3)


def make_proxy(q,x,y,mask,config):
    q=np.asarray(q,float);x=np.asarray(x,dtype=np.float32)
    y=np.asarray(y,dtype=np.float32);mask=np.asarray(mask,bool)
    N,K=q.shape
    if N<24 or K<2:raise ValueError('Need at least 24 FIT participants and >=2 latent axes')
    if not np.isfinite(q).all() or not np.all(np.isin(y[mask],[0,1])):
        raise ValueError('Nonfinite Z or nonbinary observed responses')
    rng=np.random.default_rng(int(config['seed']))
    shuffle=rng.permutation(N)
    n_fit=max(16,int(.8*N))
    a=shuffle[:n_fit];b=shuffle[n_fit:]
    stimulus=fit_stimulus_map(x[a],mask[a],config['stimulus_features'])
    # Compute only the number of allowed channels, preserving fitted stimulus standardization.
    phi=stimulus.transform(x)
    h=subject_profiles(phi,y,mask,config['stimulus_ridge'])
    degree=int(config['surrogate_degree'])
    if degree not in (2,3):raise ValueError('surrogate_degree must be 2 or 3')
    terms=monomials(K,degree)
    v=poly_design(q,terms)
    # Column normalization is fit once; afterward rotate the symmetric tensors exactly.
    rms=np.maximum(np.sqrt(np.mean(v[a]**2,axis=0)),1e-6)
    rms[0]=1.0
    vscaled=v/rms
    coeff_scaled=_safe_ridge(vscaled[a],h[a],config['surrogate_ridge'])
    coef_monomial=coeff_scaled/rms[:,None]
    coefs={0:coef_monomial[0],1:np.zeros((K,h.shape[1]))}
    if degree>=2:coefs[2]=np.zeros((K,K,h.shape[1]))
    if degree>=3:coefs[3]=np.zeros((K,K,K,h.shape[1]))
    for group,weight in zip(terms[1:],coef_monomial[1:]):
        n=len(group)
        from itertools import permutations
        positions=set(permutations(group))
        for pos in positions:coefs[n][pos]=weight/len(positions)
    predicted=v@coef_monomial
    err=((predicted[b]-h[b])**2).mean()
    baseline=((h[b]-h[a].mean(0))**2).mean()
    r2=float(1-err/max(baseline,1e-15))
    # Track frozen stimulus projection fidelity on held-out subjects.
    reconstruction=np.einsum('ntl,nl->nt',phi[b],predicted[b],optimize=True)
    clip=np.clip(reconstruction,1e-4,1-1e-4)
    binary_ce=float(-np.sum((y[b]*np.log(clip)+(1-y[b])*np.log(1-clip))*mask[b])/max(mask[b].sum(),1))
    baseline_p=float(np.mean(y[a][mask[a]]))
    naive_ce=float(-np.sum((y[b]*np.log(np.clip(baseline_p,1e-6,1-1e-6))+(1-y[b])*np.log(np.clip(1-baseline_p,1e-6,1-1e-6)))*mask[b])/max(mask[b].sum(),1))
    fidelity={'heldout_profile_r2':r2,'heldout_y_cross_entropy':binary_ce,
              'constant_y_cross_entropy':naive_ce,'profile_dimensions':h.shape[1],
              'polynomial_features':len(terms),'fit_subjects':len(a),
              'validation_subjects':len(b),'input_scale':'frozen whitened PCA q'}
    # Estimate coefficient uncertainty from two disjoint FIT-subject halves.
    # Keep the same stimulus map, polynomial basis and train-derived scaling.
    shuffled_fit=np.random.default_rng(int(config['seed'])+741).permutation(a)
    half_a,half_b=np.array_split(shuffled_fit,2)
    def half_tensor(ids):
        scaled=_safe_ridge(vscaled[ids],h[ids],config['surrogate_ridge'])
        raw=scaled/rms[:,None]
        tensors={0:raw[0].copy(),1:np.zeros_like(coefs[1])}
        if degree>=2:tensors[2]=np.zeros_like(coefs[2])
        if degree>=3:tensors[3]=np.zeros_like(coefs[3])
        from itertools import permutations
        for indices,row in zip(terms[1:],raw[1:]):
            positions=set(permutations(indices))
            for pos in positions:
                tensors[len(indices)][pos]=row/len(positions)
        return tensors
    halves=(half_tensor(half_a),half_tensor(half_b))
    fidelity['split_half_fit_subjects']=[int(len(half_a)),int(len(half_b))]
    fidelity['operator_uncertainty']='Independent FIT-half differences; heuristic scale, not an exact bootstrap confidence interval'
    return PolynomialProxy(coefs,terms,stimulus,h,q,phi,y,mask,a,b,fidelity,
                           config['stimulus_ridge'],x,halves)


def rotate_coefficients(coef,R):
    R=np.asarray(R,float)
    out={0:coef[0].copy(),1:np.einsum('ia,al->il',R,coef[1],optimize=True)}
    if 2 in coef:out[2]=np.einsum('ia,jb,abl->ijl',R,R,coef[2],optimize=True)
    if 3 in coef:out[3]=np.einsum('ia,jb,kc,abcl->ijkl',R,R,R,coef[3],optimize=True)
    return out