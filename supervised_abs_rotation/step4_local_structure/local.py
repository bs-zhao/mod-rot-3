"""Rotation-equivariant local response gradients from numerical (U,F) only.

Fit weighted local linear models of subject response profiles F to subject
coordinates U. Neighbor selection uses Euclidean distances, which are
invariant under the Step-2 orthogonal rotations. The entire local field can
therefore be fitted ONCE in an anonymous orientation and mapped to all other
orientations with an independently estimated orthogonal Procrustes transform.

Never reads theta, generative formulas, R_best, NN weights or case angle.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

EPS = 1.e-12


def orthogonal_map(ref, other, tolerance=2e-5):
    """Return H so U_other ~= U_ref @ H.T (row convention).

    SVD of cross-covariance is only a coordinate registration; it does not
    use outcomes, angle metadata, model parameters or original best R.
    """
    ref=np.asarray(ref,dtype=np.float64)
    other=np.asarray(other,dtype=np.float64)
    if ref.shape!=other.shape or ref.ndim!=2:
        raise ValueError("Pair of U matrices must have identical (N,d) shapes")
    rc=ref-ref.mean(0)
    oc=other-other.mean(0)
    uu,_,vt=np.linalg.svd(rc.T@oc,full_matrices=False)
    q=uu@vt                  # U_other ~= U_ref @ q
    reconstructed=rc@q
    rmse=float(np.sqrt(np.mean((reconstructed-oc)**2)))
    reference_scale=max(float(np.sqrt(np.mean(oc**2))),EPS)
    relative=rmse/reference_scale
    if relative>tolerance:
        raise ValueError(f"Candidate W is not an orthogonal transform: relative RMSE {relative:.4g}")
    return q.T, relative


def select_anchors(n, train, test, seed, n_anchors):
    if n_anchors<1:
        raise ValueError("At least one anchor required")
    rng=np.random.default_rng(seed)
    train=np.asarray(train,dtype=np.int64)
    test=np.asarray(test,dtype=np.int64)
    if len(np.intersect1d(train,test))!=0:
        raise ValueError("Anchor test subjects must not appear in reference TRAIN")
    if np.any(train<0) or np.any(test>=n):
        raise ValueError("Subject split out of bounds")
    return np.sort(rng.choice(test,size=min(n_anchors,len(test)),replace=False))


@dataclass
class LocalOperators:
    anchor_subject_ids: np.ndarray
    neighbor_subject_ids: np.ndarray   # (n_anchors,k)
    slope_operators: np.ndarray        # (n_anchors,d,k)
    predictions_weights: np.ndarray    # (n_anchors,k), for F predictions
    bandwidths: np.ndarray
    condition_numbers: np.ndarray
    ridge: float

    @property
    def n_anchors(self):
        return int(len(self.anchor_subject_ids))

    def apply(self, f):
        """Return B_local[anchor,latent_axis,response_feature], heldout MSE."""
        f=np.asarray(f,dtype=np.float64)
        if f.ndim!=2:
            raise ValueError("F must be subject-by-response features")
        if np.max(self.neighbor_subject_ids)>=len(f) or (
                np.max(self.anchor_subject_ids)>=len(f)):
            raise ValueError("Subject profile F does not match local operators")
        a,d,k=self.slope_operators.shape
        p=f.shape[1]
        b=np.empty((a,d,p),dtype=np.float64)
        prediction=np.empty((a,p),dtype=np.float64)
        for i in range(a):
            fn=f[self.neighbor_subject_ids[i]]
            b[i]=self.slope_operators[i]@fn
            prediction[i]=self.predictions_weights[i]@fn
        mse=float(np.mean((prediction-f[self.anchor_subject_ids])**2))
        return b,mse


def build_local_operators(u, train, anchors, *, neighbors=192, ridge_fraction=.025):
    """Precompute isotropic, weighted local slopes independent of F or Y.

    Neighbors lie in TRAIN only, anchors in TEST. Weighted ridge is relative
    to local covariance trace, hence invariant under orthogonal rotations.
    """
    u=np.asarray(u,dtype=np.float64)
    train=np.asarray(train,dtype=np.int64)
    anchors=np.asarray(anchors,dtype=np.int64)
    n,d=u.shape
    if u.ndim!=2 or not np.isfinite(u).all():
        raise ValueError("Expected finite subject coordinate U")
    if len(np.unique(train))!=len(train) or len(np.unique(anchors))!=len(anchors):
        raise ValueError("Repeated subject IDs in local fit")
    if len(np.intersect1d(train,anchors))!=0 or len(train)<=neighbors:
        raise ValueError("Local fit requires disjoint TRAIN anchors and > k TRAIN subjects")
    if neighbors<=d+2 or ridge_fraction<=0:
        raise ValueError("k must exceed d+2, ridge_fraction must be positive")

    ids=np.empty((len(anchors),neighbors),dtype=np.int64)
    slopes=np.empty((len(anchors),d,neighbors),dtype=np.float64)
    intercept=np.empty((len(anchors),neighbors),dtype=np.float64)
    widths=np.empty(len(anchors),dtype=np.float64)
    conds=np.empty(len(anchors),dtype=np.float64)
    utr=u[train]
    for t,anchor_id in enumerate(anchors):
        dist_sq=np.sum((utr-u[anchor_id])**2,axis=1)
        kth=np.argpartition(dist_sq,neighbors-1)[:neighbors]
        nearest=kth[np.argsort(dist_sq[kth],kind="stable")]
        ix=train[nearest]
        radius=max(float(np.sqrt(dist_sq[nearest[-1]])),1e-9)
        distance=np.sqrt(np.maximum(dist_sq[nearest],0.))
        weight=np.exp(-.5*(distance/radius)**2)
        weight/=weight.sum()
        xn=u[ix]
        xmean=weight@xn
        xc=xn-xmean
        gram=xc.T@(weight[:,None]*xc)
        scale=max(float(np.trace(gram))/d,1e-10)
        mat=gram+ridge_fraction*scale*np.eye(d)
        conds[t]=float(np.linalg.cond(mat))
        # Slope = (Xc'WX + alpha I)^-1 Xc'W Y_near.
        l=np.linalg.solve(mat,xc.T*weight[None,:])
        # Local prediction evaluated AT the anchor position:
        # y_hat = weighted_Ymean + (u_anchor-xmean)' * slope.
        pred_weights=weight+(u[anchor_id]-xmean)@l
        ids[t]=ix
        slopes[t]=l
        intercept[t]=pred_weights
        widths[t]=radius
        if not np.isfinite(l).all() or not np.isfinite(pred_weights).all():
            raise ArithmeticError("Nonfinite local fit")
    return LocalOperators(anchors,ids,slopes,intercept,widths,conds,ridge_fraction)


def global_fingerprint(u, f, train, ridge_fraction=.025):
    """One rotation-equivariant isotropic global linear response map B."""
    u=np.asarray(u,float)
    f=np.asarray(f,float)
    x=u[train]
    y=f[train]
    xc=x-x.mean(0)
    yc=y-y.mean(0)
    d=u.shape[1]
    gram=xc.T@xc/len(train)
    scl=max(float(np.trace(gram))/d,1e-10)
    reg=gram+ridge_fraction*scl*np.eye(d)
    b=np.linalg.solve(reg,xc.T@yc/len(train))
    pred=(u[test_dummy:=train]-x.mean(0))@b+y.mean(0)
    return b,float(np.mean((pred-y)**2))


def _entropy(p):
    return -np.sum(p*np.log(np.maximum(p,EPS)),axis=-1)


def scores(b, *, include_variation=True):
    """Structural losses of a field of local B matrices (A,d,P).

    Each loss is invariant to row sign flips and coordinate permutations,
    but can vary under general mixing of axes. All scores are smaller-is-
    simpler except controls, which must be rotation invariant.
    """
    b=np.asarray(b,dtype=np.float64)
    if b.ndim!=3 or not np.isfinite(b).all() or b.shape[0]<1:
        raise ValueError("Local B must have shape (anchors,d,features)")
    a,d,p=b.shape
    if d<2 or p<1:
        raise ValueError("At least 2 latent axes and 1 profile feature required")
    energy=b*b
    total=np.sum(energy,axis=1)          # (anchors,p), invariant under rotation
    nonzero=total[total>1e-20]
    if nonzero.size==0:
        raise ValueError("All local response gradients are zero")
    threshold=max(1e-18,float(np.median(nonzero))*1e-5)
    good=total>threshold
    if not np.any(good):
        raise ValueError("No local response-gradient signal")
    mass=energy/np.maximum(total[:,None,:],EPS)      # (anchors,d,p)
    q=np.moveaxis(mass,1,-1)[good]                  # (#valid,d)
    point_l1=np.sqrt(q).sum(-1)
    result={
        "local_point_entropy":float(np.mean(_entropy(q)/np.log(d))),
        "local_point_tail1":float(np.mean(1-np.max(q,axis=1))),
        "local_point_tail2":float(np.mean(1-np.sort(q,axis=1)[:,-2:].sum(axis=1))),
        "local_point_hoyer":float(np.mean((point_l1-1)/(np.sqrt(d)-1))),
        "local_point_l1":float(np.mean(point_l1)),
    }

    # Energy or absolute local slopes first aggregated OVER anchor positions,
    # then look for sparsity ACROSS latent axes within each response feature.
    for name,field in (("rms",np.mean(energy,axis=0)),
                       ("meanabs",np.mean(np.abs(b),axis=0)**2)):
        denom=field.sum(axis=0)
        mask=denom>threshold
        if not mask.any():
            result[f"local_{name}_entropy"]=float("nan")
            result[f"local_{name}_tail1"]=float("nan")
            result[f"local_{name}_hoyer"]=float("nan")
            continue
        probs=(field[:,mask]/denom[mask][None,:]).T
        result[f"local_{name}_entropy"]=float(np.mean(_entropy(probs)/np.log(d)))
        result[f"local_{name}_tail1"]=float(np.mean(1-probs.max(axis=1)))
        result[f"local_{name}_hoyer"]=float(np.mean(
            (np.sqrt(probs).sum(axis=1)-1)/(np.sqrt(d)-1)))

    # Rows of the gradient field can be weak or align at the same stimuli.
    # Gradient gram is a PSD matrix; its eigenvalues are invariant, but
    # off-diagonal mass depends on the chosen coordinate system.
    gram=np.einsum("adp,aep->de",b,b,optimize=True)
    denom=max(float(np.sum(gram*gram)),EPS)
    off=gram-np.diag(np.diag(gram))
    result["local_gram_offdiag_fraction"]=float(np.sum(off*off)/denom)
    group=np.sqrt(np.sum(b*b,axis=(0,2)))
    result["local_group_l1_over_l2"]=float(group.sum()/max(float(np.linalg.norm(group)),EPS))
    result["control_local_gradient_fro"]=float(np.linalg.norm(b))
    result["control_local_gradient_gram_spectrum"]=float(np.linalg.norm(np.linalg.eigvalsh(gram)))
    if include_variation and a>1:
        variation=b-b.mean(axis=0,keepdims=True)
        spread=np.sum(variation*variation,axis=(1,2)).sum()
        result["control_local_variation_fraction"]=float(
            spread/max(float(np.sum(b*b)),EPS))
        # The nonlinear part of the derivative field: variation of each
        # stimulus's gradient across the subjects, after removing the mean.
        extras=scores(variation,include_variation=False)
        for k in ("local_point_entropy","local_point_tail1","local_point_tail2",
                  "local_rms_entropy","local_rms_tail1",
                  "local_meanabs_entropy","local_meanabs_tail1",
                  "local_gram_offdiag_fraction"):
            result[f"nonlinear_{k}"]=extras[k]
    if not all(np.isfinite(x) for x in result.values()):
        raise ArithmeticError("Nonfinite structural loss")
    return result
