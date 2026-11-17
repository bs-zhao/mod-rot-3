"""Independent numerical estimators for frozen NN derivatives.

Every value-only method queries the same NN h(u,x); it cannot access analytic
truth. GP/KRR/MLP surrogates are trained on sampled *training* latent inputs,
never on evaluation subjects or true derivative labels. Exact AD is a reference
for the NN itself, not ground truth for the generative function.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
from itertools import product
import math
import numpy as np
import torch
from torch.func import jacfwd, jacrev, vmap

from .derivatives import derivative_function, block_shape


@lru_cache(None)
def stencil_weights(order, radius, degree, scheme):
    """Derivative of an interpolating or LS polynomial at the origin.

    FD uses degree=2*radius; local_poly uses an overdetermined degree.
    Chebyshev uses Chebyshev nodes/weighted interpolation on [-radius,radius].
    """
    if degree < order: raise ValueError("degree below derivative order")
    if scheme == "chebyshev":
        n=2*radius+1
        nodes=radius*np.cos((np.arange(n)+0.5)*np.pi/n)
        mat=np.polynomial.chebyshev.chebvander(nodes/radius,degree)
        deriv=np.polynomial.chebyshev.Chebyshev.basis(order).deriv(order)(0) if False else None
        # actual derivative of every basis T_k at 0
        rhs=np.array([np.polynomial.chebyshev.Chebyshev.basis(k).deriv(order)(0)/
                      radius**order for k in range(degree+1)])
        weights=np.linalg.lstsq(mat.T,rhs,rcond=1e-12)[0]
        return tuple(float(v) for v in nodes),tuple(float(v) for v in weights)
    nodes=np.arange(-radius,radius+1,dtype=np.float64)
    mat=np.stack([nodes**k for k in range(degree+1)],axis=1)
    rhs=np.array([math.factorial(order) if k==order else 0. for k in range(degree+1)])
    weights=np.linalg.lstsq(mat.T,rhs,rcond=1e-12)[0]
    return tuple(float(v) for v in nodes),tuple(float(v) for v in weights)


def _unique_tuples(block,d,p):
    """Distinct mixed multi-indices, grouped by symmetric same-letter axes."""
    inds=product(*[range(d if c=="U" else p) for c in block])
    for idx in inds:
        if all(idx[i]<=idx[j] for i in range(len(block)) for j in range(i+1,len(block)) if block[i]==block[j]):
            yield idx


def _all_equal_permutations(block,idx):
    """Equivalent indices under permutation of like-type derivative axes."""
    groups={c:[i for i,t in enumerate(block) if t==c] for c in "UX"}
    import itertools
    choices=[]
    for c in "UX":
        axes=groups[c]
        choices.append((axes,list(set(itertools.permutations([idx[i] for i in axes])))))
    out=[]
    for left,right in product(choices[0][1],choices[1][1]):
        copied=list(idx)
        for ax,value in zip(choices[0][0],left):copied[ax]=value
        for ax,value in zip(choices[1][0],right):copied[ax]=value
        out.append(tuple(copied))
    return out


def value_difference(h, block, d, p, step=.03, method="fd2"):
    """Pointwise tensor from NN-only function value queries.

    fd2, fd4, richardson, local_poly, spectral_chebyshev share a differentiator
    using separable interpolation weights. High dimension is expensive by
    design; caller records actual compute costs.
    """
    mode={"fd2":(2,4,"fd"),"fd4":(3,6,"fd"),"richardson":(2,4,"fd"),
          "local_poly":(3,4,"fd"),"spectral_chebyshev":(3,6,"chebyshev")}[method]
    radius, degree, scheme=mode
    indices=tuple(_unique_tuples(block,d,p))
    sizes=block_shape(block,d,p)
    def one_pass(u,x,stepscale=1.):
        output=torch.zeros(sizes,device=u.device,dtype=u.dtype)
        for idx in indices:
            counts=Counter((c,j) for c,j in zip(block,idx))
            groups=[]
            for (kind,j),order in counts.items():
                nodes,weights=stencil_weights(order,radius,degree,scheme)
                groups.append((kind,j,[(node*step*stepscale, weight/(step*stepscale)**order)
                                      for node,weight in zip(nodes,weights)]))
            total=torch.zeros((),device=u.device,dtype=u.dtype)
            for combo in product(*[grp[2] for grp in groups]):
                U=u.clone(); X=x.clone(); weight=1.
                for (kind,j,_),(delta,wgt) in zip(groups,combo):
                    if kind=="U": U=U+torch.nn.functional.one_hot(torch.tensor(j,device=u.device),num_classes=d).to(U)*delta
                    else: X=X+torch.nn.functional.one_hot(torch.tensor(j,device=x.device),num_classes=p).to(X)*delta
                    weight*=wgt
                total=total+weight*h(U,X)
            for equivalent in _all_equal_permutations(block,idx):
                output[equivalent]=total
        return output
    if method!="richardson": return lambda u,x:one_pass(u,x)
    # Correct leading O(h^4) error for radius2 degree4 via h and h/2.
    return lambda u,x:(16*one_pass(u,x,.5)-one_pass(u,x))/15


def hybrid_ad_fd(h,block,d,p,step=.02):
    """AD of first k-1 axes; central difference on the last axis."""
    if len(block)==1:
        return value_difference(h,block,d,p,step=step,method="fd2")
    prefix=derivative_function(h,block[:-1],"mixed")
    last=block[-1]
    m=d if last=="U" else p
    def fn(u,x):
        out=[]
        for j in range(m):
            if last=="U":
                shift=torch.nn.functional.one_hot(torch.tensor(j,device=u.device),num_classes=d).to(u)*step
                y=(prefix(u+shift,x)-prefix(u-shift,x))/(2*step)
            else:
                shift=torch.nn.functional.one_hot(torch.tensor(j,device=x.device),num_classes=p).to(x)*step
                y=(prefix(u,x+shift)-prefix(u,x-shift))/(2*step)
            out.append(y)
        return torch.stack(out,dim=-1)
    return fn


def gaussian_smoothed_function(h,d,p,sigma=.05,n_samples=16,seed=17,
                               axes="U", device="cpu",dtype=torch.float64):
    """Antithetic Gaussian convolution. CR3 should normally smooth U only."""
    rng=np.random.default_rng(seed)
    delta=rng.normal(size=(max(n_samples//2,1),d+p))
    delta=np.concatenate((delta,-delta),axis=0)
    if axes=="U": delta[:,d:]=0.
    if axes=="X": delta[:,:d]=0.
    offsets=torch.as_tensor(delta*sigma,device=device,dtype=dtype)
    def h_smooth(u,x):
        all_w=torch.cat((u,x))
        return vmap(lambda e:h((all_w+e)[:d],(all_w+e)[d:]))(offsets).mean()
    return h_smooth


def fit_value_surrogate(h, train_u, train_x, kind="krr", n_train=600,
                        n_centers=120, seed=31, device="cpu",
                        dtype=torch.float64, lengthscale=1.,
                        epochs=180):
    """Train a differentiable value-only surrogate. Inputs should be standardized.

    train_u/train_x are TRAIN-only probe coordinates; validation is an
    independent held-out subset of queries from the same probe distribution.
    """
    rng=np.random.default_rng(seed)
    if train_x.ndim==2:
        inds_u=rng.integers(0,len(train_u),size=n_train)
        inds_x=rng.integers(0,len(train_x),size=n_train)
        vv=train_u[inds_u]; xx=train_x[inds_x]
    else:
        n_subjects,n_trials=train_x.shape[:2]
        inds_u=rng.integers(0,n_subjects,size=n_train)
        inds_x=rng.integers(0,n_trials,size=n_train)
        vv=train_u[inds_u]; xx=train_x[inds_u,inds_x]
    w=torch.as_tensor(np.concatenate([vv,xx],axis=1),device=device,dtype=dtype)
    with torch.no_grad():
        vals=torch.cat([vmap(h)(batch[:,:len(train_u[0])],batch[:,len(train_u[0]):])
                        for batch in w.split(128)]).reshape(-1,1)
    perm=torch.as_tensor(rng.permutation(len(w)),device=device)
    n_fit=int(.8*len(w))
    w_fit,y_fit=w[perm[:n_fit]],vals[perm[:n_fit]]
    w_val,y_val=w[perm[n_fit:]],vals[perm[n_fit:]]
    d=train_u.shape[1]

    if kind in ("krr","gp"):
        max_centers=min(n_centers,len(w_fit)) if kind=="krr" else min(n_train//2,len(w_fit),350)
        center=w_fit[:max_centers].detach()
        # Use function-value validation ONLY. A GP uses all selected
        # observation sites and exact K+noise inversion; KRR uses Nyström.
        def kernel(a,b,ls):
            return torch.exp(-.5*torch.cdist(a/ls,b/ls)**2)
        best=None
        for ls in (lengthscale*.5,lengthscale,lengthscale*2):
            try:
                Kmm=kernel(center,center,ls)
                if kind=="gp":
                    lam=.002
                    alpha=torch.linalg.solve(Kmm+lam*torch.eye(len(center),device=device,dtype=dtype),y_fit[:len(center)])
                else:
                    K_nm=kernel(w_fit,center,ls)
                    lam=.001
                    alpha=torch.linalg.solve(K_nm.T@K_nm+lam*(Kmm+torch.eye(len(center),device=device,dtype=dtype)),
                                             K_nm.T@y_fit)
                pred=kernel(w_val,center,ls)@alpha
                err=float(torch.mean((pred-y_val)**2))
                if best is None or err<best[0]: best=(err,ls,alpha.detach())
            except torch.linalg.LinAlgError:
                continue
        if best is None: raise RuntimeError(f"{kind} kernel solve failed")
        validation, ls,alpha=best
        def surrogate(u,x):
            w=torch.cat((u,x)).unsqueeze(0)
            return (kernel(w,center,ls)@alpha).reshape(())
        return surrogate,{"type":kind,"n_fit":n_fit,"centers":len(center),
                          "lengthscale":ls,"validation_mse":validation}
    if kind=="mlp":
        model=torch.nn.Sequential(torch.nn.Linear(w.shape[1],64),
                  torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),
                  torch.nn.Linear(64,1)).to(device=device,dtype=dtype)
        opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-4)
        # Consistent mini-batches; only NN values are targets.
        for epoch in range(epochs):
            for batch in torch.randperm(n_fit,device=device).split(256):
                opt.zero_grad()
                loss=((model(w_fit[batch])-y_fit[batch])**2).mean()
                loss.backward()
                opt.step()
        model.eval()
        model.requires_grad_(False)
        validation=float(((model(w_val)-y_val)**2).mean().detach())
        def surrogate(u,x):
            return model(torch.cat((u,x))).reshape(())
        return surrogate,{"type":"mlp","n_fit":n_fit,"epochs":epochs,
                          "validation_mse":validation}
    raise ValueError(kind)


def aggregate_gaussian_stein_u(h, v_centers, x_grid, sigma=.15,
                               samples=2048, seed=71,device="cpu"):
    """Gaussian score identities for expected U derivatives of order 1-3.

    Conditional on each chosen (v,x), probe U~N(v,sigma²I). These are
    derivatives of the Gaussian-smoothed function, NOT pointwise raw fields.
    Not valid as an estimator for empirical non-Gaussian data distribution.
    """
    rng=np.random.default_rng(seed); results=[]
    dtype=torch.float64
    x_grid=x_grid if x_grid.ndim==2 else x_grid[0]
    for i,u in enumerate(v_centers):
        x=x_grid[i%len(x_grid)]
        e=rng.normal(0,1,(samples,len(u)))
        uu=torch.as_tensor(u+sigma*e,dtype=dtype,device=device)
        xx=torch.as_tensor(x,dtype=dtype,device=device).expand(samples,-1)
        with torch.no_grad(): values=vmap(h)(uu,xx).detach().cpu().numpy()
        e1=e/sigma
        s1=np.einsum("n,ni->i",values,e1)/samples
        h2=(np.einsum("n,ni,nj->ij",values,e,e)/samples -
            np.mean(values)*np.eye(len(u)))/sigma**2
        e3=np.einsum("n,ni,nj,nk->ijk",values,e,e,e)/samples
        for a in range(len(u)):
            for b in range(len(u)):
                for c in range(len(u)):
                    e3[a,b,c] -= np.mean(values*e[:,a])*(b==c)
                    e3[a,b,c] -= np.mean(values*e[:,b])*(a==c)
                    e3[a,b,c] -= np.mean(values*e[:,c])*(a==b)
        results.append((s1,h2,e3/sigma**3))
    return {k:np.mean(np.stack([v[j] for v in results]),axis=0)
            for j,k in enumerate(("U","UU","UUU"))}


METHODS={
    "ad_mixed":{"kind":"field","orders":(1,2,3)},
    "ad_forward":{"kind":"field","orders":(1,2,3)},
    "ad_reverse":{"kind":"field","orders":(1,2,3)},
    "fd2":{"kind":"field","orders":(1,2,3)},
    "fd4":{"kind":"field","orders":(1,2,3)},
    "richardson":{"kind":"field","orders":(1,2,3)},
    "hybrid":{"kind":"field","orders":(1,2,3)},
    "gaussian_smoothing":{"kind":"field","orders":(1,2,3)},
    "local_poly":{"kind":"field","orders":(1,2,3)},
    "spectral_chebyshev":{"kind":"field","orders":(1,2,3)},
    "krr":{"kind":"field","orders":(1,2,3)},
    "gp":{"kind":"field","orders":(1,2,3)},
    "mlp":{"kind":"field","orders":(1,2,3)},
    "hvp_directional":{"kind":"contraction","orders":(2,3)},
    "stein_gaussian_u":{"kind":"aggregate_only","orders":(1,2,3)},
}


def build_estimator(method,h,block,d,p,step=.03,sigma=.05,
                    mc=16,seed=19,device="cpu",dtype=torch.float64,
                    surrogate=None):
    """Return fn(u,x), metadata. Surrogates must be pre-fitted."""
    if method.startswith("ad_"):
        mode=method.replace("ad_","")
        return derivative_function(h,block,mode),{"type":"autodiff","mode":mode}
    if method in ("fd2","fd4","richardson","local_poly","spectral_chebyshev"):
        return value_difference(h,block,d,p,step,method),{"type":"value_queries","step":step,"method":method}
    if method=="hybrid":
        return hybrid_ad_fd(h,block,d,p,step),{"type":"ad_fd","step":step}
    if method=="gaussian_smoothing":
        smoothed=gaussian_smoothed_function(h,d,p,sigma,mc,seed,axes="U",device=device,dtype=dtype)
        return derivative_function(smoothed,block),{"type":"gaussian_smoothing_U_only","sigma":sigma,"mc":mc}
    if method in ("krr","gp","mlp"):
        if surrogate is None: raise ValueError("surrogate must be pre-fitted")
        return derivative_function(surrogate,block),{"type":"value_only_surrogate","kind":method}
    raise ValueError(f"{method} is not a dense field estimator")
