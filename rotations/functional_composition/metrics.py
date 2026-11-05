"""Twenty explicitly operationalized structure indicators for the response-profile proxy.

These are NOT all exact implementations of the named population quantities.
Names with 'proxy', 'bound', or 'control' document the approximation honestly.
A single suite jointly uses a selected subset in one objective; all twenty
are reported on the frozen identity and winning rotations.
"""
from __future__ import annotations
from itertools import combinations, combinations_with_replacement
import math
import numpy as np
from .surrogate import rotate_coefficients, poly_design

NAMES = {
 '01_axis_effective_support_proxy': 'Soft count of univariate degree-1/2/3 groups (not spline EDF)',
 '02_axis_curvature_total_variation_proxy': 'Mean abs axiswise second derivative of fitted response proxy',
 '03_derivative_sign_change_count': 'Sign reversals in axis derivative along sampled one-dimensional paths',
 '04_axis_degree_spectral_entropy': 'Entropy of degree energy for each univariate axis',
 '05_functional_anova_effective_order_proxy': 'Energy-weighted number of distinct latent axes per polynomial term',
 '06_pair_interaction_H_proxy': 'Energy fraction of pair-only polynomial components (not Friedman H)',
 '07_pairwise_separation_effective_rank': 'Sum of weighted SVD effective ranks of pairwise interaction monomial surfaces',
 '08_compositional_operator_soft_code': 'Soft number of scalar/power/product function modules, with equal square/product costs',
 '09_interaction_hypergraph_soft_code': 'Soft number of active distinct support hyperedges, separate from module coding',
 '10_tensor_train_rank_invariant_control': 'Effective rank of tensor unfolding; invariant/weakly sensitive, negative control',
 '11_effect_fingerprint_effective_rank': 'Sum of effective stimulus-profile ranks across subject positions for each axis',
 '12_effect_direction_drift': 'Average angular drift of conditional stimulus-effect vectors',
 '13_contextual_gradient_joint_diagonalization': 'Joint off-diagonal gradient covariance loss across stimulus-response features',
 '14_mixed_u_x_derivative_support_proxy': 'Soft channel support of d2p/du dx computed from local stimulus feature map',
 '15_learned_context_gate_soft_code': 'Axis-context effect support using X-only quantile groupings',
 '16_rotated_polynomial_soft_sparsity': 'Polynomial monomial soft sparsity, not density-orthogonal PCE',
 '17_conditional_information_ablation_proxy': 'Extra heldout response-profile MSE after deleting one latent axis, not conditional MI',
 '18_residual_HSIC_proxy': 'Kernel dependence between axis and profile residual after linear control of other axes',
 '19_discretized_synergy_lower_bound_not_PID': 'Positive lower bound from discretized conditional MI, not an exact PID',
 '20_prequential_structural_MDL_proxy': 'Subject-forward Bernoulli predictive logloss plus structural coding per stage',
}


def soft_count(g, tau):
    z=np.asarray(g,float)**2
    return z/(z+max(float(tau),1e-12)**2)


def effective_rank(values, eps=1e-14):
    z=np.asarray(values,float)**2
    total=z.sum()
    if total<eps:return 0.0
    p=z[z>eps]/total
    return float(np.exp(-np.sum(p*np.log(p))))


def _coeff_group_list(coeff):
    k=coeff[1].shape[0]
    groups=[]
    for d in sorted(x for x in coeff if x):
        for idx in combinations_with_replacement(range(k),d):
            # Symmetric tensor stores original monomial coefficient / multiplicity.
            multiplicity=math.factorial(d)
            for n in set(idx):multiplicity//=math.factorial(idx.count(n))
            vec=coeff[d][idx]
            norm=float(np.linalg.norm(vec)*np.sqrt(multiplicity))
            groups.append((idx,d,len(set(idx)),norm,vec*multiplicity))
    return groups


def support_complexity(coeff, threshold=.08):
    """Approximate code length with ALL cross and pure-square modules same base cost.

    Scales are tensor-Frobenius norms of each homogeneous degree, invariant
    under orthogonal R, so threshold is not re-tuned per rotated angle.
    """
    g=_coeff_group_list(coeff)
    norm_by_degree={d:float(np.linalg.norm(coeff[d])) for d in coeff if d}
    active=[]
    op=0.0
    hyper={}
    sparsity=0.0
    axis_only=0.0
    for inds,d,s,norm,vec in g:
        cutoff=threshold*max(norm_by_degree[d],1e-12)
        present=float(soft_count(norm,cutoff))
        # Comparable operator cost for square and product; extra degree needs more symbols.
        code=(1.0+0.15*(d-1))*present
        op+=code
        sparsity+=present
        if s==1:axis_only+=present
        if s>=2:
            group=tuple(sorted(set(inds)))
            hyper[group]=max(hyper.get(group,0.0),present)
        active.append((inds,present,norm))
    graph=float(sum((1.0+0.25*(len(group)-2))*presence for group,presence in hyper.items()))
    return dict(operator=op,hypergraph=graph,sparsity=sparsity,axis=axis_only,
                active=active,norm_by_degree=norm_by_degree)


def interaction_rank(coeff):
    k,L=coeff[1].shape
    total=0.;detail=[]
    for i,j in combinations(range(k),2):
        a=np.zeros((3,3,L))
        if 2 in coeff:a[1,1]=2*coeff[2][i,j]
        if 3 in coeff:
            a[2,1]=3*coeff[3][i,i,j]
            a[1,2]=3*coeff[3][i,j,j]
        strength=float(np.linalg.norm(a))
        if strength<1e-12:continue
        s=np.linalg.svd(a.reshape(3,-1),compute_uv=False)
        rank=effective_rank(s)
        total+=rank*float(soft_count(strength,.08*max(np.linalg.norm(coeff.get(2,np.zeros(1))),1e-12)))
        detail.append(dict(i=i,j=j,effective_rank=rank,amplitude=strength))
    return float(total),detail


def _gradient(proxy,q):
    return proxy.gradients(q)


def _rotated_grad(proxy,R,q):
    return np.einsum('ia,nal->nil',R,_gradient(proxy,q),optimize=True)


def fingerprint_and_drift(proxy,R,q):
    d=_rotated_grad(proxy,R,q)
    n,k,l=d.shape
    rank=0.;drift=0.;detail=[]
    for i in range(k):
        component=d[:,i,:]
        second=component.T@component/max(len(component),1)
        eig=np.linalg.eigvalsh(second)
        er=effective_rank(np.sqrt(np.maximum(eig,0)))
        rank+=max(0.,er-1.)  # One stable X-response mode is zero complexity.
        directions=component/np.maximum(np.linalg.norm(component,axis=1,keepdims=True),1e-9)
        mean=directions.mean(axis=0)
        norm=np.linalg.norm(mean)
        value=float(1.-min(norm,1.))
        drift+=value
        detail.append(dict(axis=i,effective_rank=er,direction_drift=value))
    return rank,drift,detail


def contextual_jointdiag(proxy,R,q, max_contexts=12):
    d=proxy.gradients(q)
    L=d.shape[-1]
    # X features play the role of prespecified, X-derived effect channels.
    losses=[]
    for j in range(min(L,max_contexts)):
        C=d[:,:,j].T@d[:,:,j]/len(q)
        if np.linalg.norm(C)<1e-10:continue
        rotated=R@C@R.T
        off=rotated-np.diag(np.diag(rotated))
        losses.append(float(np.sum(off**2)/(np.sum(C**2)+1e-12)))
    return float(np.mean(losses)) if losses else 0.0


def combined_scores(proxy,R,config,q_anchor=None):
    rotated=rotate_coefficients(proxy.coef,R)
    groups=support_complexity(rotated,config['support_threshold'])
    q=proxy.q_train[proxy.fit_ids]
    if q_anchor is not None:q=q_anchor
    elif len(q)>int(config['anchor_subjects']):
        rng=np.random.default_rng(int(config['seed'])+918)
        q=q[rng.choice(len(q),int(config['anchor_subjects']),replace=False)]
    fp,drift,detail=fingerprint_and_drift(proxy,R,q)
    jd=contextual_jointdiag(proxy,R,q)
    return dict(operator=float(groups['operator']),fingerprint=float(fp),
                jointdiag=float(jd),separation=float(interaction_rank(rotated)[0]),
                hypergraph=float(groups['hypergraph']),drift=float(drift))


def _axis_curvature_sign(proxy,R,q,coeff):
    # Mixed proxy Hessian is evaluated analytically in frozen q coordinates.
    H=proxy.hessians(q)
    Hr=np.einsum('ia,jb,nabl->nijl',R,R,H,optimize=True)
    curve=float(np.mean(np.abs(np.diagonal(Hr,axis1=1,axis2=2))))
    # Sign flips on 1D paths through up to 16 sampled points in whitened U.
    samples=q[:min(len(q),12)]@R.T
    sign_changes=0
    xlevels=np.linspace(-2.5,2.5,13)
    for axis in range(R.shape[0]):
        # Evaluate derivative of the exact rotated polynomial along the axis.
        for anchor in samples:
            g=[]
            for x in xlevels:
                u=anchor.copy();u[axis]=x
                val=coeff[1][axis].copy()
                if 2 in coeff:val+=2*np.einsum('j,jl->l',u,coeff[2][axis],optimize=True)
                if 3 in coeff:val+=3*np.einsum('j,k,jkl->l',u,u,coeff[3][axis],optimize=True)
                g.append(float(np.sum(val)))
            s=np.sign(np.array(g));s[np.abs(g)<1e-9]=0
            nz=s[s!=0]
            sign_changes+=int(np.sum(nz[:-1]*nz[1:]<0))
    return curve,sign_changes/max(len(samples),1)


def _degree_entropy(coeff):
    k=coeff[1].shape[0]
    ent=0.
    for i in range(k):
        x=[float(np.sum(coeff[1][i]**2))]
        if 2 in coeff:x.append(float(np.sum(coeff[2][i,i]**2)))
        if 3 in coeff:x.append(float(np.sum(coeff[3][i,i,i]**2)))
        x=np.array(x);x=x/max(x.sum(),1e-12)
        x=x[x>1e-12]
        ent+=float(-np.sum(x*np.log(x)))
    return ent


def _effective_order_and_pair_energy(coeff):
    groups=_coeff_group_list(coeff)
    num=sum(g[3]**2 for g in groups)
    if num<1e-15:return 0.,0.
    order=sum(g[2]*g[3]**2 for g in groups)/num
    pair=sum(g[3]**2 for g in groups if g[2]==2)/num
    return float(order),float(pair)


def _tt_rank_control(coeff):
    if 3 in coeff:
        ten=coeff[3]
        a=np.linalg.svd(ten.reshape(ten.shape[0],-1),compute_uv=False)
        b=np.linalg.svd(ten.reshape(ten.shape[0]*ten.shape[1],-1),compute_uv=False)
        return .5*(effective_rank(a)+effective_rank(b))
    return effective_rank(np.linalg.svd(coeff[2].reshape(coeff[2].shape[0],-1),compute_uv=False)) if 2 in coeff else 0.


def _mixed_u_x_support(proxy,R,q,threshold):
    # Stimulus-map finite difference is an interpolation assumption, not observed derivative data.
    x=proxy.train_x[proxy.fit_ids]
    sample=x[:min(len(x),20)].reshape(-1,x.shape[-1])
    sample=sample[np.linspace(0,len(sample)-1,min(12,len(sample)),dtype=int)]
    phi=proxy.stimulus
    dx=[]
    for j in range(sample.shape[1]):
        step=float(max(phi.std[j]*.025,1e-4))
        up=sample.copy();down=sample.copy()
        up[:,j]+=step;down[:,j]-=step
        dx.append((phi.transform(up)-phi.transform(down))/(2*step))
    dx=np.stack(dx,axis=-1)
    gu=_rotated_grad(proxy,R,q)
    mixed=np.einsum('nil,tld->nitd',gu,dx,optimize=True)
    e=np.sqrt(np.mean(mixed**2,axis=(0,2)))
    tau=threshold*max(np.linalg.norm(e),1e-12)
    return float(np.sum(soft_count(e,tau)))


def _context_gate_cost(proxy,R,q,threshold):
    x=proxy.train_x[proxy.fit_ids]
    xvals=x.reshape(-1,x.shape[-1])
    if len(xvals)>12000:
        xvals=xvals[np.linspace(0,len(xvals)-1,12000,dtype=int)]
    vals=xvals[:,0]
    thresh=np.quantile(vals,[.25,.5,.75])
    cat=np.digitize(vals,thresh)
    phis=proxy.stimulus.transform(xvals)
    means=np.stack([phis[cat==j].mean(axis=0) if np.any(cat==j)
                    else np.zeros(phis.shape[1]) for j in range(4)])
    d=_rotated_grad(proxy,R,q)
    value=np.einsum('nil,cl->nic',d,means,optimize=True)
    energy=np.sqrt(np.mean(value**2,axis=0))
    return float(np.sum(soft_count(energy,threshold*max(np.linalg.norm(energy),1e-12))))


def _predict_profiles_from_rotated(coeff,u):
    result=np.broadcast_to(coeff[0],(len(u),coeff[0].shape[0])).copy()
    result+=np.einsum('ni,il->nl',u,coeff[1])
    if 2 in coeff:result+=np.einsum('ni,nj,ijl->nl',u,u,coeff[2],optimize=True)
    if 3 in coeff:result+=np.einsum('ni,nj,nk,ijkl->nl',u,u,u,coeff[3],optimize=True)
    return result


def _ablation_proxy(proxy,R,coeff):
    ids=proxy.validation_ids
    u=proxy.q_train[ids]@R.T
    target=proxy.h_train[ids]
    full=_predict_profiles_from_rotated(coeff,u)
    base=float(np.mean((target-full)**2))
    impact=[]
    for i in range(u.shape[1]):
        altered=u.copy();altered[:,i]=0
        prediction=_predict_profiles_from_rotated(coeff,altered)
        impact.append(max(0,float(np.mean((target-prediction)**2))-base))
    total=sum(impact)
    p=np.array(impact)/max(total,1e-12)
    return float(np.sum(p*p)),[float(v) for v in impact]


def _rbf_hsic(a,b):
    a=np.atleast_2d(a);b=np.atleast_2d(b)
    def kernel(x):
        d=np.sum((x[:,None,:]-x[None,:,:])**2,axis=-1)
        positive=d[d>1e-12]
        scale=float(np.median(positive)) if len(positive) else 1.0
        return np.exp(-d/max(scale,1e-8))
    K=kernel(a);L=kernel(b)
    kc=K-K.mean(0)-K.mean(1)[:,None]+K.mean()
    lc=L-L.mean(0)-L.mean(1)[:,None]+L.mean()
    return float(np.mean(kc*lc))


def _hsic_residual_proxy(proxy,R,q):
    n=min(len(q),160)
    u=q[:n]@R.T
    h=proxy.h_train[proxy.fit_ids[:n]]
    out=[]
    for i in range(u.shape[1]):
        others=np.delete(u,i,axis=1)
        design=np.column_stack([np.ones(n),others])
        fit=np.linalg.lstsq(design,h,rcond=1e-5)[0]
        residual=h-design@fit
        out.append(_rbf_hsic(u[:,[i]],residual))
    return float(np.sum(out))


def _discrete_cmi(feature,response,context,levels_feature=4):
    # Conditional mutual information in natural logs; empirical multinomial counts.
    feature=np.asarray(feature,int);response=np.asarray(response,int);context=np.asarray(context,int)
    n=len(feature)
    total=0.
    for c in np.unique(context):
        idx=context==c
        if idx.sum()<30:continue
        y=response[idx];x=feature[idx]
        vals=np.unique(x)
        tab=np.stack([np.bincount(y[x==v],minlength=2) for v in vals],axis=0).astype(float)
        tab+=0.5 # Jeffreys smoothing; distinct levels observed per context
        tab/=tab.sum()
        px=tab.sum(axis=1);py=tab.sum(axis=0)
        mi=float(np.sum(tab*np.log(tab/(px[:,None]*py[None,:]))))
        total+=idx.sum()/n*mi
    return total


def _synergy_lower_bound(proxy,R,q):
    # Discretized CMI I(Y;Ui,Uj|Xbin)-I(Y;Ui|Xbin)-I(Y;Uj|Xbin).
    # This is a lower-bound-like synergy diagnostic, NOT a full PID decomposition.
    from itertools import combinations
    ns=min(len(q),120)
    ids=proxy.fit_ids[:ns]
    u=q[:ns]@R.T
    y=proxy.y_train[ids]
    mask=proxy.mask_train[ids]
    x=proxy.train_x[ids]
    z=np.stack([np.digitize(u[:,i],np.quantile(u[:,i],[.25,.5,.75])) for i in range(u.shape[1])],axis=1)
    qbins=np.digitize(x[...,0],np.quantile(x[...,0][mask],[.25,.5,.75]))
    max_sum=0.
    for i,j in combinations(range(u.shape[1]),2):
        aa=np.broadcast_to(z[:,i,None],y.shape)[mask]
        bb=np.broadcast_to(z[:,j,None],y.shape)[mask]
        yy=y[mask].astype(int);cc=qbins[mask]
        ia=_discrete_cmi(aa,yy,cc);ib=_discrete_cmi(bb,yy,cc)
        joint=_discrete_cmi(4*aa+bb,yy,cc)
        max_sum+=max(0.,joint-ia-ib)
    return float(max_sum)


def diagnostic_metrics(proxy,R,config,include_expensive=True):
    scores=combined_scores(proxy,R,config)
    rotated=rotate_coefficients(proxy.coef,R)
    summary=support_complexity(rotated,config['support_threshold'])
    anchor=proxy.q_train[proxy.fit_ids[:min(len(proxy.fit_ids),int(config['anchor_subjects']))]]
    curve,sign=_axis_curvature_sign(proxy,R,anchor,rotated)
    degree_ent=_degree_entropy(rotated)
    order,pair=_effective_order_and_pair_energy(rotated)
    sep,detail=interaction_rank(rotated)
    effective_rank_fp,drift,fp_detail=fingerprint_and_drift(proxy,R,anchor)
    ablation,ablation_each=_ablation_proxy(proxy,R,rotated)
    results={
        '01_axis_effective_support_proxy':float(summary['axis']),
        '02_axis_curvature_total_variation_proxy':curve,
        '03_derivative_sign_change_count':float(sign),
        '04_axis_degree_spectral_entropy':degree_ent,
        '05_functional_anova_effective_order_proxy':order,
        '06_pair_interaction_H_proxy':pair,
        '07_pairwise_separation_effective_rank':sep,
        '08_compositional_operator_soft_code':float(summary['operator']),
        '09_interaction_hypergraph_soft_code':float(summary['hypergraph']),
        '10_tensor_train_rank_invariant_control':_tt_rank_control(rotated),
        '11_effect_fingerprint_effective_rank':effective_rank_fp,
        '12_effect_direction_drift':drift,
        '13_contextual_gradient_joint_diagonalization':scores['jointdiag'],
        '14_mixed_u_x_derivative_support_proxy':_mixed_u_x_support(proxy,R,anchor,config['support_threshold']),
        '15_learned_context_gate_soft_code':_context_gate_cost(proxy,R,anchor,config['support_threshold']),
        '16_rotated_polynomial_soft_sparsity':float(summary['sparsity']),
        '17_conditional_information_ablation_proxy':ablation,
        '18_residual_HSIC_proxy':_hsic_residual_proxy(proxy,R,anchor),
        '19_discretized_synergy_lower_bound_not_PID':_synergy_lower_bound(proxy,R,anchor),
    }
    if include_expensive:
        from .mdl import prequential_mdl
        results['20_prequential_structural_MDL_proxy']=prequential_mdl(proxy,R,config)['objective']
    else:results['20_prequential_structural_MDL_proxy']=None
    return results,dict(pair_modules=detail,axis_fingerprints=fp_detail,axis_ablation=ablation_each)