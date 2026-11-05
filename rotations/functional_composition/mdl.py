"""Subject-forward, nonlinear-module predictive MDL for one frozen ZXY suite.

This is a *restricted predictive-code approximation*, NOT universal MDL.
Selection is forward and nested: each training prefix is further split into
inner-fit and inner-validation subjects. Nonlinear monomials are selected
using held-out Bernoulli cross entropy plus their structural code cost.
Selected modules are refit on the entire prefix and evaluated on future
subjects. All subsets remain inside NP TRAIN subjects, not NP TEST.
The four full linear terms are included as a rotation-invariant reference
so the question is whether nonlinear modules add predictive value.
"""
from __future__ import annotations
from itertools import combinations_with_replacement
import math
import numpy as np
from .surrogate import poly_design, rotate_coefficients, _safe_ridge


def _all_terms(k, degree):
    return [()] + [idx for d in range(1,degree+1)
                   for idx in combinations_with_replacement(range(k),d)]


def _term_amplitude(coeff, term):
    d=len(term)
    value=coeff[d][term]
    multiplicity=math.factorial(d)
    for i in set(term):
        multiplicity//=math.factorial(term.count(i))
    return float(np.linalg.norm(value*multiplicity))


def nonlinear_candidate_pool(proxy, R, config):
    k=proxy.k
    coeff=rotate_coefficients(proxy.coef,R)
    all_nonlin=[t for t in _all_terms(k,len(coeff)-1) if len(t)>=2]
    # Candidate shortlist is generated from FIT-only structural amplitudes,
    # not from true theta or from the forward validation responses.
    ordered=sorted(all_nonlin,key=lambda term:(-_term_amplitude(coeff,term),term))
    count=int(config.get('mdl_candidate_pool',18))
    if count<1:raise ValueError('mdl_candidate_pool must be >=1')
    return ordered[:count]


def _binary_nll(predicted_profiles, phi, y, mask):
    prob=np.einsum('ntl,nl->nt',phi,predicted_profiles,optimize=True)
    prob=np.clip(prob,1e-5,1-1e-5)
    losses=-(y*np.log(prob)+(1-y)*np.log1p(-prob))*mask
    return float(losses.sum()),int(mask.sum())


def _fit_profile_ridge(v,h,ids,ridge):
    design=v[ids]
    rms=np.maximum(np.sqrt(np.mean(design**2,axis=0)),1e-6)
    rms[0]=1.
    weights=_safe_ridge(design/rms,h[ids],ridge)
    return weights/rms[:,None]


def _module_code(num_nonlin, pool_size, n_subjects, n_channels, n_linear):
    """Two-part model/module code in nats; channel parameter costs per subject."""
    if num_nonlin>pool_size:raise ValueError('invalid selected module count')
    # Encode sparse support and coefficient parameters. The common linear
    # component is included, but cannot alone distinguish rotations.
    support=math.log(pool_size+1)+math.log(max(1,math.comb(pool_size,num_nonlin)))
    parameters=.5*(n_linear+num_nonlin)*n_channels*math.log(max(n_subjects,2))
    return float(support+parameters)


def prequential_mdl(proxy,R,config):
    """Compare nonlinear rotations by *nested* forward binary predictive code."""
    k=proxy.k
    ids=np.asarray(proxy.fit_ids,dtype=int).copy()
    rng=np.random.default_rng(int(config['seed'])+9041)
    rng.shuffle(ids)
    count=min(len(ids),int(config['max_mdl_subjects']))
    if count<60:raise ValueError('Need >=60 FIT subjects for nonlinear predictive MDL')
    ids=ids[:count]
    pool=nonlinear_candidate_pool(proxy,R,config)
    linear=[()] + [(i,) for i in range(k)]
    all_terms=linear+pool
    q=proxy.q_train[ids]@np.asarray(R,float).T
    V=poly_design(q,all_terms)
    H=proxy.h_train[ids]
    phi=proxy.phi_train[ids]
    Y=proxy.y_train[ids]
    mask=proxy.mask_train[ids]
    folds=max(1,min(int(config['mdl_prefix_folds']),5))
    max_extra=max(0,min(int(config.get('mdl_max_extra_modules',5)),len(pool)))
    inner_fraction=float(config.get('mdl_inner_validation_frac',.2))
    if not .05<=inner_fraction<=.45:raise ValueError('mdl_inner_validation_frac must be in [.05,.45]')
    min_gain=float(config.get('mdl_min_delta_nats',.0002))
    if min_gain<0:raise ValueError('mdl_min_delta_nats must be >=0')
    ridge=float(config['surrogate_ridge'])
    # Smaller fixed, X-only trial grid for repeated nested module selection;
    # future-subject evaluation uses ALL observed trials.
    nt=int(config.get('mdl_selection_trials',64))
    if nt<=0:raise ValueError('mdl_selection_trials must be positive')
    trial=np.unique(np.linspace(0,phi.shape[1]-1,
                                min(nt,phi.shape[1]),dtype=int))
    phi_select=phi[:,trial,:]
    y_select=Y[:,trial]
    mask_select=mask[:,trial]
    future_nll=0.;future_linear_nll=0.;future_trials=0
    accumulated_code=0.;details=[]
    final_terms=linear[:]
    for step in range(folds):
        start=int(count*(.50+.40*step/folds))
        end=int(count*(.50+.40*(step+1)/folds))
        end=min(count,max(start+1,end))
        if end<=start:continue
        n_inner_val=max(8,int(start*inner_fraction))
        n_inner_fit=start-n_inner_val
        if n_inner_fit<=len(linear)+2:continue
        inner_fit=np.arange(n_inner_fit)
        inner_val=np.arange(n_inner_fit,start)
        future=np.arange(start,end)
        chosen=list(range(len(linear)))
        available=list(range(len(linear),len(all_terms)))
        def inner_score(column_ids):
            fitted=_fit_profile_ridge(V[:,column_ids],H,inner_fit,ridge)
            p=V[inner_val][:,column_ids]@fitted
            n,obs=_binary_nll(p,phi_select[inner_val],
                               y_select[inner_val],mask_select[inner_val])
            code=_module_code(len(column_ids)-len(linear),len(pool),
                              len(inner_fit),proxy.channels,k)
            return n/max(obs,1)+code/max(obs,1)
        current=inner_score(chosen)
        selected_trace=[]
        for _ in range(max_extra):
            scores=[]
            for col in available:
                value=inner_score(chosen+[col])
                scores.append((value,col))
            if not scores:break
            next_score,col=min(scores,key=lambda x:(x[0],x[1]))
            if current-next_score<=min_gain:break
            chosen.append(col)
            available.remove(col)
            selected_trace.append(dict(term=list(all_terms[col]),
                                       validation_mdl_improvement_nats_per_trial=float(current-next_score)))
            current=next_score
        chosen_terms=[all_terms[j] for j in chosen]
        fitted=_fit_profile_ridge(V[:,chosen],H,np.arange(start),ridge)
        future_profiles=V[future][:,chosen]@fitted
        nll,obs=_binary_nll(future_profiles,phi[future],Y[future],mask[future])
        linear_fit=_fit_profile_ridge(V[:,:len(linear)],H,np.arange(start),ridge)
        linear_profiles=V[future,:len(linear)]@linear_fit
        linear_nll,_=_binary_nll(linear_profiles,phi[future],Y[future],mask[future])
        code=_module_code(len(chosen)-len(linear),len(pool),start,proxy.channels,k)
        future_nll+=nll
        future_linear_nll+=linear_nll
        future_trials+=obs
        accumulated_code+=code
        final_terms=chosen_terms
        details.append(dict(fold=step+1,prefix_subjects=start,
                            inner_fit_subjects=len(inner_fit),
                            inner_validation_subjects=len(inner_val),
                            future_subjects=len(future),future_trials=obs,
                            selected_nonlinear_modules=len(chosen)-len(linear),
                            selected_nonlinear_terms=[list(t) for t in chosen_terms[len(linear):]],
                            nonlinear_greedy_trace=selected_trace,
                            future_nats_per_trial=float(nll/max(obs,1)),
                            future_linear_only_nats_per_trial=float(linear_nll/max(obs,1)),
                            structural_code_nats=float(code)))
    if future_trials==0:raise RuntimeError('No future subjects in predictive MDL')
    prediction=future_nll/future_trials
    linear_prediction=future_linear_nll/future_trials
    structural=accumulated_code/future_trials
    return dict(objective=float(prediction+structural),
                predictive_nats_per_trial=float(prediction),
                linear_only_nats_per_trial=float(linear_prediction),
                nonlinear_prediction_gain_nats_per_trial=float(linear_prediction-prediction),
                code_nats_per_trial=float(structural),
                future_trials=int(future_trials),
                selected_module_count=len(final_terms)-1,
                selected_nonlinear_module_count=len(final_terms)-(k+1),
                selected_monomials=[list(v) for v in final_terms],
                candidate_pool=[list(v) for v in pool],
                folds=details,
                caution='Nested subject-forward sparse polynomial Bernoulli predictive-code approximation, not universal MDL; Z encodes all observed Y')
