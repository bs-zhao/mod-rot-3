"""Subject-forward predictive structural description length for a fixed rotation.

This is a deliberately *restricted* predictive-MDL approximation:
- Group selection uses fitted response-profile coefficient energy on FIT only.
- Refits chosen raw-monomial modules using progressively more FIT subjects.
- Predicts future subjects' binary choices via a limited stimulus basis.
- Reports predictive nats/trial and a separate nats/subject structure code.
It is not an exact universal-code MDL nor a claim of causal parameter recovery.
"""
from __future__ import annotations
from itertools import combinations_with_replacement
import math
import numpy as np
from .surrogate import poly_design, rotate_coefficients, _safe_ridge


def _all_terms(k,d):
    return [()] + [t for degree in range(1,d+1)
                   for t in combinations_with_replacement(range(k),degree)]


def _weighted_groups(coeff):
    groups=[]
    for d in sorted(x for x in coeff if x):
        for term in combinations_with_replacement(range(coeff[1].shape[0]),d):
            mult=math.factorial(d)
            for v in set(term):mult//=math.factorial(term.count(v))
            coef=coeff[d][term]*mult
            groups.append((term,float(np.sum(coef**2)),d))
    return groups


def selected_modules(coeff,config):
    entries=_weighted_groups(coeff)
    total=sum(e for _,e,_ in entries)
    sorted_entries=sorted(entries,key=lambda item:-item[1])
    target=float(config['mdl_energy_fraction'])
    if not 0<target<=1:raise ValueError('mdl_energy_fraction must be in (0,1]')
    chosen=[()];covered=0.
    for term,energy,degree in sorted_entries:
        if covered>=target*max(total,1e-14) and len(chosen)>1:break
        chosen.append(term)
        covered+=energy
    return chosen


def prequential_mdl(proxy,R,config):
    rng=np.random.default_rng(int(config['seed'])+9041)
    ids=proxy.fit_ids.copy()
    rng.shuffle(ids)
    limit=min(len(ids),int(config['max_mdl_subjects']))
    ids=ids[:limit]
    if limit<60:raise ValueError('Need at least 60 subjects for subject-forward MDL')
    R=np.asarray(R,float)
    coeff=rotate_coefficients(proxy.coef,R)
    selected=selected_modules(coeff,config)
    q=proxy.q_train[ids]@R.T
    design=poly_design(q,selected)
    h=proxy.h_train[ids]
    phi=proxy.phi_train[ids]
    y=proxy.y_train[ids]
    mask=proxy.mask_train[ids]
    folds=int(config['mdl_prefix_folds'])
    folds=max(1,min(folds,5))
    all_folds=[]
    total_nll=0.;total_trials=0
    # Start with half the subjects, then expand prefix and evaluate future blocks.
    for step in range(folds):
        train_end=int(limit*(0.50+0.40*step/folds))
        future_end=int(limit*(0.50+0.40*(step+1)/folds))
        future_end=max(future_end,train_end+1)
        tr=np.arange(train_end)
        te=np.arange(train_end,min(limit,future_end))
        if len(te)==0:continue
        matrix=design[tr]
        rms=np.maximum(np.sqrt(np.mean(matrix*matrix,axis=0)),1e-6)
        rms[0]=1
        coefscaled=_safe_ridge(matrix/rms,h[tr],config['surrogate_ridge'])
        coeff_fit=coefscaled/rms[:,None]
        predicted=design[te]@coeff_fit
        p=np.einsum('ntl,nl->nt',phi[te],predicted,optimize=True)
        p=np.clip(p,1e-4,1-1e-4)
        nll=-np.sum((y[te]*np.log(p)+(1-y[te])*np.log(1-p))*mask[te])
        count=int(mask[te].sum())
        total_nll+=float(nll);total_trials+=count
        all_folds.append(dict(fold=step+1,prefix_subjects=len(tr),future_subjects=len(te),
                              future_trials=count,predictive_nats_per_trial=float(nll/max(count,1))))
    nmodules=len(selected)-1
    # Encode module locations/powers and ridge coefficient channels, using number
    # of independently sampled subjects (not number of within-subject trials).
    module_space=max(len(_all_terms(proxy.k,len(proxy.coef)-1))-1,1)
    graph_nats=math.log(max(1,math.comb(module_space,min(nmodules,module_space))))
    parameter_nats=0.5*nmodules*proxy.channels*math.log(max(limit//2,2))
    description_nats=graph_nats+parameter_nats
    nll_per_trial=total_nll/max(total_trials,1)
    # Treat each participant as a unit for structural coding; report separate terms.
    per_subject_code=description_nats/max(sum(f['future_subjects'] for f in all_folds),1)
    # Nats per trial, explicitly conditional on the chosen finite stimulus basis.
    total_obj=nll_per_trial+description_nats/max(total_trials,1)
    return dict(objective=float(total_obj),predictive_nats_per_trial=float(nll_per_trial),
                code_nats_per_trial=float(description_nats/max(total_trials,1)),
                code_nats_per_future_subject=float(per_subject_code),
                selected_module_count=nmodules,
                selected_monomials=[list(v) for v in selected],
                folds=all_folds,
                caution='Restricted response-profile predictive-MDL proxy, not a universal code')