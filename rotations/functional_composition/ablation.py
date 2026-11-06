"""08/11/13 single-objective FIT experiments and nonlinear MDL20 comparison.

One complete suite: these are *ablations within functional_composition*,
not independent rotation methods or an all-suite launcher. Theta is
not accepted by any fit or MDL selection function.
"""
from __future__ import annotations
import csv
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from .optimizer import optimize
from .mdl import prequential_mdl


WEIGHTS=('operator','fingerprint','jointdiag','separation','hypergraph')
TARGETS=('operator08','fingerprint11','jointdiag13')
TARGET_KEY={'operator08':'operator','fingerprint11':'fingerprint',
            'jointdiag13':'jointdiag'}


def fit_ablations(proxy, config, joint_R):
    """Find separate coordinate systems using ONLY different FIT ZXY objectives."""
    candidates={'identity':np.eye(proxy.k),'joint':np.asarray(joint_R,float)}
    optimizations={}
    objectives={}
    if config.get('ablation_enabled',True):
        for name in TARGETS:
            c=dict(config)
            for weight in WEIGHTS:
                c['weight_'+weight]=1. if weight==TARGET_KEY[name] else 0.
            c['starts']=int(config.get('ablation_starts',2))
            c['maxiter']=int(config.get('ablation_maxiter',110))
            R,info,obj=optimize(proxy,c)
            candidates[name]=R
            optimizations[name]=info
            objectives[name]=obj
    mdls={}
    if config.get('mdl_compare_candidates',True):
        # MDL20 compares existing candidate rotations *within FIT subjects*.
        # It does NOT use the true parameter recovery score to select R.
        for name,R in candidates.items():
            mdls[name]=prequential_mdl(proxy,R,config)
        winner=min(mdls,key=lambda key:(mdls[key]['objective'],key))
        candidates['mdl20_selected']=candidates[winner].copy()
    else:
        winner=None
    return dict(rotations=candidates,optimization=optimizations,
                objectives=objectives,mdl=mdls,mdl_winner=winner)


def _serialize(d):
    if isinstance(d,np.ndarray):return d.tolist()
    if isinstance(d,(np.integer,np.floating)):return d.item()
    if isinstance(d,dict):return {str(k):_serialize(v) for k,v in d.items()}
    if isinstance(d,(list,tuple)):return [_serialize(v) for v in d]
    return d


def _sweep_ablation(proxy, ablations, config):
    steps=int(config.get('ablation_sweep_steps',25))
    if steps<3:raise ValueError('ablation_sweep_steps must be >=3')
    limit=int(config.get('ablation_sweep_pairs',6))
    rows=[]
    for name,obj in ablations['objectives'].items():
        best=ablations['rotations'][name]
        for i,j in list(combinations(range(proxy.k),2))[:limit]:
            for degrees in np.linspace(-90,90,steps):
                angle=np.deg2rad(degrees)
                B=np.eye(proxy.k)
                B[i,i]=B[j,j]=np.cos(angle)
                B[i,j]=-np.sin(angle);B[j,i]=np.sin(angle)
                for frame,base in [('identity',np.eye(proxy.k)),('optimized',best)]:
                    mat=B@base
                    rows.append(dict(target=name,pair=f'u{i+1}-u{j+1}',
                                     frame=frame,angle_deg=float(degrees),
                                     loss=float(obj.value(mat)),
                                     metric=float(obj.components(mat)[TARGET_KEY[name]])))
    return rows


def save_ablations(folder, proxy, ablations, main_R, config):
    """Save all alternative R/U BEFORE reading true theta for TEST diagnostics."""
    from . import experiment
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder)
    with np.load(folder/'rotation__matrix.npz',allow_pickle=False) as f:
        A_main=np.asarray(f['A'])
        b_main=np.asarray(f['offset'])
        Rmain=np.asarray(f['R'])
        pm=np.asarray(f['pca_mean'])
        components=np.asarray(f['pca_components'])
        eigen=np.asarray(f['pca_eigenvalues'])
    with np.load(folder/'rotation__U.npz',allow_pickle=False) as f:
        Umain=np.asarray(f['U'],float)
        tr=np.asarray(f['train_ids'],int)
        val=np.asarray(f['validation_ids'],int)
        te=np.asarray(f['test_ids'],int)
    if not np.allclose(Rmain,main_R,atol=1e-8):
        raise RuntimeError('Main persisted R differs from FIT ablation reference')
    Rdict=ablations['rotations']
    np.savez_compressed(folder/'functional__candidate_rotations.npz',**{
        'R_'+name:R for name,R in Rdict.items()})
    (folder/'functional__ablation_fit.json').write_text(
        json.dumps(dict(optimization=_serialize(ablations['optimization']),
                        predictive_mdl=_serialize(ablations['mdl']),
                        mdl_winner=ablations['mdl_winner'],
                        candidates=list(Rdict),
                        disclaimer='Each R fitted on FIT ZXY only; theta is post-fit TEST'),
                   indent=2,allow_nan=False),encoding='utf-8')
    # The joint U is the exact frozen whitening coordinate under main R,
    # so q = U_main @ R_main, and alternative U = q @ R_alternative.T.
    q_all=Umain@Rmain
    alternative={}
    for name,R in Rdict.items():
        if name=='joint':continue
        sub=folder/f'ablation__{name}'
        sub.mkdir(exist_ok=True)
        U=q_all@R.T
        # Derive A and offset from the already saved main transformation.
        A=R@Rmain.T@A_main
        b=R@Rmain.T@b_main
        if not np.allclose(U,Umain@(Rmain@R.T),atol=1e-7):
            raise RuntimeError('Alternative U coordinate orientation error')
        np.savez_compressed(sub/'rotation__matrix.npz',
                            R=R,A=A,offset=b,pca_mean=pm,
                            pca_components=components,pca_eigenvalues=eigen,
                            variant=np.array(name))
        np.savez_compressed(sub/'rotation__U.npz',U=U,
                            train_ids=tr,validation_ids=val,test_ids=te)
        alternative[name]=(sub,U)
    # The already-frozen TEST-only truth snapshot is accessed here,
    # after all candidate R/U matrices have been persisted. No theta is
    # available to the earlier fitting or MDL selection functions.
    report={}
    if not json.loads((folder/'rotation__complete.json').read_text())['evaluated']:
        report['warning']='TEST evaluation disabled'
    else:
        meta=json.loads((folder/'rotation__fit.json').read_text())
        model=meta['model']
        with np.load(folder/'rotation__test_truth.npz',allow_pickle=False) as audit:
            truth_test=np.asarray(audit['theta_test'],float)
            audit_ids=np.asarray(audit['test_ids'],int)
        if not np.array_equal(audit_ids,te):
            raise RuntimeError('Frozen TEST truth IDs do not match rotation U IDs')
        truth=np.zeros((len(Umain),truth_test.shape[1]),dtype=float)
        truth[te]=truth_test
        report['joint']=json.loads((folder/'rotation__test_evaluation.json').read_text())
        for name,(sub,U) in alternative.items():
            report[name]=experiment.evaluate_after_fit(
                truth,U,te,sub,model,f'functional_composition__{name}')
            (sub/'ablation__complete.json').write_text(
                json.dumps(dict(name=name,package='functional_composition',
                                evaluated=True),indent=2),encoding='utf-8')
    (folder/'functional__ablation_test.json').write_text(
        json.dumps(_serialize(report),indent=2,allow_nan=False),encoding='utf-8')
    data=[]
    for name,rot in Rdict.items():
        row=dict(candidate=name,
                 mdl_nats_per_trial=ablations['mdl'].get(name,{}).get('objective'),
                 mdl_nonlinear_gain=ablations['mdl'].get(name,{}).get('nonlinear_prediction_gain_nats_per_trial'),
                 matched_test_r2=report.get(name,{}).get('matched_mean_r2'))
        data.append(row)
    with (folder/'functional__ablation_comparison.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(data[0]))
        writer.writeheader();writer.writerows(data)
    # Per-candidate MDL audit: why did nonlinear degree-2/3 terms lose?
    # This includes rejected candidates, rather than just the accepted trace.
    mdl_decisions=[]
    for name,mdi in ablations['mdl'].items():
        for fold in mdi.get('folds',[]):
            for decision in fold.get('module_decisions',[]):
                mdl_decisions.append(dict(rotation_candidate=name,
                                          forward_fold=fold['fold'],
                                          prefix_subjects=fold['prefix_subjects'],
                                          inner_validation_subjects=fold['inner_validation_subjects'],
                                          nonlinear_term='*'.join('u'+str(i+1) for i in decision['term']),
                                          **{k:v for k,v in decision.items() if k!='term'}))
    if mdl_decisions:
        with (folder/'functional__mdl_candidate_decisions.csv').open('w',newline='',encoding='utf-8') as fh:
            w=csv.DictWriter(fh,fieldnames=list(mdl_decisions[0]))
            w.writeheader();w.writerows(mdl_decisions)
    if ablations['objectives']:
        sweep=_sweep_ablation(proxy,ablations,config)
        with (folder/'functional__ablation_plane_curves.csv').open('w',newline='',encoding='utf-8') as fh:
            writer=csv.DictWriter(fh,fieldnames=list(sweep[0]))
            writer.writeheader();writer.writerows(sweep)
        if config.get('make_plots',True):
            pairs=list(dict.fromkeys(row['pair'] for row in sweep))
            fig,axs=plt.subplots(len(TARGETS),len(pairs),
                                 figsize=(3.2*len(pairs),2.7*len(TARGETS)),
                                 squeeze=False)
            for row_idx,name in enumerate(TARGETS):
                for col_idx,pair in enumerate(pairs):
                    ax=axs[row_idx,col_idx]
                    for frame in ('identity','optimized'):
                        sub=[r for r in sweep if r['target']==name and r['pair']==pair and r['frame']==frame]
                        ax.plot([r['angle_deg'] for r in sub],[r['loss'] for r in sub],label=frame)
                    ax.set_title(name+' '+pair)
                    if row_idx==len(TARGETS)-1:ax.set_xlabel('Givens angle (degrees)')
                    if col_idx==0:ax.set_ylabel('Normalized objective')
                    if row_idx==0 and col_idx==0:ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(folder/'functional__ablation_plane_curves.png',dpi=160)
            plt.close(fig)
    return report
