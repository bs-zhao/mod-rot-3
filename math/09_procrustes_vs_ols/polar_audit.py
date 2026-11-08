"""Stage 3: exact polar decomposition of supervised OLS in TRAIN-whitened Z.

Identifies how much cross-axis mixing (H off diagonal) is required after
the best orthogonal Procrustes alignment. Includes paired TEST subject
bootstrap CIs. Reads frozen Stage-1 results; never refits OLS, NP, or TEST.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run import MODELS, NAMES, diagnostics, number_tag

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_BETA={'C1':.0005,'C2':.0005,'C3':.0005,'CR3':.00005}
METRICS=('mean_r2','abs_diag_mean','abs_diag_min','abs_offdiag_mean','abs_offdiag_max','diag_offdiag_gap')

def condition(beta,sparse,lo,hi,d):
    return (f'beta{number_tag(beta)}_lambda{number_tag(sparse)}_'
            f'ctx{number_tag(lo)}to{number_tag(hi)}_L{d}')

def polar_matrices(Z,T,train,whitening):
    """Closed-form TRAIN OLS/procrustes; all arrays use row-vector convention."""
    Z=np.asarray(Z,float);T=np.asarray(T,float)
    idx=np.asarray(train,int);M=np.asarray(whitening,float)
    n,d=Z.shape
    if T.shape!=(n,d) or M.shape!=(d,d):
        raise ValueError('Expected equal-dimensional standardized Z/T and square whitening')
    W=Z@M
    G=W[idx].T@W[idx]/len(idx)
    if not np.allclose(G,np.eye(d),atol=2e-7):
        raise ValueError('TRAIN whitened covariance not identity: max deviation '
                         +str(np.max(np.abs(G-np.eye(d)))))
    C=W[idx].T@T[idx]/len(idx)
    left,singular,right=np.linalg.svd(C,full_matrices=False)
    Q=left@right
    H=(right.T*singular)@right
    D=np.diag(np.diag(H))
    E=H-D
    if not np.allclose(Q@H,C,rtol=1e-8,atol=1e-9):
        raise AssertionError('Polar decomposition failed')
    pred_ols=W@C
    pred_od=W@Q
    pred_od_scaled=W@Q@D
    mse_ols=float(np.mean((pred_ols[idx]-T[idx])**2))
    mse_od_scaled=float(np.mean((pred_od_scaled[idx]-T[idx])**2))
    observed_gap=mse_od_scaled-mse_ols
    exact_gap=float(np.sum(E*E)/d)
    if not np.allclose(observed_gap,exact_gap,rtol=1e-7,atol=1e-9):
        raise AssertionError(f'Polar MSE gap identity failed: {observed_gap} != {exact_gap}')
    return dict(W=W,C=C,Q=Q,H=H,D=D,E=E,
                pred_ols=pred_ols,pred_od=pred_od,pred_od_scaled=pred_od_scaled,
                singular=singular,det_Q=float(np.linalg.det(Q)),
                train_whiten_cov=G,train_mse_ols=mse_ols,
                train_mse_orthogonal_scaled=mse_od_scaled,
                train_mse_gap=observed_gap,
                exact_gap=exact_gap,
                offdiag_H_energy_fraction=float(np.sum(E*E)/max(np.sum(H*H),1e-24)),
                offdiag_H_norm_ratio=float(np.linalg.norm(E)/max(np.linalg.norm(H),1e-12)),
                max_abs_H_offdiag=float(np.max(abs(E))),
                h_condition=float(np.linalg.cond(H)))

def metric_summary(U,T,ids):
    metric=diagnostics(U[ids],T[ids])
    output={key:float(metric[key]) for key in METRICS}
    output['per_parameter_r2']=metric['r2'].tolist()
    output['per_parameter_r']=metric['signed_diag'].tolist()
    output['correlation_matrix']=metric['corr'].tolist()
    output['uu_correlation_matrix']=metric['uu_corr'].tolist()
    return output

def bootstrap_methods(predictions,T,test_ids,reference='direct_ols_oracle',
                      reps=1000,seed=31):
    """Paired TEST-subject bootstrap. Transformations remain frozen at all draws."""
    if reps<1:raise ValueError('--bootstrap-reps must be positive')
    rng=np.random.default_rng(seed)
    idx=np.asarray(test_ids,int)
    names=list(predictions)
    if reference not in predictions:raise ValueError('Missing baseline')
    out={name:{key:np.empty(reps,dtype=float) for key in METRICS}
         for name in names}
    dim=T.shape[1]
    per={name:np.empty((reps,dim),dtype=float) for name in names}
    for b in range(reps):
        draw=rng.choice(idx,len(idx),replace=True)
        tru=T[draw]
        for name,U in predictions.items():
            met=diagnostics(U[draw],tru)
            for key in METRICS:out[name][key][b]=met[key]
            per[name][b]=met['r2']
    rows=[]
    detail=[]
    for name in names:
        for key in METRICS:
            v=out[name][key]
            diff=v-out[reference][key]
            lo,hi=np.percentile(diff,[2.5,97.5])
            rows.append(dict(method=name,reference=reference,metric=key,
                             diff_mean=float(np.mean(diff)),
                             ci95_lower=float(lo),ci95_upper=float(hi),
                             sign_consistent=bool(lo>0 or hi<0),
                             bootstrap_reps=reps))
        for j in range(dim):
            difference=per[name][:,j]-per[reference][:,j]
            lo,hi=np.percentile(difference,[2.5,97.5])
            detail.append(dict(method=name,parameter_index=j+1,
                               delta_test_r2=float(np.mean(difference)),
                               ci95_lower=float(lo),ci95_upper=float(hi)))
    return rows,detail

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def matrix_csv(path,M,names):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['axis',*names])
        for i,line in enumerate(M):
            writer.writerow([names[i],*[float(x) for x in line]])

def analyze(model,source,output,bootstrap_reps=1000,seed=31,plots=True):
    """Read unmodified Stage-1 geometry; audit POLAR decomposition on TRAIN."""
    source=Path(source);output=Path(output)
    with np.load(source/'mappings_and_U.npz',allow_pickle=False) as data:
        Z=np.asarray(data['Z_std'],float)
        T=np.asarray(data['T_std'],float)
        whitening=np.asarray(data['whitening'],float)
        ids={k:np.asarray(data[k+'_ids'],int) for k in ('train','validation','test')}
        saved_B=np.asarray(data['ols_affine__B'],float)
        saved_B_white=np.asarray(data['whiten_procrustes_od_scaled__B'],float)
        saved_oracle_pred=np.asarray(data['ols_affine__U_std'],float)
        saved_white_pred=np.asarray(data['whiten_procrustes_od_scaled__U_std'],float)
        saved_stage1_Q=np.asarray(data['whiten_procrustes_od__B'],float)
    if Z.shape!=T.shape or Z.shape[1]!=MODELS[model]:
        raise ValueError('Model/Z/T dimension mismatch')
    if not np.array_equal(np.sort(np.r_[ids['train'],ids['validation'],ids['test']]),np.arange(len(Z))):
        raise ValueError('Invalid or overlapping subject splits')
    pol=polar_matrices(Z,T,ids['train'],whitening)
    if not np.allclose(whitening@pol['C'],saved_B,rtol=1e-6,atol=1e-7):
        raise AssertionError('C does not reproduce saved original direct OLS coefficients')
    if not np.allclose(pol['pred_ols'],saved_oracle_pred,rtol=1e-6,atol=1e-7):
        raise AssertionError('Polar OLS predictions do not match saved direct OLS Oracle')
    if not np.allclose(whitening@pol['Q'],saved_stage1_Q,rtol=1e-6,atol=1e-7):
        raise AssertionError('Polar Q does not reproduce Stage-1 whitened Procrustes')
    if not np.allclose(whitening@pol['Q']@pol['D'],saved_B_white,rtol=1e-6,atol=1e-7):
        raise AssertionError('Polar scaled Q disagrees with Stage-1 saved transformation')
    if not np.allclose(pol['pred_od_scaled'],saved_white_pred,rtol=1e-6,atol=1e-7):
        raise AssertionError('Polar scaled predictions mismatch')
    predictions={
        'direct_ols_oracle':pol['pred_ols'],
        'whiten_procrustes_od':pol['pred_od'],
        'whiten_procrustes_od_scaled':pol['pred_od_scaled']
    }
    stage2=source/'stage2_joint_orthogonal_scaled'/'joint_orthogonal_scaled.npz'
    if stage2.is_file():
        with np.load(stage2,allow_pickle=False) as f:
            predictions['joint_orthogonal_scaled']=Z@np.asarray(f['B_joint_orthogonal_scaled'],float)
    test={name:metric_summary(u,T,ids['test']) for name,u in predictions.items()}
    rows,detail=bootstrap_methods(predictions,T,ids['test'],reps=bootstrap_reps,seed=seed)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'bootstrap_metric_deltas_vs_ols.csv',rows)
    for row in detail:row['parameter']=NAMES[model][row['parameter_index']-1]
    write_csv(output/'bootstrap_per_parameter_r2_deltas.csv',detail)
    for name,matrix in [('C_OLS_whitened',pol['C']),('H_symmetric_scale_mixing',pol['H']),
                        ('D_diagonal_scaling',pol['D']),('E_necessary_extra_mixing',pol['E']),
                        ('Q_orthogonal',pol['Q'])]:
        matrix_csv(output/f'{name}.csv',matrix,NAMES[model])
    np.savez_compressed(output/'polar_matrices.npz',
                        C=pol['C'],H=pol['H'],D=pol['D'],E=pol['E'],Q=pol['Q'],
                        train_ids=ids['train'],test_ids=ids['test'])
    summary={
        'model':model,'source':str(source),'samples':{k:len(v) for k,v in ids.items()},
        'oracle_untouched':True,
        'checks':dict(ols_reconstruction=True,whiten_procrustes_reconstruction=True,
                      msegap_exact=True),
        'geometry':{
            'C_OLS_whitened':pol['C'].tolist(),
            'Q_orthogonal':pol['Q'].tolist(),
            'H_symmetric':pol['H'].tolist(),
            'H_diagonal_scales':np.diag(pol['D']).tolist(),
            'E_H_offdiagonal':pol['E'].tolist(),
            'singular_values':pol['singular'].tolist(),
            'det_Q':pol['det_Q'],
            'H_condition':pol['h_condition'],
            'H_offdiag_energy_fraction':pol['offdiag_H_energy_fraction'],
            'H_offdiag_norm_ratio':pol['offdiag_H_norm_ratio'],
            'H_max_abs_offdiag':pol['max_abs_H_offdiag'],
            'train_ols_mse':pol['train_mse_ols'],
            'train_whiten_procrustes_scaled_mse':pol['train_mse_orthogonal_scaled'],
            'train_mse_gap_observed':pol['train_mse_gap'],
            'train_mse_gap_theory':pol['exact_gap']
        },
        'test_methods':test,
        'bootstrap':dict(reps=bootstrap_reps,seed=seed,
                         design='paired subjects with replacement, fixed already-fitted maps',
                         delta_reference='direct_ols_oracle',
                         ci='percentile 2.5–97.5%; descriptive, no multiplicity adjustment')
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d=MODELS[model]
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        vmax=float(np.max(abs(pol['H'])))
        vdiff=float(np.max(abs(pol['E']))) or 1.
        for ax,key,limit,label in zip(axes,('H','E'),(vmax,vdiff),
                                      ('H: scaling + mixing','E: extra mixing only')):
            matrix=pol[key]
            im=ax.imshow(matrix,cmap='coolwarm',vmin=-limit,vmax=limit)
            ax.set_xticks(range(d),NAMES[model],rotation=35,ha='right')
            ax.set_yticks(range(d),NAMES[model])
            ax.set_title(label)
            for i in range(d):
                for j in range(d):
                    val=matrix[i,j]
                    ax.text(j,i,f'{val:+.3f}',ha='center',va='center',
                            fontsize=8,color='white' if abs(val)>.62*limit else 'black')
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f'{model}: exact polar decomposition of whitened OLS')
        fig.tight_layout()
        fig.savefig(output/'polar_H_and_extra_mixing.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,key,title in zip(axes,('mean_r2','abs_diag_mean','abs_offdiag_max'),
                                ('TEST mean R2','TEST mean |diag r|','TEST max wrong |r|')):
            ax.bar(range(len(test)),[v[key] for v in test.values()])
            ax.set_xticks(range(len(test)),list(test),rotation=35,ha='right',fontsize=8)
            ax.set_title(title)
        fig.tight_layout();fig.savefig(output/'test_comparison.png',dpi=160);plt.close(fig)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=tuple(MODELS),required=True)
    p.add_argument('--beta',type=float,default=None)
    p.add_argument('--latent-dim',type=int,default=None)
    p.add_argument('--lambda-sparse',type=float,default=0.)
    p.add_argument('--context-min',type=float,default=.25)
    p.add_argument('--context-max',type=float,default=.8)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--bootstrap-reps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=31)
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    if a.beta is None:a.beta=DEFAULT_BETA[a.model]
    if a.latent_dim is None:a.latent_dim=MODELS[a.model]
    if a.latent_dim!=MODELS[a.model]:p.error('Model/latent dimension mismatch')
    if a.bootstrap_reps<1:p.error('Bootstrap reps must be >=1')
    if a.source is None:
        a.source=(ROOT/'outputs'/'math'/'09_procrustes_vs_ols'/a.model/
                  condition(a.beta,a.lambda_sparse,a.context_min,a.context_max,a.latent_dim)/
                  f'repeat{a.repeat:02d}')
    if a.output is None:a.output=a.source/'stage3_polar_audit'
    r=analyze(a.model,a.source,a.output,bootstrap_reps=a.bootstrap_reps,
              seed=a.seed,plots=not a.no_plots)
    g=r['geometry']
    print(f'[{a.model}] exact polar decomposition TRAIN check passed')
    print(f" H offdiag energy fraction: {g['H_offdiag_energy_fraction']:.8f}")
    print(f" H offdiag relative Frobenius norm: {g['H_offdiag_norm_ratio']:.6f}")
    print(f" TRAIN MSE gap (Procrustes+scale minus OLS) "
          f"observed={g['train_mse_gap_observed']:.9g} "
          f"analytic={g['train_mse_gap_theory']:.9g}")
    for name,m in r['test_methods'].items():
        print(f" TEST {name:35s} R2={m['mean_r2']:.4f} "
              f"diag|r|={m['abs_diag_mean']:.4f} wrong-max={m['abs_offdiag_max']:.4f}")
    print('Bootstrap reports and H/E matrices:',a.output)

if __name__=='__main__':main()
