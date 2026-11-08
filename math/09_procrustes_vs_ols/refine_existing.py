"""Stage 2: TRAIN-only joint orthogonal-direction + per-axis-scale optimization.

This is an independent, non-destructive analysis of outputs from
math/09_procrustes_vs_ols/run.py. The original DIRECT OLS ORACLE is read
unchanged and is always the first comparison baseline. NO NP retraining.
NO use of TEST targets to fit directions, scales, or choose best starts.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from run import NAMES, MODELS, diagnostics, number_tag

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_BETA={'C1':.0005,'C2':.0005,'C3':.0005,'CR3':.00005}

def tag(beta,sparse,cmin,cmax,d):
    return (f'beta{number_tag(beta)}_lambda{number_tag(sparse)}_'
            f'ctx{number_tag(cmin)}to{number_tag(cmax)}_L{d}')

def profile(Q, G, C, y2):
    """Best signed scale per column, and exact TRAIN mean squared loss."""
    denom=np.diag(Q.T@G@Q)
    if np.any(denom<=1e-12):
        raise ValueError('Invalid projected variance')
    numer=np.diag(Q.T@C)
    s=numer/denom
    d=len(s)
    loss=max(0.,float((y2-np.sum(numer*numer/denom))/d))
    return loss,s

def fit_joint(z,t,initial,B_ols,*,seed=31,restarts=2,maxiter=300):
    """Min TRAIN MSE over B = Q diag(s), Q in O(d), signed real scales s.

    Each optimizer chart Q0 expm(K) stays in the same determinant component.
    Because signs of s are unconstrained, the full effective model includes
    both connected components of orthogonal transformations.
    """
    z=np.asarray(z,float);t=np.asarray(t,float)
    n,d=z.shape
    if t.shape!=(n,d) or B_ols.shape!=(d,d) or initial.shape!=(d,d):
        raise ValueError('Bad optimization input shapes')
    if restarts<0 or maxiter<1:
        raise ValueError('Invalid iteration count')
    G=z.T@z/n;C=z.T@t/n;y2=float(np.sum(t*t)/n)
    inds=np.triu_indices(d,k=1)
    def skew(x):
        K=np.zeros((d,d))
        K[inds]=x
        K[inds[1],inds[0]]=-x
        return K
    u,_,vh=np.linalg.svd(B_ols,full_matrices=False)
    polar=u@vh
    rng=np.random.default_rng(seed)
    starts=[('procrustes',initial),('ols_polar',polar)]
    starts += [(f'jitter_{i+1}',initial@expm(skew(rng.normal(0,.25,len(inds[0])))))
               for i in range(restarts)]
    best=None
    audit=[]
    for name,Q0 in starts:
        initial_loss,_=profile(Q0,G,C,y2)
        result=minimize(lambda x:profile(Q0@expm(skew(x)),G,C,y2)[0],
                        np.zeros(len(inds[0])),method='L-BFGS-B',
                        options=dict(maxiter=maxiter,ftol=1e-12))
        Q=Q0@expm(skew(result.x))
        loss,scale=profile(Q,G,C,y2)
        audit.append(dict(start=name,initial_loss=initial_loss,final_loss=loss,
                          success=bool(result.success),iterations=int(result.nit),
                          evaluations=int(result.nfev)))
        if best is None or loss<best['train_loss']:
            best=dict(Q=Q,scales=scale,train_loss=loss,start=name)
    Q=best['Q'];B=Q*best['scales'][None,:]
    actual=float(np.mean((z@B-t)**2))
    if not np.allclose(Q.T@Q,np.eye(d),atol=1e-7):
        raise AssertionError('Q is not orthogonal')
    if abs(actual-best['train_loss'])>1e-8:
        raise AssertionError('Profiled vs direct TRAIN MSE disagreement')
    best.update(B=B,attempts=audit,det_Q=float(np.linalg.det(Q)))
    return best

def compare_geometry(oracle_B,B):
    """Column-wise axis directions and OLS residual in the candidate Q basis."""
    n,d=oracle_B.shape
    Q, _, vh=np.linalg.svd(B,full_matrices=False)
    # B=Q diag(s) only if its columns are mutually orthogonal.
    if not np.allclose(B.T@B,np.diag(np.diag(B.T@B)),atol=1e-7):
        return None
    q=B/np.maximum(np.linalg.norm(B,axis=0,keepdims=True),1e-14)
    C=q.T@oracle_B
    off=C-np.diag(np.diag(C))
    angles=[]
    for j in range(d):
        v=oracle_B[:,j];w=q[:,j]
        cosine=abs(np.dot(v,w))/(np.linalg.norm(v)*np.linalg.norm(w))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine,0,1)))))
    return dict(ols_in_candidate_orthogonal_basis=C.tolist(),
                ols_offdiagonal_mixing_fraction=float(np.linalg.norm(off)/max(np.linalg.norm(C),1e-14)),
                ols_axis_angle_deg=angles)

def csv_rows(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def csv_matrix(path,M,cols):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['U / T',*cols])
        for i,row in enumerate(M):
            w.writerow([f'U{i+1}',*map(float,row)])

def run_analysis(model, source, output, *, plots=True,restarts=2,maxiter=300):
    """Given one Stage-1 folder, preserve Stage-1 OLS and optimize only TRAIN."""
    source=Path(source);output=Path(output)
    old=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if old['model']!=model:raise ValueError('Stage-1 model mismatch')
    d=MODELS[model];names=NAMES[model]
    with np.load(source/'mappings_and_U.npz',allow_pickle=False) as f:
        z=f['Z_std'].copy();t=f['T_std'].copy()
        ids={k:f[k+'_ids'].copy() for k in ('train','validation','test')}
        B_ols=f['ols_affine__B'].copy()
        B_pro=f['procrustes_od__B'].copy()
        B_pro_scale=f['procrustes_od_scaled__B'].copy()
        B_white_scale=f['whiten_procrustes_od_scaled__B'].copy()
    if z.shape!=t.shape or z.ndim!=2 or z.shape[1]!=d or B_ols.shape!=(d,d):
        raise ValueError('Unexpected stored geometry')
    if not np.array_equal(np.sort(np.concatenate(list(ids.values()))),np.arange(len(z))):
        raise ValueError('Invalid subject split provenance')
    tr=ids['train'];te=ids['test']
    original_prediction=z@B_ols
    if not np.allclose(original_prediction,fake_read_old_pred(source,z),atol=1e-9):
        raise ValueError('Saved original direct OLS Oracle does not match original predictions')
    optimized=fit_joint(z[tr],t[tr],B_pro,B_ols,
                        restarts=restarts,maxiter=maxiter)
    candidates={
        'direct_ols_oracle':B_ols,
        'procrustes_od':B_pro,
        'procrustes_od_scaled':B_pro_scale,
        'whiten_procrustes_od_scaled':B_white_scale,
        'joint_orthogonal_scaled':optimized['B'],
    }
    output.mkdir(parents=True,exist_ok=True)
    summary=dict(model=model,source=str(source),n_train=len(tr),n_test=len(te),
                 baseline='direct_ols_oracle',
                 note='Never refits original OLS Oracle; TRUE T used only for supervised control.',
                 fitting_policy='Only original TRAIN subjects influence optimized Q and scales.',
                 joint_fit=dict(train_mse=optimized['train_loss'],
                                starting_point=optimized['start'],det_Q=optimized['det_Q'],
                                Q=optimized['Q'].tolist(),scales=optimized['scales'].tolist(),
                                attempts=optimized['attempts']),
                 models={})
    rows=[];per=[];pairwise=[]
    for method,B in candidates.items():
        U=z@B;test=diagnostics(U[te],t[te]);train=diagnostics(U[tr],t[tr])
        report=dict(B=B.tolist(),test_metrics={},train_metrics={},
                    test_corr=test['corr'].tolist(),
                    test_UU_corr=test['uu_corr'].tolist())
        for split,met in (('test',test),('train',train)):
            report[f'{split}_metrics']={
                k:float(met[k]) for k in
                ('abs_diag_mean','abs_diag_min','abs_offdiag_mean',
                 'abs_offdiag_max','diag_offdiag_gap','mean_r2','mean_rmse')}
        if method=='joint_orthogonal_scaled':
            report['oracle_geometry']=compare_geometry(B_ols,B)
        rows.append(dict(model=model,method=method,
                         **report['test_metrics'],train_r2=report['train_metrics']['mean_r2']))
        for i,name in enumerate(names):
            wrong=np.abs(np.delete(test['corr'][i],i))
            per.append(dict(model=model,method=method,parameter=name,
                            test_r=float(test['signed_diag'][i]),
                            test_abs_r=float(test['abs_diag'][i]),
                            test_r2=float(test['r2'][i]),
                            test_rmse_std=float(test['rmse'][i]),
                            max_wrong_abs_r=float(wrong.max())))
            for j,name_j in enumerate(names):
                if j!=i:
                    pairwise.append(dict(model=model,method=method,
                                         U_axis=f'U{i+1}',T_parameter=name_j,
                                         test_r=float(test['corr'][i,j])))
        csv_matrix(output/f'test_corr__{method}.csv',test['corr'],names)
        csv_matrix(output/f'test_UU_corr__{method}.csv',test['uu_corr'],
                   [f'U{k+1}' for k in range(d)])
        summary['models'][method]=report
    csv_rows(output/'method_comparison.csv',rows)
    csv_rows(output/'per_parameter.csv',per)
    csv_rows(output/'wrong_parameter_correlations.csv',pairwise)
    summary['test_vs_ols']={}
    base=summary['models']['direct_ols_oracle']['test_metrics']
    for method in candidates:
        m=summary['models'][method]['test_metrics']
        summary['test_vs_ols'][method]=dict(
            delta_R2=float(m['mean_r2']-base['mean_r2']),
            delta_abs_diag_r=float(m['abs_diag_mean']-base['abs_diag_mean']),
            delta_max_wrong_r=float(m['abs_offdiag_max']-base['abs_offdiag_max']))
    np.savez_compressed(output/'joint_orthogonal_scaled.npz',
                        Q=optimized['Q'],scales=optimized['scales'],
                        B_joint_orthogonal_scaled=optimized['B'],
                        B_direct_ols_oracle=B_ols,train_ids=tr,
                        test_ids=te)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',
                                       encoding='utf-8')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        methods=list(candidates)
        fig,axs=plt.subplots(1,3,figsize=(17,5))
        for ax,key,title in zip(axs,
                                ('abs_diag_mean','abs_offdiag_max','mean_r2'),
                                ('TEST mean |r(Ui,Ti)|','TEST max off-diagonal |r|','TEST mean R2')):
            ax.bar(range(len(methods)),
                   [summary['models'][m]['test_metrics'][key] for m in methods])
            ax.set_xticks(range(len(methods)),methods,rotation=38,ha='right',fontsize=8)
            ax.set_title(title);ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        fig.savefig(output/'method_comparison.png',dpi=165);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for ax,key,title in zip(axs,('test_r2','test_abs_r'),
                                ('Held-out per-parameter R2','Held-out per-parameter |r|')):
            for method in methods:
                r=[row[key] for row in per if row['method']==method]
                ax.plot(range(d),r,'o-',label=method)
            ax.set_xticks(range(d),names)
            ax.set_title(title);ax.grid(alpha=.15)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output/'parameter_breakdown.png',dpi=165);plt.close(fig)
    return summary

def fake_read_old_pred(source,z):
    with np.load(source/'mappings_and_U.npz',allow_pickle=False) as f:
        return f['ols_affine__U_std'].copy()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=tuple(MODELS))
    p.add_argument('--beta',type=float,default=None)
    p.add_argument('--latent-dim',type=int,default=None)
    p.add_argument('--lambda-sparse',type=float,default=0.)
    p.add_argument('--context-min',type=float,default=.25)
    p.add_argument('--context-max',type=float,default=.80)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--restarts',type=int,default=2)
    p.add_argument('--maxiter',type=int,default=300)
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args()
    if args.beta is None:args.beta=DEFAULT_BETA[args.model]
    if args.latent_dim is None:args.latent_dim=MODELS[args.model]
    if args.latent_dim!=MODELS[args.model]:
        p.error(f'{args.model} must use L{MODELS[args.model]}')
    if args.source is None:
        args.source=ROOT/'outputs'/'math'/'09_procrustes_vs_ols'/args.model/tag(
            args.beta,args.lambda_sparse,args.context_min,args.context_max,args.latent_dim
        )/f'repeat{args.repeat:02d}'
    if args.output is None:args.output=args.source/'stage2_joint_orthogonal_scaled'
    report=run_analysis(args.model,args.source,args.output,
                        plots=not args.no_plots,restarts=args.restarts,maxiter=args.maxiter)
    print(f'[{args.model}] original direct OLS Oracle preserved at {args.source}')
    for method,item in report['models'].items():
        m=item['test_metrics']
        print(f" {method:32s} TEST mean R2={m['mean_r2']:.4f} "
              f"diag|r|={m['abs_diag_mean']:.4f} offdiag-max={m['abs_offdiag_max']:.4f}")
    print('Saved stage-2 diagnostics to',args.output)

if __name__=='__main__':main()
