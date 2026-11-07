"""Independent CR3 benchmark of three statistical Hessian estimators.

At repo root: python math/cr3_hessian_estimators/run.py
Only 4x4 H_TT. All 500 fixed X are averaged *before* regression.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from estimators import split_subjects,fit_krr,fit_spline,fit_stein_mean,second_order_score
from reference import load_cr3,logits_and_hess_tt,truth,prior_quadrature,check_logit_values,NAMES

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'outputs/choice/CR3/s1_generate/s1_generate__dataset.npz'
DEFAULT_OUT=ROOT/'outputs/math/cr3_hessian_estimators'
CHANNELS=('binary','logit','probability')


def score_matrix(pred,true):
    pred=np.asarray(pred);true=np.asarray(true)
    diff=pred-true
    return dict(mae=float(np.mean(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2))),
                max_abs=float(np.max(np.abs(diff))),
                relative_frobenius=float(np.linalg.norm(diff)/max(np.linalg.norm(true),1e-15)),
                cosine=float(np.sum(pred*true)/max(np.linalg.norm(pred)*np.linalg.norm(true),1e-15)))


def dump_matrix(path,matrix):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['']+list(NAMES))
        for label,row in zip(NAMES,matrix):w.writerow([label]+list(np.round(row,12)))


def convert(o):
    if isinstance(o,dict):return {k:convert(v) for k,v in o.items()}
    if isinstance(o,np.ndarray):return convert(o.tolist())
    if isinstance(o,(np.floating,float)):
        x=float(o);return x if np.isfinite(x) else None
    if isinstance(o,(np.integer,int)):return int(o)
    if isinstance(o,(tuple,list)):return [convert(x) for x in o]
    return o


def plot_matrices(out,results,method,expectation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
    for col,ch in enumerate(CHANNELS):
        truth_mat=results[f'{expectation}__truth__{ch}']
        pred_mat=results[f'{expectation}__{method}__{ch}']
        arrays=[truth_mat,pred_mat,pred_mat-truth_mat]
        scale=max(1e-9,*(float(np.max(np.abs(x))) for x in arrays))
        for row,mat in enumerate(arrays):
            ax=axes[row,col]
            im=ax.imshow(mat,vmin=-scale,vmax=scale,cmap='coolwarm')
            ax.set_title(f'{ch}: {("oracle","estimated","difference")[row]}')
            ax.set_xticks(range(4),NAMES,rotation=40,ha='right',fontsize=7)
            ax.set_yticks(range(4),NAMES,fontsize=7)
            for i in range(4):
                for j in range(4):ax.text(j,i,f'{mat[i,j]:.2g}',ha='center',va='center',fontsize=7)
    fig.suptitle(f'CR3 {method}: {expectation} Hessian, theta-theta')
    fig.savefig(out/f'{method}__{expectation}__comparison.png',dpi=130)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=DEFAULT_DATA)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--methods',nargs='+',choices=('krr','spline','stein'),default=['krr','spline','stein'])
    ap.add_argument('--quad-nodes',type=int,default=8)
    ap.add_argument('--nystrom-rank',type=int,default=256)
    ap.add_argument('--spline-basis',type=int,default=6)
    ap.add_argument('--seed',type=int,default=20261106)
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    if not args.data.exists():ap.error(f'Missing {args.data}; first run python np/s1_generate.py --model CR3 --subjects 5000')
    if args.quad_nodes<3 or args.spline_basis<4 or args.nystrom_rank<16:
        ap.error('quad-nodes>=3, spline-basis>=4, nystrom-rank>=16 required')
    args.out.mkdir(parents=True,exist_ok=True)
    print('Reading CR3 dataset...',flush=True)
    T,X,channels,p_saved=load_cr3(args.data)
    print(f'Input: {len(T)} subjects, {len(X)} fixed trials, T shape {T.shape}',flush=True)
    print('Checking exact analytic CR3 probability (separate clipping)...',flush=True)
    check=check_logit_values(T,X,p_saved)
    print('CR3 probability check max average difference:',check,flush=True)
    print('Computing analytic sample Hessian TT...',flush=True)
    truth_sample=truth(T,X)
    channels['logit']=truth_sample['logit'][0]
    # Assert logit truth was computed once and handed to estimators as numbers.
    for ch in CHANNELS:
        if channels[ch].shape!=(len(T),):raise AssertionError(ch)
    print(f'Computing prior quadrature (nodes={args.quad_nodes}^4)...',flush=True)
    quad_T,quad_W,mass_errors=prior_quadrature(args.quad_nodes)
    truth_prior=truth(quad_T,X)
    splits=split_subjects(len(T),args.seed)
    summary=dict(
        data=str(args.data),subjects=len(T),trials=len(X),
        source_probability_check_max=check,
        methods=args.methods,columns=list(NAMES),
        response_channels={'binary':'500-sample mean of actual Bernoulli choices',
                           'probability':'500-sample mean of saved predictive_true probabilities',
                           'logit':'500-sample mean of true pre-sigmoid CR3 logits (not stored by S1)'},
        targets={
            'unweighted_sample':'average of H_TT(T_i,X_t) over N real T and 500 fixed stimuli',
            'unweighted_prior':'integral H_TT(theta,X_t) over true theta prior, average of fixed 500 X',
            'stein_weighted_prior':'E_prior[w(theta) H_TT]/E_prior[w(theta)], a DIFFERENT estimand'},
        split_sizes={k:len(v) for k,v in splits.items()},
        quadrature=dict(nodes_per_dim=args.quad_nodes,number_of_nodes=len(quad_T),
                        marginal_mass_errors=mass_errors),
        important_limits=[
          'Data-only estimators receive only T and 500-trial-averaged Y; no access to the analytic CR3 function.',
          'Data-only models recover the Hessian of the averaged conditional mean; not per-trial Hessian variance.',
          'KRR and spline hyperparameters selected on held-out validation subjects by function RMSE, NOT by oracle Hessian.',
          'Binary analytic comparator is Hessian of the conditional mean p, not of a random 0/1 outcome.',
          'Weighted Stein targets weighted prior Hessian only; unweighted expected Hessian is not obtained without boundary information.',
          'Validation/prediction accuracy does not establish derivative accuracy.'
        ], results={})
    result_arrays={}
    for ch in CHANNELS:
        for label,tr in [('sample',truth_sample),('prior',truth_prior)]:
            mat=np.einsum('n,nij->ij',np.full(len(T),1/len(T)) if label=='sample' else quad_W,tr[ch][1])
            result_arrays[f'{label}__truth__{ch}']=mat
            dump_matrix(args.out/f'{label}__truth__{ch}.csv',mat)

    for method in [m for m in args.methods if m in ('krr','spline')]:
        print(f'Fitting {method} on 500-trial-averaged T,Y...',flush=True)
        if method=='krr':
            models,choices=fit_krr(T,channels,splits,rank=args.nystrom_rank,seed=args.seed)
        else:
            models,choices=fit_spline(T,channels,splits,n_basis=args.spline_basis)
        summary['results'][method]={}
        for ch in CHANNELS:
            print(f' {method}/{ch}: {choices[ch]}',flush=True)
            model=models[ch]
            truth_h_test=truth_sample[ch][1][splits['test']]
            local_pred=model.hessian(T[splits['test']])
            local_rmse=float(np.sqrt(np.mean((local_pred-truth_h_test)**2)))
            local_rel=float(np.linalg.norm(local_pred-truth_h_test)/max(1e-15,np.linalg.norm(truth_h_test)))
            r=dict(selected_hyperparameters=choices[ch],
                   prediction_holdout={'rmse':float(np.sqrt(np.mean((model.predict(T[splits['test']])-channels[ch][splits['test']])**2))),
                                       'mae':float(np.mean(np.abs(model.predict(T[splits['test']])-channels[ch][splits['test']])) )},
                   local_hessian_holdout={'rmse':local_rmse,'relative_frobenius':local_rel},
                   average_hessian={})
            for label,eval_T,weights in (('sample',T,np.full(len(T),1/len(T))),('prior',quad_T,quad_W)):
                pred_h=model.hessian(eval_T)
                estimate=np.einsum('n,nij->ij',weights,pred_h)
                key=f'{label}__{method}__{ch}'
                result_arrays[key]=estimate
                dump_matrix(args.out/f'{key}.csv',estimate)
                r['average_hessian'][label]=score_matrix(estimate,result_arrays[f'{label}__truth__{ch}'])
                print(f'  {label} mean H relative Frobenius error: {r["average_hessian"][label]["relative_frobenius"]:.4g}',flush=True)
            summary['results'][method][ch]=r
        if not args.no_plots:
            for label in ('sample','prior'):
                plot_matrices(args.out,result_arrays,method,label)

    if 'stein' in args.methods:
        print('Estimating weighted average Hessian from prior score (no function fit)...',flush=True)
        W_s,psi_s=second_order_score(T)
        W_q,psi_q=second_order_score(quad_T)
        normal_s=W_s.mean();normal_q=quad_W@W_q
        summary['stein_weight']=dict(sample_mean=float(normal_s),prior_mean=float(normal_q),
                                    formula='w(T)=prod_j 16*z_j^2*(1-z_j)^2, z_j=(T_j-low_j)/(high_j-low_j)')
        summary['results']['stein']={}
        for ch in CHANNELS:
            true_sample=np.einsum('n,nij->ij',W_s/np.sum(W_s),truth_sample[ch][1])
            true_prior=np.einsum('n,nij->ij',quad_W*W_q/normal_q,truth_prior[ch][1])
            est,se,_=fit_stein_mean(T,channels[ch])
            # Quadrature evaluates IBP rhs independently with true conditional
            # mean, to validate the boundary-aware identity numerically.
            yq=truth_prior[ch][0]
            identity=np.einsum('n,n,nij->ij',quad_W,yq,psi_q)/normal_q
            summary['results']['stein'][ch]={
                'sample_estimate_vs_prior_weighted_oracle':score_matrix(est,true_prior),
                'sample_estimate_vs_sample_weighted_oracle':score_matrix(est,true_sample),
                'quadrature_stein_identity_check':score_matrix(identity,true_prior),
                'max_standard_error':float(np.max(se)),
                'standard_errors':se,
                'note':'The identity is for weighted H, not unweighted H; finite-sample estimators need not match pointwise.'}
            for label,mat in (('sample_estimate',est),('sample_oracle',true_sample),
                              ('prior_oracle',true_prior),('prior_identity',identity),('sample_se',se)):
                key=f'stein__{ch}__{label}'
                result_arrays[key]=mat;dump_matrix(args.out/f'{key}.csv',mat)
            print(f'  stein/{ch} vs prior weighted oracle: '
                  f'{summary["results"]["stein"][ch]["sample_estimate_vs_prior_weighted_oracle"]["relative_frobenius"]:.4g}',flush=True)

    with open(args.out/'summary.json','w',encoding='utf-8') as f:
        json.dump(convert(summary),f,indent=2,ensure_ascii=False,allow_nan=False)
    np.savez_compressed(args.out/'all_matrices.npz',**result_arrays)
    print('Saved:',args.out,flush=True)


if __name__=='__main__':main()
