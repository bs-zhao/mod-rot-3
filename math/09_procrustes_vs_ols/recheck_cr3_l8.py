"""Optional exploratory CR3 L8 baseline; NOT the confirmed L4 Oracle run.

This optional script targets CR3 beta=0.0001, latent_dim=8, theta_dim=4.\nThe confirmed L4 CR3 Oracle uses beta=0.00005 and the main run.py.
Direct OLS, S7-equivalent Ridge, and rectangular orthonormal Procrustes
are compared on the SAME frozen Z, theta, and subject splits. The original
L4 Procrustes results are left untouched. True theta is used only as an
ORACLE evaluation/control; not an unsupervised axis-discovery method.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
THIS=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'np'/'src'))
sys.path.insert(0,str(THIS))
from modrot3.pipeline import dataset_path, sha256
from modrot3.analysis import load_frozen, ridge_probe
from run import diagnostics, number_tag, plot_heatmap, plot_scatter

NAMES=['alpha','lambda','gamma','tau']

def condition(args,dim):
    return (f'beta{number_tag(args.beta)}_lambda{number_tag(args.lambda_sparse)}_'
            f'ctx{number_tag(args.context_min)}to{number_tag(args.context_max)}_L{dim}')

def scaled_axis_fit(Zs,Ts,idx,Q):
    """Fit separate output scale on TRAIN; orthonormal Q shape (8,4)."""
    projected=Zs[idx]@Q
    den=np.sum(projected**2,axis=0)
    if np.any(den <= 1e-12):
        raise ValueError('Degenerate Procrustes projection')
    scale=np.sum(projected*Ts[idx],axis=0)/den
    return Q*scale[None,:],scale

def rectangular_procrustes(z,t):
    """Constrained minimizer ||z Q - t||_F with Q.T Q=I_4; Q is 8x4."""
    u,s,vh=np.linalg.svd(z.T@t,full_matrices=False)
    Q=u@vh
    assert np.allclose(Q.T@Q,np.eye(t.shape[1]),atol=1e-8)
    return Q

def as_json(v):
    if isinstance(v,dict):return {str(k):as_json(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [as_json(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,(np.float32,np.float64)):return float(v)
    if isinstance(v,(np.int32,np.int64)):return int(v)
    return v

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def run_analysis(z,t,ids,output,*,s5_source='synthetic',original_s7=None,
                 l4_summary=None,plots=True):
    """Pure-array testable main calculation (Z 8D, T 4D)."""
    z=np.asarray(z,float);t=np.asarray(t,float)
    if z.ndim!=2 or t.shape!=(len(z),4) or z.shape[1]!=8:
        raise ValueError(f'Expected L8 Z and four-column T: got {z.shape} and {t.shape}')
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(t)):
        raise ValueError('Nonfinite inputs')
    tr,va,te=[np.asarray(ids[k],int) for k in ('train','validation','test')]
    if not np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(len(z))):
        raise ValueError('Invalid subject splits')
    fit_ids=np.r_[tr,va]
    mu_z=z[tr].mean(0);sd_z=z[tr].std(0)
    mu_t=t[tr].mean(0);sd_t=t[tr].std(0)
    if np.min(sd_z)<1e-10 or np.min(sd_t)<1e-10:
        raise ValueError('Near-constant variable in TRAIN')
    zs=(z-mu_z)/sd_z;ts=(t-mu_t)/sd_t
    B_train=np.linalg.lstsq(zs[tr],ts[tr],rcond=None)[0]
    # Unlike TRAIN, TRAIN+VALIDATION is not necessarily centered. Fit its intercept.
    coef_fitval=np.linalg.lstsq(np.column_stack((zs[fit_ids],np.ones(len(fit_ids)))),
                                ts[fit_ids],rcond=None)[0]
    B_fitval=coef_fitval[:8]
    b_fitval=coef_fitval[8]
    Q=rectangular_procrustes(zs[tr],ts[tr])
    B_scaled,scale=scaled_axis_fit(zs,ts,tr,Q)
    pcs=np.linalg.svd(zs[tr],full_matrices=False)[2][:4].T
    # PCA uses only TRAIN, and exactly 4 components (8D data -> 4D).
    Zpc=zs@pcs
    B_pc=np.linalg.lstsq(Zpc[tr],ts[tr],rcond=None)[0]
    ridge=ridge_probe(z,t,tr,va,te)  # EXACT existing S7 protocol
    ridge_pred=(ridge['theta_linear_pred']-mu_t)/sd_t

    U=dict(
        direct_ols_train=zs@B_train,
        direct_ols_trainval=zs@B_fitval+b_fitval,
        s7_exact_ridge=np.full_like(ts,np.nan),
        rectangular_procrustes=zs@Q,
        rectangular_procrustes_scaled=zs@B_scaled,
        pca4_ols_train=Zpc@B_pc,
    )
    U['s7_exact_ridge'][te]=ridge_pred

    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows=[];per=[];summary=dict(
        input_shapes=dict(z=list(z.shape),t=list(t.shape)),
        source=str(s5_source),
        train_subjects=int(len(tr)),validation_subjects=int(len(va)),
        test_subjects=int(len(te)),
        target='Oracle CR3 theta on held-out TEST; U dimension always four',
        methods={},prior_s7=None,
        definitions={
            'direct_ols_train':'TRAIN OLS from all 8 raw posterior-mean Z to all 4 T, including intercept in original units',
            'direct_ols_trainval':'OLS on TRAIN+VALIDATION with TRAIN normalization, held-out TEST',
            's7_exact_ridge':'Existing modrot3.analysis.ridge_probe: alpha selected on VALIDATION, fit TRAIN+VAL, TEST',
            'rectangular_procrustes':'8x4 Stiefel matrix Q.T Q = I4, NOT a 4x4 rotation',
            'rectangular_procrustes_scaled':'8x4 Procrustes plus TRAIN-fitted four axis scales',
            'pca4_ols_train':'TRAIN PCA of 8D Z to four PCs, then TRAIN OLS to T',
        },
        normalizers=dict(z_mean=mu_z,z_std=sd_z,t_mean=mu_t,t_std=sd_t),
        matrices=dict(
            B_direct_ols_train=B_train,B_direct_ols_trainval=B_fitval,
            b_direct_ols_trainval_std=b_fitval,
            Q_rectangular_procrustes=Q,B_rectangular_scaled=B_scaled,
            procrustes_scale=scale,PC4_basis=pcs,B_PC4_to_Tstd=B_pc,
            A_raw_Z_to_T=sd_t[:,None]*(B_train.T/sd_z[None,:]),
            b_raw_Z_to_T=mu_t-(sd_t[:,None]*(B_train.T/sd_z[None,:]))@mu_z,
        ),
        validation_selected_ridge_alpha=ridge['selected_ridge_alpha'],
        validation_ridge_r2=ridge['validation_r2']
    )
    for method,ui in U.items():
        ut=ui[te]
        met=diagnostics(ut,ts[te])
        summary['methods'][method]=dict(
            test_metrics={k:met[k] for k in ('abs_diag_mean','abs_diag_min',
            'abs_offdiag_mean','abs_offdiag_max','diag_offdiag_gap','mean_r2','mean_rmse')},
            test_correlation=met['corr'],test_UU_correlation=met['uu_corr'],
            per_parameter_test_r2=met['r2'],
            per_parameter_test_r=met['signed_diag'],
        )
        row=dict(method=method,**summary['methods'][method]['test_metrics'])
        rows.append(row)
        for j,name in enumerate(NAMES):
            per.append(dict(method=method,parameter=name,
                           r=float(met['signed_diag'][j]),
                           abs_r=float(abs(met['signed_diag'][j])),
                           r2=float(met['r2'][j]),rmse_std=float(met['rmse'][j]),
                           max_abs_wrong_r=float(np.max(np.abs(np.delete(met['corr'][j],j))))))
        corr_path=output/f'test_corr__{method}.csv'
        with corr_path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['U / T',*NAMES])
            for j,r in enumerate(met['corr']):
                w.writerow([f'U{j+1}',*[float(v) for v in r]])
        if plots:
            plot_heatmap(output/f'test_corr__{method}.png',met['corr'],NAMES,method)
            plot_scatter(output/f'test_scatter__{method}.png',ut,ts[te],NAMES,method)
    write_csv(output/'test_methods.csv',rows)
    write_csv(output/'test_per_parameter.csv',per)
    if original_s7 is not None and Path(original_s7).is_file():
        old=json.loads(Path(original_s7).read_text(encoding='utf-8'))
        old_r2=old.get('full_z_ridge',{}).get('test_r2')
        if old_r2 is not None:
            diff=np.asarray(ridge['test_r2'])-np.asarray(old_r2,dtype=float)
            summary['prior_s7']=dict(path=str(original_s7),stored_test_r2=old_r2,
                                     recomputed_test_r2=ridge['test_r2'],
                                     max_abs_r2_difference=float(np.max(abs(diff))),
                                     exact_match=bool(np.allclose(diff,0,rtol=0,atol=1e-8)))
    if l4_summary is not None and Path(l4_summary).is_file():
        l4=json.loads(Path(l4_summary).read_text(encoding='utf-8'))
        v=l4['test_primary']['ols_affine']
        r=summary['methods']['direct_ols_train']['test_metrics']
        summary['l4_comparison']=dict(path=str(l4_summary),
            l4_ols_mean_test_r2=v['mean_r2'],
            l8_ols_mean_test_r2=r['mean_r2'],
            delta_l8_minus_l4=float(r['mean_r2']-v['mean_r2']),
            l4_ols_mean_abs_diag_r=v['abs_diag_mean'],
            l8_ols_mean_abs_diag_r=r['abs_diag_mean'],
            l4_ols_max_offdiag_r=v['abs_offdiag_max'],
            l8_ols_max_offdiag_r=r['abs_offdiag_max'],
            warning='Models have DIFFERENT NP architectures/checkpoints; not a paired feature ablation.')
    np.savez_compressed(output/'oracle_maps_and_test_predictions.npz',
        train_ids=tr,validation_ids=va,test_ids=te,
        z_train_mean=mu_z,z_train_std=sd_z,t_train_mean=mu_t,t_train_std=sd_t,
        B_direct_ols_train=B_train,B_direct_ols_trainval=B_fitval,
        b_direct_ols_trainval_std=b_fitval,
        Q_rectangular_procrustes=Q,B_rectangular_procrustes_scaled=B_scaled,
        test_T_true=t[te],**{'test_U_std__'+k:v[te] for k,v in U.items()})
    (output/'summary.json').write_text(json.dumps(as_json(summary),indent=2)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--beta',type=float,default=.0001)
    p.add_argument('--latent-dim',type=int,default=8)
    p.add_argument('--lambda-sparse',type=float,default=0.)
    p.add_argument('--context-min',type=float,default=.25)
    p.add_argument('--context-max',type=float,default=.80)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    args.model='CR3'
    if args.latent_dim!=8:p.error('Historical CR3 Oracle audit REQUIRES --latent-dim 8')
    if args.beta<0 or args.repeat<1:p.error('Invalid beta/repeat')
    z,t,splits,source=load_frozen(args)
    with np.load(source,allow_pickle=False) as f:
        if str(f['dataset_sha256'])!=sha256(dataset_path('CR3')):
            raise ValueError('S5 dataset hash mismatch')
    cond=condition(args,8)
    if args.output is None:
        output=ROOT/'outputs'/'math'/'09_procrustes_vs_ols'/'CR3'/cond/f'repeat{args.repeat:02d}'/'oracle_l8_recheck'
    else:output=args.output
    s7file=ROOT/'outputs'/'choice'/'CR3'/'s12_z_diagnostics'/cond/f'repeat{args.repeat:02d}'/'s12__diagnostics.json'
    l4file=(ROOT/'outputs'/'math'/'09_procrustes_vs_ols'/'CR3'/
            condition(args,4)/f'repeat{args.repeat:02d}'/'summary.json')
    report=run_analysis(z,t,splits,output,s5_source=source,
                        original_s7=s7file,l4_summary=l4file,
                        plots=not args.no_plots)
    print('CR3 beta',args.beta,'L8 Z->L4 T. RESULTS:',output)
    for k,m in report['methods'].items():
        a=m['test_metrics']
        print(f'  {k:34s} mean TEST R2={a["mean_r2"]:.4f} '
              f'diag|r|={a["abs_diag_mean"]:.4f} max offdiag={a["abs_offdiag_max"]:.4f}')
        print('     r per theta:',np.round(m['per_parameter_test_r'],4),
              'R2 per theta:',np.round(m['per_parameter_test_r2'],4))
    if report['prior_s7'] is not None:
        print('Old S7 full-Z Ridge exact_match:',report['prior_s7']['exact_match'])
    if 'l4_comparison' in report:
        print('L8 TRAIN OLS minus earlier L4 OLS mean TEST R2:',
              report['l4_comparison']['delta_l8_minus_l4'])

if __name__=='__main__':
    main()
