"""Supervised geometry control: OLS affine vs orthogonal Procrustes on frozen NP Z.

All fits/normalizers use TRAIN subjects only; VALIDATION/TEST never fit a map.
No NP training, decoding or Z extraction. True theta is used ONLY for this
supervised oracle comparison, NOT for unsupervised rotation discovery.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = {'C1': 4, 'C2': 4, 'C3': 6, 'CR3': 4}
NAMES = {
    'C1': ['b1', 'b2', 'b3', 'b4'],
    'C2': ['b1', 'b2', 'b3', 'b4'],
    'C3': ['b1', 'b2', 'b3', 'b4', 'b5', 'b6'],
    'CR3': ['alpha', 'lambda', 'gamma', 'tau'],
}

def number_tag(value):
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p').replace('+', '')

def condition(args):
    return (f'beta{number_tag(args.beta)}_lambda{number_tag(args.lambda_sparse)}_'
            f'ctx{number_tag(args.context_min)}to{number_tag(args.context_max)}_L{args.latent_dim}')

def pearson_matrix(U, T):
    U = np.asarray(U, float); T = np.asarray(T, float)
    a = U-U.mean(axis=0); b = T-T.mean(axis=0)
    denom = np.linalg.norm(a, axis=0)[:, None]*np.linalg.norm(b, axis=0)[None, :]
    return np.divide(a.T @ b, denom, out=np.zeros((U.shape[1], T.shape[1])), where=denom > 1e-12)

def prepare(Z, T, train):
    Z = np.asarray(Z, float); T = np.asarray(T, float)
    train = np.asarray(train, int)
    if Z.ndim != 2 or Z.shape != T.shape or len(train) < Z.shape[1]+2:
        raise ValueError('Expected Z and theta same (N,d), with enough TRAIN subjects')
    if not np.isfinite(Z).all() or not np.isfinite(T).all():
        raise ValueError('Non-finite Z or theta')
    zm, tm = Z[train].mean(0), T[train].mean(0)
    zs, ts = Z[train].std(0), T[train].std(0)
    if np.any(zs <= 1e-10) or np.any(ts <= 1e-10):
        raise ValueError('Collapsed Z or theta coordinate; cannot standardize')
    Zs = (Z-zm)/zs
    Ts = (T-tm)/ts
    cov = Zs[train].T @ Zs[train] / len(train)
    eigen, vectors = np.linalg.eigh(cov)
    if eigen[0] <= 1e-9:
        raise ValueError('Near-singular TRAIN Z covariance; whitening unreliable')
    white = (vectors / np.sqrt(eigen)) @ vectors.T
    return Zs, Ts, dict(z_mean=zm, z_sd=zs, t_mean=tm, t_sd=ts,
                        z_cov=cov, z_eigen=eigen, whitening=white)

def procrustes(X, Y, proper=False):
    """Fit Q minimizing ||X Q - Y||_F, Q.T Q=I; proper enforces det(Q)=+1."""
    left, singular, right = np.linalg.svd(X.T @ Y, full_matrices=False)
    signs = np.ones(len(singular))
    if proper and np.linalg.det(left @ right) < 0:
        signs[-1] = -1
    Q = (left * signs) @ right
    return Q, singular

def fit_all(Zs, Ts, train, whitening):
    z, t = Zs[train], Ts[train]
    B_ols = np.linalg.lstsq(z, t, rcond=None)[0]
    Q, _ = procrustes(z, t)
    Q_so, _ = procrustes(z, t, proper=True)
    zw = Zs @ whitening
    Qw, _ = procrustes(zw[train], t)

    def scaled(B):
        train_u = z @ B
        numer = np.sum(train_u*t, axis=0)
        denom = np.sum(train_u*train_u, axis=0)
        if np.any(denom <= 1e-12):
            raise ValueError('Procrustes axis collapsed during calibration')
        scale = numer/denom
        return B*scale[None, :], scale

    B_scaled, sc = scaled(Q)
    Bw = whitening @ Qw
    Bw_scaled, scw = scaled(Bw)
    return {
        'ols_affine': dict(B=B_ols, basis='standardized Z', scale=None),
        'procrustes_od': dict(B=Q, basis='standardized Z', scale=None),
        'procrustes_so': dict(B=Q_so, basis='standardized Z', scale=None),
        'procrustes_od_scaled': dict(B=B_scaled, basis='standardized Z', scale=sc),
        'whiten_procrustes_od': dict(B=Bw, basis='whitened Z', scale=None),
        'whiten_procrustes_od_scaled': dict(B=Bw_scaled, basis='whitened Z', scale=scw),
    }

def diagnostics(U, T):
    C = pearson_matrix(U, T)
    d = len(C)
    off = np.abs(C[~np.eye(d, dtype=bool)])
    diag = np.abs(np.diag(C))
    resid = U-T
    tvar = np.sum((T-T.mean(0))**2, axis=0)
    mse = np.mean(resid*resid, axis=0)
    r2 = np.divide(np.sum(resid*resid, axis=0), tvar,
                   out=np.full(d, np.nan), where=tvar > 1e-12)
    return dict(corr=C, uu_corr=pearson_matrix(U,U),
                abs_diag_mean=float(diag.mean()), abs_diag_min=float(diag.min()),
                abs_offdiag_mean=float(off.mean()), abs_offdiag_max=float(off.max()),
                diag_offdiag_gap=float(diag.min()-off.max()),
                mean_r2=float(np.mean(1-r2)), mean_rmse=float(np.mean(np.sqrt(mse))),
                r2=(1-r2), rmse=np.sqrt(mse), mae=np.mean(abs(resid), axis=0),
                signed_diag=np.diag(C), abs_diag=diag)

def write_csv(path, names, A, rows=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['U / T', *names])
        for name, vals in zip(rows if rows is not None else names, A):
            w.writerow([name, *[f'{float(v):.12g}' for v in vals]])

def write_table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        cols=list(rows[0]); w=csv.DictWriter(f, fieldnames=cols)
        w.writeheader();w.writerows(rows)

def plot_heatmap(path, matrix, names, title, xlabel='T'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=len(names)
    fig, ax=plt.subplots(figsize=(max(5,n*0.95),max(4.4,n*.85)))
    im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(n),names,rotation=35,ha='right');ax.set_yticks(range(n),[f'U{i+1}' for i in range(n)])
    ax.set_xlabel(xlabel);ax.set_ylabel('Predicted latent U');ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v=matrix[i,j]
            ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=9,
                    color='white' if abs(v) > .65 else 'black')
    fig.colorbar(im,ax=ax,shrink=.82,label='Pearson r')
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def plot_scatter(path, U, T, names, title, max_points=1000, seed=31):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=U.shape[1]
    ids=np.arange(len(U))
    if len(ids) > max_points: ids=np.sort(np.random.default_rng(seed).choice(ids,max_points,replace=False))
    C=pearson_matrix(U,T)
    fig, axes=plt.subplots(n,n,figsize=(max(8,n*2.05),max(8,n*2.05)),squeeze=False)
    for i in range(n):
        for j in range(n):
            ax=axes[i,j]
            ax.scatter(T[ids,j],U[ids,i],s=3,alpha=.20,color='tab:blue',rasterized=True)
            ax.text(.04,.94,f'r={C[i,j]:+.3f}',transform=ax.transAxes,va='top',fontsize=8,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
            if i==n-1:ax.set_xlabel(names[j],fontsize=9)
            if j==0:ax.set_ylabel(f'U{i+1}',fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(title)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

def evaluate(model, Z, T, splits, output, source, hashes=None, plots=True, max_scatter=1000):
    """Reusable pure-array entrypoint; also used in synthetic unit tests."""
    names=NAMES[model]; n,d=np.asarray(Z).shape
    if d != len(names) or np.asarray(T).shape != (n,d):
        raise ValueError(f'{model} requires Z, T dimensions {len(names)}')
    for key in ('train','validation','test'):
        ids=np.asarray(splits[key],int)
        if ids.ndim!=1 or len(ids)<d+2 or np.any(ids<0) or np.any(ids>=n):
            raise ValueError(f'Invalid {key} split')
    all_ids=np.concatenate([np.asarray(splits[k],int) for k in ('train','validation','test')])
    if not np.array_equal(np.sort(all_ids),np.arange(n)):
        raise ValueError('Subject splits must be exhaustive and disjoint')
    Zs, Ts, norm=prepare(Z,T,splits['train'])
    fit=fit_all(Zs,Ts,splits['train'],norm['whitening'])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=dict(model=model,n_subjects=n,latent_dim=d,source=str(source),
                 provenance=hashes or {},
                 protocol='TRAIN-only normalizers and supervised oracle fits; untouched VALIDATION and TEST',
                 geometry='T_std=(T-train_mean)/train_sd; Z_std=(Z-train_mean)/train_sd; U_std=Z_std @ B',
                 note='O(d) permits reflection; SO(d) det+1. White-Procrustes is rotation AFTER whitening (not raw Z rotation).',
                 original_z_train_cov=norm['z_cov'].tolist(),
                 original_z_train_eigenvalues=norm['z_eigen'].tolist(),
                 train_z_mean=norm['z_mean'].tolist(),train_z_sd=norm['z_sd'].tolist(),
                 train_t_mean=norm['t_mean'].tolist(),train_t_sd=norm['t_sd'].tolist(),
                 methods={})
    rows=[];per_param=[];npz={};out={}
    base=diagnostics(Zs[splits['test']],Ts[splits['test']])
    summary['raw_z_baseline_test']={k:v for k,v in base.items() if np.isscalar(v)}
    for method, entry in fit.items():
        B=entry['B'];U=Zs@B
        A_std=B.T / norm['z_sd'][None,:]
        b_std=-A_std@norm['z_mean']
        A_phys=norm['t_sd'][:,None]*A_std
        b_phys=norm['t_mean']+norm['t_sd']*b_std
        orthog=float(np.linalg.norm(B.T@B-np.eye(d)))
        entry_sum=dict(basis=entry['basis'], B_standardized_Z_to_Tstd=B.tolist(),
                       A_raw_Z_to_Tstd=A_std.tolist(),b_raw_Z_to_Tstd=b_std.tolist(),
                       A_raw_Z_to_T=A_phys.tolist(),b_raw_Z_to_T=b_phys.tolist(),
                       det_B=float(np.linalg.det(B)),condition_B=float(np.linalg.cond(B)),
                       orthogonality_error_fro=orthog,
                       calibrated_axis_scale=(None if entry['scale'] is None else entry['scale'].tolist()),
                       split_metrics={})
        npz[f'{method}__B']=B;npz[f'{method}__A_raw_Z_to_T']=A_phys
        npz[f'{method}__b_raw_Z_to_T']=b_phys
        npz[f'{method}__U_std']=U
        for split in ('train','validation','test'):
            ids=np.asarray(splits[split],int)
            metrics=diagnostics(U[ids],Ts[ids]);out[(method,split)]=metrics
            write_csv(output/f'corr__{method}__{split}.csv',names,metrics['corr'])
            write_csv(output/f'uu_corr__{method}__{split}.csv', [f'U{i+1}' for i in range(d)],metrics['uu_corr'])
            fields=('abs_diag_mean','abs_diag_min','abs_offdiag_mean','abs_offdiag_max','diag_offdiag_gap','mean_r2','mean_rmse')
            entry_sum['split_metrics'][split]={k:metrics[k] for k in fields}
            rows.append(dict(method=method,split=split,**{k:metrics[k] for k in fields},
                             orthogonality_error_fro=orthog))
            for j in range(d):
                per_param.append(dict(method=method,split=split,parameter=names[j],
                                      r=float(metrics['signed_diag'][j]),abs_r=float(metrics['abs_diag'][j]),
                                      r2=float(metrics['r2'][j]),rmse_std=float(metrics['rmse'][j]),
                                      mae_std=float(metrics['mae'][j])))
        summary['methods'][method]=entry_sum
        if plots:
            met=out[(method,'test')]
            plot_heatmap(output/f'test_corr__{method}.png',met['corr'],names, f'{model} TEST {method}')
            plot_scatter(output/f'test_scatter__{method}.png',U[splits['test']],Ts[splits['test']],
                         names,f'{model} TEST {method}',max_points=max_scatter)
    np.savez_compressed(output/'mappings_and_U.npz',**npz,
                        train_ids=splits['train'],validation_ids=splits['validation'],test_ids=splits['test'],
                        T_std=Ts,Z_std=Zs,whitening=norm['whitening'])
    write_table(output/'method_split_summary.csv',rows)
    write_table(output/'per_parameter.csv',per_param)
    methods=list(fit)
    ols=out[('ols_affine','test')]
    summary['test_vs_ols']={m:dict(delta_mean_abs_diag_r=out[(m,'test')]['abs_diag_mean']-ols['abs_diag_mean'],
                                   delta_mean_r2=out[(m,'test')]['mean_r2']-ols['mean_r2'],
                                   delta_max_offdiag=out[(m,'test')]['abs_offdiag_max']-ols['abs_offdiag_max'])
                            for m in methods if m != 'ols_affine'}
    summary['test_primary']={m:summary['methods'][m]['split_metrics']['test'] for m in methods}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(16,4.6))
        colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
        vals=summary['test_primary'];xx=np.arange(len(methods))
        for ax,key,description in zip(axs,('abs_diag_mean','abs_offdiag_max','mean_r2'),
                                       ('Mean |diag Corr(U,T)| (higher)','Max |off-diag Corr(U,T)| (lower)',
                                        'Mean TEST R2 (higher)')):
            ax.bar(xx,[vals[m][key] for m in methods],color=colors[:len(methods)])
            ax.set_xticks(xx,[m.replace('procrustes','proc').replace('whiten_','white_') for m in methods],
                          rotation=50,ha='right',fontsize=8)
            ax.set_title(description)
            ax.grid(axis='y',alpha=.25)
        fig.suptitle(f'{model}: OLS affine vs Procrustes (TEST)')
        fig.tight_layout();fig.savefig(output/'test_method_comparison.png',dpi=175);plt.close(fig)
    return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=tuple(MODELS))
    p.add_argument('--beta',type=float,default=None,
                   help='NP beta; default 0.0005 for C1/C2/C3 and 0.0001 for CR3')
    p.add_argument('--lambda-sparse',type=float,default=0.)
    p.add_argument('--context-min',type=float,default=.25)
    p.add_argument('--context-max',type=float,default=.80)
    p.add_argument('--latent-dim',type=int,default=None)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--output',type=Path)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--max-scatter',type=int,default=1200)
    args=p.parse_args(argv)
    if args.beta is None: args.beta = .0001 if args.model == 'CR3' else .0005
    if args.latent_dim is None:args.latent_dim=MODELS[args.model]
    if args.latent_dim!=MODELS[args.model]:p.error(f'{args.model} needs dim={MODELS[args.model]}')
    if args.beta<0 or args.lambda_sparse<0 or args.repeat<1 or args.max_scatter<1:p.error('Invalid argument')
    sys.path.insert(0,str(ROOT/'np'/'src'))
    from modrot3.pipeline import dataset_path, sha256
    from modrot3.analysis import load_frozen
    Z,T,splits,source=load_frozen(args)
    with np.load(source,allow_pickle=False) as ar:
        if str(ar['dataset_sha256'])!=sha256(dataset_path(args.model)):
            raise RuntimeError('S5 Z and S1 dataset hash mismatch')
    if args.output is None:
        args.output=(ROOT/'outputs'/'math'/'09_procrustes_vs_ols'/args.model/
                     condition(args)/f'repeat{args.repeat:02d}')
    report=evaluate(args.model,Z,T,splits,args.output,source,
                    hashes=dict(latent_sha256=sha256(source),dataset_sha256=sha256(dataset_path(args.model))),
                    plots=not args.no_plots,max_scatter=args.max_scatter)
    print(f"[{args.model}] TRAIN-only fits; TEST n={len(splits['test'])}; output={args.output}")
    for name,v in report['test_primary'].items():
        print(f"  {name:32s} diag|r|={v['abs_diag_mean']:.4f} "
              f"offdiag-max={v['abs_offdiag_max']:.4f} R2={v['mean_r2']:.4f}")

if __name__=='__main__':
    main()
