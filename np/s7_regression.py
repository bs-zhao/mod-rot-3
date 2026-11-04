from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S7: Held-out linear Ridge recovery from full Z and PCA projections.

PCA must be run in S6 first. TRAIN standardizes; VALIDATION selects ridge
regularization; TEST is reported once per predeclared PCA dimension.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modrot3.cli import parse_args
from modrot3.pipeline import stage_folder, train_path, condition, sha256
from modrot3.analysis import (load_frozen, ridge_probe, param_names, corr,
                              write_rows, plot_corr)


def regression_report(args):
    z, theta, splits, source = load_frozen(args)
    tr, va, te=(splits[k] for k in ('train','validation','test'))
    folder=stage_folder(args, 's12_z_diagnostics')
    pca_file=folder/'s12__pca.npz'
    if not pca_file.is_file():
        raise FileNotFoundError(f'Run S6 PCA first: {pca_file}')
    with np.load(pca_file,allow_pickle=False) as f:
        if ('source_latent_sha256' not in f or
            str(f['source_latent_sha256'])!=sha256(source)):
            raise RuntimeError('PCA belongs to another extracted Z; rerun S6')
        components=np.asarray(f['transformed_z'],float)
        stored_splits=[np.asarray(f[x+'_ids'],int) for x in ('train','validation','test')]
        explained=np.asarray(f['explained_variance_ratio'],float)
        pca_weights=np.asarray(f['components'],float)
        pca_mean=np.asarray(f['train_mean'],float)
        pca_eigenvalues=np.asarray(f['explained_variance'],float)
    if any(not np.array_equal(a,b) for a,b in zip(stored_splits,(tr,va,te))):
        raise RuntimeError('PCA and Z use different subject splits')
    if components.shape!=z.shape or pca_weights.shape!=(z.shape[1],z.shape[1]):
        raise RuntimeError('PCA dimension mismatch')
    names=param_names(args.model)
    full=ridge_probe(z,theta,tr,va,te)
    curve=[]
    for k in range(1,z.shape[1]+1):
        metrics=ridge_probe(components[:,:k],theta,tr,va,te)
        curve.append(dict(n_pcs=k,mean_test_r2=float(np.mean(metrics['test_r2'])),
                          min_test_r2=float(np.min(metrics['test_r2'])),
                          **{f'r2_{name}':float(score) for name,score in zip(names,metrics['test_r2'])}))
    results=[dict(parameter=name,test_r2=full['test_r2'][j],
                  test_pearson_r=full['test_pearson_r'][j],
                  test_rmse=full['test_rmse'][j],
                  validation_r2=full['validation_r2'][j],
                  selected_ridge_alpha=full['selected_ridge_alpha'][j])
             for j,name in enumerate(names)]
    write_rows(folder/'s12__full_z_ridge.csv',results)
    write_rows(folder/'s12__pca_ridge_curve.csv',curve)
    s4=stage_folder(args,'s4_evaluate')
    s4.mkdir(parents=True,exist_ok=True)
    (s4/'s4_evaluate__summary.json').write_text(json.dumps(dict(
        model=args.model,latent_dim=args.latent_dim,
        latent_is='encoder posterior mean using ALL observed trials',
        parameter_names=names,results=results,
        source_checkpoint_sha256=sha256(train_path(args))),indent=2),encoding='utf-8')
    np.savez_compressed(s4/'s4_evaluate__linear_predictions.npz',test_ids=te,
        theta_true=theta[te],theta_linear_pred=full['theta_linear_pred'])
    diagnostics=dict(model=args.model,condition=condition(args),latent_dim=args.latent_dim,
        source_latent_sha256=sha256(source),source_checkpoint_sha256=sha256(train_path(args)),
        subjects={k:int(len(v)) for k,v in splits.items()},parameter_names=names,
        pca=dict(explained_variance_ratio=explained.tolist(),
                 cumulative_explained_variance=np.cumsum(explained).tolist(),
                 eigenvalues=pca_eigenvalues.tolist(),
                 train_mean=pca_mean.tolist(),
                 components_rows=pca_weights.tolist(),
                 theta_vs_raw_test_corr=corr(theta[te],z[te]).tolist(),
                 theta_vs_pc_test_corr=corr(theta[te],components[te]).tolist()),
        full_z_ridge=dict(test_r2=full['test_r2'],test_pearson_r=full['test_pearson_r'],
            test_rmse=full['test_rmse'],validation_r2=full['validation_r2'],
            ridge_alpha=full['selected_ridge_alpha'],
            selected_ridge_alpha=full['selected_ridge_alpha']),
        pca_k_ridge=curve,
        evaluation_policy='TRAIN PCA; TRAIN standardization; VAL ridge alpha; FIT+VAL refit; TEST scores')
    (folder/'s12__diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    pcs=np.arange(1,len(explained)+1)
    axes[0].bar(pcs,explained);axes[0].plot(pcs,np.cumsum(explained),'o-',label='Cumulative')
    axes[0].set(xlabel='PC',ylabel='Explained variance ratio',title='TRAIN PCA variance')
    axes[0].legend()
    axes[1].plot(pcs,[r['mean_test_r2'] for r in curve],'o-',label='Mean TEST R²')
    for name in names:
        axes[1].plot(pcs,[r['r2_'+name] for r in curve],label=name,alpha=.7)
    axes[1].axhline(np.mean(full['test_r2']),ls='--',color='k',label='Full Z')
    axes[1].set(xlabel='Number of top PCs',ylabel='TEST parameter R²')
    axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'s12__pca_and_regression.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,len(names),figsize=(3.1*len(names),3.3),squeeze=False)
    for j,name in enumerate(names):
        axs[0,j].scatter(theta[te,j],full['theta_linear_pred'][:,j],s=6,alpha=.28)
        axs[0,j].set_title(f'{name}: TEST R²={full["test_r2"][j]:.3f}')
        axs[0,j].set_xlabel('True theta');axs[0,j].set_ylabel('Linear prediction')
    fig.tight_layout();fig.savefig(folder/'s12__ridge_scatter.png',dpi=180);plt.close(fig)
    print(f'{args.model} complete TEST full-Z Ridge r²:',
          ', '.join(f'{p}={q:.4f}' for p,q in zip(names,full['test_r2'])))
    print('Regression results:', folder)


if __name__ == '__main__':
    regression_report(parse_args('S7: held-out linear parameter regression on Z and PCA'))