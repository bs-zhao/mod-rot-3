"""Independent post-NP PCA and held-out true-parameter linear recovery.

All PCA and normalizers are FIT only; ridge alpha selected on VALIDATION;
final ridge trained on FIT+VALIDATION; metrics are TEST only. Theta is NEVER
read by generator/trainer/z extractor/rotation optimizer.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from modlearn_rot.generators.registry import get_model
from modrot3.pipeline import stage_folder, latent_path, train_path, condition, sha256

RIDGE_GRID = (0.0, .01, .1, 1.0, 10.0, 100.0, 1000.0)


def param_names(model: str) -> list[str]:
    return (['alpha','lambda_loss','gamma','tau'] if model == 'CR3'
            else list(get_model(model).parameter_names))


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    ac=a-a.mean(0);bc=b-b.mean(0)
    den=np.sqrt((ac**2).sum(0)[:,None]*(bc**2).sum(0)[None,:])
    return np.divide(ac.T@bc,den,out=np.zeros((a.shape[1],b.shape[1])),where=den>1e-12)


def ridge_probe(x, theta, train, validation, test):
    """Matches CR3 S4 selection protocol, with immutable held-out TEST."""
    x=np.asarray(x,float);theta=np.asarray(theta,float)
    m=x[train].mean(0);s=np.maximum(x[train].std(0),1e-8)
    z=(x-m)/s
    tm=theta[train].mean(0);ts=np.maximum(theta[train].std(0),1e-8)
    y=(theta-tm)/ts
    output=np.empty((len(test),y.shape[1]),dtype=float)
    chosen=[];val_score=[]
    for j in range(y.shape[1]):
        scores=[]
        for alpha in RIDGE_GRID:
            regression=Ridge(alpha=alpha).fit(z[train],y[train,j])
            scores.append(float(r2_score(y[validation,j],regression.predict(z[validation]))))
        idx=int(np.nanargmax(scores))
        best=RIDGE_GRID[idx]
        fitted=Ridge(alpha=best).fit(z[np.r_[train,validation]],y[np.r_[train,validation],j])
        output[:,j]=tm[j]+ts[j]*fitted.predict(z[test])
        chosen.append(best)
        val_score.append(scores[idx])
    yt=theta[test]
    return dict(test_r2=[float(r2_score(yt[:,j],output[:,j])) for j in range(yt.shape[1])],
                test_pearson_r=[float(corr(yt[:,[j]],output[:,[j]])[0,0]) for j in range(yt.shape[1])],
                test_rmse=[float(np.sqrt(np.mean((output[:,j]-yt[:,j])**2))) for j in range(yt.shape[1])],
                validation_r2=val_score, selected_ridge_alpha=chosen,
                theta_linear_pred=output)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def plot_corr(matrix, rows, title, target, xlabel='Latent axes'):
    fig,ax=plt.subplots(figsize=(max(6,len(rows)*.9), max(4,matrix.shape[0]*.72)))
    vmax=1.; m=ax.imshow(matrix,vmin=-vmax,vmax=vmax,cmap='coolwarm')
    ax.set_yticks(range(len(rows)),rows)
    ax.set_xticks(range(matrix.shape[1]),[f'z{j+1}' for j in range(matrix.shape[1])])
    ax.set_xlabel(xlabel);ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(matrix[i,j])>.65 else 'black')
    fig.colorbar(m,ax=ax,label='TEST Pearson r',shrink=.85)
    fig.tight_layout();fig.savefig(target,dpi=180);plt.close(fig)


def report(args):
    source=latent_path(args)
    if not source.is_file():
        raise FileNotFoundError('Run extract first: '+str(source))
    with np.load(source,allow_pickle=False) as f:
        if str(f['checkpoint_sha256']) != sha256(train_path(args)):
            raise RuntimeError('S4 Z provenance mismatch: checkpoint was changed')
        z=np.array(f['z'],float)
        theta=np.array(f['theta_true'],float)
        splits={k:np.array(f[k+'_ids'],int) for k in ('train','validation','test')}
        if str(f['context_mode'])!='all_observed_trials':
            raise RuntimeError('Only full-trial extracted Z may be analysed')
    if z.shape[0]!=theta.shape[0] or z.shape[1]!=args.latent_dim:
        raise ValueError('Z and theta geometry inconsistent with model config')
    tr,va,te=(splits[k] for k in ('train','validation','test'))
    if not np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(len(z))):
        raise ValueError('Splits must be exhaustive and disjoint')
    names=param_names(args.model)
    full=ridge_probe(z,theta,tr,va,te)
    pca=PCA(n_components=z.shape[1],svd_solver='full').fit(z[tr])
    components=pca.transform(z)
    curve=[]
    for k in range(1,z.shape[1]+1):
        metrics=ridge_probe(components[:,:k],theta,tr,va,te)
        curve.append(dict(n_pcs=k,mean_test_r2=float(np.mean(metrics['test_r2'])),
                          min_test_r2=float(np.min(metrics['test_r2'])),
                          **{f'r2_{name}':float(score) for name,score in zip(names,metrics['test_r2'])}))
    folder=stage_folder(args,'s12_z_diagnostics')
    folder.mkdir(parents=True,exist_ok=True)
    pc_variance=pca.explained_variance_ratio_
    raw_corr=corr(theta[te],z[te]);pc_corr=corr(theta[te],components[te])
    summary=dict(model=args.model,condition=condition(args),latent_dim=args.latent_dim,
                 source_latent_sha256=sha256(source),source_checkpoint_sha256=sha256(train_path(args)),
                 subjects={k:len(v) for k,v in splits.items()},
                 parameter_names=names,
                 pca=dict(explained_variance_ratio=pc_variance.tolist(),
                          cumulative_explained_variance=np.cumsum(pc_variance).tolist(),
                          eigenvalues=pca.explained_variance_.tolist(),
                          train_mean=pca.mean_.tolist(),
                          components_rows=pca.components_.tolist(),
                          theta_vs_raw_test_corr=raw_corr.tolist(),
                          theta_vs_pc_test_corr=pc_corr.tolist()),
                 full_z_ridge=dict(test_r2=full['test_r2'],test_pearson_r=full['test_pearson_r'],
                                   test_rmse=full['test_rmse'],
                                   validation_r2=full['validation_r2'],
                                   ridge_alpha=full['selected_ridge_alpha']),
                 pca_k_ridge=curve,
                 evaluation_policy='TRAIN-only PCA/standardization; VAL-only ridge selection; final FIT+VAL fit; TEST scores')
    (folder/'s12__diagnostics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez_compressed(folder/'s12__pca.npz',components=pca.components_,
                        explained_variance=pca.explained_variance_,
                        explained_variance_ratio=pc_variance,
                        transformed_z=components,train_ids=tr,validation_ids=va,test_ids=te)
    write_rows(folder/'s12__pca_ridge_curve.csv',curve)
    results=[dict(parameter=name,test_r2=full['test_r2'][j],
                  test_pearson_r=full['test_pearson_r'][j],
                  test_rmse=full['test_rmse'][j],
                  validation_r2=full['validation_r2'][j],
                  selected_ridge_alpha=full['selected_ridge_alpha'][j])
             for j,name in enumerate(names)]
    write_rows(folder/'s12__full_z_ridge.csv',results)
    s4=stage_folder(args,'s4_evaluate');s4.mkdir(parents=True,exist_ok=True)
    (s4/'s4_evaluate__summary.json').write_text(json.dumps(
        dict(model=args.model,latent_dim=args.latent_dim,
             latent_is='encoder posterior mean using ALL observed trials',
             parameter_names=names,results=results,
             source_checkpoint_sha256=sha256(train_path(args))),indent=2),encoding='utf-8')
    np.savez_compressed(s4/'s4_evaluate__linear_predictions.npz',test_ids=te,
                        theta_true=theta[te],theta_linear_pred=full['theta_linear_pred'])
    plot_corr(raw_corr,names,f'{args.model} full Z: held-out theta–Z correlations',
              folder/'s12__theta_vs_z.png')
    plot_corr(pc_corr,names,f'{args.model} PCA: held-out theta–PC correlations',
              folder/'s12__theta_vs_pca.png',xlabel='Principal components')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(np.arange(1,len(pc_variance)+1),pc_variance)
    axes[0].plot(np.arange(1,len(pc_variance)+1),np.cumsum(pc_variance),'o-',label='Cumulative')
    axes[0].set_xlabel('PC');axes[0].set_ylabel('Explained variance ratio');axes[0].set_title('TRAIN PCA variance')
    axes[0].legend()
    axes[1].plot([r['n_pcs'] for r in curve],[r['mean_test_r2'] for r in curve],'o-',label='Mean TEST R²')
    for name in names:
        axes[1].plot([r['n_pcs'] for r in curve],[r['r2_'+name] for r in curve],label=name,alpha=.7)
    axes[1].axhline(np.mean(full['test_r2']),ls='--',c='k',label='Full Z')
    axes[1].set_xlabel('Number of top PCs');axes[1].set_ylabel('TEST parameter R²');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'s12__pca_and_regression.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,len(names),figsize=(3.1*len(names),3.3),squeeze=False)
    for j,name in enumerate(names):
        axs[0,j].scatter(theta[te,j],full['theta_linear_pred'][:,j],s=6,alpha=.28)
        axs[0,j].set_title(f'{name}: TEST R²={full["test_r2"][j]:.3f}')
        axs[0,j].set_xlabel('True theta');axs[0,j].set_ylabel('Linear prediction')
    fig.tight_layout();fig.savefig(folder/'s12__ridge_scatter.png',dpi=180);plt.close(fig)
    print(f'{args.model} complete TEST full-Z Ridge r²:',
          ', '.join(f'{p}={q:.4f}' for p,q in zip(names,full['test_r2'])))
    print('PCA explained variance:',np.round(pc_variance,5))
    print('Results:',folder)