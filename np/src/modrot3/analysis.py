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


def load_frozen(args):
    """Check provenance, full-trial origin, shapes and disjoint subject splits."""
    source = latent_path(args)
    checkpoint = train_path(args)
    if not source.exists():
        raise FileNotFoundError(f'Run S5 extraction first: {source}')
    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing trained checkpoint: {checkpoint}')
    with np.load(source, allow_pickle=False) as saved:
        if str(saved['checkpoint_sha256']) != sha256(checkpoint):
            raise RuntimeError('Saved Z belongs to a different NP checkpoint')
        if str(saved['context_mode']) != 'all_observed_trials':
            raise RuntimeError('Analysis requires Z encoded using ALL observed trials')
        z = np.asarray(saved['z'], float)
        theta = np.asarray(saved['theta_true'], float)
        split = {name: np.asarray(saved[name+'_ids'], int) for name in ('train','validation','test')}
    if z.shape[0] != theta.shape[0] or z.shape[1] != args.latent_dim:
        raise ValueError('Z/theta geometry inconsistent with --latent-dim')
    if not np.array_equal(np.sort(np.r_[split['train'],split['validation'],split['test']]),
                          np.arange(len(z))):
        raise ValueError('Subject splits must be disjoint and exhaustive')
    return z, theta, split, source