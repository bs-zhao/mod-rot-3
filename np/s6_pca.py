from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S6: PCA of frozen full-trial Z, with variance and theta-correlation plots.

PCA is fitted ONLY on TRAIN subjects and then applied to all subjects.
No regression fitting is performed in this stage. Separate TRAIN-only variance
thresholds select PC axes and original posterior-mean Z axes for descriptive
correlation matrices and pairwise scatter plots.
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.exceptions import ConvergenceWarning
import warnings
from modrot3.cli import parse_args
from modrot3.pipeline import stage_folder, sha256, train_path, condition
from modrot3.analysis import load_frozen, corr, param_names, write_rows, plot_corr



def configure_s6_parser(parser):
    parser.add_argument('--pc-var-threshold',type=float,default=0.05,
                        help='Select PC_i when individual TRAIN explained-variance ratio > threshold (default 0.05)')
    parser.add_argument('--z-var-threshold',type=float,default=0.01,
                        help='Select original z_j when TRAIN between-subject variance > threshold (default 0.01)')
    parser.add_argument('--fa-factors',type=int,default=4,
                        help='FactorAnalysis factors (default 4; capped by latent dimension)')
    parser.add_argument('--ica-max-iter',type=int,default=3000,
                        help='Maximum FastICA training iterations (default 3000)')
    parser.add_argument('--ica-seed',type=int,default=31,
                        help='TRAIN-only FastICA and FactorAnalysis random seed (default 31)')


def select_above_threshold(values,threshold,label):
    """Strict > cutoff, using TRAIN statistics. Return zero-based indices."""
    values=np.asarray(values,dtype=float)
    threshold=float(threshold)
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError(f'{label} threshold must be finite and nonnegative')
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError(f'{label} must be a finite, nonnegative 1D array')
    return np.flatnonzero(values>threshold)


def z_axis_pearson(values):
    """Signed columnwise Pearson r; constant axes get zero, including diagonal."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Correlation requires finite 2D values')
    d=values.shape[1]
    if len(values)<2:
        return np.zeros((d,d),float)
    centered=values-values.mean(0,keepdims=True)
    gram=centered.T@centered
    sd=np.sqrt(np.maximum(np.diag(gram),0))
    denom=sd[:,None]*sd[None,:]
    c=np.divide(gram,denom,out=np.zeros_like(gram),where=denom>1e-24)
    c=np.clip((c+c.T)/2,-1,1)
    np.fill_diagonal(c,np.where(sd>1e-12,1.0,0.0))
    return c


def selected_pairwise_diagnostics(folder,values,indices,kind,train,validation,test):
    """TRAIN-selected axes, ALL/TRAIN/VALIDATION/TEST correlations, ALL scatter."""
    values=np.asarray(values,dtype=float)
    indices=np.asarray(indices,dtype=int)
    names=[f'{kind}{i+1}' for i in indices]
    key='pc' if kind=='PC' else 'z'
    prefix=f's12__selected_{key}'
    result=dict(selected_indices_1based=(indices+1).tolist(),
                selected_names=names,selected_count=int(len(indices)))
    if not len(indices):
        result['status']='none_above_threshold'
        print(f'Selected {kind}: 0 axes; no pairwise graphs (lower threshold)')
        return result
    selected=values[:,indices]
    groups={'all':np.arange(len(values)),
            'train':np.asarray(train,int),
            'validation':np.asarray(validation,int),
            'test':np.asarray(test,int)}
    matrices={label:z_axis_pearson(selected[ids])
              for label,ids in groups.items()}
    for label,matrix in matrices.items():
        suffix='' if label=='all' else '_'+label
        path=folder/(prefix+f'_correlations{suffix}.csv')
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f)
            w.writerow(['dimension',*names])
            for j,row in enumerate(matrix):
                w.writerow([names[j],*[float(v) for v in row]])

    # Same color scale for independent ALL, TRAIN, TEST panels.
    d=len(names)
    fig,axs=plt.subplots(1,3,figsize=(max(12,d*2.5+4),max(4.2,d*.9+2)),
                         squeeze=False)
    for ax,label in zip(axs.flat,('all','train','test')):
        matrix=matrices[label]
        im=ax.imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1)
        ax.set_xticks(range(d),names,rotation=45,ha='right')
        ax.set_yticks(range(d),names)
        ax.set_title(f'{label.upper()} (n={len(groups[label])})')
        for i in range(d):
            for j in range(d):
                val=matrix[i,j]
                ax.text(j,i,f'{val:+.2f}',ha='center',va='center',
                        fontsize=8,color='white' if abs(val)>.65 else 'black')
    fig.suptitle(f'Selected {kind}: signed Pearson r',fontsize=12)
    fig.colorbar(im,ax=list(axs.flat),label='Pearson r',shrink=.8)
    fig.savefig(folder/(prefix+'_correlation_heatmap.png'),
                dpi=180,bbox_inches='tight')
    plt.close(fig)

    # Lower triangle scatter, diagonal histogram, upper triangle numeric r.
    fig,axs=plt.subplots(d,d,figsize=(max(5.5,d*2.3),max(5.5,d*2.3)),
                         squeeze=False)
    for i in range(d):
        for j in range(d):
            ax=axs[i,j]
            if i==j:
                ax.hist(selected[:,j],bins=30,color='tab:blue',alpha=.7)
                ax.set_title(names[j],fontsize=9)
            elif i>j:
                ax.scatter(selected[:,j],selected[:,i],
                           s=3,alpha=.15,color='gray',rasterized=True)
                ax.scatter(selected[groups['test'],j],
                           selected[groups['test'],i],s=5,alpha=.25,
                           color='tab:orange',rasterized=True)
                ax.text(.03,.95,f'r={matrices["all"][i,j]:+.3f}',
                        transform=ax.transAxes,ha='left',va='top',fontsize=8,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.75))
            else:
                ax.text(.5,.5,f'r={matrices["all"][i,j]:+.3f}',
                        transform=ax.transAxes,ha='center',va='center',fontsize=10)
                ax.axis('off')
            if i==d-1: ax.set_xlabel(names[j],fontsize=9)
            if j==0 and i!=j: ax.set_ylabel(names[i],fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'Selected {kind} pairs: ALL gray, TEST orange')
    fig.tight_layout()
    fig.savefig(folder/(prefix+'_pairwise_scatter.png'),dpi=155)
    plt.close(fig)
    result.update(status='complete',
                  correlations={k:v.tolist() for k,v in matrices.items()})
    print(f'Selected {kind} ({d}/{values.shape[1]}): {", ".join(names)}')
    print('  correlation heatmap:',folder/(prefix+'_correlation_heatmap.png'))
    print('  pairwise scatter:',folder/(prefix+'_pairwise_scatter.png'))
    print('  all-subject correlations CSV:',folder/(prefix+'_correlations.csv'))
    return result



def save_theta_component_diagnostics(folder, key, values, theta, test, names, labels):
    """TEST-only descriptive associations of frozen unsupervised axes with theta.

    No theta is passed to PCA, FastICA or FactorAnalysis; TEST is never used
    to choose the axes, their order, or their sign.
    """
    values=np.asarray(values,dtype=float)
    theta=np.asarray(theta,dtype=float)
    test=np.asarray(test,int)
    labels=list(labels)
    names=list(names)
    if (values.ndim!=2 or theta.ndim!=2 or len(values)!=len(theta)
            or values.shape[1]!=len(labels) or theta.shape[1]!=len(names)
            or len(test)<2 or not np.isfinite(values).all()
            or not np.isfinite(theta).all()):
        raise ValueError(f'Invalid {key} values/theta for TEST association')
    c=corr(theta[test],values[test])
    root=f's12__{key}_theta'
    write_rows(folder/(root+'_correlations.csv'),[
        dict(parameter=name,**{label:float(c[i,j])
                               for j,label in enumerate(labels)})
        for i,name in enumerate(names)])
    write_rows(folder/(root+'_pairwise.csv'),[
        dict(parameter=name,component=label,
             test_pearson_r=float(c[i,j]),test_pearson_r2=float(c[i,j]**2),
             test_n=int(len(test)))
        for i,name in enumerate(names)
        for j,label in enumerate(labels)])
    fig,ax=plt.subplots(figsize=(max(6.5,1.05*len(labels)+2),
                                 max(4.5,.7*len(names)+2)))
    im=ax.imshow(c,cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
    ax.set_yticks(range(len(names)),names)
    ax.set(xlabel=f'{key.upper()} component fitted on TRAIN',
           ylabel='True parameter (TEST)',
           title=f'{key.upper()} vs theta: TEST Pearson r (n={len(test)})')
    for i in range(len(names)):
        for j in range(len(labels)):
            val=float(c[i,j])
            ax.text(j,i,f'{val:+.2f}',ha='center',va='center',fontsize=9,
                    color='white' if abs(val)>.65 else 'black')
    fig.colorbar(im,ax=ax,label='Signed Pearson r')
    fig.tight_layout()
    fig.savefig(folder/(root+'_heatmap.png'),dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(len(names),len(labels),
                          figsize=(max(7,len(labels)*2.65),
                                   max(6,len(names)*2.7)),
                          squeeze=False)
    for i,name in enumerate(names):
        for j,label in enumerate(labels):
            ax=axes[i,j]
            ax.scatter(values[test,j],theta[test,i],s=7,alpha=.33,
                       rasterized=True)
            ax.text(.03,.96,f'r={c[i,j]:+.3f}\nr²={c[i,j]**2:.3f}',
                    transform=ax.transAxes,ha='left',va='top',
                    fontsize=8,bbox=dict(facecolor='white',alpha=.76,
                                         edgecolor='none'))
            if i==len(names)-1:
                ax.set_xlabel(label,fontsize=9)
            if j==0:
                ax.set_ylabel(name,fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'{key.upper()} × all theta: TEST pairwise scatter '
                 f'(n={len(test)}; descriptive only)',fontsize=12)
    fig.tight_layout()
    fig.savefig(folder/(root+'_scatter.png'),dpi=165)
    plt.close(fig)
    print(f'{key.upper()}–theta TEST signed r (rows=theta, cols={labels}):')
    print(np.array2string(c,precision=3,suppress_small=True))
    print('  Heatmap:',folder/(root+'_heatmap.png'))
    print('  Pairwise scatter:',folder/(root+'_scatter.png'))
    return dict(test_pearson_r=c.tolist(),
                theta_names=names,component_names=labels,
                test_subjects=int(len(test)),
                matrix_csv=str(folder/(root+'_correlations.csv')),
                heatmap=str(folder/(root+'_heatmap.png')),
                scatter=str(folder/(root+'_scatter.png')))


def fit_unsupervised_ica_fa(z, train, *, factor_count=4, seed=31,
                            ica_max_iter=3000):
    """TRAIN-only unsupervised ICA and 4-factor FA of original posterior means.

    z scaling is TRAIN-only; transformations apply to all subjects. FA scores
    are posterior expected factor scores, NOT uniquely identified axes.
    """
    z=np.asarray(z,dtype=float)
    train=np.asarray(train,int)
    if z.ndim!=2 or len(train)<2 or not np.isfinite(z).all():
        raise ValueError('ICA/FA require finite Z and at least two TRAIN subjects')
    if int(factor_count)<1 or int(ica_max_iter)<1:
        raise ValueError('Factor count and ICA iteration budget must be positive')
    n_axes=z.shape[1]
    k=min(int(factor_count),n_axes)
    mean=z[train].mean(axis=0)
    sd=np.maximum(z[train].std(axis=0,ddof=0),1.e-8)
    zs=(z-mean)/sd
    rng_seed=int(seed)
    ica=FastICA(n_components=n_axes,whiten='unit-variance',
                random_state=rng_seed,max_iter=int(ica_max_iter),tol=1.e-4)
    with warnings.catch_warnings(record=True) as capture_ica:
        warnings.simplefilter('always',ConvergenceWarning)
        ica.fit(zs[train])
    ic=ica.transform(zs)
    if ic.shape!=z.shape or not np.isfinite(ic).all():
        raise RuntimeError('FastICA produced invalid scores; inspect TRAIN rank')
    ica_convergence_warnings=[str(item.message) for item in capture_ica
                              if issubclass(item.category,ConvergenceWarning)]

    fa=FactorAnalysis(n_components=k,rotation='varimax',
                      random_state=rng_seed,max_iter=2000,tol=1e-4)
    with warnings.catch_warnings(record=True) as capture_fa:
        warnings.simplefilter('always',ConvergenceWarning)
        fa.fit(zs[train])
    factors=fa.transform(zs)
    if factors.shape!=(len(z),k) or not np.isfinite(factors).all():
        raise RuntimeError('FactorAnalysis produced invalid factor scores')
    fa_convergence_warnings=[str(item.message) for item in capture_fa
                             if issubclass(item.category,ConvergenceWarning)]
    info=dict(
        train_mean=mean,train_std=sd,
        ic_scores=ic,ica_components=ica.components_,
        ica_mixing=ica.mixing_,ica_n_iter=int(ica.n_iter_),
        ica_convergence_warnings=ica_convergence_warnings,
        factors=factors,fa_components=fa.components_,
        fa_noise_variance=fa.noise_variance_,
        fa_n_iter=int(fa.n_iter_),
        fa_convergence_warnings=fa_convergence_warnings,
        factor_count=k,ica_seed=rng_seed,
    )
    return info


def pca_report(args):
    z, theta, splits, source = load_frozen(args)
    tr, va, te = (splits[k] for k in ('train','validation','test'))
    fitted = PCA(n_components=z.shape[1], svd_solver='full').fit(z[tr])
    scores = fitted.transform(z)
    explained = fitted.explained_variance_ratio_
    folder = stage_folder(args, 's12_z_diagnostics')
    folder.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(source)
    np.savez_compressed(folder/'s12__pca.npz',
        components=fitted.components_, explained_variance=fitted.explained_variance_,
        explained_variance_ratio=explained, transformed_z=scores,
        train_mean=fitted.mean_, source_latent_sha256=np.array(source_sha),
        train_ids=tr, validation_ids=splits['validation'], test_ids=te)
    rows=[dict(pc=i+1, explained_variance_ratio=float(v),
               cumulative_variance=float(np.sum(explained[:i+1]))) for i,v in enumerate(explained)]
    write_rows(folder/'s12__pca_variance.csv',rows)
    pc_threshold=float(getattr(args,'pc_var_threshold',0.05))
    z_threshold=float(getattr(args,'z_var_threshold',0.01))
    z_train_variances=np.var(z[tr],axis=0,ddof=1)
    pc_indices=select_above_threshold(explained,pc_threshold,'PC ratio')
    z_indices=select_above_threshold(z_train_variances,z_threshold,'Original Z variance')
    write_rows(folder/'s12__z_train_variances.csv',[
        dict(dimension=f'z{j+1}',train_variance=float(value),
             threshold=z_threshold,selected=(j in z_indices))
        for j,value in enumerate(z_train_variances)])
    pc_selected=selected_pairwise_diagnostics(
        folder,scores,pc_indices,'PC',tr,va,te)
    z_selected=selected_pairwise_diagnostics(
        folder,z,z_indices,'z',tr,va,te)
    pc_selected.update(threshold=pc_threshold,
                       selection_rule='Individual TRAIN PC explained-variance ratio > threshold',
                       selected_explained_variance_ratio=explained[pc_indices].tolist())
    z_selected.update(threshold=z_threshold,
                      selection_rule='Original coordinate TRAIN variance (ddof=1) > threshold',
                      all_train_variances=z_train_variances.tolist())
    (folder/'s12__selected_dimensions.json').write_text(
        json.dumps(dict(pc=pc_selected,z=z_selected),indent=2,allow_nan=False),
        encoding='utf-8')
    names=param_names(args.model)
    # Every PC (including below-threshold PCs) is audited against all theta.
    pc_theta=save_theta_component_diagnostics(
        folder,'pc',scores,theta,te,names,
        [f'PC{j+1}' for j in range(scores.shape[1])])
    factor_count=int(getattr(args,'fa_factors',4))
    seed=int(getattr(args,'ica_seed',31))
    maxiter=int(getattr(args,'ica_max_iter',3000))
    unsup=fit_unsupervised_ica_fa(z,tr,factor_count=factor_count,
                                 seed=seed,ica_max_iter=maxiter)
    np.savez_compressed(
        folder/'s12__ica.npz',
        transformed_z=unsup['ic_scores'],components=unsup['ica_components'],
        mixing=unsup['ica_mixing'],n_iter=np.array(unsup['ica_n_iter']),
        train_mean=unsup['train_mean'],train_std=unsup['train_std'],
        source_latent_sha256=np.array(source_sha),
        train_ids=tr,validation_ids=va,test_ids=te)
    np.savez_compressed(
        folder/'s12__factor_analysis.npz',
        transformed_z=unsup['factors'],
        components=unsup['fa_components'],
        noise_variance=unsup['fa_noise_variance'],
        train_mean=unsup['train_mean'],train_std=unsup['train_std'],
        n_iter=np.array(unsup['fa_n_iter']),
        source_latent_sha256=np.array(source_sha),
        train_ids=tr,validation_ids=va,test_ids=te)
    ic_theta=save_theta_component_diagnostics(
        folder,'ic',unsup['ic_scores'],theta,te,names,
        [f'IC{j+1}' for j in range(unsup['ic_scores'].shape[1])])
    fa_theta=save_theta_component_diagnostics(
        folder,'fa',unsup['factors'],theta,te,names,
        [f'FA{j+1}' for j in range(unsup['factor_count'])])
    component_report=dict(
        methodology='TRAIN-only ICA and FactorAnalysis; standardized by TRAIN '
            'mean/std; TEST theta used only for descriptive correlations',
        pca_theta=pc_theta,ica_theta=ic_theta,fa_theta=fa_theta,
        ica=dict(n_components=int(unsup['ic_scores'].shape[1]),
                 n_iter=unsup['ica_n_iter'],seed=seed,max_iter=maxiter,
                 warnings=unsup['ica_convergence_warnings'],
                 algorithm='FastICA, whiten=unit-variance'),
        factor_analysis=dict(n_factors=unsup['factor_count'],
                 n_iter=unsup['fa_n_iter'],rotation='varimax',
                 warnings=unsup['fa_convergence_warnings'],
                 caveat='When n_factors equals n_original_variables, '
                 'the factor model can be underidentified; factors are not '
                 'uniquely determined or guaranteed to match generating theta.'),
    )
    (folder/'s12__component_theta_summary.json').write_text(
        json.dumps(component_report,indent=2,allow_nan=False),encoding='utf-8')
    raw_corr=corr(theta[te], z[te])
    pc_corr=corr(theta[te], scores[te])
    report=dict(model=args.model,condition=condition(args),source_latent_sha256=source_sha,
        parameter_names=names, train_subjects=int(len(tr)),
        explained_variance_ratio=explained.tolist(),
        cumulative_explained_variance=np.cumsum(explained).tolist(),
        eigenvalues=fitted.explained_variance_.tolist(),
        selected_pc_analysis=pc_selected,
        selected_z_analysis=z_selected,
        component_theta_analysis=component_report,
        theta_vs_raw_test_corr=raw_corr.tolist(),theta_vs_pc_test_corr=pc_corr.tolist(),
        policy='PCA TRAIN only; TEST parameter correlations for descriptive evaluation only')
    (folder/'s12__pca_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plot_corr(raw_corr,names,f'{args.model} held-out theta vs Z',folder/'s12__theta_vs_z.png')
    plot_corr(pc_corr,names,f'{args.model} held-out theta vs PCA',
              folder/'s12__theta_vs_pca.png',xlabel='Principal components')
    fig,ax=plt.subplots(figsize=(8,4.5))
    axis=np.arange(1,len(explained)+1)
    ax.bar(axis, explained, label='Individual PC')
    ax.plot(axis,np.cumsum(explained),'o-',label='Cumulative')
    ax.set(xlabel='Principal component',ylabel='Explained variance ratio',title=f'{args.model} TRAIN PCA')
    ax.legend()
    fig.tight_layout();fig.savefig(folder/'s12__pca_variance.png',dpi=180);plt.close(fig)
    print('PCA explained variance:', np.round(explained,5))
    print('ICA iterations:',unsup['ica_n_iter'],
          '| warnings:',unsup['ica_convergence_warnings'])
    print('FactorAnalysis factors:',unsup['factor_count'],
          '| warnings:',unsup['fa_convergence_warnings'])
    print(f'PC threshold > {pc_threshold:g}: {len(pc_indices)} retained')
    print(f'Original Z variance threshold > {z_threshold:g}: {len(z_indices)} retained')
    print('PCA results:',folder)


if __name__ == '__main__':
    pca_report(parse_args('S6: PCA analysis of all-trial latent Z',
                          configure_parser=configure_s6_parser))