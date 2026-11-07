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
from sklearn.decomposition import PCA
from modrot3.cli import parse_args
from modrot3.pipeline import stage_folder, sha256, train_path, condition
from modrot3.analysis import load_frozen, corr, param_names, write_rows, plot_corr



def configure_s6_parser(parser):
    parser.add_argument('--pc-var-threshold',type=float,default=0.05,
                        help='Select PC_i when individual TRAIN explained-variance ratio > threshold (default 0.05)')
    parser.add_argument('--z-var-threshold',type=float,default=0.01,
                        help='Select original z_j when TRAIN between-subject variance > threshold (default 0.01)')


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
    raw_corr=corr(theta[te], z[te])
    pc_corr=corr(theta[te], scores[te])
    report=dict(model=args.model,condition=condition(args),source_latent_sha256=source_sha,
        parameter_names=names, train_subjects=int(len(tr)),
        explained_variance_ratio=explained.tolist(),
        cumulative_explained_variance=np.cumsum(explained).tolist(),
        eigenvalues=fitted.explained_variance_.tolist(),
        selected_pc_analysis=pc_selected,
        selected_z_analysis=z_selected,
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
    print(f'PC threshold > {pc_threshold:g}: {len(pc_indices)} retained')
    print(f'Original Z variance threshold > {z_threshold:g}: {len(z_indices)} retained')
    print('PCA results:',folder)


if __name__ == '__main__':
    pca_report(parse_args('S6: PCA analysis of all-trial latent Z',
                          configure_parser=configure_s6_parser))