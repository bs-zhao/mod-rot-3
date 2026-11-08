from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S7: Held-out linear Ridge recovery from full Z, selected raw Z and PCA.

PCA must be run in S6 first. TRAIN standardizes; VALIDATION selects ridge
regularization; TEST is reported once per predeclared PCA dimension. An additional
analysis uses only PCs whose individual TRAIN explained-variance ratio exceeds
a configurable threshold (default 0.10). Additionally, S7 compares the
four true theta variables with four independently regressed theta-hat columns
for all Z, TRAIN-variance-selected Z, and TRAIN-ratio-selected PCs on TEST.
"""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from modrot3.cli import parse_args
from modrot3.pipeline import stage_folder, train_path, condition, sha256
from modrot3.analysis import (load_frozen, ridge_probe, param_names, corr,
                              write_rows, plot_corr)


def configure_s7_parser(parser):
    parser.add_argument('--pca-threshold', '--pca-variance-threshold',
                        dest='pca_threshold', type=float, default=0.10,
                        help='Retain only PCA components with individual TRAIN '
                             'explained_variance_ratio strictly greater than '
                             'this threshold (default: 0.10). S7 only.')
    parser.add_argument('--z-var-threshold',type=float,default=0.01,
                        help='Retain original Z dimensions whose TRAIN sample '
                             'variance (ddof=1) strictly exceeds this absolute '
                             'threshold (default: 0.01); S7 only.')


def select_pcs_above_threshold(explained_variance_ratio, threshold=0.10):
    """1-based PC indices are for reporting; returned integer indices are 0-based."""
    ratio = np.asarray(explained_variance_ratio, dtype=float)
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError('--pca-threshold must be finite and in [0, 1)')
    if ratio.ndim != 1 or not len(ratio) or not np.all(np.isfinite(ratio)):
        raise ValueError('Invalid PCA explained-variance ratios')
    if np.any(ratio < 0):
        raise ValueError('PCA explained-variance ratios must be nonnegative')
    selected = np.flatnonzero(ratio > threshold)
    if not len(selected):
        raise ValueError(
            f'No PCA component has individual explained variance ratio '
            f'> {threshold:g}; lower --pca-threshold')
    return selected



def select_z_above_threshold(z, train, threshold=0.01):
    """Original-coordinate Z variance, selected on TRAIN only (ddof=1)."""
    z=np.asarray(z,dtype=float)
    train=np.asarray(train,dtype=int)
    threshold=float(threshold)
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError('--z-var-threshold must be finite and nonnegative')
    if z.ndim!=2 or len(train)<2 or not np.isfinite(z).all():
        raise ValueError('Need finite 2D Z and at least two TRAIN subjects')
    variance=np.var(z[train],axis=0,ddof=1)
    selected=np.flatnonzero(variance>threshold)
    if not len(selected):
        raise ValueError(f'No original Z dimension has TRAIN variance '
                         f'> {threshold:g}; lower --z-var-threshold')
    return selected,variance


def theta_hat_correlation_audit(folder, theta_test, names, fits, *,
                                test_ids, z_variance_threshold, pca_threshold,
                                selected_z_indices, selected_pc_indices):
    """TEST true theta (rows) vs separately fitted theta-hat (columns).

    'fits' are already fit by ridge_probe, which uses TRAIN fit, VAL alpha,
    FIT+VAL refit and untouched TEST. The audit uses no theta for selection.
    """
    truth=np.asarray(theta_test,dtype=float)
    npar=len(names)
    if truth.ndim!=2 or truth.shape[1]!=npar:
        raise ValueError('TEST theta shape does not match parameter names')
    schemes=('full_z','selected_z','selected_pc')
    if set(fits)!=set(schemes):
        raise ValueError(f'Expected exactly the three regression schemes: {schemes}')
    audits={}
    figs,axs=plt.subplots(1,3,figsize=(16.7,5.1),squeeze=False)
    for ax,key in zip(axs.flat,schemes):
        model=fits[key]
        prediction=np.asarray(model['theta_linear_pred'],dtype=float)
        if prediction.shape!=truth.shape or not np.isfinite(prediction).all():
            raise ValueError(f'Invalid TEST theta-hat predictions for {key}')
        matrix=corr(truth,prediction)
        # Write both the TRUE-theta row label and predicted-theta column label.
        with (folder/f's12__theta_hat_corr_{key}.csv').open(
            'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f)
            w.writerow(['true_theta / predicted_theta',
                        *[f'pred_{name}' for name in names]])
            for i,name in enumerate(names):
                w.writerow([name,*[float(v) for v in matrix[i]]])
        diagonal=np.diag(matrix)
        offdiag=matrix[~np.eye(npar,dtype=bool)]
        audits[key]=dict(
            test_pearson_r_true_vs_predicted=matrix.tolist(),
            mean_diagonal_r=float(np.mean(diagonal)),
            mean_abs_offdiagonal_r=float(np.mean(np.abs(offdiag)))
                if len(offdiag) else 0.,
            max_abs_offdiagonal_r=float(np.max(np.abs(offdiag)))
                if len(offdiag) else 0.,
            diagonal_test_r2=model['test_r2'],
            test_samples=int(len(truth)),
        )
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(npar),[f'ĥ{n}' for n in names],
                      rotation=40,ha='right')
        ax.set_yticks(range(npar),names)
        ax.set_xlabel('Predicted theta-hat (one Ridge per parameter)')
        ax.set_ylabel('True theta')
        ax.set_title({'full_z':'All original Z',
                      'selected_z':'Variance-filtered original Z',
                      'selected_pc':'Variance-filtered PCs'}[key])
        for i in range(npar):
            for j in range(npar):
                v=float(matrix[i,j])
                ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=9,
                        color='white' if abs(v)>.65 else 'black')
        for i in range(npar):
            ax.add_patch(Rectangle((i-.5,i-.5),1,1,fill=False,
                                   ec='black',lw=1.4))
        fig,one=plt.subplots(figsize=(max(6,npar*.9+2),max(4.6,npar*.8+2)))
        im1=one.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
        one.set_xticks(range(npar),[f'pred_{name}' for name in names],
                       rotation=40,ha='right')
        one.set_yticks(range(npar),names)
        one.set(xlabel='Independently predicted theta-hat',
                ylabel='True theta',
                title=f'TEST true theta x predicted theta-hat: {key}')
        for i in range(npar):
            for j in range(npar):
                v=float(matrix[i,j])
                one.text(j,i,f'{v:+.3f}',ha='center',va='center',
                         color='white' if abs(v)>.65 else 'black',fontsize=9)
            one.add_patch(Rectangle((i-.5,i-.5),1,1,fill=False,
                                    ec='black',lw=1.4))
        fig.colorbar(im1,ax=one,label='Signed Pearson r',shrink=.8)
        fig.tight_layout()
        fig.savefig(folder/f's12__theta_hat_corr_{key}.png',dpi=180)
        plt.close(fig)
    figs.suptitle('TEST true theta (rows) vs predicted theta-hat (columns): '
                  'diagonal = matched, off-diagonal = cross-parameter')
    figs.colorbar(im,ax=list(axs.flat),label='Signed Pearson r',shrink=.75)
    figs.savefig(folder/'s12__theta_hat_corr_3way.png',
                 dpi=180,bbox_inches='tight')
    plt.close(figs)
    np.savez_compressed(
        folder/'s12__theta_hat_test_predictions.npz',
        theta_true=truth,
        test_ids=np.asarray(test_ids,dtype=int),
        parameter_names=np.asarray(names),
        theta_hat_full_z=np.asarray(fits['full_z']['theta_linear_pred']),
        theta_hat_selected_z=np.asarray(fits['selected_z']['theta_linear_pred']),
        theta_hat_selected_pc=np.asarray(fits['selected_pc']['theta_linear_pred']),
        selected_z_indices_1based=np.asarray(selected_z_indices)+1,
        selected_pc_indices_1based=np.asarray(selected_pc_indices)+1,
        z_var_threshold=np.asarray(z_variance_threshold),
        pca_threshold=np.asarray(pca_threshold))
    report=dict(
        matrix_orientation='row=true theta; column=theta-hat predicted '
                           'from an independently trained Ridge model',
        evaluation_policy='TRAIN feature standardization; VALIDATION picks '
                          'Ridge alpha per true parameter; TRAIN+VAL refit; '
                          'TEST theta only for final Pearson correlations',
        selected_original_z_indices_1based=(np.asarray(selected_z_indices)+1).tolist(),
        selected_pc_indices_1based=(np.asarray(selected_pc_indices)+1).tolist(),
        z_var_threshold=float(z_variance_threshold),
        pca_threshold=float(pca_threshold),
        test_true_theta_baseline_r=corr(truth,truth).tolist(),
        methods=audits)
    (folder/'s12__theta_hat_corr_3way.json').write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('TEST TRUE θ versus predicted θ-hat (rows true, columns predicted):')
    for key in schemes:
        m=np.asarray(audits[key]['test_pearson_r_true_vs_predicted'])
        print(f'  {key}: diagonal mean r={np.mean(np.diag(m)):.4f}; '
              f'off-diagonal mean |r|={audits[key]["mean_abs_offdiagonal_r"]:.4f}')
    print('Three-way theta-hat heatmap:',folder/'s12__theta_hat_corr_3way.png')
    return report


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
    threshold=float(getattr(args,'pca_threshold',0.10))
    retained=select_pcs_above_threshold(explained,threshold)
    z_threshold=float(getattr(args,'z_var_threshold',0.01))
    selected_z_indices,train_z_variances=select_z_above_threshold(
        z,tr,z_threshold)
    selected_z_names=[f'z{i+1}' for i in selected_z_indices]
    selected_pc_names=[f'PC{i+1}' for i in retained]
    explained_retained=explained[retained]
    full=ridge_probe(z,theta,tr,va,te)
    selected_z=ridge_probe(z[:,selected_z_indices],theta,tr,va,te)
    filtered=ridge_probe(components[:,retained],theta,tr,va,te)
    theta_hat_audit=theta_hat_correlation_audit(
        folder,theta[te],names,
        dict(full_z=full,selected_z=selected_z,selected_pc=filtered),
        test_ids=te,z_variance_threshold=z_threshold,pca_threshold=threshold,
        selected_z_indices=selected_z_indices,selected_pc_indices=retained)
    theta_hat_audit['train_original_z_variances']=train_z_variances.tolist()
    filtered_summary=dict(
        threshold=threshold,
        criterion='individual TRAIN explained_variance_ratio > threshold (strict)',
        retained_pc_count=int(len(retained)),
        retained_pc_indices_1based=(retained+1).tolist(),
        retained_pc_names=selected_pc_names,
        retained_explained_variance_ratio=explained_retained.tolist(),
        retained_variance_fraction=float(np.sum(explained_retained)),
        test_r2=filtered['test_r2'],
        mean_test_r2=float(np.mean(filtered['test_r2'])),
        min_test_r2=float(np.min(filtered['test_r2'])),
        test_pearson_r=filtered['test_pearson_r'],
        test_rmse=filtered['test_rmse'],
        validation_r2=filtered['validation_r2'],
        selected_ridge_alpha=filtered['selected_ridge_alpha'],
        fitting_policy='TRAIN standardization, VALIDATION alpha, FIT+VAL refit, TEST scores',
    )
    filtered_rows=[
        dict(parameter=name,pca_threshold=threshold,
             retained_pc_count=len(retained),retained_pcs=','.join(selected_pc_names),
             selected_variance_fraction=float(np.sum(explained_retained)),
             test_r2=filtered['test_r2'][j],
             test_pearson_r=filtered['test_pearson_r'][j],
             test_rmse=filtered['test_rmse'][j],
             validation_r2=filtered['validation_r2'][j],
             selected_ridge_alpha=filtered['selected_ridge_alpha'][j])
        for j,name in enumerate(names)]
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
    # One auditable table containing BOTH independently fitted regressions.
    # All eight raw Z columns are included jointly, not fitted one at a time.
    comparison_rows=[]
    for label,metrics,features in (
        ('full_z',full,[f'Z{i+1}' for i in range(z.shape[1])]),
        ('threshold_pca',filtered,selected_pc_names),
    ):
        for j,name in enumerate(names):
            comparison_rows.append(dict(
                representation=label,parameter=name,feature_count=len(features),
                features=','.join(features),
                pca_threshold=(threshold if label=='threshold_pca' else ''),
                test_r2=metrics['test_r2'][j],
                test_pearson_r=metrics['test_pearson_r'][j],
                test_rmse=metrics['test_rmse'][j],
                validation_r2=metrics['validation_r2'][j],
                selected_ridge_alpha=metrics['selected_ridge_alpha'][j]))
    comparison_deltas=[
        dict(parameter=name,
             full_z_test_r2=full['test_r2'][j],
             threshold_pca_test_r2=filtered['test_r2'][j],
             delta_test_r2_pca_minus_full_z=filtered['test_r2'][j]-full['test_r2'][j],
             full_z_test_pearson_r=full['test_pearson_r'][j],
             threshold_pca_test_pearson_r=filtered['test_pearson_r'][j],
             full_z_test_rmse=full['test_rmse'][j],
             threshold_pca_test_rmse=filtered['test_rmse'][j])
        for j,name in enumerate(names)]
    write_rows(folder/'s12__ridge_input_comparison.csv',comparison_rows)
    write_rows(folder/'s12__ridge_input_deltas.csv',comparison_deltas)
    # All three representations are also compared by standard TEST Ridge metrics.
    three_way_rows=[]
    for key,fit,features in (
        ('full_z',full,[f'z{i+1}' for i in range(z.shape[1])]),
        ('selected_z',selected_z,selected_z_names),
        ('selected_pc',filtered,selected_pc_names),
    ):
        for j,name in enumerate(names):
            three_way_rows.append(dict(
                representation=key,parameter=name,
                n_features=len(features),features=','.join(features),
                test_r2=fit['test_r2'][j],
                test_pearson_r=fit['test_pearson_r'][j],
                test_rmse=fit['test_rmse'][j],
                validation_r2=fit['validation_r2'][j],
                selected_ridge_alpha=fit['selected_ridge_alpha'][j]))
    write_rows(folder/'s12__theta_hat_ridge_3way.csv',three_way_rows)
    write_rows(folder/'s12__full_z_ridge.csv',results)
    write_rows(folder/'s12__pca_ridge_curve.csv',curve)
    write_rows(folder/'s12__pca_threshold_ridge.csv',filtered_rows)
    (folder/'s12__pca_threshold_ridge.json').write_text(
        json.dumps(filtered_summary,indent=2),encoding='utf-8')
    np.savez_compressed(folder/'s12__pca_threshold_predictions.npz',
        test_ids=te,theta_true=theta[te],
        theta_threshold_pca_pred=filtered['theta_linear_pred'],
        pca_threshold=np.asarray(threshold),
        retained_pc_indices_1based=retained+1,
        retained_explained_variance_ratio=explained_retained)
    s4=stage_folder(args,'s4_evaluate')
    s4.mkdir(parents=True,exist_ok=True)
    (s4/'s4_evaluate__summary.json').write_text(json.dumps(dict(
        model=args.model,latent_dim=args.latent_dim,
        latent_is='encoder posterior mean using ALL observed trials',
        parameter_names=names,results=results,
        pca_variance_threshold_ridge=filtered_summary,
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
        pca_variance_threshold_ridge=filtered_summary,
        theta_hat_correlation_audit=theta_hat_audit,
        selected_z_ridge=dict(
            train_variance_threshold=z_threshold,
            selected_dimensions=selected_z_names,
            test_r2=selected_z['test_r2'],
            test_pearson_r=selected_z['test_pearson_r'],
            selected_ridge_alpha=selected_z['selected_ridge_alpha']),
        ridge_input_comparison=dict(
            full_z_feature_count=int(z.shape[1]),
            selected_pc_count=int(len(retained)),
            selected_pc_names=selected_pc_names,
            full_z_mean_test_r2=float(np.mean(full['test_r2'])),
            threshold_pca_mean_test_r2=float(np.mean(filtered['test_r2'])),
            delta_mean_test_r2_pca_minus_full_z=float(
                np.mean(filtered['test_r2'])-np.mean(full['test_r2'])),
            per_parameter_deltas=comparison_deltas),
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
    # A separate plot preserves the original full-Z and PCA-k figures.
    fig,axs=plt.subplots(1,len(names),figsize=(3.1*len(names),3.3),squeeze=False)
    for j,name in enumerate(names):
        axs[0,j].scatter(theta[te,j],filtered['theta_linear_pred'][:,j],
                         s=6,alpha=.28)
        axs[0,j].set_title(f'{name}: TEST R²={filtered["test_r2"][j]:.3f}')
        axs[0,j].set_xlabel('True theta')
        axs[0,j].set_ylabel('Threshold PCA prediction')
    fig.suptitle(f'{args.model}: {len(retained)} PCs with individual variance > {threshold:g}')
    fig.tight_layout()
    fig.savefig(folder/'s12__pca_threshold_scatter.png',dpi=180)
    plt.close(fig)
    # Compare both fitted models on identical TEST subjects and parameter scales.
    fig,axs=plt.subplots(2,len(names),figsize=(3.1*len(names),7.0),squeeze=False)
    for j,name in enumerate(names):
        true=theta[te,j]
        all_values=np.concatenate((
            true,full['theta_linear_pred'][:,j],
            filtered['theta_linear_pred'][:,j]))
        low,high=float(np.min(all_values)),float(np.max(all_values))
        padding=max((high-low)*.05,1e-6)
        low-=padding
        high+=padding
        for row,(label,metrics) in enumerate((
            ('All Z',full),('Selected PCs',filtered))):
            ax=axs[row,j]
            ax.scatter(true,metrics['theta_linear_pred'][:,j],s=6,alpha=.28)
            ax.plot([low,high],[low,high],ls='--',lw=1,color='gray')
            ax.set(xlim=(low,high),ylim=(low,high),
                   xlabel='True theta',ylabel=f'{label} prediction',
                   title=f'{name} | TEST R²={metrics["test_r2"][j]:.3f}')
            ax.set_aspect('equal',adjustable='box')
    fig.suptitle(
        f'{args.model}: Full Z ({z.shape[1]}) vs '
        f'threshold PCs ({len(retained)}; explained ratio > {threshold:g})')
    fig.tight_layout()
    fig.savefig(folder/'s12__ridge_input_comparison_scatter.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(max(7,1.6*len(names)+2),4.5))
    positions=np.arange(len(names))
    width=.38
    ax.bar(positions-width/2,full['test_r2'],width,label=f'All {z.shape[1]} Z')
    ax.bar(positions+width/2,filtered['test_r2'],width,
           label=f'Selected {len(retained)} PCs')
    ax.set_xticks(positions,names)
    ax.set_ylabel('TEST parameter R²')
    ax.set_title(f'{args.model}: Full Z vs PCA threshold > {threshold:g}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'s12__ridge_input_comparison_r2.png',dpi=180)
    plt.close(fig)
    print(f'=== {args.model} S7: BOTH linear parameter regressions ===')
    print(f'Full-Z predictors: Z1-Z{z.shape[1]} ({z.shape[1]} jointly)')
    print(f'PCA individual explained-variance threshold: > {threshold:g}')
    print(f'Retained PCA components: {len(retained)} / {len(explained)} '
          f'({", ".join(selected_pc_names)}); '
          f'variance explained: {100*np.sum(explained_retained):.2f}%')
    for label,metrics in (
        ('FULL Z (all raw Z)',full),
        ('SELECTED PCs (thresholded PCA)',filtered),
    ):
        print(f'--- {label} ---')
        print(f'Mean TEST R²: {np.mean(metrics["test_r2"]):.4f}')
        for j,name in enumerate(names):
            print(f'  {name}: TEST R²={metrics["test_r2"][j]:.4f}, '
                  f'Pearson r={metrics["test_pearson_r"][j]:+.4f}, '
                  f'RMSE={metrics["test_rmse"][j]:.4f}, '
                  f'VAL R²={metrics["validation_r2"][j]:.4f}, '
                  f'Ridge alpha={metrics["selected_ridge_alpha"][j]:g}')
    print('--- SELECTED PCs minus FULL Z: TEST R² changes ---')
    for row in comparison_deltas:
        print(f'  {row["parameter"]}: '
              f'{row["delta_test_r2_pca_minus_full_z"]:+.4f}')
    print('Selected original Z (TRAIN var > '
          f'{z_threshold:g}): {", ".join(selected_z_names)}')
    print('Three-way Ridge CSV:',folder/'s12__theta_hat_ridge_3way.csv')
    print('Input comparison CSV:', folder/'s12__ridge_input_comparison.csv')
    print('Comparison scatter:', folder/'s12__ridge_input_comparison_scatter.png')
    print('Comparison R² plot:', folder/'s12__ridge_input_comparison_r2.png')
    print('Regression results:', folder)


if __name__ == '__main__':
    regression_report(parse_args('S7: held-out linear parameter regression on Z and PCA',
                                 configure_parser=configure_s7_parser))