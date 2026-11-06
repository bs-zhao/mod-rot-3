from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S4: Display and plot original NP checkpoint validation/test behavior.

This stage reads fixed-context test metrics saved by the S3 trainer; it does
not retrain or rerun inference. If S5's matching frozen all-trial Z exists,
it also reports the posterior means' distributions and per-coordinate KL.
No true simulator theta is used.
"""
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modrot3.cli import parse_args
from modrot3.pipeline import train_path, stage_folder, condition, sha256, dataset_path, latent_path
from modlearn_rot.training.continuous import load_checkpoint


def _posterior_z_statistics(mu, logvar, splits):
    """Descriptive original-coordinate mu statistics; exact diagonal-Gaussian KL.

    All-subject summaries describe S5's full-trial posterior, not the training
    stochastic-context distribution. TRAIN-only PCA matches S6's convention.
    """
    mu=np.asarray(mu,dtype=np.float64)
    lv=np.asarray(logvar,dtype=np.float64)
    if mu.ndim!=2 or mu.shape!=lv.shape or mu.shape[0]<2 or mu.shape[1]<1:
        raise ValueError('S5 posterior mean and logvar must match nonempty 2D shapes')
    if not np.isfinite(mu).all() or not np.isfinite(lv).all():
        raise ValueError('Non-finite S5 posterior means or log-variances')
    if np.any(lv>700):
        raise ValueError('S5 posterior log-variance too large to evaluate KL')
    n,dim=mu.shape
    train=np.asarray(splits['train'],dtype=int)
    if len(train)<2:
        raise ValueError('At least two TRAIN subjects required for PCA diagnostics')
    all_ids=np.concatenate([np.asarray(splits[k],dtype=int)
                            for k in ('train','validation','test')])
    if not np.array_equal(np.sort(all_ids),np.arange(n)):
        raise ValueError('S5 subject splits are not disjoint and exhaustive')
    var_post=np.exp(lv)
    kl_per_subject_axis=.5*(np.square(mu)+var_post-1.-lv)
    if not np.isfinite(kl_per_subject_axis).all():
        raise ValueError('Non-finite per-subject posterior KL')
    rows=[]
    for j in range(dim):
        z=mu[:,j]
        q=np.percentile(z,[1,5,25,50,75,95,99])
        rows.append(dict(
            dimension=f'z{j+1}',n_subjects=n,
            mean=float(np.mean(z)),
            variance=float(np.var(z,ddof=0)),
            std=float(np.std(z,ddof=0)),
            minimum=float(np.min(z)),
            p01=float(q[0]),p05=float(q[1]),p25=float(q[2]),
            median=float(q[3]),p75=float(q[4]),p95=float(q[5]),
            p99=float(q[6]),maximum=float(np.max(z)),
            mean_abs=float(np.mean(np.abs(z))),
            posterior_variance_mean=float(np.mean(var_post[:,j])),
            posterior_variance_median=float(np.median(var_post[:,j])),
            posterior_logvar_mean=float(np.mean(lv[:,j])),
            mean_kl_nats=float(np.mean(kl_per_subject_axis[:,j])),
            aggregate_posterior_variance=float(np.var(z,ddof=0)+
                                               np.mean(var_post[:,j])),
        ))
    # TRAIN-only PCA for a basis-invariant diagnostic of posterior means.
    cov=np.cov(mu[train],rowvar=False,ddof=1)
    cov=np.atleast_2d(cov)
    eig=np.maximum(np.linalg.eigvalsh(cov)[::-1],0.)
    denom=float(np.sum(eig**2))
    effective_dim=float(np.sum(eig)**2/denom) if denom>0 else 0.
    total=float(eig.sum())
    ratios=np.divide(eig,total,out=np.zeros_like(eig),where=total>0)
    # Correlations across original posterior-mean Z coordinates; rank of
    # covariance is NOT necessarily the number of noncollapsed posterior axes.
    def safe_pearson_cross_axes(values):
        values=np.asarray(values,dtype=np.float64)
        if values.ndim!=2 or values.shape[1]!=dim:
            raise ValueError('Unexpected Z correlation input')
        if len(values)<2:
            return np.zeros((dim,dim),dtype=float)
        centered=values-values.mean(axis=0,keepdims=True)
        c=centered.T@centered
        scale=np.sqrt(np.maximum(np.diag(c),0.))
        d=scale[:,None]*scale[None,:]
        out=np.divide(c,d,out=np.zeros((dim,dim),float),where=d>1e-24)
        np.fill_diagonal(out,np.where(scale>1e-12,1.,0.))
        return np.clip(out,-1.,1.)
    correlation_matrices=dict(
        all=safe_pearson_cross_axes(mu),
        train=safe_pearson_cross_axes(mu[train]),
        validation=safe_pearson_cross_axes(mu[np.asarray(splits['validation'],int)]),
        test=safe_pearson_cross_axes(mu[np.asarray(splits['test'],int)]),
    )
    summary=dict(
        definition='S5 all-observed-trials posterior q(z|all trials): distribution '
                   'of posterior means, NOT samples from q(z|context)',
        latent_dim=dim,
        subjects=n,
        split_sizes={k:int(len(splits[k])) for k in ('train','validation','test')},
        moments_ddof=0,
        per_dimension=rows,
        posterior_mean_correlations={
            name:matrix.tolist() for name,matrix in correlation_matrices.items()
        },
        correlation_definition='Pearson correlations across subjects of raw '
            'posterior means; any zero-variance coordinate yields 0 correlations '
            'and a 0 diagonal as an explicitly defined plotting convention',
        mean_total_kl_nats=float(np.mean(np.sum(kl_per_subject_axis,axis=1))),
        sum_coordinate_mean_kl_nats=float(sum(r['mean_kl_nats'] for r in rows)),
        train_mean_pca_eigenvalues=eig.tolist(),
        train_mean_pca_explained_variance_ratio=ratios.tolist(),
        train_mean_pca_participation_ratio=effective_dim,
        interpretation='Small posterior-mean variance alone is NOT proof of '
                       'posterior collapse; inspect posterior variance and KL. '
                       'PCA axes are rotations, not original z coordinates.',
    )
    return rows,summary


def _save_z_diagnostics(args, folder, checkpoint_sha, dataset_sha, model_state):
    """Use S5 only when present and proven to match this S3 checkpoint."""
    source=latent_path(args)
    if not source.is_file():
        print('Z distributions not generated: matching S5 file is absent:',source)
        print('Run np/s5_extract_z.py with the same --model/--beta/'
              '--latent-dim/--repeat, then rerun np/s4_evaluate.py.')
        return dict(status='skipped_missing_s5',source=str(source))
    with np.load(source,allow_pickle=False) as f:
        required={'z','z_logvar','checkpoint_sha256','dataset_sha256',
                  'context_mode','train_ids','validation_ids','test_ids'}
        if not required.issubset(f.files):
            raise RuntimeError('S5 file lacks posterior diagnostics or provenance fields')
        if str(f['checkpoint_sha256'])!=checkpoint_sha:
            raise RuntimeError('S5 Z is from a different S3 checkpoint')
        if str(f['dataset_sha256'])!=dataset_sha:
            raise RuntimeError('S5 Z is from a different S1 dataset')
        if str(f['context_mode'])!='all_observed_trials':
            raise RuntimeError('S5 Z must be encoded from all observed trials')
        mu=np.asarray(f['z'],dtype=np.float64)
        lv=np.asarray(f['z_logvar'],dtype=np.float64)
        ids={k:np.asarray(f[k+'_ids'],dtype=np.int64)
             for k in ('train','validation','test')}
    if mu.ndim!=2 or mu.shape[1]!=int(model_state['model_config']['latent_dim']):
        raise RuntimeError('S5 latent dimensions differ from S3')
    for k in ('train','validation','test'):
        expected=np.asarray(model_state['subject_splits'][k],dtype=np.int64)
        if not np.array_equal(ids[k],expected):
            raise RuntimeError(f'S5 and S3 differ in {k} subject split')
    rows,summary=_posterior_z_statistics(mu,lv,ids)
    summary['status']='complete'
    summary['source_s5']=str(source)
    summary['source_s5_sha256']=sha256(source)
    write_path=folder/'s3_evaluate__z_dimension_stats.csv'
    with write_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder/'s3_evaluate__z_diagnostics.json').write_text(
        json.dumps(summary,indent=2),encoding='utf-8')

    # Each z panel shows ALL subjects, with split outlines for stability.
    dim=mu.shape[1]
    ncols=min(4,dim)
    nrows=math.ceil(dim/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,3.1*nrows),
                          squeeze=False)
    colors={'train':'tab:blue','validation':'tab:orange','test':'tab:green'}
    for j,ax in enumerate(axes.flat):
        if j>=dim:
            ax.axis('off')
            continue
        values=mu[:,j]
        edges=np.histogram_bin_edges(values,bins=30)
        if edges[0]==edges[-1]:
            edges=np.linspace(edges[0]-.5,edges[0]+.5,31)
        ax.hist(values,bins=edges,density=True,alpha=.30,
                color='gray',label='ALL')
        for split_name in ('train','validation','test'):
            sel=ids[split_name]
            if len(sel):
                ax.hist(values[sel],bins=edges,density=True,
                        histtype='step',lw=1.2,
                        color=colors[split_name],
                        label=split_name.upper())
        ax.axvline(0,color='black',ls=':',lw=1,alpha=.6)
        record=rows[j]
        ax.set_title(f'z{j+1}: mean={record["mean"]:.3g}, '
                     f'var={record["variance"]:.3g}\n'
                     f'KL={record["mean_kl_nats"]:.3g} nats')
        ax.set_xlabel('Posterior mean mu (all observed trials)')
        ax.set_ylabel('Density')
        if j==0:
            ax.legend(fontsize=7)
    fig.suptitle(f'{args.model}: full-trial posterior means by original Z axis '
                 f'(beta={args.beta:g})',fontsize=13)
    fig.tight_layout()
    fig.savefig(folder/'s3_evaluate__z_dimension_histograms.png',dpi=180)
    plt.close(fig)

    idx=np.arange(dim)
    fig,axs=plt.subplots(1,2,figsize=(max(10,dim*1.4),4.1))
    axs[0].bar(idx-.18,[r['variance'] for r in rows],width=.36,
               label='Between-subject Var(mu)')
    axs[0].bar(idx+.18,[r['posterior_variance_mean'] for r in rows],
               width=.36,label='Mean posterior Var(z|context)')
    axs[0].axhline(1,ls=':',color='gray',lw=1,label='Prior Var = 1')
    axs[0].set(xlabel='Original latent coordinate',ylabel='Variance',
               title='Mean variance vs posterior uncertainty')
    axs[0].set_xticks(idx,[r['dimension'] for r in rows])
    axs[0].legend(fontsize=7)
    axs[1].bar(idx,[r['mean_kl_nats'] for r in rows])
    axs[1].set(xlabel='Original latent coordinate',
               ylabel='Mean KL to N(0,1) (nats)',
               title='Per-axis diagonal Gaussian KL')
    axs[1].set_xticks(idx,[r['dimension'] for r in rows])
    fig.tight_layout()
    fig.savefig(folder/'s3_evaluate__z_variance_and_kl.png',dpi=180)
    plt.close(fig)

    # Save every split's numeric matrix, plus a labeled, annotated heatmap.
    corr_matrices=summary['posterior_mean_correlations']
    for split_name,values in corr_matrices.items():
        csv_path=folder/f's3_evaluate__z_correlations_{split_name}.csv'
        with csv_path.open('w',newline='',encoding='utf-8') as fp:
            writer=csv.writer(fp)
            writer.writerow(['dimension']+[f'z{i+1}' for i in range(dim)])
            for i,row in enumerate(values):
                writer.writerow([f'z{i+1}',*row])
    fig,axs=plt.subplots(1,2,figsize=(max(10,dim*1.3),5.3),squeeze=False)
    for ax,split_name in zip(axs[0],('all','train')):
        mat=np.asarray(corr_matrices[split_name],dtype=float)
        im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(np.arange(dim),[f'z{i+1}' for i in range(dim)],
                      rotation=45,ha='right')
        ax.set_yticks(np.arange(dim),[f'z{i+1}' for i in range(dim)])
        ax.set_title(f'{split_name.upper()} posterior-mean Pearson r')
        for i in range(dim):
            for j in range(dim):
                val=mat[i,j]
                ax.text(j,i,f'{val:+.2f}',ha='center',va='center',
                        fontsize=7,color='white' if abs(val)>.65 else 'black')
    fig.colorbar(im,ax=list(axs[0]),shrink=.75,label='Pearson r')
    fig.savefig(folder/'s3_evaluate__z_correlation_heatmap.png',
                dpi=180,bbox_inches='tight')
    plt.close(fig)

    eig=np.asarray(summary['train_mean_pca_eigenvalues'])
    x=np.arange(1,dim+1)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].bar(x,eig)
    axs[0].set(xlabel='PC of posterior means (TRAIN only)',
               ylabel='Absolute PCA eigenvalue',
               title='Between-subject latent eigen-spectrum')
    axs[1].bar(x,summary['train_mean_pca_explained_variance_ratio'])
    axs[1].plot(x,np.cumsum(summary['train_mean_pca_explained_variance_ratio']),
                'o-',label='Cumulative ratio')
    axs[1].set(xlabel='PC',ylabel='Explained variance ratio',
               title=f'Participation ratio = '
                     f'{summary["train_mean_pca_participation_ratio"]:.2f}')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(folder/'s3_evaluate__z_pca_spectrum.png',dpi=180)
    plt.close(fig)

    print('Original Z dimension diagnostics: posterior mean mu across '
          f'{len(mu)} subjects (ALL observed trials)')
    print('  Axis         Mean         Std      Var(mu)   E[Var(z|C)]     Mean KL')
    for r in rows:
        print(f'  {r["dimension"]:<6} {r["mean"]:>11.5g} '
              f'{r["std"]:>11.5g} {r["variance"]:>12.5g} '
              f'{r["posterior_variance_mean"]:>13.5g} '
              f'{r["mean_kl_nats"]:>11.5g}')
    print(f'  Sum mean KL (nats/subject): {summary["mean_total_kl_nats"]:.6g}')
    print(f'  TRAIN posterior-mean PCA participation ratio: '
          f'{summary["train_mean_pca_participation_ratio"]:.4f}')
    print('  Posterior Z histogram:',folder/'s3_evaluate__z_dimension_histograms.png')
    print('  Per-axis stats:',write_path)
    print('  Posterior-mean Pearson correlation matrix (ALL subjects):')
    print(np.array2string(np.asarray(corr_matrices['all']),precision=3,
                          suppress_small=True))
    print('  Correlation heatmap:',folder/'s3_evaluate__z_correlation_heatmap.png')
    print('  Correlation matrices: s3_evaluate__z_correlations_{all,train,validation,test}.csv')
    print('  Variance and KL plot:',folder/'s3_evaluate__z_variance_and_kl.png')
    return summary

def report(args):
    ckpt=train_path(args)
    if not ckpt.is_file():
        raise FileNotFoundError('Train first: '+str(ckpt))
    state=load_checkpoint(ckpt)
    if state.get('extra_state',{}).get('dataset_sha256')!=sha256(dataset_path(args.model)):
        raise RuntimeError('NN checkpoint dataset provenance mismatch')
    folder=stage_folder(args,'s3_evaluate')
    folder.mkdir(parents=True,exist_ok=True)
    hist=state['history']
    summary=dict(model=args.model,condition=condition(args),
                 best_epoch=int(state['best_epoch'])+1,
                 actual_epochs=len(hist['train_loss']),
                 model_config=state['model_config'],
                 training_config=state['training_config'],
                 validation=state['validation_metrics'],
                 test=state['test_metrics'],
                 checkpoint_sha256=sha256(ckpt),
                 metric_protocol='original frozen validation/test subject context masks; sampled choice reconstruction targets')
    (folder/'s3_evaluate__summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (folder/'s3_evaluate__history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch',*hist.keys()])
        writer.writeheader()
        for i in range(len(hist['train_loss'])):
            writer.writerow(dict(epoch=i+1,**{k:v[i] for k,v in hist.items()}))
    fig,axs=plt.subplots(2,3,figsize=(13,7.5))
    for ax,key in zip(axs.flat,('loss','reconstruction','accuracy','brier','kl','sparse')):
        ax.plot(hist['train_'+key],label='TRAIN')
        ax.plot(hist['validation_'+key],label='VAL')
        ax.axvline(int(state['best_epoch']),ls='--',c='k',alpha=.4)
        ax.set_title(key);ax.set_xlabel('Epoch');ax.legend()
    fig.suptitle(f'{args.model}: original binary-choice NP learning curves')
    fig.tight_layout();fig.savefig(folder/'s3_evaluate__learning_curves.png',dpi=170);plt.close(fig)
    # S4 is usable before S5. When S5 exists, append descriptive Z/KL
    # diagnostics without altering the frozen checkpoint or S5 file.
    summary['z_diagnostics']=_save_z_diagnostics(
        args,folder,summary['checkpoint_sha256'],sha256(dataset_path(args.model)),state)
    (folder/'s3_evaluate__summary.json').write_text(
        json.dumps(summary,indent=2),encoding='utf-8')
    print(args.model,'original NP validation metrics:',state['validation_metrics'])
    print(args.model,'original NP TEST metrics:',state['test_metrics'])
    print('Saved NN evaluation:',folder)

if __name__ == '__main__':
    report(parse_args('S4: evaluate saved NP checkpoint and training curves'))