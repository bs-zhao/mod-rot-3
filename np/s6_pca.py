from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S6: PCA of frozen full-trial Z, with variance and theta-correlation plots.

PCA is fitted ONLY on TRAIN subjects and then applied to all subjects.
No regression fitting is performed in this stage.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from modrot3.cli import parse_args
from modrot3.pipeline import stage_folder, sha256, train_path, condition
from modrot3.analysis import load_frozen, corr, param_names, write_rows, plot_corr


def pca_report(args):
    z, theta, splits, source = load_frozen(args)
    tr, te = splits['train'], splits['test']
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
    names=param_names(args.model)
    raw_corr=corr(theta[te], z[te])
    pc_corr=corr(theta[te], scores[te])
    report=dict(model=args.model,condition=condition(args),source_latent_sha256=source_sha,
        parameter_names=names, train_subjects=int(len(tr)),
        explained_variance_ratio=explained.tolist(),
        cumulative_explained_variance=np.cumsum(explained).tolist(),
        eigenvalues=fitted.explained_variance_.tolist(),
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
    print('PCA results:',folder)


if __name__ == '__main__':
    pca_report(parse_args('S6: PCA analysis of all-trial latent Z'))