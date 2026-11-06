"""Export the completed fixed-method audit to a user-facing folder.

Selects matching current source hashes, never theta recovery. Requires all four
benchmarks complete. Figures/CSV/JSON remain ignored, research notes are tracked.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import argparse
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rotations.stimulus_contrast.data import sha256
from rotations.stimulus_contrast.run import write_json,write_csv


def find_current(model):
    candidates=[]
    base=ROOT/'outputs'/'choice'/model/'rotations'/'stimulus_contrast'/'beta0p0005_lambda0_ctx0p25to0p8_L8'/'repeat01'
    for run in base.glob('run_*'):
        if not (run/'complete.json').exists():continue
        fit=json.loads((run/'fit.json').read_text())
        hashes=fit['environment'].get('source_sha256',{})
        if not hashes:continue
        if all((ROOT/path).exists() and sha256(ROOT/path)==value for path,value in hashes.items()):
            candidates.append((run,fit))
    if len(candidates)!=1:raise RuntimeError(f'{model}: expected one exact-source completed run, got {len(candidates)}')
    return candidates[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    rows=[];manifest={};details={};models=['C1','C2','C3','CR3'];runs={}
    for model in models:
        run,fit=find_current(model);runs[model]=run
        evaluation=json.loads((run/'evaluation.json').read_text());details[model]=evaluation
        manifest[model]=dict(run=str(run),fit=fit)
        for method,report in evaluation.items():
            rows.append(dict(model=model,method=method,
                **{k:report[k] for k in ('train_matched_test_mean_r2','test_optimal_mean_r2','min_r2','offtarget_mean_r2','offtarget_max_abs_r','contrast_loss')},
                parameter_r2=json.dumps(report['parameter_r2']),source_run=str(run)))
        target=dest/'details'/model;target.mkdir(parents=True,exist_ok=True)
        for name in ['evaluation.json','fit.json','comparison.csv','reference_angle_curves.csv','reference_angle_curves.png','contrast_logit__r2.png','identity__r2.png','supervised_reference__r2.png']:
            shutil.copy2(run/name,target/name)
        if (run/'oracle_structure_audit'/'summary.json').exists():
            shutil.copy2(run/'oracle_structure_audit'/'summary.json',target/'decoder_fidelity.json')
        if (run/'profile_fidelity_audit.json').exists():
            shutil.copy2(run/'profile_fidelity_audit.json',target/'profile_fidelity.json')
    write_csv(dest/'all_results.csv',rows);write_json(dest/'signed_correlations.json',details);write_json(dest/'provenance.json',manifest)
    fig,ax=plt.subplots(figsize=(10,4.7));positions=np.arange(4);width=.25
    for off,(name,label,color) in enumerate([('identity','PCA Identity','#68768a'),('contrast_logit','Stimulus contrast (rejected)','#d65b4b'),('supervised_reference','Supervised diagnostic','#3b8c78')]):
        vals=[details[m][name]['train_matched_test_mean_r2'] for m in models]
        bars=ax.bar(positions+(off-1)*width,vals,width,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
    ax.set_xticks(positions,models);ax.set_ylim(0,1.08);ax.set_ylabel('TEST mean Pearson r² (TRAIN axis matching)')
    ax.set_title('Fixed objective audit | exploratory data');ax.legend(frameon=False,loc='upper center',ncol=3,fontsize=8)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(dest/'stage_summary.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(12,15),squeeze=False)
    for row,m in enumerate(models):
        for col,name in enumerate(['identity','contrast_logit','supervised_reference']):
            report=details[m][name];c=np.array(report['signed_test_corr']);ax=axes[row,col]
            ax.imshow(c*c,vmin=0,vmax=1,cmap='Blues')
            ax.set_xticks(range(c.shape[1]),[f'u{i+1}' for i in range(c.shape[1])]);ax.set_yticks(range(c.shape[0]),report['parameter_names'])
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):ax.text(j,i,f'{c[i,j]**2:.2f}',ha='center',va='center',fontsize=8,color='white' if c[i,j]**2>.55 else 'black')
            ax.set_title(f'{m} | {name}',fontsize=10)
    fig.tight_layout();fig.savefig(dest/'all_model_heatmaps.png',dpi=160);plt.close(fig)
    analytic=ROOT/'outputs'/'research'/'stimulus_contrast'/'analytic_v1'
    for name in ['analytic.json','toy_counterexamples.png']:shutil.copy2(analytic/name,dest/name)
    for m in ['C1','CR3']:
        base=ROOT/'outputs'/'choice'/m/'rotations'/'stimulus_contrast'
        shutil.copy2(base/'channel_audit_v1'/'comparison.csv',dest/f'{m}_channel_basis_audit.csv')
        shutil.copy2(base/'weight_audit_v2'/'comparison.csv',dest/f'{m}_weight_audit.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,m in zip(axes,['C1','CR3']):
        pairs=np.load(runs[m]/'profile_fidelity_loss_pairs.npz')['TEST_anchors_loss_pairs']
        ax.scatter(pairs[:,1],pairs[:,0],s=18,alpha=.7)
        limits=[float(pairs.min()),float(pairs.max())];ax.plot(limits,limits,c='gray',lw=1)
        ax.set_xlabel('Frozen decoder raw-X contrast loss');ax.set_ylabel('Profile proxy raw-X contrast loss')
        ax.set_title(f'{m}: same subjects, X and loss\nr={np.corrcoef(pairs.T)[0,1]:.3f} across 64 fixed rotations')
    fig.tight_layout();fig.savefig(dest/'profile_objective_fidelity.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(output=str(dest),source_runs={k:str(v) for k,v in runs.items()}),indent=2))


if __name__=='__main__':main()
