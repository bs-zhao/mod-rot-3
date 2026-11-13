"""Save every retained criterion's raw angle curves and loss-selected R/T plots.

The finite search consumes only frozen W/Y scores and case W. Evaluation is a
separate function, called after R and its loss have been written and hashed.
Continuous-search results can replace the finite solution only if their exact
criterion, channel and source score hash match. No original supervised R is read.
"""
import argparse
import csv
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import run as wave
from record_rotation_optima import WAVES
from write_approximate_shortlist import CANDIDATES

MODELS=('CR3','C1','C2','C3')
OUTPUT=wave.ROOT/'outputs/supervised_abs_rotation/retained_metric_rotation_gallery'
_DATASET_DIGESTS={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read_source(model,tag,metric,channel):
    source=wave.io.default_step2(wave.ROOT,model)
    # CH and CH2 use exactly the same estimator and occupancy=20. CH2 is
    # the completed four-model repeat; this alias is recorded in the manifest.
    effective='CH2' if tag=='CH' and model!='CR3' else tag
    folder=source.parent/'step5_matrix_metrics'/WAVES[effective]
    if not (folder/'reveal.json').exists():return None
    p=json.loads((folder/'protocol.json').read_text())
    if p['status']!='COMPLETE':return None
    digest=sha(folder/'blind_scores.csv')
    if digest!=p['scores_sha256']:raise ValueError(f'Changed scores: {folder}')
    mapping=json.loads((folder/'reveal.json').read_text())['mapping']
    groups={}
    with (folder/'blind_scores.csv').open() as f:
        for r in csv.DictReader(f):
            if r['channel']==channel and r['metric']==metric and r['anonymous_case'] in mapping:
                groups.setdefault(r.get('estimator',''),{})[mapping[r['anonymous_case']]]=float(r['loss'])
    if len(groups)!=1:raise ValueError(f'Expected one estimator for {model}/{tag}/{metric}: {list(groups)}')
    losses=next(iter(groups.values()))
    index=json.loads((source/'index.json').read_text())
    if set(losses)!={c['id'] for c in index['cases']}:raise ValueError('Incomplete angle curve')
    if not np.isfinite(list(losses.values())).all():raise ValueError('Nonfinite loss')
    return dict(source=source,folder=folder,index=index,losses=losses,source_tag=effective,scores_sha256=digest)


def freeze_grid_search(info,output,metric,channel):
    """Exhaustive loss minimization over saved W frames; no T is available here."""
    output.mkdir(parents=True,exist_ok=True)
    losses=info['losses'];chosen=min(losses,key=lambda k:(losses[k],k!='baseline',k))
    source=info['source'];index=info['index'];stage1=Path(index['source_stage1'])
    counts=np.load(source/'trial_counts.npy',allow_pickle=False)
    offsets=np.r_[0,np.cumsum(counts)[:-1]]
    z=np.load(stage1/'Zstd.npy',allow_pickle=False)
    w=np.load(source/'cases'/chosen/'W.npy',mmap_mode='r',allow_pickle=False)
    u=np.asarray(w[offsets,:index['Z_dim']],dtype=float)
    a,error=wave.io.registration(z,u);r=a.T
    np.save(output/'R.npy',r);np.save(output/'U.npy',z@r.T)
    result=dict(status='FROZEN',search_scope='exhaustive saved case grid',
        global_optimality_proven=False,metric=metric,channel=channel,chosen_case=chosen,
        loss=losses[chosen],baseline_loss=losses['baseline'],grid_points=len(losses),
        source_scores_sha256=info['scores_sha256'],source_tag=info['source_tag'],
        all_subjects=len(z),R_application='U = Zstd @ R.T; equivalently U_column = R @ Zstd_column',
        coordinate_registration_relative_error=error,T_used_by_search=False,
        original_saved_supervised_R_read=False,R_sha256=sha(output/'R.npy'),U_sha256=sha(output/'U.npy'))
    wave.json_file(output/'search_freeze.json',result)
    return result


def evaluate_frozen(info,output):
    """T is read only after the immutable search result exists."""
    frozen=json.loads((output/'search_freeze.json').read_text())
    for filename in ('R.npy','U.npy'):
        if sha(output/filename)!=frozen[filename[0]+'_sha256']:raise ValueError('Changed frozen rotation')
    stage1=Path(info['index']['source_stage1']);manifest=json.loads((stage1/'manifest.json').read_text())
    path=Path(manifest['source_dataset'])
    with np.load(path,allow_pickle=False) as f:
        t=np.asarray(f['parameters'],dtype=float)
        for key in ('subject_index','trial_index','trial_counts'):
            if not np.array_equal(f[key],np.load(info['source']/(key+'.npy'),allow_pickle=False)):
                raise ValueError('Original T subject/trial alignment failed')
    u=np.load(output/'U.npy',allow_pickle=False);z=np.load(stage1/'Zstd.npy',allow_pickle=False)
    if t.ndim!=2 or t.shape[0]!=u.shape[0] or not np.isfinite(t).all():raise ValueError('Invalid original T')
    stat=path.stat();signature=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if signature not in _DATASET_DIGESTS:_DATASET_DIGESTS[signature]=sha(path)
    if _DATASET_DIGESTS[signature]!=manifest['source_dataset_sha256']:raise ValueError('Original T dataset SHA mismatch')
    model=info['index'].get('model',info['source'].name)
    shared=OUTPUT/'source_coordinates'/model;shared.mkdir(parents=True,exist_ok=True)
    for name,value in [('T.npy',t),('Zstd.npy',z)]:
        if not (shared/name).exists():np.save(shared/name,value)
    def correlation(x,y):
        x=x-x.mean(0);y=y-y.mean(0)
        return x.T@y/np.sqrt(np.sum(x*x,0)[:,None]*np.sum(y*y,0)[None,:])
    pearson=correlation(u,t);original=correlation(z,t)
    sr=spearmanr(np.column_stack((u,t)),axis=0).statistic[:u.shape[1],u.shape[1]:]
    np.save(output/'pearson_U_T.npy',pearson);np.save(output/'spearman_U_T.npy',sr)
    for name,array in (('pearson_U_T',pearson),('spearman_U_T',sr),('pearson_Z_T',original)):
        with (output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['axis']+[f'T{i+1}' for i in range(t.shape[1])])
            writer.writerows([[f'U{i+1}' if name!='pearson_Z_T' else f'Z{i+1}',*row] for i,row in enumerate(array)])
    wave.json_file(output/'evaluation.json',dict(T_source=str(path),T_source_declared_sha256=manifest['source_dataset_sha256'],
        T_dataset_SHA_verified=True,shared_Zstd_and_T=str(shared.resolve()),
        parameters_shape=list(t.shape),subject_trial_alignment_verified=True,T_read_after_R_frozen=True,
        T_used_to_choose_R=False,no_axis_permutation_or_sign_alignment_in_heatmap=True,
        search_freeze_sha256=sha(output/'search_freeze.json')))
    return pearson,original


def heatmap(ax,array,title):
    ax.imshow(array,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
    ax.set_xticks(range(array.shape[1]),[f'T{i+1}' for i in range(array.shape[1])])
    ax.set_yticks(range(array.shape[0]),[f'U{i+1}' for i in range(array.shape[0])])
    ax.set_title(title,fontsize=10)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):ax.text(j,i,f'{array[i,j]:.2f}',ha='center',va='center',fontsize=9,
            color='white' if abs(array[i,j])>.65 else 'black')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);inventory=[];summary=[]
    for number,candidate in enumerate(CANDIDATES,1):
        label,tag,metric=candidate[:3];channel=candidate[3] if len(candidate)>3 else 'logit_nn'
        folder=OUTPUT/f'{number:02d}_{tag}_{channel}';folder.mkdir(exist_ok=True)
        wave.json_file(folder/'criterion.json',dict(number=number,label=label,wave=tag,metric=metric,channel=channel))
        fig,axes=plt.subplots(2,2,figsize=(13,9));hfig,haxes=plt.subplots(2,2,figsize=(11,9));raw=[]
        for model,ax,hax in zip(MODELS,axes.flat,haxes.flat):
            info=read_source(model,tag,metric,channel)
            if info is None:
                ax.set_title(f'{model}: source not complete');hax.set_title(f'{model}: source not complete')
                inventory.append(dict(number=number,wave=tag,model=model,status='SOURCE_PENDING'));continue
            meta={c['id']:c for c in info['index']['cases']};losses=info['losses']
            pairs=sorted({tuple(c['pair_1based']) for c in meta.values() if c.get('pair_1based')})
            for plane_index,pair in enumerate(pairs):
                points=[(0,losses['baseline'],'baseline')]+sorted((int(meta[k]['angle_deg']),v,k)
                    for k,v in losses.items() if tuple(meta[k].get('pair_1based') or [])==pair)
                ax.plot([p[0] for p in points],[p[1] for p in points],marker='.',label=f'U{pair[0]}-U{pair[1]}',lw=1,
                    color=plt.get_cmap('tab20')(plane_index))
                raw.extend(dict(model=model,pair=str(pair),angle_deg=a,case=k,loss=v) for a,v,k in points)
            ax.set_title(f'{model}: {len(losses)} saved rotations');ax.set_xlabel('Rotation angle (degrees)');ax.set_ylabel('Raw criterion loss')
            ax.set_xticks(range(0,91,15));ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=3)
            result=freeze_grid_search(info,folder/model/'grid_search',metric,channel)
            corr,_=evaluate_frozen(info,folder/model/'grid_search')
            heatmap(hax,corr,f'{model}: grid minimum = {result["loss"]:.6g}')
            inventory.append(dict(number=number,wave=tag,model=model,status='COMPLETE_GRID',source=str(info['folder'])))
            summary.append(dict(number=number,wave=tag,model=model,channel=channel,metric=metric,
                minimum_loss=result['loss'],baseline_loss=result['baseline_loss'],chosen_case=result['chosen_case'],
                search_scope=result['search_scope'],output=str((folder/model/'grid_search').resolve())))
        title=f'{number:02d} / {tag} / {channel}\n'+'\n'.join(textwrap.wrap(metric,100))
        fig.suptitle(title,fontsize=10);fig.tight_layout(rect=(0,0,1,.92))
        hfig.suptitle(title+'\nPearson corr(U, T); signed axes, no post-hoc matching',fontsize=9)
        hfig.tight_layout(rect=(0,0,1,.89))
        for ext in ('png','pdf'):
            fig.savefig(folder/f'four_model_angle_curves.{ext}',dpi=170)
            hfig.savefig(folder/f'four_model_grid_minimum_U_T_heatmaps.{ext}',dpi=170)
        plt.close(fig);plt.close(hfig);wave.csv_file(folder/'angle_curves.csv',raw)
        wave.csv_file(OUTPUT/'inventory.csv',inventory);wave.csv_file(OUTPUT/'grid_search_summary.csv',summary)
        print(f'[Gallery] {number}/44 {tag} {channel}: saved curves, R, U, correlations',flush=True)
    wave.json_file(OUTPUT/'protocol.json',dict(status='COMPLETE' if all(r['status']=='COMPLETE_GRID' for r in inventory) else 'SOURCE_PENDING',
        candidates=len(CANDIDATES),completed_model_candidates=len(summary),expected_model_candidates=4*len(CANDIDATES),
        T_policy='evaluation only, after R and U hash freeze',global_optimality_proven=False))


if __name__=='__main__':main()
