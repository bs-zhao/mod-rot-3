"""Exact loss selection over all saved scored frames and matching frozen searches.

This is a finite candidate-pool search, not a claim of a global optimum. Source
W stress frames and continuous searches are included only for the same metric,
response channel and immutable source score table.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run as wave
from write_approximate_shortlist import CANDIDATES
from export_candidate_rotation_gallery import OUTPUT,MODELS,read_source,sha,evaluate_frozen,heatmap


def select_saved_frames(info,output,metric,channel):
    source=info['source'];folder=info['folder'];index=info['index'];output.mkdir(parents=True,exist_ok=True)
    mapping=json.loads((folder/'reveal.json').read_text())['mapping']
    with (folder/'blind_scores.csv').open() as f:
        scores=[r for r in csv.DictReader(f) if r['metric']==metric and r['channel']==channel]
    counts=np.load(source/'trial_counts.npy',allow_pickle=False);offsets=np.r_[0,np.cumsum(counts)[:-1]]
    z=np.load(Path(index['source_stage1'])/'Zstd.npy',allow_pickle=False);d=z.shape[1]
    anchor=np.load(source/'cases'/mapping['case_000']/'W.npy',mmap_mode='r',allow_pickle=False)
    anonymous_u=np.asarray(anchor[offsets,:d],dtype=float)
    a,error=wave.io.registration(z,anonymous_u)
    geometry=folder/'W_only_geometry.npz';saved_geometry=None
    if geometry.exists():
        with np.load(geometry,allow_pickle=False) as f:saved_geometry={k:f[k].copy() for k in f.files}
    candidates=[]
    for row in scores:
        name=row['anonymous_case'];q=None;r=None
        if saved_geometry is not None:
            k=int(name.rsplit('_',1)[1])+(len(mapping) if name.startswith('audit_frame_') else 0)
            if k<len(saved_geometry['membership']):q=saved_geometry['directions'][saved_geometry['membership'][k]].T
        if q is not None:r=(a@q).T
        elif name in mapping:
            w=np.load(source/'cases'/mapping[name]/'W.npy',mmap_mode='r',allow_pickle=False)
            b,_=wave.io.registration(z,np.asarray(w[offsets,:d],dtype=float));r=b.T
        if r is not None:candidates.append(dict(loss=float(row['loss']),R=r,origin=str(folder),frame=name,
            kind='saved case grid' if name in mapping else 'saved off-grid stress frame'))
    source_protocol=json.loads((folder/'protocol.json').read_text())
    if source_protocol.get('primary_blind_metric')==metric and source_protocol.get('primary_blind_channel')==channel:
        frozen=json.loads((folder/'blind_freeze.json').read_text());file=folder/'blind_rotations.npz'
        if sha(file)!=frozen['rotation_scores_sha256']:raise ValueError('Frozen radial GP search changed')
        with np.load(file,allow_pickle=False) as f:
            for name in ('random','optimized'):
                for j,(q,loss) in enumerate(zip(f[name],f[name+'_loss'])):
                    candidates.append(dict(loss=float(loss),R=(a@q).T,origin=str(folder),frame=f'{name}_{j}',kind='prior continuous exact-loss search'))
    legacy_kernel={
        'with_U_main_Ui_innovation_with_context_raw_UX_columns_worst_soft_count_eps010':'waveBK2_complete_response_blind_seed456'}
    if channel=='logit_nn' and metric in legacy_kernel:
        legacy=folder.parent/legacy_kernel[metric]
        if (legacy/'protocol.json').exists():
            p=json.loads((legacy/'protocol.json').read_text());frozen=json.loads((legacy/'blind_freeze.json').read_text())
            if p['source_scores_sha256']!=info['scores_sha256']:raise ValueError('Conditional-kernel source mismatch')
            if sha(legacy/'blind_rotations.npz')!=frozen['rotations_sha256'] or sha(legacy/'blind_exact_scores.csv')!=frozen['exact_scores_sha256']:
                raise ValueError('Legacy conditional-kernel search changed')
            with np.load(legacy/'blind_rotations.npz',allow_pickle=False) as f:frames=f['exact_evaluated'].copy()
            with (legacy/'blind_exact_scores.csv').open() as f:legacy_rows=list(csv.DictReader(f))
            for j,(q,row) in enumerate(zip(frames,legacy_rows)):
                candidates.append(dict(loss=float(row['loss']),R=(a@q).T,origin=str(legacy),frame=f'exact_evaluated_{j}',kind='prior continuous exact-loss search'))
    if info['source_tag']=='AU' and channel=='logit_nn':
        legacy=folder.parent/'waveAW_tree_blind_bootstrap_seed425'
        if (legacy/'protocol.json').exists():
            p=json.loads((legacy/'protocol.json').read_text())
            if p['original_scores_sha256']!=info['scores_sha256']:raise ValueError('Tree search source mismatch')
            for name,digest in p['blind_file_sha256'].items():
                if sha(legacy/name)!=digest:raise ValueError('Frozen tree search changed')
            with np.load(legacy/'blind_rotation_matrices.npz',allow_pickle=False) as f:
                frames=f['optimized'].copy();random_frames=f['random'].copy()
            with (legacy/'blind_optimization_scores.csv').open() as f:history=list(csv.DictReader(f))
            with (legacy/'blind_random_scores.csv').open() as f:random_rows=list(csv.DictReader(f))
            for j,q in enumerate(frames):
                value=min(float(row['loss']) for row in history if int(row['start'])==j)
                candidates.append(dict(loss=value,R=(a@q).T,origin=str(legacy),frame=f'optimized_{j}',kind='prior continuous exact-loss search'))
            for j,(q,row) in enumerate(zip(random_frames,random_rows)):
                candidates.append(dict(loss=float(row['loss']),R=(a@q).T,origin=str(legacy),frame=f'random_{j}',kind='prior exact-scored random search'))
    # Matching prior exact continuous searches. An unrelated loss is never used
    # to assign a score to a rotation, even if its method name looks similar.
    for other in sorted(folder.parent.glob('wave*/protocol.json')):
        p=json.loads(other.read_text());parent=other.parent
        if p.get('status')!='COMPLETE':continue
        if p.get('metric')!=metric or p.get('channel')!=channel:continue
        if p.get('source_scores_sha256')!=info['scores_sha256']:continue
        rotations=parent/'blind_rotations.npz'
        if not rotations.exists():continue
        digest=sha(rotations)
        if p.get('frames_sha256')!=digest:continue
        with np.load(rotations,allow_pickle=False) as f:
            q=np.asarray(f['optimized']);losses=np.asarray(f['optimized_loss'])
        if q.ndim==2:q=q[None]
        for j,(frame,loss) in enumerate(zip(q,np.atleast_1d(losses))):
            candidates.append(dict(loss=float(loss),R=(a@frame).T,origin=str(parent),frame=f'optimized_{j}',kind='prior continuous exact-loss search'))
    # The original BU2/BU6 files predate source-score hashes in protocols.
    # Bind them to the current source by replaying the exact frozen objective,
    # rather than trusting a family-name match or silently omitting them.
    if info['source_tag']=='BU':
        legacy=folder.parent/'waveBU6_local_Shapley_refinement_seed489'
        p=json.loads((legacy/'protocol.json').read_text())
        if p['status']!='COMPLETE' or sha(legacy/'refined_frames.npz')!=p['frames_sha256'] or sha(legacy/'blind_refined_scores.csv')!=p['scores_sha256']:
            raise ValueError('Legacy BU refinement changed')
        replay_path=output/'legacy_BU6_exact_replay.json'
        replay=json.loads(replay_path.read_text()) if replay_path.exists() else None
        with np.load(legacy/'refined_frames.npz',allow_pickle=False) as f:frames=f['frames'].copy()
        with (legacy/'blind_refined_scores.csv').open() as f:legacy_rows=list(csv.DictReader(f))
        if replay is None or replay['source_scores_sha256']!=info['scores_sha256'] or replay['frames_sha256']!=p['frames_sha256']:
            from shapley_profile_blind import Scorer
            import torch
            from threadpoolctl import threadpool_limits
            with threadpool_limits(4):
                torch.set_num_threads(4)
                scorer=Scorer(source,folder,output/'legacy_replay_scratch')
                values=[scorer.score(q) for q in frames]
            expected=[float(r['final_loss']) for r in legacy_rows]
            if not np.allclose(values,expected,rtol=1e-8,atol=1e-10):raise ArithmeticError('Legacy exact loss replay failed')
            replay=dict(source_scores_sha256=info['scores_sha256'],frames_sha256=p['frames_sha256'],losses=values)
            wave.json_file(replay_path,replay)
            del scorer;torch.cuda.empty_cache()
        for j,(q,loss) in enumerate(zip(frames,replay['losses'])):
            candidates.append(dict(loss=float(loss),R=(a@q).T,origin=str(legacy),frame=f'refined_{j}',kind='prior continuous exact-loss search, replay verified'))
    # Fresh searches performed for this gallery also use the identical frozen
    # criterion. Their R already uses the original Zstd coordinate convention.
    current=output.parent/'continuous_search/search_freeze.json'
    if current.exists():
        p=json.loads(current.read_text())
        if p['source_scores_sha256']!=info['scores_sha256'] or p['metric']!=metric or p['channel']!=channel:
            raise ValueError('Fresh continuous criterion mismatch')
        if sha(current.parent/'R.npy')!=p['R_sha256']:raise ValueError('Fresh R changed')
        candidates.append(dict(loss=p['loss'],R=np.load(current.parent/'R.npy',allow_pickle=False),origin=str(current.parent),
            frame='optimized',kind='fresh continuous exact-loss search'))
    chosen=min(candidates,key=lambda r:r['loss']);r=chosen['R']
    if not np.allclose(r@r.T,np.eye(d),atol=1e-8):raise ArithmeticError('Invalid R')
    np.save(output/'R.npy',r);np.save(output/'U.npy',z@r.T)
    wave.csv_file(output/'candidate_frame_losses.csv',[{k:v for k,v in row.items() if k!='R'} for row in candidates])
    result=dict(status='FROZEN',search_scope='all available exact-scored case/stress frames plus matching frozen continuous optimizers',
        metric=metric,channel=channel,loss=chosen['loss'],baseline_loss=info['losses']['baseline'],
        grid_minimum_loss=min(info['losses'].values()),selected_frame=chosen['frame'],selected_origin=chosen['origin'],
        selected_kind=chosen['kind'],candidate_frames=len(candidates),global_optimality_proven=False,
        all_subjects=len(z),source_scores_sha256=info['scores_sha256'],source_tag=info['source_tag'],
        coordinate_registration_relative_error=error,T_used_by_search=False,original_saved_supervised_R_read=False,
        R_application='U = Zstd @ R.T; equivalently U_column = R @ Zstd_column',
        R_sha256=sha(output/'R.npy'),U_sha256=sha(output/'U.npy'))
    wave.json_file(output/'search_freeze.json',result)
    return result


def main():
    summary=[];pages=[]
    for number,c in enumerate(CANDIDATES,1):
        label,tag,metric=c[:3];channel=c[3] if len(c)>3 else 'logit_nn'
        folder=OUTPUT/f'{number:02d}_{tag}_{channel}';fig,axes=plt.subplots(2,2,figsize=(11,9))
        for model,ax in zip(MODELS,axes.flat):
            info=read_source(model,tag,metric,channel)
            if info is None:raise ValueError(f'Missing source: {model}/{tag}')
            out=folder/model/'best_available_search';result=select_saved_frames(info,out,metric,channel)
            corr,_=evaluate_frozen(info,out)
            heatmap(ax,corr,f'{model}: loss={result["loss"]:.6g}\n{result["selected_kind"]}')
            summary.append(dict(number=number,label=label,model=model,wave=tag,channel=channel,metric=metric,
                loss=result['loss'],grid_minimum_loss=result['grid_minimum_loss'],selected_kind=result['selected_kind'],
                candidate_frames=result['candidate_frames'],output=str(out)))
        fig.suptitle(f'{number:02d} / {tag} / {channel}\nPearson corr(U, T): best available exact-loss rotation\nNo T in loss or rotation selection; global optimum not proven',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.9))
        for ext in ('png','pdf'):fig.savefig(folder/f'four_model_best_available_U_T_heatmaps.{ext}',dpi=170)
        plt.close(fig);pages.append(dict(number=number,folder=folder.name,label=label,metric=metric))
        wave.csv_file(OUTPUT/'best_available_search_summary.csv',summary)
        print(f'[BestAvailable] {number}/44 {tag}: four-model R/U/heatmaps saved',flush=True)
    from html import escape
    cards=[]
    for p in pages:
        name=p['folder'];cards.append(f'<article><h2>{p["number"]:02d}. {escape(p["label"])}</h2>'
            f'<p><code>{escape(p["metric"])}</code></p><p><a href="{name}/angle_curves.csv">曲线 CSV</a> · '
            f'<a href="{name}/four_model_angle_curves.pdf">曲线 PDF</a> · '
            f'<a href="{name}/four_model_best_available_U_T_heatmaps.pdf">热力图 PDF</a></p>'
            f'<img loading="lazy" src="{name}/four_model_angle_curves.png">'
            f'<img loading="lazy" src="{name}/four_model_best_available_U_T_heatmaps.png">'
            '<p>'+ ' · '.join(f'<a href="{name}/{m}/best_available_search/">{m}: R、U、相关矩阵、搜索记录</a>' for m in MODELS)+'</p></article>')
    (OUTPUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>44 个保留指标：四库旋转结果</title>'
        '<style>body{max-width:1300px;margin:40px auto;font-family:Arial,sans-serif;padding:0 24px;background:#f5f6f8;color:#202630}'
        'article{background:white;padding:24px;margin:32px 0;border-radius:12px}img{width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#145ca8}</style>'
        '<h1>44 个保留候选：旋转曲线与参数相关热力图</h1>'
        '<p>每项包括 CR3、C1、C2、C3；使用全部 5000 个被试。Y 仅使用已保存的神经网络 logit 或 p。</p>'
        '<p>横轴为保存 case 的 0–90° 角度，每条线对应一个旋转平面；纵轴为原始指标值，越小越优。0° 是原 case 的基准坐标。</p>'
        '<p>热力图使用该指标在全部已评分 case、额外压力旋转点及精确匹配的已完成连续搜索中的最低 loss 所对应的 R。'
        '这叫“当前可用结果最优”，不是全局最优证明；具体来源见搜索汇总。T 只在 R 保存并冻结后用于评估，不参与选择。</p>'
        '<p>U = Zstd @ R.T；未按 T 调整符号、重排 U。CH 在 C1/C2/C3 使用同一方法的完整 CH2 重复，来源已记录。</p>'
        '<p><a href="best_available_search_summary.csv">最佳可用搜索汇总 CSV</a> · <a href="grid_search_summary.csv">角度网格搜索汇总 CSV</a></p>'
        +''.join(cards)+'</html>',encoding='utf-8')
    wave.json_file(OUTPUT/'best_available_protocol.json',dict(status='COMPLETE',candidates=44,model_candidates=len(summary),
        all_subjects=5000,global_optimality_proven=False,T_used_by_search=False))


if __name__=='__main__':main()
