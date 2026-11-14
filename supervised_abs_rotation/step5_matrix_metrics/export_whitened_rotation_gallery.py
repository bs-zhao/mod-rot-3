"""Post-rotation ZCA whitening of every saved U, with T-only evaluation."""
import csv
import hashlib
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

import run as wave
from write_approximate_shortlist import CANDIDATES

MODELS=('CR3','C1','C2','C3')
KINDS=('best_available_search','grid_search','continuous_search')
SOURCE=wave.ROOT/'outputs/supervised_abs_rotation/retained_metric_rotation_gallery'
OUTPUT=SOURCE/'whitened'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def zca_whiten(u):
    """Population covariance over all subjects; no T and no axis reordering."""
    u=np.asarray(u,dtype=np.float64)
    if u.ndim!=2 or not np.isfinite(u).all():raise ValueError('Invalid U')
    mean=u.mean(0);centered=u-mean;cov=centered.T@centered/len(u)
    eigenvalues,vectors=np.linalg.eigh(cov)
    threshold=max(float(eigenvalues.max()),1e-30)*u.shape[1]*np.finfo(float).eps*10
    keep=eigenvalues>threshold
    inverse=np.zeros_like(eigenvalues);inverse[keep]=1/np.sqrt(eigenvalues[keep])
    a=(vectors*inverse)@vectors.T
    white=centered@a;expected=vectors[:,keep]@vectors[:,keep].T
    actual=white.T@white/len(white)
    error=float(np.max(np.abs(actual-expected)))
    if error>1e-9:raise ArithmeticError('Whitened covariance failed numerical verification')
    return white,mean,a,dict(eigenvalues=eigenvalues.tolist(),rank=int(keep.sum()),dimension=u.shape[1],
        relative_eigenvalue_cutoff=threshold/max(float(eigenvalues.max()),1e-30),
        covariance_error_to_expected_projector=error,covariance_error_to_identity=float(np.max(np.abs(actual-np.eye(u.shape[1])))),
        mean_maximum_absolute=float(np.max(np.abs(white.mean(0)))),covariance_ddof=0)


def corr(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    return (x.T@y)/np.sqrt(np.sum(x*x,axis=0)[:,None]*np.sum(y*y,axis=0)[None,:])


def save_matrix(folder,name,array,row='U',column='T'):
    np.save(folder/(name+'.npy'),array)
    with (folder/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['axis']+[f'{column}{j+1}' for j in range(array.shape[1])])
        writer.writerows([[f'{row}{i+1}',*values] for i,values in enumerate(array)])


def heatmap(ax,values,title):
    im=ax.imshow(values,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(values.shape[1]),[f'T{j+1}' for j in range(values.shape[1])])
    ax.set_yticks(range(values.shape[0]),[f'U{i+1}' for i in range(values.shape[0])])
    ax.set_title(title,fontsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9,
                color='white' if abs(values[i,j])>.65 else 'black')
    return im


def original_T(model,z,t):
    """Verify relocated original parameters against the saved evaluation copy."""
    source=wave.io.default_step2(wave.ROOT,model)
    stage=source.parent/'stage1_data'
    manifest=json.loads((stage/'manifest.json').read_text())
    # The entire repo moved. Rebase only the declared repository-relative path;
    # do not rewrite any historical manifests or frozen experiment hashes.
    declared=Path(manifest['source_dataset']);parts=declared.parts
    position=next(i for i,p in enumerate(parts) if p=='mod-rot-3')
    dataset=wave.ROOT.joinpath(*parts[position+1:])
    if sha(dataset)!=manifest['source_dataset_sha256']:raise ValueError('Original T dataset SHA mismatch')
    with np.load(dataset,allow_pickle=False) as saved:
        if not np.array_equal(saved['parameters'],t):raise ValueError('Saved T differs from original parameters')
        for key in ('subject_index','trial_index','trial_counts'):
            if not np.array_equal(saved[key],np.load(source/(key+'.npy'),allow_pickle=False)):
                raise ValueError('Subject alignment mismatch')
    if not np.array_equal(z,np.load(stage/'Zstd.npy',allow_pickle=False)):raise ValueError('Zstd mismatch')
    return dict(original_dataset=str(dataset),dataset_sha256=manifest['source_dataset_sha256'],
        subject_alignment_verified=True,original_T_verified=True)


def process(source,output,z,t):
    frozen=json.loads((source/'search_freeze.json').read_text())
    for name in ('U','R'):
        if sha(source/(name+'.npy'))!=frozen[name+'_sha256']:raise ValueError('Saved rotation changed')
    u=np.load(source/'U.npy',allow_pickle=False);r=np.load(source/'R.npy',allow_pickle=False)
    if u.shape[0]!=5000 or not np.allclose(u,z@r.T,atol=1e-10):raise ValueError('Invalid U or R convention')
    white,mean,a,audit=zca_whiten(u);output.mkdir(parents=True,exist_ok=True)
    np.save(output/'U_whitened.npy',white);np.save(output/'U_mean.npy',mean)
    np.save(output/'ZCA_matrix.npy',a);np.savetxt(output/'ZCA_matrix.csv',a,delimiter=',')
    b=a@r;offset=-mean@a
    np.save(output/'combined_transform_B.npy',b);np.save(output/'centering_offset.npy',offset)
    if not np.allclose(z@b.T+offset,white,atol=1e-10):raise ArithmeticError('Combined transform convention failed')
    result=dict(status='FROZEN',method='ZCA symmetric inverse square root',
        source=str(source.relative_to(SOURCE)),source_U_sha256=frozen['U_sha256'],source_R_sha256=frozen['R_sha256'],
        all_subjects=len(u),T_used_for_whitening=False,T_used_for_rotation_selection=False,
        whitening_formula='U_white = (U - mean(U)) @ A; A = Cov(U)^(-1/2), population covariance',
        combined_formula='U_white = Zstd @ B.T + offset; B = A @ R; offset = -mean(U) @ A',
        combined_transform_is_orthogonal=bool(np.allclose(b@b.T,np.eye(b.shape[0]),atol=1e-10,rtol=0.)),
        rank_deficient_policy='pseudoinverse on positive numerical eigenspace',
        U_whitened_sha256=sha(output/'U_whitened.npy'),ZCA_matrix_sha256=sha(output/'ZCA_matrix.npy'),**audit)
    save_json(output/'whitening_freeze.json',result)
    # T enters evaluation only after the whitening transform and new U froze.
    before=corr(u,t);after=corr(white,t)
    spearman=corr(rankdata(white,axis=0),rankdata(t,axis=0))
    save_matrix(output,'pearson_Uwhite_T',after);save_matrix(output,'spearman_Uwhite_T',spearman)
    save_matrix(output,'pearson_U_T_before',before)
    save_matrix(output,'pearson_Uwhite_Uwhite',corr(white,white),column='U')
    save_matrix(output,'covariance_Uwhite',white.T@white/len(white),column='U')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    heatmap(axes[0],before,'Before whitening: Pearson corr(U, T)')
    im=heatmap(axes[1],after,'After ZCA whitening: Pearson corr(U_white, T)')
    fig.subplots_adjust(left=.07,right=.88,bottom=.11,top=.82,wspace=.3)
    fig.colorbar(im,cax=fig.add_axes((.91,.16,.018,.58)),label='Pearson r')
    fig.suptitle(f'{source.parent.parent.name} / {source.parent.name} / {source.name}',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(output/f'before_after_U_T_heatmaps.{ext}',dpi=150)
    plt.close(fig)
    # Independent numerical check of the written array and Pearson matrix.
    check=np.load(output/'U_whitened.npy',allow_pickle=False)
    check_corr=np.corrcoef(np.column_stack((check,t)),rowvar=False)[:check.shape[1],check.shape[1]:]
    if not np.allclose(check_corr,after,atol=1e-12):raise ArithmeticError('Saved correlation mismatch')
    save_json(output/'evaluation.json',dict(T_used_after_whitening_freeze=True,T_used_to_fit_whitening=False,
        whitening_freeze_sha256=sha(output/'whitening_freeze.json'),Pearson_recomputed_independently=True))
    return after,result


def add_data_links():
    """Make numerical outputs directly accessible in the saved HTML viewer."""
    path=OUTPUT/'index.html';html=path.read_text(encoding='utf-8')
    for number,c in enumerate(CANDIDATES,1):
        channel=c[3] if len(c)>3 else 'logit_nn'
        name=f'{number:02d}_{c[1]}_{channel}'
        for kind in KINDS:
            for model in MODELS:
                relative=f'{name}/{model}/{kind}'
                old=f'<a href="{relative}/before_after_U_T_heatmaps.png">{model} 白化前后对照</a>'
                if f'href="{relative}/pearson_Uwhite_T.csv"' in html:continue
                html=html.replace(old,old+
                    f' [<a href="{relative}/pearson_Uwhite_T.csv">Pearson CSV</a> / '
                    f'<a href="{relative}/spearman_Uwhite_T.csv">Spearman CSV</a> / '
                    f'<a href="{relative}/U_whitened.npy">新 U</a> / '
                    f'<a href="{relative}/">全部数据</a>]')
    path.write_text(html,encoding='utf-8')


def main():
    OUTPUT.mkdir(exist_ok=True);save_json(OUTPUT/'protocol.json',dict(status='RUNNING',method='ZCA',T_used_to_fit_whitening=False))
    # Snapshot completed source runs. Incomplete continuous optimizations have no
    # frozen U and are not silently substituted with another rotation.
    runs={p.parent for p in SOURCE.glob('*/*/*/search_freeze.json') if p.parent.name in KINDS}
    shared={}
    for model in MODELS:
        folder=SOURCE/'source_coordinates'/model
        z=np.load(folder/'Zstd.npy',allow_pickle=False);t=np.load(folder/'T.npy',allow_pickle=False)
        shared[model]=(z,t)
    rows=[];cards=[];provenance={}
    for number,candidate in enumerate(CANDIDATES,1):
        label,tag,metric=candidate[:3];channel=candidate[3] if len(candidate)>3 else 'logit_nn'
        name=f'{number:02d}_{tag}_{channel}';dest=OUTPUT/name;dest.mkdir(exist_ok=True)
        links=[]
        for kind in KINDS:
            selected={model:SOURCE/name/model/kind for model in MODELS if SOURCE/name/model/kind in runs}
            if not selected:continue
            fig,axes=plt.subplots(2,2,figsize=(10,9))
            for model,ax in zip(MODELS,axes.flat):
                if model not in selected:
                    ax.set_axis_off();ax.set_title(f'{model}: no frozen continuous result');continue
                after,audit=process(selected[model],dest/model/kind,*shared[model])
                heatmap(ax,after,f'{model}: after ZCA whitening')
                rows.append(dict(candidate=number,label=label,model=model,kind=kind,
                    covariance_error=audit['covariance_error_to_identity'],rank=audit['rank'],dimension=audit['dimension'],
                    source_U_sha256=audit['source_U_sha256'],U_whitened_sha256=audit['U_whitened_sha256'],
                    output=str((dest/model/kind).relative_to(OUTPUT))))
            fig.suptitle(f'{number:02d} / {tag} / {channel} / {kind}\nPearson corr(U_white, T); ZCA fitted on U only',fontsize=10)
            fig.subplots_adjust(left=.07,right=.89,bottom=.06,top=.87,hspace=.33,wspace=.25)
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1,1),cmap='coolwarm'),
                cax=fig.add_axes((.93,.15,.015,.65)),label='Pearson r')
            filename=f'four_model_{kind}_ZCA_U_T_heatmaps'
            for ext in ('png','pdf'):fig.savefig(dest/(filename+'.'+ext),dpi=160)
            plt.close(fig)
            links.append(f'<h3>{escape(kind)}</h3><a href="{name}/{filename}.pdf">PDF</a>'
                f'<img loading="lazy" src="{name}/{filename}.png"><p>'+
                ' · '.join(f'<a href="{name}/{model}/{kind}/before_after_U_T_heatmaps.png">{model} 白化前后对照</a>' for model in selected)+'</p>')
        cards.append(f'<article><h2>{number:02d}. {escape(label)}</h2><code>{escape(metric)}</code>'+''.join(links)+'</article>')
        print(f'[ZCA] {number}/44 {tag}: {len(rows)} rotations saved',flush=True)
    # Verify T provenance after whitening all U, without feeding T to its fit.
    for model,(z,t) in shared.items():provenance[model]=original_T(model,z,t)
    if len(rows)!=len(runs):raise ArithmeticError('Not every frozen rotation was whitened')
    with (OUTPUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUTPUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>旋转后 ZCA 白化相关矩阵</title>'
        '<style>body{max-width:1200px;margin:36px auto;padding:0 20px;font:16px Arial;background:#f4f6f8;color:#202630}'
        'article{background:white;padding:22px;margin:30px 0;border-radius:12px}img{width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#145ca8}</style>'
        f'<h1>旋转后 ZCA 白化：U 与原始参数 T 的相关矩阵</h1><p>44 个候选，{len(rows)} 组已冻结旋转结果，每组全部 5000 个被试。</p>'
        '<p>先旋转得到 U，再对中心化 U 做 ZCA 白化：U_white = (U − mean(U)) Cov(U)<sup>−1/2</sup>。'
        '协方差按全部被试计算（ddof=0）。白化只使用 U，T 仅在新 U 冻结后用于相关评估；未按 T 调整符号或排列。</p>'
        '<p>原旋转结果保持不变。白化后的组合变换 B = A R 通常不再是正交旋转矩阵；本次未重新搜索 R 或用原指标给白化结果打分。</p>'
        '<p>每项保留当前搜索最优、网格最优和已完成连续搜索三类结果。连续搜索缺失的数据库明确留空。</p>'
        '<p><a href="summary.csv">数值核验汇总</a> · <a href="../index.html">原始旋转图集</a></p>'+''.join(cards)+'</html>',encoding='utf-8')
    verification=dict(status='PASS',method='ZCA',candidates=44,rotations=len(rows),
        counts_by_kind={kind:sum(r['kind']==kind for r in rows) for kind in KINDS},
        all_subjects=5000,all_full_rank=all(r['rank']==r['dimension'] for r in rows),
        maximum_covariance_identity_error=max(r['covariance_error'] for r in rows),
        source_R_U_hashes_verified=True,combined_transform_verified=True,Pearson_independently_recomputed=True,
        T_used_to_fit_whitening=False,T_provenance=provenance)
    save_json(OUTPUT/'verification.json',verification);save_json(OUTPUT/'protocol.json',dict(status='COMPLETE',**{k:v for k,v in verification.items() if k!='status'}))
    original_index=SOURCE/'index.html'
    html=original_index.read_text(encoding='utf-8')
    if 'id="zca-whitening-link"' not in html:
        html=html.replace('<h1>','<p id="zca-whitening-link"><a href="whitened/index.html">查看全部旋转后 ZCA 白化相关矩阵及白化前后对照</a></p><h1>',1)
        original_index.write_text(html,encoding='utf-8')
    add_data_links()
    print(f'[ZCA COMPLETE] {len(rows)} rotations; maximum covariance error {verification["maximum_covariance_identity_error"]:.3g}',flush=True)


if __name__=='__main__':
    with threadpool_limits(4):main()
