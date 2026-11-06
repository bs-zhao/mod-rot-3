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
    table='\n'.join('| '+m+' | '+' | '.join(f"{details[m][name]['train_matched_test_mean_r2']:.3f}" for name in ['identity','contrast_logit','supervised_reference'])+' |' for m in models)
    text=f'''# 旋转研究阶段总结 · 2026-11-06

**尚未得到四模型共同成功的新旋转方法。本阶段形成了可复查的否定结果，并区分了目标设计、刺激表示和代理函数保真度问题。**

已实际运行新方法 stimulus-contrast covariance rotation：利用冻结 decoder 对白化 PCA 坐标的 logit 梯度，扣除所有刺激共有的二阶矩，再做正交联合对角化。四个模型采用同一损失、配置及选择规则，没有使用真实参数选择旋转、权重或条件。

| 模型 | PCA Identity | 新方法 | 监督正交参考（仅诊断） |
|---|---:|---:|---:|
{table}

表中为 Pearson 平均匹配 r²；在 TRAIN 确定配对，再应用到 TEST。TEST 自身最优匹配另列于 `all_results.csv`。每参数结果、交叉载荷和完整有符号矩阵见 `signed_correlations.json` 及 `details/`，不能只按均值判断。

## 三项决定性发现

1. **目标本身可以偏好错误坐标。** 对完全正确的 C1 解析函数，真实轴 loss 为 5/6，四轴均匀混合后反而降到 13/24；指定的幂函数 toy 也偏好约45°错误混合。六次初始化在已检验模型上得到相同错误优点，不能通过增加优化步数解决。新候选已否定。
2. **刺激表示改变了 JointDiag 的含义。** 保持同一 C1 profile 函数不变，仅正交改写刺激特征基，恢复可从约0.943降到0.391。它是特定刺激基下的简单性假设；原始 trial 与投影特征通道不能混作同一指标。
3. **profile R²接近1仍可能损失结构。** CR3 历史系数 profile R²约0.9997，但相同 TEST 被试和 X 上，逐刺激概率 R²约0.853，概率梯度相对误差约50%，同一定义的 raw-X 旋转 loss 相关约0.852。这个比较直接针对冻结 decoder，没有真实参数坐标换算误差。

旧自适应组合的归一化另经重现核实：改为 Identity 等贡献可使 CR3 从约0.388升至0.470，仍低于 Identity 0.484，未解决主问题。

## 实现核验与适用范围

- 列约定 q=W(z−μ)、u=Aq；行实现 U=Q Aᵀ。W 包含 TRAIN PCA 与白化，A 是该空间的正交矩阵。原始8维 latent 的剩余部分保留，完整预测分布等价；截断误差单列。
- 正交性、矩阵方向、预测分布、自动微分/有限差分检查均已执行。弱轴 float32 差分异常另用 float64 排查，不能误判为 AD 错误。8 项相关测试通过。
- 当前结果是探索性。C3 是本次补齐的首次训练；CR3 不同 beta 属于条件敏感性检查，均不能替代新数据或独立 NP seed 验证。没有声称四模型共同成功或独立重复成功。
- 方法代码、配置、必要研究记录已提交到研究分支；实验输出与 checkpoint 保留在本机 ignored outputs，原有未跟踪 `outputs.zip` 未改动。

## 查看文件

- `stage_summary.png`：四模型比较。
- `all_model_heatmaps.png`：Identity、新方法、监督参考的完整 r²热力图。
- `toy_counterexamples.png`：解析反例与角度—loss 曲线。
- `profile_objective_fidelity.png`：代理函数与原 decoder 的同定义 loss 比较。
- `all_results.csv`、`signed_correlations.json`：全部预定候选、逐参数与交叉相关。
- `provenance.json`：实际输出路径、环境、种子、代码版本与源文件哈希。
- `研究记录.md`：假设、尝试、失败、数值追查、下一步决定。

下一轮先要求目标通过线性参数函数和幂函数反例，再研究有明确刺激表示定义、同时控制代理误差的功能描述复杂度；不继续用同一 TEST 挑最优权重或刺激基。
'''
    (dest/'阶段总结.md').write_text(text,encoding='utf-8')
    shutil.copy2(ROOT/'research'/'RESEARCH_LOG_2026-11-06.md',dest/'研究记录.md')
    print(json.dumps(dict(output=str(dest),source_runs={k:str(v) for k,v in runs.items()}),indent=2))


if __name__=='__main__':main()
