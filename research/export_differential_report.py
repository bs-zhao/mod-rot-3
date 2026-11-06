"""Export frozen prospective candidate results, not best-TEST runs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'np/src')]
import argparse,csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rotations.differential_program.toys import run_toys

MODELS=['C1','C2','C3','CR3']
CONDITION='beta0p0005_lambda0_ctx0p25to0p8_L8'


def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def csv_rows(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def find_equivariant(model,repeat,seed,frozen):
    base=ROOT/'outputs/choice'/model/'rotations/differential_equivariant'/CONDITION/f'repeat{repeat:02d}'
    candidates=[]
    for run in base.glob('run_*'):
        if not (run/'complete.json').exists():continue
        fit=json.loads((run/'fit.json').read_text())
        if fit['config']['seed']!=seed:continue
        if fit['source_hashes']!=frozen['source_sha256']:continue
        expected=dict(frozen['config'],seed=seed)
        if fit['config']!=expected:continue
        candidates.append((run,fit,json.loads((run/'evaluation.json').read_text())))
    if len(candidates)!=1:raise RuntimeError(f'{model} repeat{repeat} seed{seed}: expected one frozen run, got {len(candidates)}')
    return candidates[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--preview',action='store_true');args=p.parse_args()
    frozen=json.loads((ROOT/'research/FROZEN_DIFFERENTIAL_CANDIDATE_2026-11-06.json').read_text())
    for path,digest in frozen['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen method source changed: '+path)
    cases={};manifest={};rows=[];details={}
    for model in MODELS:
        for seed in [31,47]:
            key=f'{model}_repeat01_anchor{seed}';cases[key]=find_equivariant(model,1,seed,frozen)
    independent=None
    if not args.preview:
        independent=find_equivariant('CR3',2,31,frozen);cases['CR3_repeat02_anchor31']=independent
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    for key,(run,fit,evaluation) in cases.items():
        manifest[key]=dict(path=str(run),fit=fit,complete=json.loads((run/'complete.json').read_text()),frozen_source_match=True)
        details[key]=evaluation;target=dest/'details'/key;target.mkdir(parents=True,exist_ok=True)
        training=ROOT/'outputs/choice'/fit['model']/'s3_train'/fit['condition']/f"repeat{fit['repeat']:02d}"/'s3_train__summary.json'
        manifest[key]['np_training']=json.loads(training.read_text())
        shutil.copy2(training,target/'np_training_summary.json')
        learning=ROOT/'outputs/choice'/fit['model']/'s3_evaluate'/fit['condition']/f"repeat{fit['repeat']:02d}"/'s3_evaluate__learning_curves.png'
        if learning.exists():shutil.copy2(learning,target/'np_learning_curves.png')
        for filename in ['fit.json','evaluation.json','comparison.csv','complete.json','mapping.npz','rotations.npz','initializations.npz','supervised_reference.npz','supervised_angle_curves.csv','identity__r2.png','equivariant_code__r2.png','equivariant_l1__r2.png','supervised_reference__r2.png','finite_equivariance_audit.json','bootstrap_symmetries.json']:
            if (run/filename).exists():shutil.copy2(run/filename,target/filename)
        for name,result in evaluation.items():
            rows.append(dict(case=key,method=name,**{metric:result[metric] for metric in ['mean_r2','test_optimal_mean_r2','min_r2','off_mean_r2','off_max_abs_r']},
                             parameter_r2=json.dumps(result['parameter_r2']),source_run=str(run)))
    csv_rows(dest/'all_frozen_results.csv',rows);write_json(dest/'signed_correlations.json',details);write_json(dest/'provenance.json',manifest)
    write_json(dest/'frozen_protocol.json',frozen)
    local=[];other=[]
    # Keep diagnostic branches separate. No run is ranked by recovery.
    for family in ['differential_program','differential_functional','differential_calibrated']:
        for model in MODELS:
            for run in (ROOT/'outputs/choice'/model/'rotations'/family/CONDITION/'repeat01').glob('run_*'):
                if not (run/'complete.json').exists():continue
                fit=json.loads((run/'fit.json').read_text());evaluation=json.loads((run/'evaluation.json').read_text())
                item=dict(family=family,model=model,anchor_seed=fit['config']['seed'],path=str(run),fit=fit,evaluation=evaluation)
                other.append(item)
    write_json(dest/'diagnostic_branches.json',other)
    primary={model:cases[f'{model}_repeat01_anchor31'][2] for model in MODELS}
    fig,axes=plt.subplots(1,2 if independent else 1,figsize=(15 if independent else 10,5.5),squeeze=False)
    ax=axes[0,0];positions=np.arange(4);w=.25
    styles=[('identity','PCA Identity','#6d7c91'),('equivariant_code','Frozen functional objective','#237e70'),('supervised_reference','Supervised diagnostic','#c3984e')]
    for index,(name,label,color) in enumerate(styles):
        values=[primary[model][name]['mean_r2'] for model in MODELS]
        bars=ax.bar(positions+(index-1)*w,values,w,label=label,color=color);ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
    ax.set_xticks(positions,MODELS);ax.set_title('Original NP trainings | exploratory');ax.set_ylabel('TEST mean Pearson r² | TRAIN matching');ax.set_ylim(0,1.08)
    if independent:
        ax=axes[0,1]
        for index,(name,label,color) in enumerate(styles):
            values=[primary['CR3'][name]['mean_r2'],independent[2][name]['mean_r2']]
            bars=ax.bar(np.arange(2)+(index-1)*w,values,w,label=label,color=color);ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
        ax.set_xticks([0,1],['Original NP','New NP seed']);ax.set_ylim(0,1.08);ax.set_title('CR3 | frozen before new training result')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(dest/'main_results.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5));positions=np.arange(4)
    a=[primary[m]['equivariant_code']['mean_r2'] for m in MODELS]
    b=[cases[f'{m}_repeat01_anchor47'][2]['equivariant_code']['mean_r2'] for m in MODELS]
    ax.plot(positions,a,'o-',label='FIT anchors 31');ax.plot(positions,b,'s--',label='FIT anchors 47')
    ax.scatter(positions,[primary[m]['identity']['mean_r2'] for m in MODELS],marker='x',c='gray',label='Identity')
    ax.set_xticks(positions,MODELS);ax.set_ylim(.3,1);ax.set_ylabel('TEST mean r²');ax.set_title('Anchor sensitivity | same NP, not independent training');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(dest/'anchor_sensitivity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(15,16))
    for i,model in enumerate(MODELS):
        for j,name in enumerate(['identity','equivariant_code','supervised_reference']):
            report=primary[model][name];c=np.array(report['signed_test_corr']);ax=axes[i,j]
            ax.imshow(c*c,vmin=0,vmax=1,cmap='Blues');ax.set_title(model+' | '+name,fontsize=10)
            ax.set_yticks(range(len(report['parameter_names'])),report['parameter_names']);ax.set_xticks(range(c.shape[1]),[f'u{h+1}' for h in range(c.shape[1])]);ax.tick_params(labelsize=9)
            for row in range(c.shape[0]):
                for col in range(c.shape[1]):ax.text(col,row,f'{c[row,col]**2:.2f}',ha='center',va='center',fontsize=8,color='white' if c[row,col]**2>.55 else 'black')
    fig.tight_layout(w_pad=3,h_pad=2);fig.savefig(dest/'all_model_heatmaps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,model in zip(axes.flat,MODELS):
        fit=cases[f'{model}_repeat01_anchor31'][1]
        for item in fit['symmetries']:
            ax.plot(np.arange(len(item['eigenvalues']))+1,item['eigenvalues'],'o-',label=item['property']+(' accepted' if item['accepted'] else ' rejected'))
        ax.set_title(model);ax.set_xlabel('Generalized eigenvalue rank');ax.set_ylabel('Relative functional residual');ax.set_ylim(0,1.08);ax.legend(fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(dest/'functional_role_spectra.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    for ax,model in zip(axes.flat,MODELS):
        run=cases[f'{model}_repeat01_anchor31'][0]
        audit=json.loads((run/'finite_equivariance_audit.json').read_text())
        values=audit['rows'];ax.plot([r['step'] for r in values],[100*r['weighted_residual_fraction'] for r in values],'o-')
        ax.set_title(model+' | '+values[0]['property']);ax.set_xlabel('Finite q displacement');ax.set_ylabel('Non-equivariant change energy (%)')
        ax.set_ylim(0,max(2,1.25*max(100*r['weighted_residual_fraction'] for r in values)))
    fig.suptitle('Held-out subjects and new actual X | no theta in audit',fontsize=12)
    fig.tight_layout();fig.savefig(dest/'finite_functional_roles.png',dpi=160);plt.close(fig)
    toy,curves=run_toys();write_json(dest/'analytic_gates.json',toy);csv_rows(dest/'analytic_angle_curves.csv',curves)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,name in zip(axes,['power','bilinear','linear_stimulus']):
        data=[row for row in curves if row['toy']==name];true=next(row['module_code'] for row in data if row['angle']==0)
        ax.plot([row['angle'] for row in data],[row['module_code']/true for row in data]);ax.axvline(0,c='gray',lw=.8)
        ax.set_title(name);ax.set_xlabel('Rotation angle (degrees)');ax.set_ylabel('Module code / true-axis code')
    fig.tight_layout();fig.savefig(dest/'analytic_gates.png',dpi=160);plt.close(fig)
    trades=[]
    for key in ['CR3_repeat01_anchor31']+(['CR3_repeat02_anchor31'] if independent else []):
        evaluation=cases[key][2]
        for name,value in evaluation['equivariant_code']['parameter_r2'].items():
            base=evaluation['identity']['parameter_r2'][name]
            trades.append(dict(case=key,parameter=name,identity_r2=base,primary_r2=value,change=value-base,supervised_reference=evaluation['supervised_reference']['parameter_r2'][name]))
    csv_rows(dest/'CR3_parameter_tradeoffs.csv',trades)
    if independent:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,key,title in zip(axes,['CR3_repeat01_anchor31','CR3_repeat02_anchor31'],['Original NP','New NP seed']):
            ev=cases[key][2];names=ev['equivariant_code']['parameter_names'];positions=np.arange(len(names))
            for offset,name,color in [(-.16,'identity','#6d7c91'),(.16,'equivariant_code','#237e70')]:
                vals=[ev[name]['parameter_r2'][p] for p in names];bars=ax.bar(positions+offset,vals,.32,label=name,color=color);ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
            ax.scatter(positions,[ev['supervised_reference']['parameter_r2'][p] for p in names],marker='D',s=24,c='#c3984e',label='supervised reference')
            ax.set_xticks(positions,names,rotation=10);ax.set_ylim(0,1.05);ax.set_title(title);ax.set_ylabel('TRAIN-matched TEST r²')
        axes[0].legend(fontsize=8,frameon=False);fig.tight_layout();fig.savefig(dest/'CR3_parameter_tradeoffs.png',dpi=160);plt.close(fig)
    table='\n'.join(f"| {m} | {primary[m]['identity']['mean_r2']:.3f} | {primary[m]['equivariant_code']['mean_r2']:.3f} | {cases[m+'_repeat01_anchor47'][2]['equivariant_code']['mean_r2']:.3f} |" for m in MODELS)
    validation='独立训练复核尚未完成；本预览不是最终交付。'
    if independent:
        ie=independent[2];improved=ie['equivariant_code']['mean_r2']>ie['identity']['mean_r2']
        validation=f"新 CR3 NP 训练 seed 的固定方法复核：Identity={ie['identity']['mean_r2']:.3f}，主候选={ie['equivariant_code']['mean_r2']:.3f}，监督参考={ie['supervised_reference']['mean_r2']:.3f}。"+('平均收益保持。' if improved else '平均收益未复现，不能宣称稳定成功。')+' 数据和被试split相同，属于训练seed复核，不是独立新数据验证。'
    text=f'''# 新旋转目标：设计与实测

本轮实现并运行了新的目标。最有依据的候选是：**先由冻结函数检出近似整体平移/增益方向，再在其正交补上最小化完整微分模块描述代价。** 没有按模型编号指定参数、刺激区域或loss分支。

| 模型 | Identity | 固定候选：anchor31 | 相同方法：anchor47 |
|---|---:|---:|---:|
{table}

数值为 TRAIN 定配对、TEST 评价的平均 Pearson r²；已有数据上的结果是探索性的。{validation}

## 为什么这些目标比旧 JointDiag 更有希望

1. **完整微分模块 code**：同时计算纯平方、纯立方和混合导数；`ab`是一项乘积，错误旋转后两个平方都付代价。再用统一的latent–X混合算子识别线性参数的不同刺激作用。三个解析toy和4维隐藏旋转gate通过；纯模块目标已在C1–C3达到约0.942/0.951/0.933。
2. **轴向微分递推**：比较沿轴的一、二、三阶作用需要多少状态模式。它保留真实乘法，但在线性参数模型上不可辨识，因此没有被包装成通用主方法。
3. **相对功能残差目标**：对“刺激恒定的变化”和“与整个logit profile成比例的变化”分别求广义Rayleigh最小方向，以实际效应能量归一化，并检查唯一性、活动性和抽样间隙。相同规则在前三模型检出平移，在CR3检出增益；这些角色由F决定，不使用θ名称。

主候选是第3项支持的轴约束加第1项目标；阈值、主候选名称和源码在查看新NP结果前已冻结。L1只是预声明消融，没有按TEST挑获胜版本。

## 不能省略的限制

- CR3原训练的均值由0.484升至0.660，最弱参数由0.287升至0.579，非匹配平均r²由0.094降到0.036；但alpha由约0.632降到0.582，仍有参数代价。anchor47均值约0.591，仍有抽样敏感性。
- C1–C3接近监督参考；CR3的alpha/gamma仍未充分分开。不能仅据均值宣布四模型完整成功。
- 局部X尺度校准和全局X投影都做过误差控制；前者未单独解决CR3，后者的CR3高阶导数保真不足，均不当作成功结果。
- 有限幅度检查在未参与角色检测的TEST被试和实际X上进行，不加载θ：C1–C3的平移残差约0.4%–1.4%，CR3增益残差约8%–10%。是近似功能角色，不是纯净机制轴的证明。
- 尚未覆盖四模型各自的独立训练重复和新数据seed。函数等变是明确先验，不保证任意生成机制可辨识。

## 文件

`main_results.png`、`all_model_heatmaps.png`、`anchor_sensitivity.png`、`functional_role_spectra.png`、`analytic_gates.png`用于展示；完整有符号矩阵见`signed_correlations.json`，逐参数代价见`CR3_parameter_tradeoffs.csv`。`details/`含映射、旋转、所有初始化及诊断；`provenance.json`和`frozen_protocol.json`记录来源、配置、源码和冻结协议。大数据与checkpoint保留在原项目ignored outputs。

详细数学定义、失败原因和下一步决定见`研究记录.md`。代码已提交研究分支；原有未提交文件及实验均保留。
'''
    (dest/'设计与结果.md').write_text(text,encoding='utf-8')
    shutil.copy2(ROOT/'research/DIFFERENTIAL_OBJECTIVES_2026-11-06.md',dest/'研究记录.md')
    print(json.dumps(dict(output=str(dest),cases=list(cases),independent_complete=bool(independent)),indent=2))


if __name__=='__main__':main()
