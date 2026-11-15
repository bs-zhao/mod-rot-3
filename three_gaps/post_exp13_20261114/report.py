"""Rebuild a transparent report from files that actually exist; never invent runs."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import core as C

EXPS=['exp14_probability','exp15_finite_scale','exp16_local_context','exp17_reliability','exp18_matrix_only','exp19_model_library']

def mdtable(df):
    if len(df)==0: return '尚无已完成的正式批次。\n'
    cols=[' '.join(str(v) for v in x if str(v)) if isinstance(x,tuple) else str(x) for x in df.columns]
    rows=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,row in df.iterrows():
        vals=[]
        for val in row:
            if isinstance(val,float): vals.append(f'{val:.4f}')
            else: vals.append(str(val))
        rows.append('| '+' | '.join(vals)+' |')
    return '\n'.join(rows)+'\n'

def main():
    baseline=[]
    for f in (C.HERE/'baseline/results').glob('*/metrics.csv'):
        baseline.append(pd.read_csv(f))
    base=pd.concat(baseline,ignore_index=True) if baseline else pd.DataFrame()
    if len(base): base.to_csv(C.HERE/'baseline/results/metrics_all.csv',index=False)
    text=['# 实测结果与实施边界','',
          '本页由 `report.py` 从实际 CSV/完成标记生成。`pilot` 仅检查代码通路，排除正式汇总。所有四模型 TEST 都已被历史工作查看，本页属于探索性结果。',
          '', '## EXP13 冻结与原入口复现','']
    if len(base): text.append(mdtable(base[base.channel=='prob'][['model','repeat','mcc','leakage','max_leakage']]))
    text+=['原 float32 优化器、24×250、Haar 200 seed 1、起点 seed 7；归档 manifest 见 `baseline/results/archive_manifest.json`。',
           '', '## 六条路线的实际覆盖','']
    allframes=[]; counts=[]; byexp={}
    for exp in EXPS:
        frames=[]; complete=0; partial=0
        for f in (C.HERE/exp/'results').glob('**/metrics.csv'):
            if 'pilot' in f.relative_to(C.HERE).parts: continue
            tag=f.relative_to(C.HERE/exp/'results').parts[0]
            if (C.HERE/exp/'results'/tag/'ABORTED.json').exists(): continue
            try: df=pd.read_csv(f)
            except pd.errors.EmptyDataError: continue
            df['result_path']=str(f.relative_to(C.HERE)).replace('\\','/')
            df['experiment']=exp; df['complete']=(f.parent/'COMPLETE.json').exists()
            df['tag']=tag
            if df['complete'].iloc[0]: complete+=1
            else: partial+=1
            frames.append(df)
        if frames:
            df=pd.concat(frames,ignore_index=True); allframes.append(df); byexp[exp]=df
            for tag,dd in df.groupby('tag'):
                cc=dd.groupby('result_path').complete.first()
                counts.append(dict(experiment=exp,tag=tag,completed_configs=int(cc.sum()),
                                   in_progress_configs=int((~cc).sum()),candidate_rows=len(dd)))
        else: counts.append(dict(experiment=exp,tag='-',completed_configs=0,in_progress_configs=0,candidate_rows=0))
    text.append(mdtable(pd.DataFrame(counts)))
    # Separate exploratory baseline tuning and unit/PCA/beta sensitivity from new losses.
    for prefix,label in [('tuning_','完整 tuning 网格（探索性，不按 TEST 选最终方案）'),
                         ('robustness_','可用 checkpoint 的 beta / PCA / X 单位敏感性')]:
        tables=[]
        for f in (C.HERE/'baseline/results').glob(prefix+'*/**/metrics.csv'):
            try: dd=pd.read_csv(f)
            except pd.errors.EmptyDataError: continue
            dd['scenario']=f.parent.name; dd['complete']=(f.parent/'COMPLETE.json').exists()
            tables.append(dd)
        if tables:
            dd=pd.concat(tables,ignore_index=True); dd.to_csv(C.HERE/('baseline_'+prefix.strip('_')+'_metrics.csv'),index=False)
            text += ['', '## '+label,'',
                     mdtable(dd.groupby(['model','repeat','channel']).agg(candidates=('method','count'),
                                  completed=('complete','all'),mcc_min=('mcc','min'),mcc_max=('mcc','max')).reset_index()),
                     'MCC 范围描述全部预设候选，不是无监督选择规则；完整逐项结果见对应 baseline CSV。']
    for exp in EXPS:
        text+=['',f'## {exp}','']
        if exp not in byexp:
            text+=['正式批次尚未产出 metrics；不能把实现或试跑写成完成验证。']; continue
        df=byexp[exp]; df=df[df.complete]
        if exp=='exp19_model_library':
            for tag,dd in df.groupby('tag'):
                if 'model' in dd and dd.model.notna().any():
                    text += [f'批次 `{tag}`：冻结 synthetic score 迁移到原 NP，未在 benchmark T 上重新训练。',
                             mdtable(dd[['model','repeat','channel','method','mcc','leakage','max_leakage']])]
                    continue
                text += [f'批次 `{tag}`：测试函数族与训练/validation 分离；负对照单列，不能平均成成功率。',
                         mdtable(dd.groupby(['split','family','method'],dropna=False)[['mcc','leakage','axis_error','abstain']].mean().reset_index())]
            continue
        cr=df[df.model=='CR3']
        if len(cr)==0:
            text+=['四模型批次进行中，尚无已完成的 CR3 正式设置。']; continue
        # All CR3 candidates, not a TEST-selected winner. Full table linked below.
        cr=cr.copy(); cr['evidence']=cr['result_path'].str.split('/').str[-2]
        cols=['tag','repeat','channel','evidence','method','mcc','leakage','max_leakage']
        text += ['CR3 全候选（所有方法按预设规则搜索后评价，未按 TEST MCC 选解）：','',mdtable(cr[cols].sort_values(['tag','evidence','method']))]
    text+=['', '## 必须保留的限制与后续工作','',
           '- EXP14/17 正式字段预算为 300 TRAIN 被试和全部冻结刺激；EXP15 为 8×8 中心点的查询实验，EXP16 为 64×64（逐点 128 点），EXP18 拟合 64×64 真实试次。采样量、估计器与优化器差异不能混同于 loss 改善。',
           '- EXP15 的 UX 是真实 bank 最近邻配对的方向响应，不是完整环境 UX Hessian；全尺度、全矩阵有限差分都在每个 R 重新查询。',
           '- 可靠性共识不能证明正确；独立 surrogate 仍共享原 NP 表示，可能保留系统偏差。',
           '- EXP18 按 validation 被试的 Y-MSE 选超参；post-fit teacher 导数只用于审计。choice 的 NP 导数参考不是真实条件均值的导数。',
           '- EXP19 使用合成监督和一个小型固定 grammar；不宣称通用 learned simplicity。径向反例应拒绝唯一恢复；满响应 Jacobian 秩的反例仍可能偏好另一套轴。',
           '- 已补充可用 CR3 beta checkpoint、保留全部维数的 PCA 和 X 单位敏感性；仍未完成新 S1 数据、NP repeat 3–5、更高维 PCA/残差协议、更大有限扰动预算、有限差分局部张量、新刺激设计 NP 重训与大规模 grammar。',
           '', '完整逐候选机器可读结果：[all_metrics.csv](all_metrics.csv)。每个结果子目录含 config、rotations、diagnostics、全部平面与 U–T 热图；运行命令和失败栈在 `logs/`。']
    if allframes: pd.concat(allframes,ignore_index=True).to_csv(C.HERE/'all_metrics.csv',index=False)
    (C.HERE/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(pd.DataFrame(counts).to_string(index=False))

if __name__=='__main__': main()
