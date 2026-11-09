"""Compare fixed single metrics across output scales; never select using TEST."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parent/'results'


def main():
    labels=['s01_single_metric_atlas','s04_probability_single_atlas'];models=['C1','C2','C3','CR3']
    metrics=['hs_ux_rms_l1','hs_ux_rms_soft003','hs_ux_rms_soft01','hs_ux_rms_lp05',
        'hs_ux_signed_l1','hs_ux_sd_l1','hs_ux_local_l1','hs_ux_median_l1','hs_ux_q90_l1','hs_ux_q99_l1',
        'hs_ux_rms_row_tail1_k1','hs_ux_rms_column_tail1_k1','hs_ux_rms_column_tail2sq_k2',
        'hs_uu_self_rms_l1','hs_uu_off_rms_l1','hs_uu_self_local_l1','hs_uu_off_rms_energy_fraction',
        'hs_uu_graph_rows_soft_t0.03','hs_nonlinearity_self_cap_t0.03','hs_uu_third_repeat_l1',
        'hs_uu_third_mixed_l1','hs_ux_profile1_l1','hs_jd_profile1_strength0_selfl1_w0',
        'hs_jd_mean_profile1_strength0','hs_full_rms_l1','hs_full_rms_soft003','hs_full_q90_l1']
    states={'FLAT_PLANE':0,'PLANE_COUNTEREXAMPLE':1,'NEEDS_JOINT_OPTIMIZATION':2}
    matrix=np.zeros((len(metrics),8));annotations={};columns=[];counts=[]
    for ci,(label,model) in enumerate((lab,m) for lab in labels for m in models):
        a=json.loads((ROOT/label/model/'atlas.json').read_text(encoding='utf8'))
        lookup={r['metric']:r for r in a['results']}
        columns.append(a['channel']+'\n'+model)
        for ri,metric in enumerate(metrics):
            r=lookup[metric];matrix[ri,ci]=states[r['state']]
            annotations[ri,ci]='F' if r['state']=='FLAT_PLANE' else str(round(max(p['axis_error'] for p in r['planes']),1))
        counts.append(dict(channel=a['channel'],model=model,
            rejected=sum(r['state']=='PLANE_COUNTEREXAMPLE' for r in a['results']),
            flat=sum(r['state']=='FLAT_PLANE' for r in a['results']),
            surviving=sum(r['state']=='NEEDS_JOINT_OPTIMIZATION' for r in a['results'])))
    fig,ax=plt.subplots(figsize=(13,12));fig.subplots_adjust(left=.4,right=.98,top=.91,bottom=.12)
    ax.imshow(matrix,cmap=ListedColormap(['#dedede','#eda69e','#a6d4e9']),vmin=-.5,vmax=2.5,aspect='auto')
    ax.set_xticks(range(8),columns);ax.xaxis.tick_top()
    ax.set_yticks(range(len(metrics)),[m.removeprefix('hs_') for m in metrics],fontsize=9)
    for (ri,ci),text in annotations.items():ax.text(ci,ri,text,ha='center',va='center',fontsize=9)
    ax.axvline(3.5,color='white',linewidth=4)
    ax.set_title('Single Hessian metrics: analytic rotation screens',pad=42,fontsize=15)
    fig.text(.4,.065,'Gray F: unresolved flat plane   Red: sampled plane counterexample\n'
        'Blue: needs full rotation optimization (NOT a Gap 1 pass)\n'
        'Numbers: largest sampled plane-minimum axis error, degrees. Same metric across all models.',fontsize=10)
    folder=ROOT/'single_metric_study';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'single_metric_screen_comparison.png',dpi=160);plt.close(fig)
    lines=['# 单一 Hessian 指标：两种输出尺度的首轮比较','',
        '本轮暂停组合与权重探索。共 9 个实现族、314 个去重后的单项定义；统计量和阈值变体均明确列出，'
        '不把它们当作 314 种独立思想。每个定义在四模型一致使用，并分别测 logit 与概率输出。','',
        '| 输出 | 模型 | 平面反例 | 平坦方向 | 待完整优化 |','|---|---|---:|---:|---:|']
    for r in counts:lines.append('| {channel} | {model} | {rejected} | {flat} | {surviving} |'.format(**r))
    lines+=['','“待完整优化”不等于 Gap 1 成功。四模型均存活的同一单项才进入多起点完整旋转检查；'
        'Gap 1 真正通过后才检查 WY inference，再进入 decoder 验证。','',
        '## 完整优化与推断验证','',
        '| 轮次 | 模型 | 指标 | 全空间轴误差 | 结果 |',
        '|---|---|---|---:|---|']
    rounds=['s02_single_joint_confirm','s05_single_soft003_g2','s06_single_soft003_g2_degree2',
        's07_single_soft003_g2_fullrbf','s08_single_soft003_g2_smoother']
    for label in rounds:
        for path in sorted((ROOT/label).glob('**/summary.json')):
            j=json.loads(path.read_text(encoding='utf8'));a=j['args']
            passed=j['gap1_angle_pass'] if j['stage']=='G1' else j['gap2_audit']['pass_gate']
            verdict=j['stage']+(' PASS' if passed else ' FAIL')
            lines.append(f'| {label} | {a["model"]} | {a["metric"]} | {j["worst_axis_error_degrees"]:.4f}° | {verdict} |')
    lines+=['','G2 不仅看全空间轴误差，还检查各平面最低点、曲线相关和相对误差。'
        '因此表中轴误差低于 6° 仍可能 G2 失败。未列模型表示该推断分支尚未运行该模型，不能拼接不同分支冒充四模型通过。',
        '当前尚未为这个单指标运行 G3；必须先有同一推断规则在四模型通过 G2。','',
        '需要分清的结论：','',
        '- C1/C2/C3 的 logit 对参数线性，纯 UU/UUU 单项的零场是解析不可辨识；更好的 inference 不能制造缺失的信息。',
        '- 概率的 Hessian 会因 sigmoid 曲率引入额外耦合，所以“logit 不可辨识”不能直接推及概率，但额外曲率也不保证正确轴最简单。',
        '- 符号平均、RMS、SD、局部绝对值、分位数要分开看；前者可能抵消，SD 会删除恒定交互，分位数改变对少数极端点的敏感性。',
        '- 行尾部与列尾部描述不同的交互稀疏性；允许 top-k 项时，还可能出现代价为零的旋转自由度。',
        '- 只压 UU 非对角可能把真正的交互转为自曲率；所以记录“偏好错轴”，不把较低 loss 当作成功。','',
        '![代表单项](single_metric_screen_comparison.png)','',
        '[全部 logit 单项](../s01_single_metric_atlas/REPORT.md) · '
        '[全部概率单项](../s04_probability_single_atlas/REPORT.md)']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[:15]))


if __name__=='__main__':main()
