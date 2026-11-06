"""Descriptive tables and complete angle plots; never selects an optimizer."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def safe_spearman(a,b):
    a=np.asarray(a);b=np.asarray(b);good=np.isfinite(a)&np.isfinite(b)
    if good.sum()<3 or np.ptp(a[good])<1e-10 or np.ptp(b[good])<1e-10:return np.nan
    return float(spearmanr(a[good],b[good]).statistic)


def report(out,paths,plots=True):
    out=Path(out);summaries=[];positions=[];optima=[];agreements=[];associations=[];availability=[]
    all_values=[]
    for path in paths:
        frame=pd.read_csv(path/'metric_values.csv');all_values.append(frame)
        catalog=json.loads((path/'catalog.json').read_text());metrics=catalog['metric_names']
        complete=json.loads((path/'complete.json').read_text());ref=json.loads((path/'reference.json').read_text())
        summaries.append(dict(model=path.name,reference_test_mean_r2=np.mean(ref['test_diagonal_r2']),
            reference_test_min_r2=np.min(ref['test_diagonal_r2']),**{k:complete[k] for k in ['cases','unique_cases','historical_labels','scalar_metrics','axis_families']}))
        meta=pd.read_csv(path/'case_metadata.csv')
        scans=frame[frame.family.isin(['reference','plane','joint_path','random_rotation'])]
        unique=scans.merge(meta[['case_id','digest']],on='case_id').drop_duplicates(['digest','response'])
        for response,g in unique.groupby('response',sort=False):
            star=g[g.family=='reference'].iloc[0]
            for metric in metrics:
                v=g[metric].dropna();base=star[metric]
                positions.append(dict(model=path.name,response=response,metric=metric,reference=base,
                    n_valid=len(v),fraction_below_reference=float((v<base-1e-10).mean()) if len(v) else np.nan,
                    fraction_above_reference=float((v>base+1e-10).mean()) if len(v) else np.nan,
                    minimum=v.min(),maximum=v.max()))
                availability.append(dict(model=path.name,response=response,metric=metric,valid_cases=int(g[metric].notna().sum()),total_cases=len(g)))
                associations.append(dict(model=path.name,response=response,metric=metric,
                    spearman_with_parameter_recovery=safe_spearman(g[metric],g.recovery_mean_r2),
                    note='Posthoc supervised association; not used for candidate selection'))
        for (response,plane),g in frame[frame.family=='plane'].groupby(['response','plane']):
            for metric in metrics:
                valid=g.dropna(subset=[metric]);zero=g[g.angle==0][metric].iloc[0]
                if valid.empty:continue
                low=valid.loc[valid[metric].idxmin()];high=valid.loc[valid[metric].idxmax()]
                optima.append(dict(model=path.name,response=response,plane=plane,metric=metric,
                    reference_value=zero,minimum_angle=low.angle,minimum_value=low[metric],
                    maximum_angle=high.angle,maximum_value=high[metric],range=valid[metric].max()-valid[metric].min()))
        truth=unique[unique.response=='true_logit'].set_index('case_id')
        for response,g in unique.groupby('response'):
            joined=g.set_index('case_id').join(truth[metrics],rsuffix='_oracle')
            for metric in metrics:
                agreements.append(dict(model=path.name,response=response,metric=metric,
                    spearman_landscape_vs_true_logit=safe_spearman(joined[metric],joined[metric+'_oracle'])))
        if plots:draw(path,frame,metrics)
    summary=pd.DataFrame(summaries);summary.to_csv(out/'reference_quality.csv',index=False)
    pd.DataFrame(positions).to_csv(out/'reference_metric_positions.csv',index=False)
    pd.DataFrame(optima).to_csv(out/'all_plane_metric_optima.csv',index=False)
    pd.DataFrame(agreements).to_csv(out/'response_landscape_agreement.csv',index=False)
    pd.DataFrame(associations).to_csv(out/'metric_recovery_associations_DIAGNOSTIC.csv',index=False)
    pd.DataFrame(availability).to_csv(out/'metric_availability.csv',index=False)
    pd.concat(all_values).to_csv(out/'all_metric_values.csv',index=False)
    text=['# UXY 指标景观：第一项独立实验','',
        '本实验是监督参考坐标附近的描述性指标比较，不是新的无监督旋转方法。A* 仅用 NP TRAIN 上的原始 8 维 Z 回归标准化真实参数构造；全部指标计算器只接收 U、X、Y 及固定数值设置。',
        '', '五种 Y：实际 observed choice、decoder 概率的固定 Bernoulli 抽样、decoder logit、真实生成 logit、真实 logit 加固定高斯噪声。噪声标准差为 TRAIN 校准真实 logit 标准差的 25%。所有坐标和角度共享同一批实际试次、同一组响应和噪声。',
        '', '## 参考坐标实际恢复水平','',summary.to_markdown(index=False),
        '', '参考值不预设为 1；A* 为一般线性回归映射，通常是矩形/斜交映射。随后才对它的 K 维输出作正交旋转。历史坐标可能属于不同子空间/变换族，不把它们冒充为 U* 的纯旋转。',
        '', '## 指标及图形','',
        '每模型包含所有坐标对的平面扫描、一条全部平面共同变化的联合路径、固定随机正交旋转，以及来源哈希匹配的既有旋转。0° 是 U*；±90° 平面旋转可以是符号/置换等价，不要求唯一最优恰为 0°。联合路径不是整个 SO(K) 空间。',
        '', 'axis_metrics.npz 保存逐轴指标；metric_values.csv 保存标量聚合。RMS 与熵是同一估计器的两种摘要，不算独立证据。各模型 catalog.json 列出全部名称。',
        '', '每模型 curves_<Y>.pdf 包含所有标量指标的全部平面/联合角度曲线；joint_path_all_metrics.png 画所有指标的联合路径，颜色是相对 0° 的逐行缩放变化，原始数值见 CSV。历史坐标结果同样在 CSV 中，未按参数恢复挑选指标。',
        '', '## 必须保留的限制','',
        '- 非参数依赖估计只取每人一个实际试次，避免把重复的 U 当作独立被试。回归部分使用多个实际试次，但交叉验证按被试分组。',
        '- NP 的 S5 Z 已编码全部观察 choice，当前样本也已在此前研究中查看；不能把这些 CV 分数解释为全新的未见试次预测验证。',
        '- 标准化固定在旋转前；不逐轴重新标准化旋转结果。各历史坐标只进行 TRAIN 中心化和一个统一 RMS 缩放。',
        '- pdCor 是描述性偏距离相关，不等于条件独立性检验。CODEC 有限样本可负，分母为零时记 NA。',
        '- 连续 kNN CMI 不适用于本实现的二元 Y，故记 NA；没有用微小 jitter 把二元标签伪装成连续数据。',
        '- coarsened_X_cmi 只控制 X 的两个 PC 的粗分层，不能称为完整高维 CMI。adaptive_grid_nmi_proxy 不是 MIC/MICe。交互信息 proxy 不是 PID synergy。',
        '- GAM 是有限 B-spline 加性 ridge；不能声称包含任意交互。删除变量的 CV partial/semipartial R²可负。样条结点用无标签的描述性输入确定。',
        '- proxy_* 来自 U/X 二次张量积 ridge，是低阶函数近似的结构指标，不是 MLP 或真实函数的导数。相应分组 CV R²必须一起查看。扫描对同一个已拟合 proxy 作精确张量变换，避免逐角度重拟合误差。',
        '- 完整欧氏 RBF、完整多项式预测和完整导数能量等作为旋转不变量负对照；这些量不应被当作选轴目标。',
        '- 二元 choice 的有限样本噪声、近邻维数灾难、确定性 logit 的条件依赖饱和，都可能使指标平坦或失真；不强制解释为 U* 最好。',
        '', '所有参考排名、极值角度和指标与参数恢复的关联均为事后诊断，不把它们包装为已验证的共同优化目标。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def draw(path,frame,metrics):
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    for response,g in frame.groupby('response',sort=False):
        scan=g[g.family.isin(['plane','joint_path'])]
        planes=list(scan[scan.family=='plane'].plane.unique())
        with PdfPages(path/f'curves_{response}.pdf') as pdf:
            for start in range(0,len(metrics),12):
                fig,axes=plt.subplots(4,3,figsize=(15,15))
                for ax,metric in zip(axes.flat,metrics[start:start+12]):
                    for index,plane in enumerate(planes):
                        values=scan[scan.plane==plane].sort_values('angle')
                        ax.plot(values.angle,values[metric],lw=.8,alpha=.65,color=plt.cm.tab20(index%20))
                    joint=scan[scan.family=='joint_path'].sort_values('angle')
                    ax.plot(joint.angle,joint[metric],'k--',lw=1.5)
                    zero=g[g.family=='reference'][metric].iloc[0]
                    if np.isfinite(zero):ax.scatter([0],[zero],c='red',s=12,zorder=5)
                    else:ax.text(.5,.5,'Not applicable / undefined',transform=ax.transAxes,ha='center',fontsize=8)
                    ax.axvline(0,c='gray',lw=.5);ax.set_title(metric,fontsize=8);ax.tick_params(labelsize=7)
                    ax.set_xlabel('Rotation angle (degrees)',fontsize=8)
                for ax in axes.flat[len(metrics[start:start+12]):]:ax.axis('off')
                fig.suptitle(f'{path.name} | {response} | red=U*; dashed=joint path; colored=all planes',fontsize=12)
                fig.tight_layout(rect=(0,0,1,.97));pdf.savefig(fig);plt.close(fig)
    responses=list(frame.response.unique())
    fig,axes=plt.subplots(1,len(responses),figsize=(25,max(16,len(metrics)*.25)),sharey=True)
    for ax,response in zip(axes,responses):
        g=frame[(frame.response==response)&(frame.family=='joint_path')].sort_values('angle')
        reference=frame[(frame.response==response)&(frame.family=='reference')].iloc[0]
        raw=g[metrics].to_numpy().T-reference[metrics].to_numpy(dtype=float)[:,None]
        valid=np.isfinite(raw).any(1)
        scale=np.max(np.abs(np.nan_to_num(raw)),axis=1)
        zero=scale<1e-8*np.maximum(1,np.abs(reference[metrics].to_numpy(dtype=float)))
        scaled=np.divide(raw,scale[:,None],out=np.zeros_like(raw),where=scale[:,None]>0)
        scaled[zero]=0;scaled[~valid]=np.nan
        ax.imshow(np.ma.masked_invalid(scaled),aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
        ticks=np.linspace(0,len(g)-1,7,dtype=int);ax.set_xticks(ticks,g.angle.iloc[ticks].astype(int))
        ax.set_yticks(range(len(metrics)),metrics,fontsize=6);ax.set_title(response,fontsize=10)
        ax.set_xlabel('Joint-path angle');ax.axvline(np.flatnonzero(g.angle.to_numpy()==0)[0],c='black',lw=.6)
    fig.suptitle(path.name+' | all metric changes from U* | independently row-scaled; gray/blank=NA',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(path/'joint_path_all_metrics.png',dpi=160);plt.close(fig)
