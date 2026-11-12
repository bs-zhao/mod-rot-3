"""Readable retained candidates, including displaced optima; no automatic veto."""
import csv

import run as wave
import record_rotation_optima as optima


CANDIDATES=[
    ('树：条件 UX 函数的 10% L2 规则成本','AU','UX_function_tree_leaves_relative_L2_eps100_Xworst'),
    ('多项式近似后：同一树规则成本','AY_poly5','UX_function_tree_leaves_relative_L2_eps100_Xworst'),
    ('GP 后验函数：3% UX 小尾部','BA2','derivative_pooled_UX_columns_worst_soft_count_eps003'),
    ('固定径向 GP：点态 UX 软支持','BB','point_UX_columns_Xmean_subject_mean_soft_count_eps010'),
    ('固定径向 GP、秩变换：3% UX 小尾部','BB2','pooled_UX_columns_worst_soft_count_eps003'),
    ('条件核：10% UX 软支持','BC2','Ui_innovation_with_context_raw_UX_columns_worst_soft_count_eps010'),
    ('条件核：1% UX 分数支持','BC2','Ui_innovation_with_context_raw_UX_columns_worst_fractional_count_eps001'),
    ('双侧残差：10% UX 软支持','BG','poly5_rbf128_Ui_innovation_with_context_raw_UX_columns_worst_soft_count_eps010'),
    ('层级组 GP：10% UX 软支持','BJ','poly3_rbf32_posterior_mean_UX_columns_worst_soft_count_eps010'),
    ('各 U 子集高斯核：10% UX 软支持','BM','poly3_rbf8_posterior_mean_UX_columns_worst_soft_count_eps010'),
    ('多项式 GP 学习全局输出变换：1% UX 分数支持','BR','Ui_innovation_with_context_raw_UX_columns_worst_fractional_count_eps001'),
    ('纯 U 与 UX：10% 最差分量支持','BK','with_U_main_Ui_innovation_with_context_raw_UX_columns_worst_soft_count_eps010'),
    ('纯 U 与 UX：10% 平均分量支持','BK','with_U_main_Ui_innovation_with_context_raw_UX_columns_mean_soft_count_eps010'),
]


def main():
    optima.main()
    output=wave.ROOT/'outputs/supervised_abs_rotation/rotation_optima_including_nearby'
    with (output/'four_model_optima.csv').open() as stream:rows=list(csv.DictReader(stream))
    table=[];selected=[]
    for label,tag,metric in CANDIDATES:
        found={r['model']:r for r in rows if r['wave']==tag and r['channel']=='logit_nn' and r['metric']==metric}
        cells=[];planes=[]
        for model in ('CR3','C1','C2','C3'):
            if model not in found:cells.append('运行中');planes.append('—');continue
            row=found[model];selected.append(dict(candidate_label=label,**row))
            cells.append(f"{row['minimum_grid_angle_deg']}° / {float(row['minimum_grid_relative_change_percent']):+.6g}%")
            planes.append(f"{row['success_planes']}/{row['n_planes']}")
        table.append('| '+label+' | '+' | '.join(cells)+' | '+'，'.join(planes)+' |')
    text='''# 保留的近似候选：最优位置和连续分数差

每行使用相同定义、同一输出通道和同一参数策略评估四库。
单元格为“已观测网格最优角度 / 相对 baseline 的分数变化”。负值表示比 baseline 更低。
0° 表示 baseline 是网格最小值之一。75° 与 15° 在允许轴置换时均对应 15° 轴偏移。
这些是探索性保留项，不按 MSE=.01、是否恰好 0° 或某个小差距自动淘汰。

| 同一指标 | CR3 | C1 | C2 | C3 | 严格平面计数（仅诊断） |
|---|---|---|---|---|---|
'''+ '\n'.join(table)+'''

全部最优平面、最小值并列项、15°/75° 点和完整曲线跨度见 `four_model_optima.csv`。
更多相关变体见 `four_model_overview.csv`；多个变体高度相关，不代表独立验证。

网格之外的反例另行保留：BC2 的旧盲搜方向低约 12.1%；1% 分数支持只低约
0.107%，但在 24 次成对重新估计的被试子集中始终更低。BK 的新盲搜差距约 2.04%。
这些结果不会因网格位置接近就被隐藏，也不会抹去网格上的近似信号。
已冻结盲搜结果的轴匹配角和分数差见同级目录的
`../frozen_blind_axis_positions/axis_positions.csv`：分数近似不等于坐标恢复近似。
'''
    (output/'approximate_candidate_shortlist.md').write_text(text,encoding='utf-8')
    wave.csv_file(output/'approximate_candidate_shortlist.csv',selected)
    print('[Retained shortlist]',str(output/'approximate_candidate_shortlist.md'))


if __name__=='__main__':main()
