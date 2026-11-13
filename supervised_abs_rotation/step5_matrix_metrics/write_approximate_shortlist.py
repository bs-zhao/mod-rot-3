"""Readable retained candidates, including displaced optima; no automatic veto."""
import csv
import json

import run as wave
import record_rotation_optima as optima


CANDIDATES=[
    ('一致多项式核方差剖面：3% UX 支持（p_nn）','CG','nested_polynomial_variance_profile_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('一致多项式核方差能量：3% UX 支持（p_nn）','CG','nested_polynomial_variance_energy_UX_columns_mean_soft_count_eps003','p_nn'),
    ('留一预测方差 Shapley：最差 X 列 3% UX 支持（logit）','CF','PRESS_variance_UX_columns_worst_soft_count_eps003'),
    ('留一预测方差 Shapley：3% UX 支持（p_nn）','CF','PRESS_variance_UX_columns_mean_soft_count_eps003','p_nn'),
    ('留一预测方差 Shapley：1% UX 分数支持（p_nn）','CF','PRESS_variance_UX_columns_mean_fractional_count_eps001','p_nn'),
    ('自适应条件 Shapley＋未归因分量：3% UX 支持（p_nn）','CE','resolved_with_unassigned_profile_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('GCV 自动选逼近器的条件 Shapley：3% UX 支持（p_nn）','CD','shapley_profile_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('径向核 256 中心条件 Shapley：3% UX 支持（p_nn）','CB','shapley_profile_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('五次多项式＋径向核条件 Shapley：3% UX 支持（p_nn）','CC','shapley_profile_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('留一预测增益剖面：3% UX 支持（p_nn）','CA','conditional_PRESS_projection_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('留一预测增益能量：3% UX 支持（p_nn）','CA','conditional_PRESS_energy_UX_columns_mean_soft_count_eps003','p_nn'),
    ('条件正交投影剖面：3% UX 支持（p_nn）','BZ','conditional_span_projection_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('条件正交投影能量：3% UX 支持（p_nn）','BZ','conditional_span_energy_UX_columns_mean_soft_count_eps003','p_nn'),
    ('树小波局部变化：3% UX 支持（p_nn）','BY','haar_node_UX_columns_mean_soft_count_eps003','p_nn'),
    ('树小波按 U 轴汇总：3% UX 支持（p_nn）','BY','haar_axis_UX_columns_mean_soft_count_eps003','p_nn'),
    ('条件交互子集剖面：3% UX 支持（p_nn）','BX','conditional_dividend_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('条件交互子集能量：1% UX 分数支持（p_nn）','BX','conditional_dividend_energy_UX_columns_mean_fractional_count_eps001','p_nn'),
    ('同向导数剖面：10% UX 分数支持','BW','norm_lift_UX_operator_columns_worst_fractional_count_eps010'),
    ('条件 Shapley 函数剖面：3% UX 支持（p_nn）','BU','shapley_profile_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('条件 Shapley，64 核中心：3% UX 支持（p_nn）','BU3','shapley_profile_UX_operator_columns_mean_soft_count_eps003','p_nn'),
    ('学习输出变换的算子模态：10% UX 分数支持（p_nn）','BT','Ui_innovation_with_context_UX_operator_columns_worst_fractional_count_eps010','p_nn'),
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
    table=[];selected=[];basins=[]
    for candidate in CANDIDATES:
        label,tag,metric=candidate[:3];channel=candidate[3] if len(candidate)>3 else 'logit_nn'
        found={r['model']:r for r in rows if r['wave']==tag and r['channel']==channel and r['metric']==metric}
        cells=[];planes=[]
        for model in ('CR3','C1','C2','C3'):
            if model not in found:
                folder=wave.io.default_step2(wave.ROOT,model).parent/'step5_matrix_metrics'/optima.WAVES[tag]
                if (folder/'protocol.json').exists():
                    status=json.loads((folder/'protocol.json').read_text())['status']
                    cells.append('未完成' if status!='COMPLETE' else '未记录此项')
                else:cells.append('未运行')
                planes.append('—');continue
            row=found[model];selected.append(dict(candidate_label=label,**row))
            if tag in ('BU','BU3','CB','CC','CD','CE','CF','CG'):
                bands={int(b['score_tolerance_percent']):b for b in json.loads(row['near_optimal_grid_bands_json'])}
                cells_band=[str(bands[p]['maximum_axis_deviation_deg'])+'°' for p in (1,3,15)]
                rank=f"{row['baseline_rank']}/{row['grid_points']}"
                percentile=f"{float(row['baseline_strictly_better_other_grid_points_percent']):.3g}%"
                value=row['baseline_regret_relative_grid_span_percent'];regret=f'{float(value):.3g}%' if value else '无幅度'
                basins.append('| '+tag+' / '+model+' | '+rank+' | '+percentile+' | '+regret+' | '+' | '.join(cells_band)+' |')
            cells.append(f"{row['minimum_grid_angle_deg']}° / {float(row['minimum_grid_relative_change_percent']):+.6g}%")
            planes.append(f"{row['success_planes']}/{row['n_planes']}")
        table.append('| '+label+' | '+' | '.join(cells)+' | '+'，'.join(planes)+' |')
    basin_text='''

同一 Shapley 候选的“近最优网格区间”也保留。下面给出分数落在已观测网格最小值
1% / 3% / 15% 内的点所达到的最大轴偏移。90° 置换端点不重复计数。
这描述当前网格的定位信息，不是新淘汰标准，也不包含后续连续旋转搜索点。
如果较小分数带也覆盖 45°，就不能只凭一个 15° 最优点声称轴已精确定位。
“更低分点占比”以其余全部给定网格点为分母，包含 90° 置换端点；它不是随机旋转概率。
这样角度、相对分数差和网格排名百分比都各自记录，不把 15° 和 15% 混用。
“距最优占曲线幅度”把基准与最小值的差除以整个网格的最大最小差；曲线没有幅度时留空。
它能区分低分差来自接近最优，还是来自整个指标对旋转都几乎不变。

| 候选 / 数据库 | 基准排名 | 更低分点占比 | 距最优占曲线幅度 | 1% 分数带 | 3% 分数带 | 15% 分数带 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(basins)
    text='''# 保留的近似候选：最优位置和连续分数差

每行使用相同定义、同一输出通道和同一参数策略评估四库。
单元格为“已观测网格最优角度 / 相对 baseline 的分数变化”。负值表示比 baseline 更低。
0° 表示 baseline 是网格最小值之一。75° 与 15° 在允许轴置换时均对应 15° 轴偏移。
这些是探索性保留项，不按 MSE=.01、是否恰好 0° 或某个小差距自动淘汰。

| 同一指标 | CR3 | C1 | C2 | C3 | 严格平面计数（仅诊断） |
|---|---|---|---|---|---|
'''+ '\n'.join(table)+basin_text+'''

全部最优平面、最小值并列项、15°/75° 点和完整曲线跨度见 `four_model_optima.csv`。
更多相关变体见 `four_model_overview.csv`；多个变体高度相关，不代表独立验证。

条件 Shapley 的 CR3 新盲搜低约 0.275%，轴匹配平均 12.7°、最大 17.3°；
BU6 更细搜索进一步找到低约 1.130%、最大轴偏移 7.524° 的独立随机起点结果。
这些属于保留的近似结果，搜索未证明全局最优；C2/C3 的部分随机起点仍有远轴近分数。
BT 的 CR3 旧冻结方向低约 15%—18%，
因此其近似网格位置不能单独作为坐标恢复证据。

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
