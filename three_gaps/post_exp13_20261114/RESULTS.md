# 实测结果与实施边界

本页由 `report.py` 从实际 CSV/完成标记生成。`pilot` 仅检查代码通路，排除正式汇总。所有四模型 TEST 都已被历史工作查看，本页属于探索性结果。

## EXP13 冻结与原入口复现

| model | repeat | mcc | leakage | max_leakage |
|---|---|---|---|---|
| C1 | 1 | 0.9688 | 0.0212 | 0.0576 |
| C1 | 2 | 0.9689 | 0.0202 | 0.0611 |
| C2 | 1 | 0.9645 | 0.0599 | 0.1523 |
| C2 | 2 | 0.9677 | 0.0427 | 0.1450 |
| C3 | 1 | 0.9600 | 0.0464 | 0.1143 |
| C3 | 2 | 0.9615 | 0.0478 | 0.1525 |
| CR3 | 1 | 0.8173 | 0.1646 | 0.3677 |
| CR3 | 2 | 0.8122 | 0.1832 | 0.3781 |

原 float32 优化器、24×250、Haar 200 seed 1、起点 seed 7；归档 manifest 见 `baseline/results/archive_manifest.json`。

## 六条路线的实际覆盖

| experiment | completed_configs | in_progress_configs | candidate_rows |
|---|---|---|---|
| exp14_probability | 12 | 1 | 385 |
| exp15_finite_scale | 1 | 1 | 8 |
| exp16_local_context | 0 | 0 | 0 |
| exp17_reliability | 2 | 1 | 30 |
| exp18_matrix_only | 0 | 0 | 0 |
| exp19_model_library | 1 | 0 | 30 |


## exp14_probability

CR3 全候选（所有方法按预设规则搜索后评价，未按 TEST MCC 选解）：

| tag | repeat | channel | evidence | method | mcc | leakage | max_leakage |
|---|---|---|---|---|---|---|---|
| full | 1 | prob | CR3_rep1_decoder | AUX_BUU | 0.7760 | 0.2019 | 0.5475 |
| full | 1 | prob | CR3_rep1_decoder | B__combo | 0.8057 | 0.1944 | 0.3622 |
| full | 1 | prob | CR3_rep1_decoder | p__combo | 0.8173 | 0.1646 | 0.3677 |


## exp15_finite_scale

四模型批次进行中，尚无已完成的 CR3 正式设置。

## exp16_local_context

正式批次尚未产出 metrics；不能把实现或试跑写成完成验证。

## exp17_reliability

四模型批次进行中，尚无已完成的 CR3 正式设置。

## exp18_matrix_only

正式批次尚未产出 metrics；不能把实现或试跑写成完成验证。

## exp19_model_library

批次 `grammar_v1`：测试函数族与训练/validation 分离；负对照单列，不能平均成成功率。
| split | family | method | mcc | leakage | axis_error | abstain |
|---|---|---|---|---|---|---|
| test | composite | EXP13 | 0.9192 | 0.1087 | 0.0806 | 0.0000 |
| test | composite | learned | 0.9925 | 0.0457 | 0.0076 | 0.0000 |
| test | difference_square | EXP13 | 0.8801 | 0.1359 | 0.1197 | 0.3333 |
| test | difference_square | learned | 0.8244 | 0.2227 | 0.1762 | 0.6667 |
| test | radial | EXP13 | 0.7340 | 0.3409 | 0.2670 | 1.0000 |
| test | radial | learned | 0.7522 | 0.3157 | 0.2491 | 1.0000 |
| test | units | EXP13 | 0.9974 | 0.0315 | 0.0026 | 0.0000 |
| test | units | learned | 0.9160 | 0.1442 | 0.0844 | 0.3333 |
| validation | cubic | EXP13 | 0.9927 | 0.0461 | 0.0073 | 0.0000 |
| validation | cubic | learned | 0.9847 | 0.0573 | 0.0153 | 0.0000 |


## 必须保留的限制与后续工作

- EXP14/17 正式字段预算为 300 TRAIN 被试和全部冻结刺激；EXP15 为 8×8 中心点的查询实验，EXP16 为 64×64（逐点 128 点），EXP18 拟合 64×64 真实试次。采样量、估计器与优化器差异不能混同于 loss 改善。
- EXP15 的 UX 是真实 bank 最近邻配对的方向响应，不是完整环境 UX Hessian；全尺度、全矩阵有限差分都在每个 R 重新查询。
- 可靠性共识不能证明正确；独立 surrogate 仍共享原 NP 表示，可能保留系统偏差。
- EXP18 按 validation 被试的 Y-MSE 选超参；post-fit teacher 导数只用于审计。choice 的 NP 导数参考不是真实条件均值的导数。
- EXP19 使用合成监督和一个小型固定 grammar；不宣称通用 learned simplicity。径向反例应拒绝唯一恢复；满响应 Jacobian 秩的反例仍可能偏好另一套轴。
- 尚未完成：新 S1 数据、NP repeat 3–5、beta/PCA 大网格、更大有限扰动预算、有限差分的局部张量、新固定/连续刺激设计的 NP 重训与大规模 grammar。

完整逐候选机器可读结果：[all_metrics.csv](all_metrics.csv)。每个结果子目录含 config、rotations、diagnostics、全部平面与 U–T 热图；运行命令和失败栈在 `logs/`。
