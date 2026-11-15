# EXP18 · 严格矩阵估计与平滑 surrogate

独立运行：`python three_gaps/post_exp13_20261114/exp18_matrix_only/run.py --model CR3 --y-source nn --channel prob --tag round1`。其他来源为 `analytic` 和 `choice`（后者强制 p 目标，不能取 logit）。

采集 TRAIN 64 被试×64 真实试次和 VALIDATION 64 被试×64 试次，先保存 W,Y/subject IDs/原行号。`surrogates.py` 只接收这些矩阵，不持有原 decoder/T/解析函数。多项式总阶 2/3/4×ridge、384 基函数 kernel、最多 384 观测的 GP、3 个 tanh MLP 和 UU 等向有限扰动正则 MLP，均按 held-out 被试 Y-MSE 选超参/训练步数。choice 使用平方误差作为条件概率估计，不声称已经恢复真实条件均值。

analytic 的 W 是真实标准化 T 的固定 Haar 混合，属于 Gap2；nn/choice 的 W 用原 NP V，属于 Gap3/观测数据。这些不是相同潜在坐标，相关矩阵和 Oracle 分开记录。矩阵拟合/旋转完成后，`audit.py` 才读取 teacher 导数，报告结构误差与 landscape 排序，不能反馈到拟合选择。

旋转结构先在固定 16×16 TRAIN 网格估计。原 NP 的固定/连续 X 重训、splines、更多独立训练子集和全量结构估计仍属于后续扩展。
