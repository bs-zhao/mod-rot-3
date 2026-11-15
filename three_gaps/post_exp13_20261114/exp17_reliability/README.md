# EXP17 · 可靠性与共识

独立运行：`python three_gaps/post_exp13_20261114/exp17_reliability/run.py --model CR3 --channel prob --subjects 300 --trials 500 --tag round1`。

TRAIN 被试、试次各拆成 2/4/8 组，共同 200 Haar seed 1 用于 T-free ρ 估计，独立 seed 91 的 1000 Haar 检查 Spearman 与低 loss 20% 区域排名。逐组优化后报告有符号排列等价距离和交叉子集 loss。原等权、max(0,ρ) 权重、均值+1 SD 都预先固定。

SD 下限 1e-9；所有 ρ 为零时记录 no_signal 并不输出强行恢复的 ρ 方法。一个 decoder 的 bootstrap 一致性不能排除系统性偏差。EXP17B 的独立固定-Z surrogate 位于 EXP18：3 个初始化与 80% TRAIN 被试子采样、同一 encoder/Z，原 NP 坐标不跨网络平均。
