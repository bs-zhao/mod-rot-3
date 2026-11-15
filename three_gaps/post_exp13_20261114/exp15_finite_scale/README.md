# EXP15 · 多尺度有限响应

独立运行：`python three_gaps/post_exp13_20261114/exp15_finite_scale/run.py --model CR3 --repeat 1 --tag round1 --subjects 8 --trials 8 --starts 24 --steps 250`。

预设 h=.1/.25/.5/1 个旋转后 U 的标准差。每个 R 都逆映射到原 frozen V，重新查询一阶、UU 轴向和混合有限差分；不把有限差分当张量旋转。相同中心点、X 配对、起点和提案种子固定。4 个单尺度组合与多尺度等权组合均按 Haar 标准差缩放。

X 使用标准化距离最近的不同 bank 点。其混合量是支持域内方向 chord 响应在 X 列上的投影，**不是完整环境 UX Hessian，也不是与原 EXP13 UX 完全相同的观测量**。对连续 X 也使用同一个配对协议，不利用模型角色。大尺度 U 支持域用 TRAIN Mahalanobis 99% 椭球覆盖率诊断，不按 R 删除点。

首轮 8×8 中心点；优化是共同随机数的 greedy Givens 搜索，预算与基线 Adam 的目标调用次数相近，算力成本不同。`matched_limit.py` 给出同样采样、配对与搜索器的 autograd 极限对照。扩大到 64×64、128×128 和完整字段前，不能宣称大规模结果。

`--only` 可分别运行 `h0.1_UU`、`h0.25_UX`、`h0.5_gradient`、`h1.0_UUoff` 等单项；正式默认组合并非按 T 选择。
