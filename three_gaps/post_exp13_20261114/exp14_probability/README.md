# EXP14 · Probability Hessian 分解

独立运行：`python three_gaps/post_exp13_20261114/exp14_probability/run.py --model CR3 --repeat 1 --source decoder --subjects 300 --trials 500 --tag mechanism_v1`。

34 个预先固定的候选：η、A、B、p 的单项和组合，UX/UU 的全部 4×4 配对，无权/一阶 sigmoid 权重的梯度外积，一阶梯度及 X 聚类梯度协方差，κ=.25/.5/2/4，以及 tanh/softplus/asinh 输出变换。κ=0/1 对应 A/p，identity/sigmoid 对应 η/p。来源 `analytic` 是真实标准化参数坐标的 Gap1，不与 NN 原 V 坐标混称。

`decomposition.json` 保存原坐标与 Oracle 处的 E[A²]、E[B²] 和有符号 2E[AB]；负交叉项不参与非负稀疏度。`identity_check.json` 对照独立缓存的 p 导数，误差必须 <1e-4。优化不读取 T；Oracle 和所有平面仅在保存搜索旋转后诊断。

每个来源/模型/重复的全部候选与终点在 `results/<tag>/`；正式结论以顶层 RESULTS 和两次重复为准。
