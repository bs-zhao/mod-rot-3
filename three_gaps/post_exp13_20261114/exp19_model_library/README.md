# EXP19 · 模型族与 learned score

独立运行：`python three_gaps/post_exp13_20261114/exp19_model_library/run.py --tag grammar_v1`。

小型明确 grammar：加法/乘法/有限阶幂、输出 sigmoid、tanh、sin、有界 exp；每族 3 个固定种子生成配置。训练族 additive/products/shared/sigmoid，validation 族 cubic，测试族 composite/实际 X 列量纲改变/满响应秩的二次反例/径向不可识别。公式族不跨 split。

从有限 W,Y 拟合 degree 2–4 sieve（只按 validation Y-MSE 选阶），提取 6 个对有符号轴排列不变的结构摘要，训练 24×24 tanh score 网络预测旋转的轴匹配误差。合成 Q/T 可用于监督；不声称 T-free meta-training。输入行排列不变，latent 正交重参数化的候选得分等变由数值检查验证。

score 在 validation 族选择训练步数，然后冻结，对未见族搜索，与固定 EXP13 比较。报告局部响应 Jacobian 秩、低谷多解、拒绝恢复和反例；局部满秩不保证真轴是最简单轴。拒绝规则用拟合出的局部梯度秩比和最低 loss 多起点分歧，不读取测试 T。测试 T 只评价。

这是**小型 grammar 首轮**，不是大规模模型库或普适 learned simplicity。直接预测 R 网络是计划的另一条可选路线，尚未作为本轮成果。

`grammar_v1` 的测试族曾在 pilot 中被查看，因此保留探索性标签。`frozen_holdout.py` 使用 SHA256 固定的 grammar_v1 权重，对额外 rational/softplus_mix/sine_product/radial_two_blocks 四族测试，不重新训练、早停或校准 meta-score；`transfer_np.py` 把同一冻结 score 直接用于原四模型的两个重复和两个通道。这条迁移使用 decoder-autograd 字段，单列于严格 WY 模型库。
