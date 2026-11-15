# EXP16 · 局部与上下文结构

独立运行：`python three_gaps/post_exp13_20261114/exp16_local_context/run.py --model CR3 --channel prob --tag round1`。

比较总体 neff、逐点 neff 的均值、按点总能量加权的局部 neff，以及固定 X 聚类、第一主成分分位数、重叠近邻和 V 的欧氏聚类。所有上下文不随 R 变化；V 欧氏距离在正交坐标变换下不变。组合分别使用 UUd_share 或 UU 非对角 neff，不把互补比例相加。

总体/context 使用 64×64 字段；逐点使用其中固定的 128 点以控制计算量。低能量点按旋转前字段阈值 1e-20 固定跳过，报告比例。梯度上下文单列；有限差分局部张量尚未纳入首轮。
