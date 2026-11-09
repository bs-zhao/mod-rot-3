# Hessian 指标研究历程与完整方法档案（截至 2026-11-09）

> **新专题（2026-11-09）**：[旋转简单性与稀疏损失：CR3 06/07 实证、Toy M001–M024、ZZ/ZX 高阶结构和三个技术 gap](README_SPARSITY_SIMPLICITY_AND_FUNCTIONAL_ROTATION_20261109.md)。聚焦“多主作用 + 小残余”的 L1/L2/Top-k/Soft-L0/Log-sum/Capped 惩罚，并区分解析成功、Formula-Y 回归成功及 NN-Y 成功。


> **新专题（2026-11-09）**：[从 Hessian 函数到数值指标：非线性、交互、条件主效应与 RMS](README_HESSIAN_FUNCTIONAL_METRICS_AND_EFFECTS_20261109.md)。重点归档本轮对话逐步推导的例子、三种曲率能量的精确分解、RMS 为什么不抵消符号，以及数学领域中的典型用途。
>
> **性质**：研究讨论的历史汇总（history），不是最终定论、运行命令、实验结果汇报或已经全部实现的算法。收录本项目截至目前围绕 Hessian 的**数学对象、统计量、可优化简单性指标、数值估计、旋转、函数交互、与 Sobol 的关系及反例**。为便于回溯，也记录与 Hessian 紧密相关的梯度、ANOVA、Sobol 等**非 Hessian** 对照指标，但明确标注其类别。
>
> **术语约定**：历史讨论中用过 Z、T、U、X。先在一般问题中把输入统一写为 \(W\in\mathbb R^d\)；需要区分时令 \(W=(Z,X)\)，\(Z\) 为潜在/参数方向，\(X\) 为题目或刺激变量；\(U=RZ\) 为旋转坐标。这里 \(A_*\) 是通过真参数监督回归得到的诊断映射，**不能进入无监督寻找旋转的 Loss**。
>
> **相关主文件**：[总框架](../README.md) · [矩阵 M01–M20](../HESSIAN_TO_MATRIX_METRICS.md) · [函数算子 D×P](../HESSIAN_FUNCTIONAL_OPERATIONS.md) · [积分 Q](../HESSIAN_INTEGRATION_AND_AVERAGING.md) · [WY 估计 E01–E14](../HESSIAN_FROM_ZXY_METHODS.md) · [四个真实生成模型](../TOY_GENERATIVE_MODELS.md) · [旋转计划](../ORTHOGONAL_ROTATION_AND_HESSIAN_PLAN.md)。**本历史文档不重置这些文件的现有编号，也不声称每个候选指标已经跑通。**

## 0. 研究问题与必须分离的层次

给定 \(Y=f(W)\) 或仅观察到数据 \(\mathcal D=\{(W_s,Y_s)\}_{s=1}^N\)。研究：

1. **数学真值是什么**：\(H_f(w)=\nabla_w^2f(w)\) 是局部矩阵，随 w 改变，绝不是天然固定数值矩阵。
2. **要如何汇总**：取点、条件/先验积分、逐元素 RMS、矩阵 Gram、谱函数、局部先评分等，得到不同维数的输出。
3. **什么是效应**：\(H_{ii}\) 是沿第 i 个坐标的自身曲率；\(H_{ij}\,(i\ne j)\) 是两个边际作用的局部交互；它们不是 W 本身的统计相关系数。
4. **如何定义简单性**：对角曲率、非对角交互、每个 Z 影响多少 X、稀疏性、可加性与高阶效应的权衡；一个指标能算出数字，不代表它应被最小化。
5. **如何仅从 WY 估计**：函数/导数估计、直接平均导数泛函估计和有限样本识别约束。
6. **如何验证旋转**：解析真轴处应为**极小值**（若是简单性 Loss），formula-Y 矩阵估计应复现解析曲线，NN-Y 矩阵估计应与 formula-Y 及解析有一致角度模式，最终才检验旋转 U 与真实 T 一一对应、非对应相关低。
7. **对三种输出分别分析**：logit \(\eta\)、概率 \(p=\sigma(\eta)\)、随机 choice；同名 Hessian 不等于同一个目标。
8. **对函数与输入分布都要声明**：在 W 或 X 上取经验平均、对 T 真实先验积分、对 Z 经验分布积分，是不同统计目标。

数学表达应写成一条带有标签的路径：

\[
\boxed{\text{生成函数/数据来源}\to
\text{Hessian 估计器 E}\to
\text{局部 }H(W)\ \text{或直接泛函}\to
\text{积分与评价域 Q}\to
\text{D/P/M 指标}\to
\text{标量简单性 k}\to L(R).}
\]

**禁止混淆**：Hessian 的准确性、Hessian 汇总指标的数学意义、Loss 的旋转最小性，以及最终真参数恢复，是四个不同问题。

---

## 1. 从函数到 Hessian：矩阵本身及导数阶数

\[
H_{ij}(w)=\partial_{w_i}\partial_{w_j}f(w),\qquad
H(w)\in\operatorname{Sym}(d),
\quad g(w)=\nabla f(w).
\]

| 数学对象 | 输出 | 衡量的内容 / 限制 |
|---|---|---|
| \(f(w)\) | 标量函数 | 生成输出；本身的可加性与非线性可通过函数值评估 |
| \(g_i=\partial_i f\) | d 个标量函数 | 各维边际效应大小、符号、单调性 |
| \(H_{ii}=\partial_{ii}f\) | 单个标量函数 | 该坐标的**条件自身曲率**；可能随其他维度变化 |
| \(H_{ij}=\partial_{ij}f,\ i\ne j\) | 单个标量函数 | 改变 \(w_j\) 时 \(w_i\) 的边际效应如何变化 |
| \(H(w)\) | d×d 矩阵值函数 | 全部局部二阶作用；不自动等于固定矩阵 |
| \(\partial_kH_{ij}=\partial_{ijk}f\) | 三阶张量 | Hessian 元素如何继续变化，三阶效应 |
| \(\partial_{kl}H_{ij}=\partial_{ijkl}f\) | 四阶张量 | 曲率变化的二阶变化；高阶结构与平滑性 |
| \(v^\top H(w)v\) | 标量函数 | 方向 v 上的二阶方向导数 |
| \(H(w)v\) | 向量函数 | Hessian-vector product；更常是计算技术而非函数复杂度指标 |

若函数在连通矩形域上 \(C^2\)，所有混合导数 \(H_{ij}\equiv0\;(i\ne j)\) 当且仅当 \(f(w)=c+\sum_i f_i(w_i)\)。**仅有 \(\mathbb E[H_{ij}]=0\) 绝不能推出可加性**。高阶交互也能使混合二阶导数非零：例如 \(f=w_1w_2w_3\) 时 \(H_{12}=w_3\)。

自身曲率也可能来自交互：\(f(w_1,w_2)=w_1^2(1+w_2)\)，则 \(H_{11}=2(1+w_2)\)，这不是与其他维度无关的“纯主效应非线性”。

---

## 2. 两个独立数学操作：去掉 W 依赖 D × 输出形状 P

| | **保持原 d×d 形状** | **变成标量、向量或其他矩阵形状** |
|---|---|---|
| **仍含 W** | \(H(W)\)、\(H(W)^2\)、\(|H(W)|_{\rm el}\) | \(\operatorname{tr}H(W)\)、\(\operatorname{diag}H(W)\)、\(H(W)v\)、\(P^\top H(W)P\) |
| **完全消除 W** | \(H(w_0)\)、\(\mathbb E_Q H(W)\)、逐元素 RMS/Q99、\(\mathbb E_Q[H^2]\) | \(\operatorname{tr}H(w_0)\)、\(\mathbb E_Q\lambda_{\max}(H)\)、\(\sup_W\|H\|_2\)、\(\operatorname{Cov}_Q(\operatorname{vec}H)\) |

**D** 决定剩余自由变量；**P** 决定数值/函数的输出形状；**Q** 单独定义评价分布、权重、区域和求积实现；**k** 决定怎样把输出变成待最小化的标量。这四个层次不应合并。

### 2.1 D01–D10：从 Hessian 函数获得有限代表值的操作

| 编号 | 方法 | 公式/说明 | 典型目的 |
|---|---|---|---|
| D01 | 固定点取值 | \(H(w_0)\) | 驻点局部几何、Newton、单点诊断 |
| D02 | 有限加权求和 | \(\sum_s q_s H(w_s)\) | 固定设计/经验样本平均 |
| D03 | 对指定测度求期望 | \(\mathbb E_Q[H(W)]\) | 总体有符号平均 Hessian |
| D04 | 条件期望/部分积分 | \(\mathbb E_X[H(Z,X)\mid Z]\) | 留下 Z 相关矩阵场，需继续积分才能固定 |
| D05 | 轨迹/路径积分或平均 | \(T^{-1}\sum_tH(w_t)\)、曲线积分 | 沿优化路径/受控扫描的曲率 |
| D06 | 指数移动平均 | \(M_t=\rho M_{t-1}+(1-\rho)H(w_t)\) | 在线平滑，仅代表该路径 |
| D07 | 取极限 | \(\lim_{t\to\infty}H(w_t)\) | 渐近局部状态；可能不存在 |
| D08 | 上下确界 | \(\sup_{w\in\Omega}|H_{ij}(w)|\) | 最坏情况曲率/导数界 |
| D09 | 分位数/中位数/截尾 | \(Q_\alpha(|H_{ij}|)\) | 典型值与尾部稳健统计 |
| D10 | Loewner 矩阵界 | 寻找 \(M\) 使 \(H(w)\preceq M\) | 二阶全局界；矩阵序不是全序，一般无唯一最小界 |

另外，依照指定距离 \(d(\cdot,\cdot)\) 的**最佳常数代表矩阵**：

\[
M^*=\arg\min_M\mathbb E_Q[d(H(W),M)].
\]

平方 Frobenius 距离下解为 \(\mathbb E_QH\)；非平方 Frobenius 距离下解是相应的几何中位数（可能不唯一）；Huber、max-error 等给出不同代表。它是一个更一般的统计决策视角，未必有独立“标准 Hessian 指标”名称。

### 2.2 P：不要求先取固定矩阵的形状变换

| 类型 | 典型操作 | 数学目的/限制 |
|---|---|---|
| P0 同形矩阵 | \(H^2=HH\)、\(H^k\)、\(H\odot H\)、\(|H|_{\rm el}\) | **矩阵平方与逐元素平方不同** |
| P0 谱矩阵函数 | \(H_+,H_-,|H|_{\rm op}=(H^2)^{1/2}\) | 分离正负特征曲率；对称方阵要求 |
| P1 标量 | Trace \(=\Delta f\) | 拉普拉斯算子、扩散、调和分析；正交旋转不变 |
| P1 标量 | Determinant、logdet | 体积变化、Monge–Ampère、概率局部集中；logdet 实数有域限制 |
| P1 标量 | \(\lambda_{\max},\lambda_{\min}\)、负特征值个数/Morse index | 凸性、鞍点、局部不稳定方向 |
| P1 标量 | Frobenius / spectral / nuclear norm | 总曲率强度/最大幅度/奇异值总和；正交不变 |
| P1 标量 | \(v^\top Hv\)、rank、\(\kappa_2(H)\) | 方向曲率、退化、数值病态；奇异时条件数无穷 |
| P2 向量 | diag(H)、有序 eigenvalues(H)、Hv | 坐标自身曲率、谱、方向作用 |
| P3 子矩阵 | \(H_{SS}\)、\(P^\top HP\)、\(HP\) | 变量子集、固定低维子空间曲率 |
| P3 变形 | Schur complement \(H_{AA}-H_{AB}H_{BB}^{-1}H_{BA}\) | 变量消元，需要分块和可逆性 |
| P3 近似分解 | 低秩/Kronecker 因子 | 压缩曲率近似；K-FAC 通常是 Fisher/Gauss–Newton 近似，非一般 Hessian 恒等式 |

**操作顺序**：对固定矩阵投影 P，\(E[P^\top HP]=P^\top E[H]P\)，\(\operatorname{tr}(EH)=E\operatorname{tr}H\)；但一般有 \(\lambda_{\max}(EH)\le E\lambda_{\max}(H)\)，\(E[H^2]\ne(EH)^2\)，\(E\det H\ne\det(EH)\)。不能把“先平均再取范数”和“逐点先取范数再平均”混为一谈。

---

## 3. 完整数值矩阵指标 M01–M20（保留原仓库编号）

令 \(\langle\cdot\rangle=\mathbb E_Q[\cdot]\)，\(H\) 为局部 Hessian 或其指定子块；\(\odot\) 是逐元素乘积，\(|H|_{\rm el}\) 为逐元素绝对值；矩阵的开方若无特别说明是逐元素开方。除注明者外，各统计量都作用于每个 \(H_{ij}(W)\) 的 Q 分布。

| ID | 指标（历史讨论） | 数学定义 | 性质/用途 |
|---|---|---|---|
| **M01** | Signed Mean | \(\bar H=\langle H\rangle\) | 平均有符号作用，可能正负抵消 |
| **M02** | Mean Absolute | \(\langle|H|_{\rm el}\rangle\) | 平均绝对曲率，避免抵消 |
| **M03** | RMS | \(\sqrt{\langle H\odot H\rangle}\) | 曲率能量的平方根，突出局部大值 |
| **M04** | Mean Square | \(\langle H\odot H\rangle\) | 逐元素二阶矩，不是 \(HH\) |
| **M05** | Variance | \(\langle(H-\bar H)\odot(H-\bar H)\rangle\) | 曲率随 W 的波动 |
| **M06** | SD | \(\sqrt{M05}\) | 波动的原尺度 |
| **M07** | Generalized Lp | \((\langle|H|_{\rm el}^p\rangle)^{1/p}\) | p 调节尾部大曲率的权重 |
| **M08** | Median Absolute | \(\operatorname{median}_Q|H_{ij}|\) | 典型值，稳健 |
| **M09** | Absolute Quantile | \(Q_{\alpha,Q}(|H_{ij}|)\) | Q90/Q95/Q99 尾部指标 |
| **M10** | Maximum Absolute | \(\sup_{\Omega_Q}|H_{ij}|\) | 极端局部作用；可能无穷 |
| **M11** | Threshold Exceedance | \(\Pr_Q(|H_{ij}|>\tau)\) | 超过指定幅度的频率 |
| **M12** | Sign Consistency | \(|\langle\operatorname{sign}H_{ij}\rangle|\) | 符号稳定性，不是幅度 |
| **M13** | Positive/Negative Parts | \(\langle\max(H,0)\rangle,\langle\max(-H,0)\rangle\) | **逐元素**正负幅度，两张矩阵 |
| **M14** | Trimmed/Winsorized | 逐元素截尾/缩尾统计 | 对异常曲率更稳健 |
| **M15** | Row Gram | \(\langle HH^\top\rangle\) | 行方向二阶作用；方阵对称 H 时是 \(\langle H^2\rangle\) |
| **M16** | Column Gram | \(\langle H^\top H\rangle\) | 列方向二阶作用；矩形块尤其有用 |
| **M17** | Spectral Absolute | \(\langle(H^2)^{1/2}\rangle\) | **谱矩阵绝对值**平均，非逐元素绝对值 |
| **M18** | Hessian Element Covariance | \(\operatorname{Cov}(\operatorname{vec}H)\) | Hessian 元素之间共同变化；可从 d²×d² 降到独立对称元素空间 |
| **M19** | Conditional/ANOVA Variance | \(E_Z\operatorname{Var}_{X|Z}(H_{ij}|Z)\) 等 | 区分由 Z、X 和它们共同调节的 Hessian 变化 |
| **M20** | Higher-order Hessian Variation | \(E|\partial_kH_{ij}|\)、RMS\((\partial_kH_{ij})\)、\(\partial_{kl}H_{ij}\) 的统计 | 三阶/四阶导数，反映曲率随其他变量变化 |

**重要恒等式**（逐元素）：\(M03^2=M01^2+M05\)。所以 RMS 不会发生正负抵消，而 mean 会。例如曲率在半数点上 +5、半数点上 −5，signed mean 为 0，Mean Absolute 与 RMS 都为 5；若 90% 为 0、10% 为 10，则 Mean Absolute=1、RMS=\(\sqrt{10}\)。Var=0 也**不代表**曲率小：恒定曲率 100 的 variance 是 0。

**重要细节**：M15–M17 等是算子级统计；M18 是**不同尺寸矩阵**；M19/M20 可能需要条件设计或高阶导数。历史编号 M01–M20 **不代表这些全部已被同一程序实现**。精确定义和代码状态以 [原指标表](../HESSIAN_TO_MATRIX_METRICS.md) 为准。

---

## 4. 直接变成标量的局部/整体曲率指标（不必先有固定数值矩阵）

| 类型 | 标量指标 | 常见数学用途/能否区分旋转 |
|---|---|---|
| 局部总曲率 | \(\|H(w_0)\|_F^2=\sum_{ij}H_{ij}(w_0)^2\) | 曲率能量；完整对称 H 的正交旋转不变 |
| 局部最大幅度 | \(\|H(w_0)\|_2=\max|\lambda_i(H)|\) | 局部光滑度、最大方向作用；正交不变 |
| 全局光滑界 | \(\sup_{w\in\Omega}\|H(w)\|_2\) | 梯度 Lipschitz 常数的界；定义域/可微条件必需 |
| 强凸界 | \(\inf_{w\in\Omega}\lambda_{\min}(H(w))\) | 正时可给强凸性保证 |
| 迹/拉普拉斯 | \(\operatorname{tr}H(w)\) 或 \(E\operatorname{tr}H\) | PDE 中 \(\Delta f\)；正交不变 |
| 行列式 | \(\det H(w)\) 或 \(E\det H\) | Monge–Ampère、梯度映射体积；正交不变 |
| LogDet | \(\log\det H(w)\)（域内） | Laplace 近似与概率局部精度；不能对任意不定 H 取实数 logdet |
| 谱和条件数 | \(\lambda_\min,\lambda_\max,\sum|\lambda|,\kappa_2\) | 驻点分类、局部稳定性、病态分析 |
| 局部曲率符号 | 正/负特征值个数 | 鞍点/Morse 指数；非直接“简单” Loss |
| 方向曲率 | \(v^\top H(w)v\) | 特定方向作用；v 是否随坐标变换决定其含义 |
| Local-first 交互 | \(\mathbb E_Q[\sum_{i<j}H_{ij}(W)^2]\) | 均方混合二阶作用，不会正负抵消 |
| Matrix-first 交互 | \(\sum_{i<j}(\mathbb E_QH_{ij})^2\) | 平均混合导数后评分，容易符号抵消 |
| Grad Gram / Cov | \(E[gg^\top]\)、\(\operatorname{Cov}(g)\) | 梯度效应共同变化，**不是 Hessian**，也不等于交互 |

**警告**：完整 Hessian 的 trace、谱、行列式、Frobenius norm 在正交重参数化下具有不变性。作为**唯一旋转 Loss**时常是平坦曲线，不能识别参数轴；它们适合做诊断/尺度和几何约束，不代表有辨识力。

---

## 5. 从 Hessian 提取“自身非线性、两两交互、整体交互”

本节先令 \(W=(W_1,\ldots,W_d)\)，不区分 Z/X。

### 5.1 单维自身非线性

\[
N_i^{\rm total}=\sqrt{E_Q[H_{ii}(W)^2]},\quad
N_i^{\rm abs}=E_Q|H_{ii}|,\quad
N_i^{\rm signed}=E_QH_{ii}.
\]

这分别衡量沿 i 的总体曲率 RMS、绝对曲率幅度与平均方向；signed mean 可能正负抵消。其他可选：\(Q_{.95}|H_{ii}|\)、最大曲率、\(\operatorname{Var}(H_{ii})\)、Hessian 更高阶变化。

**条件自身曲率不等于纯主效应**：\(H_{ii}\) 可能依赖其他变量。更细的条件分解（有限二阶矩）为：

\[
\boxed{
E[H_{ii}^2]
=E[(E[H_{ii}\mid W_i])^2]
+E[\operatorname{Var}(H_{ii}\mid W_i)].
}
\]

分别定义 \(N_i^{\rm main-curv}\) 与 \(N_i^{\rm modulated-curv}\) 为后两项开平方。这个**数学恒等式始终成立**；如果想把第一个等同于由固定边际分布求得的 partial-dependence 主效应曲率，还需要 W 各维独立及微分与积分可交换等条件。

例子：\(f(w_1,w_2)=w_1^2(1+w_2)\)，且 \(w_2\sim N(0,1)\)、输入独立，\(H_{11}=2(1+w_2)\)。总体曲率 RMS=\(2\sqrt2\)，主效应曲率 RMS=2，调制曲率 RMS=2。\(f=w_1^2w_2\) 时可有总体曲率 2、主效应曲率 0、调制曲率 2。

### 5.2 两两函数交互与所有交互

\[
\boxed{I_{ij}^{H}=\sqrt{E_Q[H_{ij}(W)^2]},\quad i\ne j;}
\qquad
L_{\rm cross-H}=\sum_{i<j}E_Q[H_{ij}^2].
\]

这是 **crossed derivative-based global sensitivity measure (Crossed-DGSM)** 类指标，含二阶以上的共同参与；它**不是** \(W_i,W_j\) 的 Pearson correlation、MI 或 HSIC。

另一种是分母固定的规范化版本 \(I_{ij}^{H}/c_{ij}\)（必须声明与 W 单位、作用尺度有关的 \(c_{ij}\)），或将总曲率能量分配到自曲率和混合曲率的比例；这种“归一化”改变指标解释，不能悄悄把完整 Hessian Frobenius 范数当作可变旋转 Loss。

### 5.3 梯度共同变化（相关但非 Hessian 指标）

\[
G_{ij}=E_Q[g_i(W)g_j(W)],\quad
C^g_{ij}=\operatorname{Cov}_Q(g_i(W),g_j(W)).
\]

这些反映局部**边际效应**的方向/共同波动，可能用于梯度 active subspaces / Fisher-like 函数算子；但若 \(f=w_1w_2\)、独立零均值 W，\(H_{12}=1\) 而 \(\operatorname{Cov}(g_1,g_2)=0\)。即使 f 可加，相关输入也能让梯度协方差不为零。

输入本身的 Corr、dCor、MI、HSIC 是**统计依赖**，不能直接视为函数交互；对函数的非线性/交互解释不能由输入相关代替。

---

## 6. 块结构：Z-self、Z–Z、Z–X、X–X、整体

仅当研究明确需要分组时，将 \(W=(Z,X)\)，Hessian 分块：

\[
H=\begin{bmatrix}H_{ZZ}&H_{ZX}\\H_{XZ}&H_{XX}\end{bmatrix}.
\]

| 关注对象 | 指标示例 | 解释/注意 |
|---|---|---|
| Z-self | \(\sum_iE[H_{z_i z_i}^2]\) | 每个 Z 维度自身的非线性（可能由 X/Z 调制） |
| Z–Z offdiag | \(\sum_{i<j}E[H_{z_i z_j}^2]\) | 各 Z 维度相互调节 |
| Z–X total | \(E\|H_{ZX}\|_F^2\) | Z 对 X 的全部交叉二阶敏感性；**对任意正交 Z 旋转不变** |
| Z–X row support | 每个 Z_i 的 \(H_{z_i x_j}\) 中有多少显著作用 | 研究“一个 Z 是否影响尽可能少的 X”；需要坐标相关稀疏度 |
| Z–X row sparsity | 每行 L1、L0 近似、Top-k tail、entropy/Gini/Hoyer 等 | 多个小但非零的 X 作用是否受到足够惩罚；必须事先确定尺度与阈值 |
| X–X | \(E[H_{x_i x_j}^2]\)、diag/offdiag 分别计算 | X 自身曲率与刺激变量交互；X 无旋转时可作为背景/诊断 |
| Gram Z direction | \(E[H_{ZX}H_{ZX}^\top]\) | Z 行效应指纹的相似性（输出 d_Z×d_Z）；谱与 trace 对 Z 正交旋转不变 |
| Gram X direction | \(E[H_{ZX}^\top H_{ZX}]\) | X 作用方向的 Gram；仅旋转 Z 时完全不变 |
| Hessian fingerprints | 对每个 Z_i 汇总其跨 X、w、条件的效应向量及差分/TV/曲率 | 后续作用稀疏度、簇分离、指纹稳定性；属于候选结构指标，并非一个固定定义 |
| Conditional block | \(E_Z\operatorname{Var}_X(H_{ij}\mid Z)\) | 某导数元素随 X 调制程度，与 Z 内部效应不同 |

**对角/非对角必须分开**：历史中明确提出 ZZ 和 XX 块都需分开报告 diag 与 offdiag，不能仅输出整个块的总体相似度或总体 norm。

**特别注意：**统计独立的 Z 与 X、Z 内部统计独立、以及函数 \(f(Z,X)\) 中没有交互是三个不同命题。仅用 Z 的 ICA/HSIC/JADE 不会自动发现功能意义上的真参数轴。

---

## 7. 简单性（Loss）候选：和 Hessian 汇总指标分开

拿到数值矩阵 M、局部 H 场或函数指纹之后，还需要定义一个**越小越简单**的标量目标 \(L\)。

| 类别 | 讨论过的 Loss / 函数 | 主要偏好/风险 |
|---|---|---|
| Offdiag L2 energy | \(\sum_{i<j}M_{ij}^2\)；或局部 \(E\sum_{i<j}H_{ij}^2\) | 偏好可加/解耦；未必偏好真参数轴 |
| Offdiag L1 | \(\sum_{i<j}|M_{ij}|\) | 促进稀疏；受尺度影响 |
| 全元素 L1/L2 | \(\sum_{ij}|M_{ij}|\)、\(\|M\|_F^2\) | L1 坐标相关；完整对称 M 的 L2 在正交旋转下可能平坦 |
| Diag energy | \(\sum_iE[H_{ii}^2]\) | 偏好每维自身线性；可能与最少 Z–Z 交互冲突 |
| Effective support | L0/\(\ell_p,p<1\) 平滑近似、Top-k tail、软阈值计数 | 区分如 [2,2,.01,.001] 与 [2,2,.001,.001]；对微小值、缩放敏感 |
| Concentration/entropy | 归一化 row entropy、Gini、Hoyer、参与比、能量集中度 | 关注是否集中在少量 X 或 Z 方向；归一化选择影响结论 |
| Group sparsity | 行/列 group lasso、Top-k group tails | 一个 Z 对尽可能少 X、或 X 被尽量少 Z 调节 |
| Matrix rank / nuclear norm | 奇异值 tail、effective rank、nuclear norm | 偏好低维共同作用，可能不区分具体坐标轴 |
| Joint diagonalization | 多个局部/条件 Gram 或 Hessian 的共同非对角损失 | 一个全局旋转同时简化多张矩阵，而非单点各自对角化 |
| Nonlinearity + interaction | \(L=\lambda_{\rm self}L_{\rm self}+\lambda_{\rm cross}L_{\rm cross}\) | 必须明确相对权重；等权时有时完全平坦 |
| Gradient/Hessian regularity | smoothness、TV、higher derivative norms、monotonicity violations | 对函数局部振荡和变化敏感；可能鼓励过平滑 |
| Function-class loss | 最佳加性残差、条件仿射残差 | 不求 Hessian 的对照；衡量函数值层面简单性 |
| Cross-angle fingerprint stability | 角度扫描、bootstrap、模型/响应通道一致性 | 是筛选/验证指标，不应直接混入待优化 Loss |
| HSIC / ICA / JADE | 统计独立性、四阶累积量 | 是候选对照；变量独立性不等于功能作用独立 |

尤其讨论过的“稀疏度不能忽略很小的剩余效应”：例如 [2,0.011,0.001] 与 [2,0.001,0.001]，以及 [2,2,0.01,0.001] 与 [2,2,0.001,0.001]。**标准 L2** 的平方会把很小的项压得更小，未必符合期望的相对敏感性；应研究 L1、非凸稀疏近似、阈值风险、log/entropy 和 Top-k tail，但不能为了匹配具体数值对任意挑阈值。

对任何候选 Loss 需检查：
- 相同输入单位/scale 与 Z 标准化；否则参数缩放本身能改变惩罚；
- 真实轴在允许旋转族（2D/3D/4D、\(SO(d)\)/\(O(d)\)）是否为**极小值**且有明显优势；
- 平坦、不唯一、符号翻转与置换的等价性；
- 参数真轴可能本来包含非零交互，最小非加性坐标不必等于真轴。

---

## 8. 旋转公式、等变性与已讨论的反例

令 \(U=RZ\)，R 为正交矩阵，固定相同物理点评价并对 Q 做一致推前：

\[
H_{UU}(u,x)=R H_{ZZ}(R^\top u,x)R^\top,\qquad
H_{UX}(u,x)=R H_{ZX}(R^\top u,x).
\]

因此：
- \(E[H_{UU}]=R(EH_{ZZ})R^\top\)；Gram \(E[H_{UX}H_{UX}^\top]\) 也相应合同变换；
- 局部/整体 Frobenius norm、Trace、Spectrum、Determinant 等对对称全 ZZ Hessian 的正交变换不变；
- 对 \(H_{ZX}\) **整块** Frobenius norm 不变，单行/单元素 sparsity 与列的归属则可能变化；
- **先旋转局部 H 再逐元素取绝对值/RMS/分位数**不等于先聚合再旋转；
- 如果不是正交旋转而是一般可逆线性变换，需使用相应的链式法则合同变换；没有一般的正交谱不变性。

### 8.1 单点对角化不是成功

任意实对称 \(H(w_0)\) 都可正交对角化，得到单点 offdiag 为零。但不同点可能要求不同角度。真正要检验：**同一个 R 能否同时使多个 w/条件处的场变简单**，并偏好理论参数轴，而不是仅仅某个平均矩阵的 eigenvectors。

### 8.2 关键反例 \(f(a,b;x_1,x_2)=ax_1+abx_2\)

比较真轴 \((a,b)\) 与 \(u_1=(a+b)/\sqrt2,\ u_2=(a-b)/\sqrt2\)。固定 x：

\[
f(u_1,u_2;x)=\frac{x_1}{\sqrt2}(u_1+u_2)+\frac{x_2}{2}(u_1^2-u_2^2).
\]

- 原轴：每个参数自身线性；有 a–b 混合交互。
- 45°：对 \(u_1,u_2\) **完全加性**；但二者都出现平方项。
- 原轴 \(H_{ZZ}=\left[\begin{smallmatrix}0&x_2\\x_2&0\end{smallmatrix}\right]\)；45° \(H_{UU}=\operatorname{diag}(x_2,-x_2)\)。
- \(L_{\rm ZZ-offdiag}\) 偏好 **45°**；\(L_{\rm Z-self-diag}\) 偏好 **0°**。
- 如果 \(a,b\overset{\rm iid}{\sim}N(0,1)\) 且 x 为常数，定义 Sobol 方差比，则

\[
S_{12}(\theta)=\frac{x_2^2}{x_1^2+x_2^2}\cos^2(2\theta),\qquad
\sum_i N_i^{\rm nonlinear}(\theta)
=\frac{x_2^2}{x_1^2+x_2^2}\sin^2(2\theta).
\]

两者**等权总和恒定**，不能唯一偏好理论轴。\(N_i^{\rm nonlinear}\) 此处是**主效应函数偏离最佳线性函数的方差比**，不是 Hessian RMS，注意区分。
- 若连 \(x_1,x_2\) 也独立标准正态，整体原轴有 Z–X 纯两变量贡献 1/2 与 Z–Z–X 三变量贡献 1/2；45° 后全部变成 Z–X 纯两变量贡献，Z–Z–X 为零。**纯 Z–Z Sobol** 两者都为 0；**所有 Z–X（分组）交互总贡献**两者都为 1。这说明只分“全部 ZX”不够，必须考虑具体 Z 行、效应阶数、Z-self 与 ZZ。
- 原轴 \(H_{ZX}=\left[\begin{smallmatrix}1&b\\0&a\end{smallmatrix}\right]\)；45° \(H_{UX}=\left[\begin{smallmatrix}1/\sqrt2&u_1\\1/\sqrt2&-u_2\end{smallmatrix}\right]\)。其整块 Frobenius 能量相同，具体元素/行稀疏不同。

**重要解释**：这个反例证明“最小交互”可以把交互转换为自身非线性，**不证明**真轴一定应该更简单；需要另立目标（如真实生成式的 effect alignment、单维条件线性与 X 作用指纹简洁度），且从没有真参数先验的纯 WY 数据出发通常不能保证唯一识别理论语义。

---

## 9. 三种 Y：Logit、Probability、Choice 的 Hessian 不能混淆

对选择生成模型：

\[
\eta(W)=\mathrm{logit},\quad p(W)=\sigma(\eta(W)),\quad
C\mid W\sim\mathrm{Bernoulli}(p(W)).
\]

链式法则：

\[
\nabla p=p(1-p)\nabla\eta,
\quad
H_p=p(1-p)H_\eta+
p(1-p)(1-2p)\nabla\eta\nabla\eta^\top.
\]

| Y 通道 | 解析 Hessian 是什么 | 指标意义和误区 |
|---|---|---|
| Logit | \(H_\eta\) | log-odds / 生成效用尺度上的曲率与交互 |
| Probability | \(H_p\) | sigmoid 后选择概率尺度曲率；即使 \(\eta\) 加性，也可引入概率交互 |
| Binary choice | **单次 0/1 没有连续意义下的 Hessian** | 要以 \(f(W)=E[C\mid W]=p(W)\) 为目标；先进行概率回归，或做二元 GLM/GAM |

- Sigmoid 饱和时 \(p(1-p)\to0\)，概率 Hessian 可能变小，不能据此断言 logit 无交互。
- Bernoulli 负对数似然的 **expected Hessian / Fisher**（以输入 w 参数化并对 C 条件期望，正则条件下）为 \(p(1-p)\nabla\eta\nabla\eta^\top\)，这是**NLL 曲率**，不是 \(H_p\) 或 \(H_\eta\)。它在 \(p\approx0.5\) 处强调可区分性。
- Bernoulli choice 的总方差为 \(\operatorname{Var}(C)=\operatorname{Var}(p(W))+E[p(1-p)]\)，所以用 choice 原始方差做函数型 Sobol 分母会把不可预测 Bernoulli 噪声混进去；要研究的函数型目标通常仍是 p。
- Choice 用平方风险比较受限与完整概率函数类时，期望平方风险中的 \(E[p(1-p)]\) 对不同候选函数相同，理想总体风险差可反映函数近似损失；有限数据要分组交叉验证。
- **不能要求三个 Y 的 Hessian 数值/交互完全一致**：它们是不同定义的数学目标。可以分别比较旋转角度**模式一致性**，并明确输出尺度所引入的变化。

项目四个 toy：C1/C2/C3 的 logit 对真 T 线性，故 \(H_{TT}(\eta)=0\)；C2/C3 对 X 可非线性、\(H_{TX}\) 可变化；CR3 CPT-like \((\alpha,\lambda,\gamma,\tau)\) 的效用更复杂。CR3 共享 500 个 X，零金额与分数幂可能导致某些 X 方向导数未定义。详见 [真实模型](../TOY_GENERATIVE_MODELS.md)。

---

## 10. Q：积分对象、测度、求积方式必须另列

| 轴 | 讨论过的取法 | 需要避免的混淆 |
|---|---|---|
| 积分对象 | 不积分 / 仅 X / 仅 Z/T / 联合 W / 条件积分 | 仅 X 积分后仍可能是 Z 的函数 |
| Z/T 测度 | 模拟被试 T 经验分布、真实 T 先验、NP 抽取 Z 经验分布 | T 先验不等于 NP Z 分布 |
| X 测度 | 固定试次 X bank、随机生成 X、条件子集/分层 X | CR3 固定500试次并不等于连续 X 先验积分 |
| 联合方式 | 原始配对数据、经验 Z × 固定 X、假定独立乘积测度 | 有无重新组合观测点关系到可识别性 |
| 权重 | subject 等权、trial 等权、importance、weighted Stein、条件权重 | 权重改变总体目标 |
| 算法 | 有限和、Monte Carlo、Quasi-MC/Sobol **数列**、quadrature | Sobol QMC **不是** Sobol 方差敏感性指数 |
| 有效域 | 零金额、分数幂、边界/不可微点 mask | 只在可微区域条件平均不能称为全域真平均 |
| 统计不确定性 | bootstrap（按被试重抽）、积分收敛、有效样本量 | 5000×500 trials 不等于 250 万个独立被试 |

CR3 的两种典型定义：

\[
M_{\rm sample}=\frac1N\sum_{s=1}^N\frac1{500}\sum_{t=1}^{500}H(T_s,X_t);
\qquad
M_{\rm prior}=\int \left(\frac1{500}\sum_tH(T,X_t)\right)\pi(T)dT.
\]

前者是**经验被试求和**，后者是**真实参数先验积分**；二者的 X 都仍是固定 bank 的离散平均。

---

## 11. 只有 WY 数据时：到底估计 H(W) 还是直接估计指标？

| 估计目标 | 具体目标 | 典型估计路线 |
|---|---|---|
| 局部 Hessian | \(H(w_0)\) | 局部多项式、可微全局函数拟合、核导数 |
| 整个局部字段 | \(\hat H(w)=\nabla^2\hat f(w)\) | 多项式/样条/KRR/GP/代理 NN |
| 平均 Hessian | \(E_QH(W)\) | 对估计字段求均值；在密度/边界条件下直接二阶 Stein |
| RMS/Variance/Q99 | \(E H_{ij}^2\)、\(\operatorname{Var}H_{ij}\)、quantile | 一般需要更丰富的导数信息或专门非线性泛函估计 |
| Fisher/NLL curvature | 负对数似然的期望 Hessian 或 score 外积 | 与响应函数 Hessian **不同目标** |

### 11.1 仓库对应的数值估计器 E01–E14

| 编号 | 方法 | 是否能恢复局部 H | 主要局限 |
|---|---|---|---|
| E01 | Global polynomial sieve | 是 | 阶数、ridge、边界振荡；已有实现 |
| E02 | Tensor-product/low-rank separable bases | 是 | 结构截断偏差；已有实现 |
| E03 | Local polynomial/quadratic | 是 | 高维邻域稀疏；候选 |
| E04 | Penalized B-splines/P-splines | 是 | 光滑惩罚与结点选择；已有部分 T-only |
| E05 | GAM/ANOVA interaction splines | 受函数类限制 | 拟合阶段可能提前强制简单性；候选 |
| E06 | Gaussian RBF KRR/Nyström | 是 | 核长度尺度决定导数；已有部分实现 |
| E07 | Gaussian Processes | 是 | 需足够光滑核，计算量与不确定性；候选 |
| E08 | Orthogonal polynomial/spectral series | 是 | 截断与正交测度；Chebyshev 部分实现 |
| E09 | Finite difference/Savitzky–Golay | 需要近邻密网格或可查询 f | 固定稀疏 WY 不能凭空获得偏移响应 |
| E10 | Second-order Stein / integration by parts | 通常只能直接平均 H | 密度 score、边界项/带权目标；已有 T-only weighted 版本 |
| E11 | Gaussian smoothing / randomized zeroth-order | 平滑函数的 H | 需 off-grid 函数查询，改变目标 |
| E12 | Independent smooth NN surrogate + autodiff | 是（对代理） | Y 预测准确不保证真实二阶导准确 |
| E13 | Binary logistic GLM/smooth sieve | 是（对概率函数） | 适合 choice，但取决于模型结构 |
| E14 | HVP/matrix-free | 否；是导数计算技术 | 不是独立 WY 统计估计器 |

**二阶 Stein 典型恒等式**：对标准正态 \(W\sim N(0,I)\) 且满足可积边界条件，

\[
E[\nabla^2 f(W)]
=
E[f(W)(WW^\top-I)].
\]

从 Y 直接估计平均 Hessian 不必先求局部字段。但这个等式**不能直接给出** \(E[H_{ij}^2]\) 或 Q99。对于一般密度 q 可用二阶 score 恒等式，截断先验的边界项不能忽略；仓库的加权 Stein 估计的是**加权**平均，并非自动等于无权 prior expectation。

**有限 WY 的识别限制**：任意两个光滑函数可以在观察到的每个 W 点给出完全相同 Y，但在中间位置有不同导数与交互；所以要声明光滑性、函数类、数据支持域和外推假设。对于 CR3 共用固定 500 X，即使被试人数无限增大也不能凭空识别任意连续 X 插值的 Hessian。预测 Y 的误差小，**不等于 Hessian 的误差小**。

---

## 12. 与 Hessian 并行的函数结构指标：Sobol / Functional ANOVA / 加性残差

这一节只为说明 Hessian 的**定位**，以下并非 Hessian 指标。

在独立输入、\(f\in L^2\) 下，functional ANOVA 分解为 \(f=f_0+\sum_i f_i+\sum_{i<j}f_{ij}+\cdots\)，Sobol 贡献为各成分方差除以 \(V=\operatorname{Var}(f(W))\)。

| 研究对象 | Hessian 路线 | 非 Hessian 互补定义 |
|---|---|---|
| 边际单维非线性 | \(E[H_{ii}^2]\)（**包括条件调制**） | \(\eta_i^2-r_i^2\) 或主效应减最佳线性拟合的方差比例 |
| 条件单维非线性 | \(E[H_{ii}^2]\)、主曲率与调制曲率分解 | \(\inf_{a,b}E[(f-a(W_{-i})-b(W_{-i})W_i)^2]/V\) |
| 纯二阶交互 | 混合 Hessian **不区分是否纯二阶** | \(S_{ij}\) |
| 两维共同参与所有阶交互 | \(E[H_{ij}^2]\)（Crossed-DGSM） | \(S_{ij}^{\supseteq}=\sum_{A\supseteq\{i,j\}}S_A\) |
| 一个变量参与所有交互 | 汇总与 i 有关的混合导数 | \(S_{T_i}-S_i\) |
| 整体函数非加性 | \(\sum_{i<j}E[H_{ij}^2]\) | \(1-\sum_iS_i\)，或最佳加性逼近平方残差/V |
| 效应共同变化 | \(\operatorname{Cov}(g_i,g_j)\)、Gradient Gram | 不是 Sobol 纯交互，也不是 W 输入相关 |

**数学联系**：在独立输入并满足适当 Poincaré 条件下，Crossed-DGSM \(\nu_{ij}=E[(\partial_{ij}f)^2]\) 可以给 Sobol 的 superset interaction importance 一个含分布常数的上界（Roustant et al., 2014）。它们不能因存在这个不等式就视为同一个指标。

**模型/估计方法不是指标本身**：GAM/SS-ANOVA/COSSO、Sparse PCE、MARS、EBM/FAST、GP surrogate 是函数估计/结构建模路线；Friedman H、ALE、Shapley-Taylor、iRF、RuleFit 是其他函数交互描述/发现工具。Chatterjee ξ、dCor、HSIC、MIC、RESET 检验、GAM EDF/p-value 等测的是依赖、诊断或形状复杂度，不应混成一个具有统一方差解释的非线性强度。有限样本与相关 W 下，经典 Sobol 分解不能直接照搬。

**绝对重要性不等于旋转最优**：例如 \(f=w_1w_2\) 旋转 45° 后变为 \((u_1^2-u_2^2)/2\)；纯交互消失，而自曲率出现。Sobol 交互偏好 45°、自非线性偏好原轴，等权总体可保持不变。若 Z 与 X 分组，只计算全部 group Z–X interaction 也可能对 Z 内部旋转保持不变。

---

## 13. 现有实验和验证应如何理解

历史上实施的分析包括：
- CR3 解析式直接 Hessian 与用真实 (T,X,Y) 从数值拟合的对照；
- 三通道 Y（logit、probability、choice）、模拟被试经验平均与真先验积分；
- 多种基函数/核/加权 Stein 的 WY Hessian 估计；
- NP 的 Z 经监督 \(A_*Z\approx T\) 对齐后，在同一 Z/X 接口比较解析式 Y、NN decoder Y、解析 Oracle Hessian；
- 对 ZZ、ZX、XX 和 full 矩阵比较 raw error 与**pattern similarity**，并把 ZZ/XX 的 diag 和 offdiag 分开；
- 两两平面角、3D/4D/联合旋转、多个 Hessian/稀疏性指标的旋转损失曲线；
- 区分 analytic 成功、formula-Y 数值估计失败、NN-Y 数值估计失败三层 gap。

**不能根据本历史讨论直接推断**“某指标在四模型全部成功”、“ZZ/ZX 某一类全部成功”或“已找到统一可展示的无监督旋转 Loss”；这些需要对应的测试结果和独立证据。尤其**真解析指标优良但公式生成 Y 的数值估计失败**，说明问题可能先在 Hessian 的统计估计器，不应直接归咎于 NN decoder。

### 每个指标/旋转实验建议保存的信息

| 分类 | 最少字段 |
|---|---|
| 输入 | model, output_channel, data_source, W block, coordinate_system, n_subjects, Xset hash |
| Hessian | oracle/estimator ID, derivative order, local grid, undefined mask |
| 统计 Q | integration axes, sample vs prior, weighting, domain, numerical quadrature |
| 数学指标 | M01–M20/D/P, matrix_first vs local_first, output shape |
| 简单性 k | L1/L2/offdiag/diag/support/group-sparsity/TV，明确最小化还是最大化 |
| 旋转 | R parameterization, scale convention, sign/permutation, angle sampling, seeds |
| 准确性 | RMSE/MAE、relative norm、pattern correlation、rank/top-k/sign/diag/offdiag |
| 结构成功 | analytic true-axis margin, formula-Y vs analytic angle-curve pattern, NN-Y vs analytic pattern |
| 真参数核验 | signed-permutation matched correlation、off-diagonal cross-correlations、out-of-sample T prediction（只评价不训练 Loss） |

针对已有研究提出的“好指标”标准：**解析和 NN 生成 Y 的角度模式一致**，并且简单性 Loss 在正确轴附近 0°/90°（考虑符号与置换）达到**极小值**，而不是仅有高相关或最大值。要测试 3/4 维联合旋转，单独通过六张两两角度曲线不等于全局旋转成功。

---

## 14. 讨论中的重要数学结论（供以后快速回顾）

1. \(H(W)\) 一般是 **d×d 的函数矩阵**；取点、均值、RMS、Gram、谱、投影等输出彼此不同。
2. D（消除变量）和 P（改变形状）是两个独立维度；Q（测度/求积）与 k（简单性）也独立。
3. Signed Mean 会抵消；Mean Absolute、RMS、Variance、Q99、Gram 会保留不同信息。
4. 逐元素 \(H\odot H\) **不等于**矩阵 \(H^2\)；\(\operatorname{RMS}(H)\) 不等于 \(|EH|\)。
5. 一张固定 Hessian 的 eigendirections 通常只是局部或平均的几何方向，不保证理论参数轴。
6. Trace、全矩阵 Frobenius、谱等正交不变量单独不能辨识正交旋转。
7. 单维 \(H_{ii}\) 可能被其他变量调制；总体自身曲率要与独立主效应分开。
8. \(H_{ij}\) 衡量函数交互，不等于输入 Corr/MI/HSIC，也不等于梯度 covariance。
9. 交互最小化可以把乘法交互变成单变量平方项；“最少交互”不自动等于“总体表达最简单”或“真参数轴”。
10. Sobol 衡量**函数值方差贡献**，Hessian 衡量**局部导数强度**；二者在适当条件下有理论界但不等价。
11. Z–X 的**整体能量**对 Z 正交旋转不变；若要找 Z 轴，应研究每个 Z 的 X 效应分配/稀疏度/稳定性，而不是只看总量。
12. Choice 的 Hessian 指条件均值 p 的 Hessian；logit/p 的 Hessian 由 sigmoid 链式法则联系，不能强求相同。
13. WY 预测准确不保证导数准确；直接估计 \(E[H]\) 的 Stein 公式不自动提供 RMS/Q99。
14. 真先验、模拟被试与 NP Z 经验分布是不同 Q；CR3 的固定 X bank 与连续设计空间也不同。
15. Oracle 恢复、formula-Y 估计准确、decoder-Y 估计准确、真轴监督评价，是需要逐级跨越的不同技术鸿沟。

---

## 15. 参考文献与现有 README 交叉索引

下面列出讨论中用到的**方法来源**；论文证明的是相应一般数学/统计方法，不表示在本项目中已验证能恢复四个模型的理论坐标。

- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. [Publisher](https://www.routledge.com/Local-Polynomial-Modelling-and-Its-Applications-Monographs-on-Statistics/Fan/p/book/9780203748725).
- Eilers, P. H. C., & Marx, B. D. (1996). *Flexible Smoothing with B-splines and Penalties*. [DOI](https://doi.org/10.1214/ss/1038425655).
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). [Publisher](https://www.routledge.com/Generalized-Additive-Models-An-lntroduction-with-R-Second-Edition/Wood/p/book/9781498728331).
- Li, K.-C. (1992). *On Principal Hessian Directions for Data Visualization and Dimension Reduction: Another Application of Stein's Lemma*. [DOI](https://doi.org/10.1080/01621459.1992.10475264). 平均 Hessian 与降维。
- Sobol', I. M. (2001). *Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates*. [DOI](https://doi.org/10.1016/S0378-4754(00)00270-6).
- Sobol, I. M., & Kucherenko, S. (2009). *Derivative based global sensitivity measures and their link with global sensitivity indices*. [DOI](https://doi.org/10.1016/j.matcom.2009.01.023).
- Roustant, O., Fruth, J., Iooss, B., & Kuhnt, S. (2014). *Crossed-derivative based sensitivity measures for interaction screening*. [DOI](https://doi.org/10.1016/j.matcom.2014.05.005).
- Friedman, J. H., & Popescu, B. E. (2008). *Predictive learning via rule ensembles*. [DOI](https://doi.org/10.1214/07-AOAS148). Interaction H-statistic.
- Apley, D. W., & Zhu, J. (2020). *Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models*. [DOI](https://doi.org/10.1111/rssb.12377). ALE effects.
- Hooker, G. (2007). *Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables*. [DOI](https://doi.org/10.1198/106186007X237892).
- Pearlmutter, B. A. (1994). *Fast Exact Multiplication by the Hessian*. [DOI](https://doi.org/10.1162/neco.1994.6.1.147).
- Baydin et al. (2018). *Automatic Differentiation in Machine Learning: a Survey*. [JMLR](https://www.jmlr.org/papers/v18/17-468.html).
- 关于高阶 WY 函数估计、Stein score、核微分与方法状态，请见 [E01–E14 完整目录及其文献](../HESSIAN_FROM_ZXY_METHODS.md)。
- 关于 CR3 零金额导数、不同积分测度、weighted Stein 的严格定义，请见 [积分规范](../HESSIAN_INTEGRATION_AND_AVERAGING.md)。
- 关于 M01–M20 更严格的输出形状与状态，请见 [数值矩阵指标](../HESSIAN_TO_MATRIX_METRICS.md)。

> **归档原则**：这里的“指标”既包括标准文献中的数学对象，也包括本项目讨论的候选简单性/稀疏性 Loss。标准数学恒等式、条件性理论结论、研究假设和未经验证的候选方法要始终分开标注；新增候选指标应先定义输入、输出、尺度、旋转不变性与反例，再考虑实现。