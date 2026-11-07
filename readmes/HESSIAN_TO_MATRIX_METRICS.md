# 独立实验维度 A：Hessian 函数 → 数值矩阵的不同指标

> 与 [研究总框架](README.md)、[数值 ZXY → Hessian 估计器](HESSIAN_FROM_ZXY_METHODS.md)、[积分/求平均维度 Q](HESSIAN_INTEGRATION_AND_AVERAGING.md)并列。
>
> **目的**：在已知函数型或离散局部 Hessian 场之后，定义从该场到固定数值矩阵的多种统计指标。**矩阵指标不同于下游标量简单性指标**。
>
> 本文用指标编号 M01–M20。变量 a 表示矩阵统计指标，不代表旋转矩阵 A 或 analytic 数据源 A。

## 1. 核心定义

对于同一函数 f(U,X)，Hessian 是随坐标变化的矩阵值函数：

\[
H(U,X)=\nabla_{(U,X)}^2 f(U,X)
=\begin{bmatrix}H_{UU}(U,X)&H_{UX}(U,X)\\ H_{XU}(U,X)&H_{XX}(U,X)\end{bmatrix}.
\]

Q 是**单独指定的积分测度与数值规则**：例如经验 Z × CR3 固定 500 试次 X 等权。对这个 Q，记

\[
\langle g\rangle_Q=\mathbb E_Q[g(U,X)].
\]

然后由**矩阵指标 a**指定从局部 Hessian 得到的数值矩阵：

\[
\boxed{M_a(Q)=\mathcal A_a(\{H(U_s,X_t)\},Q).}
\]

最后标量简单性指标 k 再给 M_a 评分：

\[
L_{a,k}=S_k(M_a).
\]

**所以三个实验维度独立**：
1. Q：在什么分布、哪些点、什么权重下进行求积；
2. a：用哪种矩阵函数指标转换局部 Hessian；
3. k：对已得到的数值矩阵，如何定义“简单”的标量损失。

例如同一 Q 下，M01 Mean、M03 RMS 与 M09 Q99 产生不同的矩阵；同一 M03 RMS 下，改变 Q 为真实 T 先验或模拟 Z 经验分布，仍会产生不同的矩阵。

## 2. M01–M20 矩阵指标库

矩阵逐元素绝对值记作 \(|H|_{\rm el}\)，逐元素积记 \(\odot\)。下面的统计对每个 Hessian 元素按同一 Q 的点位和权重计算。

### A. 逐元素强度、变化、稳健性

| ID | 名称 | 生成数值矩阵的公式 | 作用 |
|---|---|---|---|
| M01 | Signed Mean | \(\langle H\rangle_Q\) | 有符号平均，正负易抵消 |
| M02 | Mean Absolute | \(\langle |H|_{\rm el}\rangle_Q\) | 平均绝对曲率 |
| M03 | RMS | \(\sqrt{\langle H\odot H\rangle_Q}\) | 曲率均方根，保留能量 |
| M04 | Mean Square | \(\langle H\odot H\rangle_Q\) | 曲率逐元素二阶矩 |
| M05 | Variance | \(\langle(H-\bar H)\odot(H-\bar H)\rangle_Q\) | 曲率在点之间的变化 |
| M06 | Standard Deviation | \(\sqrt{M05}\) | 波动的原单位大小 |
| M07 | Generalized Lp | \(\left(\langle|H|_{\rm el}^p\rangle_Q\right)^{1/p}\) | 通过 p 调节强曲率权重 |
| M08 | Median Absolute | \(\operatorname{median}_Q|H_{ij}|\) 逐元素 | 典型作用、较抗极端值 |
| M09 | Absolute Quantile | \(Q_{\alpha,Q}(|H_{ij}|)\) 如 Q90、Q99 | 强局部曲率尾部 |
| M10 | Maximum Absolute | \(\max_{\Omega_Q}|H_{ij}|\) | 极端局部值，敏感 |
| M11 | Threshold Exceedance | \(\Pr_Q(|H_{ij}|>\tau)\) | 大曲率出现概率 |
| M12 | Sign Consistency | \(\left|\langle\operatorname{sign} H_{ij}\rangle_Q\right|\) | 符号是否一致，不衡量幅度 |
| M13 | Positive / Negative Parts | \(\langle\max(H,0)\rangle_Q\) 与 \(\langle\max(-H,0)\rangle_Q\) | 输出两块矩阵，防止符号抵消 |
| M14 | Trimmed / Winsorized | 各元素按 Q 截尾/缩尾后再平均 | 稳健平均，需给截尾比例 |

以上是**逐元素统计量**；例如 M04 是逐元素平方平均，**不是**矩阵乘积的平均。逐元素恒等式

\[
M03\odot M03=M01\odot M01+M05
\]

表明 RMS 同时反映平均强度与波动。

### B. 整体矩阵算子（不能与逐元素指标混淆）

| ID | 名称 | 数学公式 | 输入/输出意义 |
|---|---|---|---|
| M15 | Row Gram / Hessian second moment | \(\langle HH^\top\rangle_Q\) | 对 UU 或 UX 输入，输出行方向 Gram；维度由行数决定 |
| M16 | Column Gram | \(\langle H^\top H\rangle_Q\) | 对矩形 UX/ZX，输出 X 方向列 Gram |
| M17 | Spectral matrix absolute | \(\left\langle (H^2)^{1/2}\right\rangle_Q\) | H **必须对称方阵**，与逐元素绝对值不同 |
| M18 | Covariance of Hessian elements | \(\operatorname{Cov}_Q(\operatorname{vec}H)\) | 输出更高维协方差矩阵，不再是原始 ZZ 大小 |

对 \(H_{UX}\in\mathbb R^{d_U\times d_X}\)，M15 是 \(d_U\times d_U\)，M16 是 \(d_X\times d_X\)。对 4×4 对称 ZZ，M18 若使用 16 个向量化元素为 16×16，若仅用 10 个独立元素则为 10×10；必须记录选择。

### C. 条件变化与高阶导数

| ID | 指标族 | 生成方式 | 数据要求 |
|---|---|---|---|
| M19 | Conditional / ANOVA Variance | \(\mathbb E_Z[\operatorname{Var}_{X|Z}(H_{ij}|Z)]\)、\(\mathbb E_X[\operatorname{Var}_{Z|X}(H_{ij}|X)]\) 或 Z/X/ZX ANOVA 方差分解 | 需要 Q 的条件、联合分布和配对规则 |
| M20 | Higher-order Hessian Variation | \(\mathbb E_Q[|\partial_{u_k}H_{ij}|]\)、\(\mathbb E_Q[|\partial_{x_l}H_{ij}|]\)、它们的 RMS 等 | 必须提供三/四阶局部导数字段 |

M19 的 ANOVA 分解示例：若 Z 与 X 在 Q 下独立，且对应 Hessian 元素平方可积，则

\[
H_{ij}(Z,X)=\mu_{ij}+a_{ij}(Z)+b_{ij}(X)+r_{ij}(Z,X).
\]

可以形成 \(\operatorname{Var}(a_{ij})\)、\(\operatorname{Var}(b_{ij})\)、\(\mathbb E[r_{ij}^2]\) 三类固定数值矩阵。其中 \(\mathbb E_Z[\operatorname{Var}_X(H_{ij}|Z)]\) 一般包含 X 主效应的方差加残余交互的方差，而不是单纯 X 主效应方差。

M20 是对 Hessian **函数的变化**作统计，不能只从一个已经聚合好的 M01–M18 矩阵反推出三阶或四阶导数。必须向上游 Hessian 估计模块申请这些高阶导数字段。

## 3. 具体对照：同一个解析 Hessian，四种不同矩阵

令

\[
f(z_1,z_2,x)=z_1^3+z_1z_2x,\quad
H_{ZZ}(z_1,z_2,x)=
\begin{pmatrix}6z_1 & x\\x&0\end{pmatrix}.
\]

Q 取独立的标准正态 Z1 与 X。于是：

\[
M01=\begin{pmatrix}0&0\\0&0\end{pmatrix},\quad
M02=\sqrt{2/\pi}\begin{pmatrix}6&1\\1&0\end{pmatrix},
\]

\[
M03=\begin{pmatrix}6&1\\1&0\end{pmatrix},\quad
M05=\begin{pmatrix}36&1\\1&0\end{pmatrix}.
\]

全部源于**同一个真实解析 Hessian**。所以聚合转换本身是影响轴恢复结果的独立研究问题，不是 Hessian 提取器精度的一个子指标。

## 4. Q 与 a 的独立比较

| 固定积分方案 Q | 改变矩阵指标 a | 输出 |
|---|---|---|
| CR3 的全部 S5 Z × 固定 500 个 X，等权 | M01 | 平均 ZZ |
| 同一个 Q | M02 | Mean Absolute ZZ |
| 同一个 Q | M03 | RMS ZZ |
| 同一个 Q | M05 | Variance ZZ |
| 同一个 Q | M09（0.99） | Q99 ZZ |
| 同一个 Q | M15 | Gram ZZ |

反过来**固定 a=M03 RMS**，改变 Q 为 T 真实先验、Z 经验分布或不同的 X 条件，可以单独检查积分测度带来的影响。这是与简单性 k、旋转方式无关的实验对照。

只有对 X 做一次积分，一般留下依赖 Z 的矩阵函数 \(M_a(Z)\)；这还不是一个固定矩阵。要么固定 Z=z0，要么再对 Z 进行第二阶段运算。

## 5. 旋转的运算顺序：逐元素聚合与坐标变化不可交换

对正交旋转 U=RZ，局部 ZZ Hessian 满足

\[
H_{UU}(U,X)=RH_{ZZ}(Z,X)R^\top.
\]

对于 Signed Mean M01，在变换后的同一评价测度上

\[
M01(H_{UU})=R\,M01(H_{ZZ})R^\top.
\]

但对于逐元素绝对值、RMS、方差、分位数、硬 threshold 等，通常

\[
M_a(RHR^\top)\ne R\,M_a(H)R^\top.
\]

**正确做法：先旋转每一个局部 Hessian，再做 M02/M03/M09 等转换**，不能先求原空间的 RMS 再旋转 RMS 矩阵。对于对称算子 Gram/M15、谱矩阵绝对值/M17，正交旋转下存在相应的矩阵等变性质，但它们与元素 RMS 的意义不同。

## 6. 矩阵指标 a 与标量简单性 k 的区别

第一条路径是：

\[
L_{\rm matrix-first}=S_k(M_a(H;Q)).
\]

例如 a=M03 给出 RMS ZZ 数值矩阵，k=offdiagonal energy 再给出标量损失。

第二条同样合法但**不同**的路径是：

\[
L_{\rm local-first}=\mathbb E_Q[S_k(H(U,X))].
\]

它先对局部 Hessian 计算标量简单性再积分，**没有经过一个固定的数值矩阵**。一般情况下

\[
S_k(\mathbb E_Q H)\ne \mathbb E_Q[S_k(H)].
\]

两条路径要用单独的 pipeline_mode 记录；不能将前者的数值矩阵指标称为后者的局部 loss。

## 7. 建议规范与现有代码范围

每次评估至少保存：matrix_metric_id、hessian_block、derivative_order、integration_id、coordinate_system、matrix_shape、weights、fraction_defined、rotation_order、pipeline_mode、simplicity_metric_id，以及最终 loss。

- **当前 math/06–07** 已对部分 mean、RMS、variance、分位数及高阶导数字段进行实验；
- **M01–M20 是研究指标分类和将来的独立接口约定，不代表仓库已经完整实现所有 20 个矩阵指标**；
- 若某个估计器（如直接 Stein）只输出加权平均 Hessian，**只能选择与该输出兼容的 a**，不能声称已经获得局部 Hessian 进而计算 Q99/RMS；
- 若矩阵维度改变（Gram / covariance），下游标量简单性 k 的意义和矩阵形状约束也要同步检查。

**总框架：生成模型 / NP / Y 来源 → Hessian 估计器 → 积分方案 Q → 数值矩阵指标 a → 标量简单性 k → 旋转损失。**