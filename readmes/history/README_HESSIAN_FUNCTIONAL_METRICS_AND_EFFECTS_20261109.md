# Hessian 函数的数学变换、曲率、交互与主效应分解：讨论归档

> **归档日期：2026-11-09**。本文独立整理本轮对话的概念与数学结论：从解析式得到 \(H(W)\)，逐步讨论固定矩阵、数值标量、不同尺寸的矩阵；统计平均与操作顺序；常见数学用途；单个维度的非线性、变量间函数交互、统计依赖；最后重点说明为什么 Hessian 对角线的曲率未必等于一个可独立分离的主效应，以及如何得到三个有意义的数字。**这里只假设真实解析函数已知**，不混入仅有 \((W,Y)\) 样本时的函数拟合/导数误差；后者留待下一阶段。
>
> **关联文档**：[完整 Hessian 研究历史](README.md) · [两个正交数学操作](../HESSIAN_FUNCTIONAL_OPERATIONS.md) · [M01–M20 固定数值矩阵指标](../HESSIAN_TO_MATRIX_METRICS.md) · [积分方案 Q](../HESSIAN_INTEGRATION_AND_AVERAGING.md) · [从 ZXY 估计 Hessian](../HESSIAN_FROM_ZXY_METHODS.md)。
>
> 本文首先只用 \(W\) 表示一个完整的向量，不人为拆分 \(W=(Z,X)\)。这是**数学讨论的历史总结**，不是声称指标已在 C1/C2/C3/CR3 上验证或全部实现。

## 0. 起点：Hessian 是一个矩阵值函数

设 \(f:\Omega\subseteq\mathbb R^n\to\mathbb R\) 二阶可微，

\[
g(W)=\nabla f(W),\qquad
H(W)=\nabla^2 f(W)
=\big[H_{ij}(W)\big]_{i,j=1}^n,\qquad
H_{ij}(W)=\frac{\partial^2 f(W)}{\partial W_i\partial W_j}.
\]

当 \(f\in C^2\)，\(H(W)\) 对称。矩阵**形状**固定为 \(n\times n\)，但每个元素 \(H_{ij}(W)\) 仍可能随 W 变化。解析式给出的是**整个 Hessian 场**，不是天然的一张固定数值矩阵。

研究后续操作需要独立回答：

1. **是否消除 W 的依赖？** 对点取值、取期望/积分、沿路径平均、极限、极值、分位数、矩阵界。
2. **是否改变输出形状？** 保持 \(n\times n\)；变为向量、标量或不同大小的矩阵。
3. **先后顺序如何？** 先逐点变换 Hessian 再汇总，还是先汇总再对结果做谱/范数等变换。
4. **积分采用什么测度 Q？** 评价域、输入联合分布、权重、数值积分精度与剩余自由变量。
5. **数学结果用于什么目的？** 局部形状、凸性/光滑性、统计信息、全局作用、交互、主效应等。输出一个数字**不自动意味着**它就是一个合理的“简单性损失”。

## 1. 两个正交操作：消除变量依赖 D × 改变输出形状 P

| | **形状保持 \(n\times n\)** | **形状改变** |
|---|---|---|
| **仍依赖 W** | \(H(W)\)、\(H(W)^2\)、\(|H(W)|_{\mathrm{entry}}\) | \(\operatorname{tr}H(W)\)、\(\lambda_{\max}(H(W))\)、\(\operatorname{diag}H(W)\)、\(P^\top H(W)P\) |
| **不依赖 W** | \(H(W_0)\)、\(\mathbb E_Q H\)、逐元素 RMS/Q99、\(\mathbb E_Q[H^2]\) | \(\operatorname{tr}H(W_0)\)、\(\mathbb E_Q[\lambda_{\max}H]\)、\(\sup_W\|H(W)\|_2\)、\(\operatorname{Cov}(\operatorname{vec}H)\) |

注意：只对部分输入积分（如固定一个坐标、对其他坐标积分），一般**仍然含有剩余自由变量**，所以不属于完全固定的数值结果。

### 1.1 消除变量依赖的常见方法

- **点评价**：\(M=H(W_0)\)。保留该点全部局部二阶信息；并不代表其它位置。
- **带权有限和、积分或期望**：\(M=\mathbb E_Q[H(W)]\)。保留平均有符号曲率；正负可能抵消。**测度 Q 是独立选择**，没有一个无条件唯一的“总体平均 Hessian”。
- **轨迹平均与指数滑动平均**：对特定轨迹 \(W_t\) 汇总 Hessian，描述该轨迹而非必然描述指定先验。
- **条件平均/部分积分**：\(\mathbb E[H\mid W_i=w_i]\)，通常输出 \(w_i\) 的函数；可继续取点或积分。
- **极限**：\(\lim_{t\to\infty}H(W_t)\)，需要指定路径且极限存在。
- **逐元素中位数、分位数、上确界/下确界**：通常可以得到固定形状的数值矩阵，但保留的是稳健典型值或尾部/最坏情形。
- **Loewner 矩阵界**：寻找常数 \(M\) 使 \(H(W)\preceq M\) 对所有 W 成立；这是矩阵偏序意义的约束，**一般不存在唯一的“最小”上界矩阵**，需增加优化准则。

### 1.2 改变输出形状的常见方法

- **局部标量**：Trace \(=\Delta f\)、Determinant、LogDet（满足定义域条件）、\(\lambda_{\max/\min}\)、矩阵范数、方向二次型 \(v^\top Hv\)、rank、负特征值个数、条件数。
- **局部向量**：\(\operatorname{diag}H(W)\)、特征值向量、Hessian-vector product \(H(W)v\)。
- **不同尺寸矩阵**：固定子块 \(H_{SS}\)、子空间投影 \(P^\top H P\)、Schur 补、低秩或 Kronecker 近似等。
- **形状不变的逐点矩阵操作**：矩阵乘法 \(H^2=HH\)，逐元素平方 \(H\odot H\)，矩阵谱绝对值 \((H^2)^{1/2}\)（实对称），逐元素绝对值 \(|H|_{\mathrm{entry}}\)；这些操作**不能相互混淆**。

### 1.3 运算先后顺序的重要性

线性操作通常可与固定测度下的期望交换（满足相应可积条件）：

\[
\mathbb E_Q[\operatorname{tr}H]=\operatorname{tr}(\mathbb E_Q H),\qquad
\mathbb E_Q[P^\top H P]=P^\top(\mathbb E_Q H)P
\]

其中 P 不依赖 W。非线性操作一般不交换：

\[
\mathbb E_Q[\lambda_{\max}(H)]\ge \lambda_{\max}(\mathbb E_Q H),\quad
\mathbb E_Q[\det H]\ne \det(\mathbb E_Q H)\ \text{(通常)},
\]

\[
\mathbb E_Q[HH^\top]\ne (\mathbb E_Q H)(\mathbb E_Q H)^\top\ \text{(通常)}.
\]

因此 \(\mathbb E_Q[S(H(W))]\) 与 \(S(\mathbb E_Q[H(W)])\) 通常不一样。协方差等**场级联合统计**还可能需要多个矩，未必能写成一次逐点变换加一次期望。

## 2. 固定 \(n\times n\) 数值矩阵的常见指标

在同一测度 Q 下，令 \(\langle h\rangle=\mathbb E_Q[h(W)]\)、\(\odot\) 表示逐元素乘积。

| 操作 | 数值矩阵 \(M\) | 保留信息/局限 |
|---|---|---|
| Signed Mean | \(\mathbb E_Q[H]\) | 保留有符号平均曲率；可能完全抵消 |
| Mean Absolute | \(\mathbb E_Q[|H|_{\mathrm{entry}}]\) | 各元素平均绝对作用强度；丢符号 |
| RMS | \(\sqrt{\mathbb E_Q[H\odot H]}\) | 各元素曲率均方根；强调大值；丢符号 |
| Mean Square | \(\mathbb E_Q[H\odot H]\) | 各元素二阶矩、能量 |
| Elementwise Variance | \(\operatorname{Var}_Q(H_{ij})\) | 元素随 W 的波动；恒定但很强的曲率可为零 |
| Median/Quantile | \(\operatorname{med}_Q(|H_{ij}|)\)、\(Q_{.99,Q}(|H_{ij}|)\) | 典型或尾部作用 |
| Supremum | \(\sup_{W\in\Omega}|H_{ij}(W)|\) | 局部极值，受定义域和奇异点影响 |
| Matrix Gram | \(\mathbb E_Q[HH^\top]\) | 整个矩阵的方向作用；与逐元素平方不一样 |
| Spectral Abs | \(\mathbb E_Q[(H^2)^{1/2}]\) | 对称矩阵的谱强度、方向结构 |
| Representative Matrix | \(\arg\min_M\mathbb E_Q[d(H(W),M)]\) | 指定距离下的最优常数代表；Frobenius 平方损失的解就是 \(\mathbb E_QH\) |

也有按分位数取常数矩阵、条件方差分解、向量化的元素协方差 \(\operatorname{Cov}(\operatorname{vec}H)\)，但最后一种**输出维度通常改变为 \(n^2\times n^2\)**，不能误认为原尺寸 Hessian。

对于一个局部或全域曲率的数值标量，也可以先逐点计算 \(\|H(W)\|_F\)、\(\operatorname{tr}H(W)\)、\(\lambda_{\max}(H(W))\)，再取期望/极值。**标量输出不一定来自一张固定数值矩阵。**

### 2.1 例子：同一真实 Hessian 得到截然不同的固定矩阵

\[
f(w_1,w_2)=w_1^3+\tfrac12w_1w_2^2,\qquad
H(W)=\begin{pmatrix}6w_1&w_2\\w_2&w_1\end{pmatrix}.
\]

若 \(W_1,W_2\) 独立且均为标准正态：

\[
\mathbb E H=\begin{pmatrix}0&0\\0&0\end{pmatrix},\qquad
\mathrm{RMS}_{\mathrm{entry}}(H)=
\begin{pmatrix}6&1\\1&1\end{pmatrix},
\]

\[
\operatorname{Var}_{\mathrm{entry}}(H)=
\begin{pmatrix}36&1\\1&1\end{pmatrix},\qquad
\mathbb E[H^2]=
\begin{pmatrix}37&0\\0&2\end{pmatrix}.
\]

**结论**：即便解析 Hessian 完全正确，汇总为哪一个矩阵仍是独立数学选择；\(\mathbb E[H\odot H]\) 不等于 \(\mathbb E[H^2]\)。

## 3. 常见数学领域为什么计算这些 Hessian 指标

| 领域/目的 | 典型操作 | 关键问题及条件 |
|---|---|---|
| 局部优化与二阶检验 | \(H(W_0)\)、\(H^{-1}\nabla f\)、HVP、特征值符号 | 当前方向的曲率、驻点极值与鞍点；非凸 Hessian 求逆要谨慎 |
| 全局光滑性、凸性 | \(\sup_\Omega\|H\|_2\)、\(\inf_\Omega\lambda_{\min}\)、\(H(W)\succeq0\) | 梯度 Lipschitz 上界、强凸性；取决于定义域和正则性 |
| PDE/几何 | \(\operatorname{tr}H=\Delta f\)、\(\det H\) | 扩散/调和与 Monge–Ampère 的局部体积变换 |
| 统计信息 | 负对数似然 Hessian 的条件期望 | 在正确模型及正则条件下给 Fisher 信息，而非任意 \(\mathbb E H\) 都是 Fisher |
| 贝叶斯局部近似 | 众数处负对数后验 Hessian 的逆与 logdet | Laplace 近似的不确定性和体积修正；需合适的局部正定性 |
| 统计降维 | 平均 Hessian 的特征向量（pHd） | 寻找响应函数的重要低维方向；依赖模型及分布条件 |
| 曲率强度和异质性 | RMS、Mean Abs、Variance、Quantile | 描述局部作用强度、随位置波动与尾部曲率；**不是普遍有统一理论解释的单一标准** |
| 方向结构和矩阵算子 | \(\mathbb E[H^2]\)、谱分解、协方差 \(\operatorname{Cov}(\operatorname{vec}H)\) | 功能方向、曲率元素共同变化；需给出具体 estimand |
| 消元与计算降维 | 固定子块、投影、Schur 补、Kron 近似 | 关注特定子空间、降低计算量；近似不等于精确 Hessian |

**重要的跨领域区分**：某个指标在优化/几何/统计中有成熟用途，**不自动**等于它是好的坐标旋转简单性指标。比如 \(\operatorname{tr}H\)、特征值、\(\|H\|_F\) 在正交坐标旋转下不变；若只优化这类不变量，就没有角度辨识力。

## 4. 单变量效应、非线性、函数交互、统计依赖：四个不同概念

假设 \(\partial_i f\)、\(H_{ij}\) 在所选 Q 下有足够矩。

| 希望研究什么 | 自然的数学对象 | 典型数字 | 主要局限 |
|---|---|---|---|
| **单个维度的边际效应强度** | \(g_i(W)=\partial_i f(W)\) | \(\sqrt{\mathbb E[g_i^2]}\) | 一阶敏感度不是非线性程度 |
| **单个维度自身局部非线性** | \(H_{ii}(W)=\partial_i^2 f(W)\) | \(\sqrt{\mathbb E[H_{ii}^2]}\) | 曲率可能依赖其它变量，未必是独立主效应 |
| **两维函数交互** | \(H_{ij}(W)=\partial_i\partial_j f(W),\,i\ne j\) | \(\sqrt{\mathbb E[H_{ij}^2]}\) | 混合曲率不是 W_i/W_j 的统计相关 |
| **边际作用的共同波动** | 梯度场 \((g_i(W),g_j(W))\) | \(\operatorname{Cov}_Q(g_i,g_j)\) | 共同波动不等于真实的函数交互 |
| **输入维度的统计依赖** | \(p(W_i,W_j)\) | Pearson、MI、distance covariance、HSIC | 仅从 f 的 Hessian 无法识别输入分布的依赖 |

两个最小反例：

- \(f(w_1,w_2)=w_1+w_2\)，即便 \(W_1,W_2\) 高度相关，\(H_{12}=0\)：**统计依赖不意味着函数交互**。
- \(f(w_1,w_2)=w_1w_2\)，即便 \(W_1,W_2\) 统计独立，\(H_{12}=1\)：**函数交互不意味着统计依赖**。

若 \(f\) 在连通矩形域上是 \(C^2\)，且所有混合二阶导 \(H_{ij}\equiv0\)（对所有 \(i\ne j\)），则 \(f(W)\) 在原坐标下加性可分 \(c+\sum_i f_i(W_i)\)。这里要求**处处**为零，单纯 \(\mathbb E H_{ij}=0\) 不足以推出可加性。

### 4.1 对角曲率与非对角曲率

\[
H_{ii}(W)=\frac{\partial}{\partial W_i}\left(\frac{\partial f}{\partial W_i}\right)
\]

刻画固定其他维度时，\(W_i\) 自身的**局部二阶曲率**。

\[
H_{ij}(W)=\frac{\partial}{\partial W_j}\left(\frac{\partial f}{\partial W_i}\right),\quad i\ne j
\]

刻画 \(W_j\) 改变时 \(W_i\) 的边际作用如何变化，即**局部函数交互**。

非零 Hessian 对角线并不意味着「存在一个可加分离出来的非线性 \(f_i(W_i)\)」；原因是 \(H_{ii}(W)\) 自身可能仍然依赖 \(W_{-i}\)。

## 5. 重点：为什么自身 Hessian 曲率不一定是独立主效应？

考虑

\[
f(w_1,w_2)=w_1^2(1+w_2)=w_1^2+w_1^2w_2.
\]

则

\[
\partial_1 f=2w_1(1+w_2),\qquad
H_{11}(w_1,w_2)=2(1+w_2).
\]

当 \(w_2=0\)，同一维度 \(w_1\) 的曲率为 2；当 \(w_2=2\)，曲率为 6。**这说明 \(w_1\) 的自身局部曲率受 \(w_2\) 调节。**

若只计算一个全域对角线 RMS，便将 \(w_1^2\) 自身非线性与 \(w_1^2w_2\) 交互调节混在一起。可以使用积分测度分解来区分这两部分。

## 6. 三个可以分别计算为数字的曲率指标

对于维度 i，固定一个联合分布 Q；假定 \(H_{ii}\) 平方可积。记 \(W_{-i}\) 为除 i 外的所有坐标。

### 6.1 Total curvature：总体自身曲率能量

\[
\boxed{N_i^{\rm total}=
\sqrt{\mathbb E_Q[H_{ii}(W)^2]}.}
\]

它计算的是沿 i 坐标的二阶曲率**总体幅度**，包括被其他变量调节的部分。

### 6.2 Conditional-mean curvature：在给定 \(W_i\) 条件下平均的曲率能量

先定义

\[
m_i(w_i)=\mathbb E_Q[H_{ii}(W)\mid W_i=w_i],
\]

再计算

\[
\boxed{N_i^{\rm main}=
\sqrt{\mathbb E_{W_i}[m_i(W_i)^2]}.}
\]

**重要限定**：这个 “main” 名称严格来说首先表示**条件平均曲率分量**。当各输入相互独立，且可交换微分与对 \(W_{-i}\) 的积分时，它才直接等于平均 partial-dependence 函数

\[
f_i(w_i)=\mathbb E_{W_{-i}}[f(w_i,W_{-i})]
\]

的二阶导数 \(f_i''(w_i)\) 的 RMS。对于相关输入，\(\mathbb E[H_{ii}|W_i]\) 不可直接等同于 \(\frac{d^2}{dw_i^2}\mathbb E[f(W)\mid W_i=w_i]\)，因为条件分布也随 \(w_i\) 变化。主效应的定义取决于**所选分布与边际/条件解释**。

### 6.3 Modulation curvature：其他变量调节自身曲率的能量

\[
\boxed{
N_i^{\rm modulation}
=\sqrt{\mathbb E_{W_i}
[\operatorname{Var}_Q(H_{ii}(W)\mid W_i)]}.}
\]

固定 \(W_i\) 后，让其他变量按照条件分布变化，观察 \(H_{ii}\) 波动程度。

### 6.4 精确分解恒等式（不是近似）

由条件期望的正交投影/全方差分解：

\[
\boxed{
(N_i^{\rm total})^2
=
(N_i^{\rm main})^2
+
(N_i^{\rm modulation})^2.
}
\]

**这个能量恒等式对同一个联合测度 Q 下的平方可积 \(H_{ii}\) 成立，即使输入变量相关。** 但要把 \(N_i^{\rm main}\) 解释成 independent input 下的 partial-dependence 主效应曲率，需要额外的独立性和微分交换条件。

### 6.5 数值例一：独立自身曲率与调制共存

当 \(f(w_1,w_2)=w_1^2(1+w_2)\)，\(W_2\sim N(0,1)\) 且与 \(W_1\) 独立，则

\[
H_{11}=2(1+W_2),\quad
N_1^{\rm total}=2\sqrt2,\quad
N_1^{\rm main}=2,\quad
N_1^{\rm modulation}=2.
\]

验证：\((2\sqrt2)^2=2^2+2^2=8\)。

这给出三个不同问题的答案：曲率总强度、平均主效应曲率、其他变量调节造成的曲率波动。

### 6.6 数值例二：没有平均主效应曲率，却有条件非线性

令 \(f(w_1,w_2)=w_1^2w_2\)，\(W_2\sim N(0,1)\) 与 \(W_1\) 独立。

\[
H_{11}=2W_2,\quad
N_1^{\rm total}=2,\quad
N_1^{\rm main}=0,\quad
N_1^{\rm modulation}=2.
\]

也就是说 \(W_1\) **在每个给定 \(W_2\ne0\) 的条件下都有非线性**，但按独立零均值 \(W_2\) 积分后，其平均主效应为零，曲率能量全部体现为 modulation。

## 7. RMS 为什么不会发生正负抵消？

对一个 Hessian 元素 \(h(W)=H_{ii}(W)\)：

\[
\text{Signed Mean}=\mathbb E[h],\quad
\text{Mean Absolute}=\mathbb E[|h|],\quad
\boxed{\text{RMS}=\sqrt{\mathbb E[h^2]}}.
\]

RMS 是**先平方（去符号）→期望→开平方**。因此与 Signed Mean 不同，正负曲率不会相互抵消。

若 \(h=+5\) 与 \(-5\) 各出现一半，则：

| 指标 | 结果 |
|---|---:|
| Signed Mean | 0 |
| Mean Absolute | 5 |
| RMS | 5 |

若 \(h=0\) 的概率 .9、\(h=10\) 的概率 .1，则：

| 指标 | 结果 |
|---|---:|
| Mean Absolute | 1 |
| RMS | \(\sqrt{10}\approx3.162\) |
| Maximum Absolute | 10 |

RMS 对较大曲率更敏感；它不会出现符号抵消，但**会丢失凸/凹的方向信息**。若同时关心曲率强弱和方向，可报告 \((\mathbb E H_{ii},\sqrt{\mathbb E H_{ii}^2})\) 作为一对数字。

恒等式：

\[
\mathbb E[h^2]=(\mathbb E h)^2+\operatorname{Var}(h)
\]

说明曲率能量不仅包含平均曲率大小，也包含曲率随位置的波动。**Variance 本身不能代替 RMS**：常数 \(h\equiv100\) 方差是 0，但曲率非常大。

## 8. 另一条“单变量非线性”路径：不用 Hessian 也能得到数字

对独立输入的 partial-dependence 主效应

\[
f_i(w_i)=\mathbb E_{W_{-i}}[f(w_i,W_{-i})],
\]

先寻找最优一元线性近似

\[
(a_i^*,b_i^*)
=\arg\min_{a,b}\mathbb E_{W_i}
[(f_i(W_i)-a-bW_i)^2].
\]

定义

\[
\boxed{
N_i^{\rm linear-residual}
=\sqrt{\mathbb E_{W_i}[
(f_i(W_i)-a_i^*-b_i^*W_i)^2]}.
}
\]

这个指标回答**平均主效应函数在函数值意义上不能被直线解释的程度**。它不同于 Hessian 曲率 RMS：两者单位、对函数形状/输入分布的敏感性与所反映的“非线性”都不同。不要将函数值残差标为 Hessian-only 统计。

同样，Sobol/functional ANOVA 指标衡量的是输入分布下的**函数值方差贡献**，混合 Hessian RMS 衡量的是**导数级的交互作用幅度**；二者不能相互替代。

## 9. 进一步的限制与后续方向

1. **尺度**：改变 W_i 的单位或 Y 的单调链接（如把 logit 变为 probability），会改变 Hessian、非线性强度和交互强度；需明确输出尺度与坐标单位。
2. **分布**：期望、RMS、条件方差、分位数依赖 Q；在不同输入分布下，同一解析函数的数值指标可不同。
3. **局部 vs 全局**：固定一点的 Hessian 不能描述整个定义域；对所有点的 Signed Mean 为 0 也不能推出局部 Hessian 恒为 0。
4. **正交旋转不变性**：Trace、谱、Frobenius norm 在同一物理点经正交相似变换不变，不能单独分辨具体坐标轴；逐元素对角/非对角等则与坐标系有关。
5. **三/四阶导数**：如 \(H_{ii}(W)\) 随其他变量变化，\(\partial_jH_{ii}=\partial_{iij}f\) 定量描述曲率调节的变化率；若需完整更高阶结构，可考虑高阶导数，但增加平滑性要求。
6. **不把相关性等同交互**：输入统计相关系数、函数混合偏导、梯度协方差，是不同对象。
7. **仅有 (W,Y)**：后续需要先验证 Hessian 或目标积分泛函是否可由离散数据识别/可靠估计；预测 Y 很准不保证二阶导数准。数据来源、估计器误差、矩阵汇总误差、真实数学结构分别评价。
8. **不能滥用“主效应”命名**：\(N_i^{\rm main}\) 是 Hessian 条件平均的 RMS；只有在适当独立性下与 partial-dependence 主效应曲率一致。
9. **不是所有函数交互都只需要 Hessian**：如果目标是函数整体非可加部分的方差或可分性，functional ANOVA、交互残差或受控变化可能比局部混合 Hessian 更直接。

## 10. 本次讨论凝练成的数学结论

- \(H(W)\) 是矩阵**函数**；固定值、积分、逐元素非线性汇总、矩阵算子、标量/向量投影是不同操作。
- 是否消除 W 依赖与是否改变输出形状是两个**正交因素**；操作顺序一般不可交换。
- 对角元素 \(H_{ii}\) 是条件自身曲率，非对角 \(H_{ij}\) 是局部函数交互；二者不同于统计依赖。
- 对角线 RMS \(\sqrt{\mathbb E[H_{ii}^2]}\) 不会发生正负抵消，但包括平均主效应曲率与其他变量对曲率的调制。
- 通过条件期望，可以严格分解平方能量为 \(\mathbb E[(\mathbb E[H_{ii}|W_i])^2]\) 和 \(\mathbb E[\operatorname{Var}(H_{ii}|W_i)]\)。
- 条件平均曲率与独立的 partial-dependence 主效应的联系需要**独立性与可交换微分/积分**等额外条件。
- Mean Absolute、RMS、Variance、Quantile、Matrix Gram、Hessian covariance 回答不同的数学问题；使用同一个解析 Hessian 并不会自动决定哪种摘要“正确”。
- 数学上成立的数值统计量，不必然是好的旋转目标、好的模型复杂度指标或能从有限 \((W,Y)\) 样本稳定估计的量。

---

**历史定位**：本文优先归档“从 \(H(W)\) 出发到数字/矩阵，以及变量非线性与函数交互的严格区分”。更广泛的 C1/C2/C3/CR3、旋转反例、数值估计和 Sobol 讨论仍见已有的 [history/README.md](README.md) 及其链接文档。