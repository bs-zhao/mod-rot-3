# Rotation simplicity / sparsity metrics: 数学定义、CR3 证据、Toy 对照与研究决策（2026-11-09）

> **文档性质**：Mod-Rot-3 项目的研究历史归档，尤其记录 2026-11-07 至 2026-11-09 的讨论、实验与反例。不是“所有指标已成功”的声明，也不是用于训练时泄露真实参数的方案。
>
> **核心问题**：给定冻结解码器或连续生成函数 \(f(U,X)\)，以及来自模型的潜变量 \(Z\)，能否只从函数结构与观测数据构建简单性损失 \(L(A)\)，求得 \(U=AZ\)，使各 \(U_i\) 对应真实模型参数的独立可解释轴？
>
> **重点**：简单性 ≠ 稀疏性 ≠ 正交不相关 ≠ 统计独立 ≠ 参数可恢复性。必须单独验证“真实方向的解析极小值”“从解析 Y 估计的同一指标是否恢复”和“NN Y 路径是否保留该规律”。
>
> 本文件新增于 readmes/history，不取代 [Hessian 全历史](README.md) 与 [Hessian 函数运算专题](README_HESSIAN_FUNCTIONAL_METRICS_AND_EFFECTS_20261109.md)。

---

## 1. 记号、目标和不能偷用的真值

四个真实参数 \(T=(\alpha,\lambda,\gamma,\tau)\in\mathbb R^4\)；学习到的潜表示 \(Z\in\mathbb R^4\)，刺激 \(X\in\mathbb R^6\)。

目前 CR3 诊断流程先使用真实参数做一次**有监督基准回归**：

\[
\widehat T=A^* Z+b,\qquad A^*\in\mathbb R^{4\times4}.
\]

对于参数维度 \(i,j\) 的二维角度扫描，

\[
R_{ij}(\phi)\in SO(4),\quad
A(\phi)=R_{ij}(\phi)A^*,\quad
U(\phi)=A(\phi)Z+b.
\]

后续希望最大限度不依赖 \(T\)：直接从 \((Z,X,Y)\) 或已知的冻结解码器函数，优化 \(A\)：

\[
A_{\rm discovered}
=\arg\min_{A\in\mathcal A} L\big(f(AZ,X)\big).
\]

**关键限制**：\(A^*\) 及对应的真实 \(T\) 可以用来构建 *oracle 对照、角度诊断和最终验证*，不能进入将来声称“无监督发现”的 loss 本身。当前实验先围绕 \(A^*\) 评价结构，尚未完成自动恢复的完整证明。

输出形式要分开：

\[
\eta=\text{CR3 logit},\qquad
p=\sigma(\eta),\qquad
C\sim{\rm Bernoulli}(p).
\]

解析 Choice 的可微参照实际上是 \(\mathbb E[C\mid T,X]=p\)；离散抽样 choice 本身不具有一般意义的连续 Hessian。NN 的硬 choice 与解析 Bernoulli sample 还存在标签生成机制差异。

---

## 2. 我们希望简单的究竟是什么？

| 目标 | 数学对象 | 直觉 | 风险/反例 |
|---|---|---|---|
| 效应稀疏（effect sparsity） | 梯度、Hessian 或其他作用向量 | 尽量少的显著作用 | 稀疏不保证参数身份正确 |
| 参数—刺激稀疏（ZX） | \(H_{UX}=\partial_U\partial_X f\) | 某个 U 只受到少数 X 作用，或一个 X 只受少数 U 作用 | 行和列定义完全不同 |
| 参数间可分离（ZZ） | \(H_{UU}\) 的非对角项 | 少量跨参数作用 | \(abx\) 可能自然需要乘积 |
| 低阶非线性 | \(\partial_{U_i}^3 f,\partial_{U_i}^4f\)、扫描残差 | 单参数方向低阶、平滑 | “曲率大”不等于“复杂” |
| 功能性共同方向 | 梯度与 X 派生算子组成的矩阵集 | 一组算子尽量同时对角化 | 算子构造、中心化与积分顺序影响结果 |
| 全局可加性 | 固定其他变量的 1D/2D 函数扫描、矩形差分 | 不同 U 对响应的作用尽量可拆分 | 扫描可能出训练联合分布 |
| 统计独立（ICA/JADE/HSIC） | \(P(U)\) 而非仅 \(f\) | 旋转后潜变量在分布上独立 | 高斯潜变量下正交旋转的独立性可能不唯一 |
| 直接参数恢复 | \(U\leftrightarrow T\) 的相关矩阵 | 对角对应高、off-diagonal 低 | 只用于实验验证，不得作为无监督目标 |

这些是不同且可能冲突的目标。**某个指标对“真实函数”在 0° 最小，是要被检验的经验/数学命题，而非稀疏性一词自动保证的结论。**

---

## 3. 从 Hessian 场到可比较的损失：均值不够

对 \(W=(U,X)\) 的函数 \(f\)：

\[
H(W)=\nabla_W^2 f(W)
=\begin{pmatrix}
H_{UU}(W)&H_{UX}(W)\\ H_{XU}(W)&H_{XX}(W)
\end{pmatrix}.
\]

它首先是**矩阵值函数**，通常不是常数矩阵。过去分别研究过：

- 先得到 \(\bar H=\mathbb E_W H(W)\)，再评价稀疏性；
- 先取 \(\mathbb E|H_{ij}(W)|\) 或 \(\sqrt{\mathbb E H_{ij}(W)^2}\)，再评价各元素强度；
- 在每个 \(W\) 上直接计算行/列稀疏性，然后对 \(W\) 求均值、分位数或最坏情况；
- 保留 \(H(W)\) 的变化，用三/四阶导数、受控扫描或方差分解刻画“曲率本身的非线性”。

这些操作不可随意交换：

\[
|\mathbb E H_{ij}|\leq\mathbb E|H_{ij}|
\leq\sqrt{\mathbb E H_{ij}^2}.
\]

例如 \(f(u)=u^3\) 的 \(H_{uu}=6u\)：在对称分布上 \(\mathbb E H_{uu}=0\)，但 \(H_{uu}\) 并不恒为零；而 \(f(u)=u^2\) 的 \(H_{uu}=2\) 不随自身变量变化。

**普通整体 L2 / Frobenius 不是寻找正交旋转的好方向损失**。例如对正交 \(R\)：

\[
\|RH_{UU}R^\top\|_F=\|H_{UU}\|_F,\qquad
\|RH_{UX}\|_F=\|H_{UX}\|_F.
\]

同一物理函数与同一积分点下，整体二阶能量对旋转不敏感。必须改用**非对角、逐行/列、Top-k、排序残差、有效支持或跨多个算子的共同结构**，才能产生方向差异。数值拟合出现的总能量变化还可能是估计器误差，不能视为可靠的旋转信号。

---

## 4. 重要的稀疏性设计需求：惩罚“小而多余”的作用

讨论中的两个代表性例子（不是仅针对这两个数值设计）：

\[
[2,\ 0.011,\ 0.001]\quad\text{vs}\quad[2,\ 0.001,\ 0.001],
\]

\[
[2,\ 2,\ 0.01,\ 0.001]\quad\text{vs}\quad[2,\ 2,\ 0.001,\ 0.001].
\]

目标是：**允许不止一个大的真实作用，但多出来的相对小作用仍应明显提高复杂度损失**。因此不能固定假设“每个 Z 只能关联一个 X”；应测试多个 \(k\)，也不能只看最大的作用。

第 06 号实验在
[math/06_cr3_hessian_rotation_landscapes/sparsity.py](../../math/06_cr3_hessian_rotation_landscapes/sparsity.py)
实现了丰富的稀疏损失。对某条作用向量 \(v\) ，先令

\[
m=\max_j|v_j|,\qquad s_j=|v|_{(j)}/m,
\quad s_1\geq s_2\geq\cdots\geq0.
\]

当 \(m=0\) 时另行处理。这样大多数损失对整体正比例缩放、符号翻转和坐标置换不变；**并不意味着对一般连续旋转不变**。归一化到当前最大元素也意味着极小值的数值解释是相对的，而非物理导数的原始幅度。

### 4.1 稀疏性指标族（全部有实现）

| 家族 | 代表公式 | 想测量什么 | 特点/局限 |
|---|---|---|---|
| L1/L2、Hoyer | \((\|s\|_1/\|s\|_2-1)/(\sqrt d-1)\) | 非零作用是否集中 | 对较大的多余作用敏感；对非常小的额外作用可能不够 |
| Sublinear \(L_p\) | \(\sum_j s_j^p,\ p=0.25,0.5,0.75\) | 加重小非零项的相对惩罚 | 非凸，近零处敏感 |
| Entropy/Rényi/Gini | \(p_j=s_j/\sum s\)，熵或 \(1-\sum p_j^2\) | 有效支持数与集中度 | 属于强度分布而非硬非零个数 |
| Top-k Tail L1 | \(\sum_{j>k}s_j\) | 第 k 大作用以后的残余 | \(k=1,\ldots,4\) 并列评估 |
| Top-k Tail L2 | \((\sum_{j>k}s_j^2)^{1/2}\) | 残余作用能量 | 对残余组合更平滑 |
| Top-k Tail L2² | \(\sum_{j>k}s_j^2\) | 大残余的平方惩罚 | 不应固定唯一的 k |
| Next-k | \(s_{k+1}\) | 第 k+1 大的“额外作用” | 对作用大小排序变动敏感 |
| Soft-L0 | \(\sum_j(1-e^{-s_j/t})\) | 平滑近似有效非零数 | 大作用逐渐饱和，小作用依然影响结果 |
| Capped L1 | \(\sum_j\min(s_j/t,1)\) | 截断的有效支持 | 阈值附近有折点 |
| Capped L2² | \(\sum_j\min((s_j/t)^2,1)\) | 对小额外作用按平方惩罚 | 是 capped-square，不能误称为普通整体 L2 |
| Log-sum | \(\sum_j\log(1+s_j/t)/\log(1+1/t)\) | 非凸、连续的近似支持惩罚 | 更强调小/中强度额外作用 |
| Fraction | \(\sum_j (s_j/t)/(1+s_j/t)\) | 平滑饱和型有效支持 | 参数 t 决定敏感区间 |
| Adaptive support | \(\min_{0\leq k\leq d}\{k\,t^2+\sum_{j>k}s_j^2\}\) | 联合权衡“几个主要作用”和尾部能量 | 不预先规定主作用个数 |
| Hard support diagnostics | \(\#\{j:s_j>t\}\) | 不同阈值下的经验有效支持 | **只做诊断**，不宜单独作为最终连续优化损失 |

本轮使用相对阈值 \(t=0.0005,0.001,0.005,0.02\)。不应根据一次“最好看”的曲线事后选单一阈值而不做新数据验证。

**数学上容易误解的两点**：

1. 归一化 Hoyer、熵等对所有维度同时缩小不敏感。因此应另报 Hessian Frobenius/RMS 大小，防止把“整体很弱”误当成“作用集中”。
2. \(\|H\|_F\) 的标量、某个行向量的 L2、Top-k *尾部* L2 是三种完全不同的指标；前者通常不能决定正交旋转，后两者可能可以。

### 4.2 ZX 的行稀疏与列稀疏不是一回事

\(H_{UX}\in\mathbb R^{4\times6}\)。对于某个 \(U_i\)：

- **行稀疏**：\(H_{UX}[i,:]\)，问“一个参数方向与多少种 X 效应关联？”
- **列稀疏**：\(H_{UX}[:,k]\)，问“一个 X 刺激特征会影响多少个参数方向？”

讨论中重点提出“一个 Z 对应的 X 尽量少”，所以**行稀疏**尤其重要，但列稀疏是独立的、值得并列测试的目标。

聚合方式也不能偷换：

\[
L_{\rm rowMean}=\frac14\sum_i\ell(H_{UX}[i,:]),\qquad
L_{\rm rowMax}=\max_i\ell(H_{UX}[i,:]).
\]

Mean 会掩盖最坏参数轴；Max 更严格，却容易放大估计噪声和“最差行切换”。第 06/07 号实验均实现了相应版本。此外可以先对每个局部 \((U,X)\) 计算，再求均值/分位数，而不只在平均 Hessian 上评分。

### 4.3 单位归一化与重复导数

四个参数（\(\alpha,\lambda,\gamma,\tau\)）和六个 X 的原始单位不同，直接比较 Hessian 系数可能产生伪稀疏。应对照原始单位和标准化单位，例如：

\[
\widetilde H_{U_iX_k}
=\sigma_{U_i}\sigma_{X_k}H_{U_iX_k}.
\]

第 06 号保留 X 标准差校准版；第 07 号的高阶张量则乘上每个相应输入坐标的标准差，包括重复索引（如 \(\sigma_{U_i}^2\sigma_{X_k}\,\partial_{U_i}^2\partial_{X_k}f\)）。

注意：**在原坐标上先标准化再旋转**与**每次旋转后再按新轴 SD 归一化**并不一定产生相同的 SO(4) 优化问题。必须固定实验约定并报告，避免把坐标标尺改变误解为参数恢复。

---

## 5. ZZ：不仅是非对角稀疏，还包括自身的高阶复杂度

\[
H_{UU}(U,X)\in\mathbb R^{4\times4},\quad
H_{ii}(U,X)=\partial_{U_i}^2f(U,X).
\]

- 非对角 \(H_{ij},i\neq j\)：**二阶混合函数交互**；
- 对角 \(H_{ii}\)：沿 \(U_i\) 的二阶曲率，但它**可能依赖其他 U 或 X**，不能未经条件化就称为可分离的“单变量主效应”；
- 只看平均 \(H_{ii}\) 无法捕捉其沿自身 \(U_i\) 进一步变化。

示例：

\[
f_1(u,x)=u^2g(x),\quad H_{uu}^{(1)}=2g(x);
\qquad
f_2(u,x)=u^3g(x),\quad H_{uu}^{(2)}=6u\,g(x).
\]

两者对 X 的依赖可以很强；但只有第二个的 Hessian 依赖**自身 u**。由此讨论了两种成熟的复杂度思路：

\[
L_{{\rm selfVar},i}
=
\mathbb E_{u_{-i},x}
\big[\operatorname{Var}_{u_i}(H_{ii}\mid u_{-i},x)\big],
\]

\[
L_{{\rm third},i}
=
\mathbb E\left[(\partial_{U_i}H_{ii})^2\right]
=\mathbb E[(\partial_{U_i}^3f)^2].
\]

实际第 07 号对固定锚点 \(u_{-i},X\) 做**受控扫描**，不是凭原始观测直接无偏估计某个条件分布的真实条件方差。相关版本还包含曲率 TV、三阶导数 L1/RMS、四阶导数、二次/三次逼近残差。

### 5.1 高阶导数分类

| 家族 | 典型导数 | 含义 |
|---|---|---|
| ZZ 自身三阶 | \(\partial_{U_i}^3f\) | 二次及以下沿自身轴为零；能区分 \(u^2\) 与 \(u^3\) |
| ZZ 跨 Z 三阶 | \(\partial_{U_i}^2\partial_{U_j}f\)、\(\partial_{U_i}\partial_{U_j}\partial_{U_k}f\) | 非线性参数交互 |
| ZZ–X 三阶 | \(\partial_{U_i}^2\partial_{X_k}f\) | 某个 ZZ 曲率如何受 X 调制 |
| ZX–X 三阶 | \(\partial_{U_i}\partial_{X_k}^2f\) | ZX 效应如何随同一 X 改变 |
| ZZ 四阶 | \(\partial_{U_i}^4f\)、\(\partial_{U_i}^2\partial_{U_j}^2f\)、\(\partial_{U_i}^3\partial_{U_j}f\) | 更高阶的自身与跨参数非线性 |
| ZZ–XX / Z–XXX 四阶 | \(\partial_{U_i}^2\partial_{X_k}^2f\)、\(\partial_{U_i}\partial_{X_k}^3f\) | 刺激调制的四阶变化 |

第 07 号的主要统计聚合包括 mean absolute、RMS、STD、P90、P99、Max，也包括 mean-Hessian/RMS-Hessian 的 Top-k、非凸稀疏性、多尺度损失与局部逐行聚合。

高阶“曲率粗糙度”不是“严格多项式次数”。例如 \(u^4\) 与 \(u^3\) 的大小比较依赖系数、测度和区间；第三阶能量大并不数学等价于“函数更复杂”这一普遍排序。

### 5.2 Hessian 之外的函数结构：扫描与可加性

- 受控单轴扫描：固定其余 U 与 X，仅改变 \(U_i\)，计算曲率方差/TV、一次至四次多项式相对 MSE、DCT 高频能量；
- 二维 ZZ 扫描：固定其余轴，变化 \(U_i,U_j\)，计算二维函数减去两项主效应后的残差能量（functional-ANOVA 式 additivity gap）；
- Z–X 的可加性缺口：固定其他变量，测量 \(U_i\) 与 X 作用能否相加拆分。

需要查验扫描的支持范围（项目已有 Mahalanobis/OOD95 诊断）；单轴或双轴反事实点可能离开经验 \(P(U)\) 的联合支持。有限差分及局部回归的高阶误差会被此类指标放大。

---

## 6. Toy 项目的 M001–M024：与第 07 号的异同

参考上传/保存的 modlearn-rot-toy 项目中的 catalog/README_METRICS.md。其指标针对**连续函数**，T11-id 对应 CR3 logit，T11-sig 对应 CR3 probability。M 编号是 Toy 项目内部定义，**不能与本仓库旧的 M01–M20 Hessian 矩阵指标混为一谈**。

| Toy 指标 | 主对象 | 对 Mod-Rot-3 第 07 号的关系 |
|---|---|---|
| M001 XU group L1、M002 concentration | 每个 U 轴的 XU 导数能量 | 与 ZX 稀疏同族，但聚合/归一化不同 |
| M003 local XU L1 | 每个 X 导数跨 U 轴的局部稀疏 | 更接近 ZX **列**方向，而非第 07 号的 row-wise Z→X 定义 |
| M004/M005 | U 梯度随 X 的协方差、未中心化梯度 Gram | 第 07 号缺少完全对应的梯度算子 |
| **M006** | UU 非对角 Hessian 平方能量占比 | 接近第 07 号 zz_offdiag_total_energy_fraction_loss；积分与坐标缩放有差别 |
| M007 | UU 对角能量占比 | 与 M006 互补而非独立新规律 |
| M008–M011 | 梯度/Hessian 总能量 | 正交旋转下不变量或控制量；不应作为独立方向选择损失 |
| M012–M016 | 单轴 affine、函数形状、秩一、二阶 roughness、矩形差分 | 与受控 sweep / additivity 类似，但扫描中心、节点与公式不完全一样 |
| **M017/M018** | Level / First-X / Second-X 的功能算子联合对角化 JD | 第 07 号尚未完整移植；最有价值的互补家族 |
| M019–M022 | 上述 JD 的单一算子子族 | 有助于分辨是哪类 X 效应决定方向 |
| M023 | sigmoid 权重的梯度 Gram 非对角损失 | 功能加权算子；**不等于连续响应真实似然 Fisher** |
| M024 | 上述算子 trace | 正交旋转不变量，适合检验实现 |

JD 的核心做法不是简单压小某个导数，而是先构造多个关于 U 的函数向量 \(\psi_\alpha\)，然后形成

\[
K_\alpha=\mathbb E_U[\psi_\alpha(U)\psi_\alpha(U)^\top],
\]

\[
L_{\rm JD}(R)=
\frac1{|\mathcal A|}\sum_{\alpha\in\mathcal A}
\|\operatorname{off}(R\widetilde K_\alpha R^\top)\|_F^2.
\]

Toy 的关键顺序：**先对 X 积分得到 \(\psi_\alpha\)，再形成外积、对 U 积分**。随意换成 \(\mathbb E_{U,X}[gg^\top]\) 不是同一个指标。

这体现两条可能互补的研究路线：第 07 号偏重 *effect sparsity / low-order complexity*，Toy 的 JD 偏重 *shared functional axes*。不要在新实验里用相同名称掩盖公式差异。

---

## 7. 三条路径 × 三种 Y：避免把估计器问题误判成 NN 问题

每个旋转角度下对照：

1. **解析真值（analytic）**：CR3 解析函数的精确二阶 Hessian；三阶/四阶由精确 Hessian 的有限差分求得并做步长收敛检查。
2. **解析 Y 回归（formula_y）**：只用 \(U,X,Y_{\rm formula}\) 拟合相同的张量多项式/正则化 surrogate，再对 surrogate 求局部导数。
3. **NN Y 回归（nn_y）**：冻结解码器产生 Logit/Probability/Hard choice 的 Y；再用同一个 surrogate 拟合求导，**不是对 NN 权重直接 autodiff**。

这一组三路对照对应 \(3\times3=9\) 个来源—Y 组合。所有源在同一旋转角度、同一 held-out U×X 评估点上比较。

对任何局部导数张量 \(K\)，严格恒等式为

\[
\underbrace{\widehat K_{\rm NN-Y}-K_{\rm exact}}_{\text{总差异}}
=
\underbrace{\widehat K_{\rm formula-Y}-K_{\rm exact}}_{\text{解析 Y 下的估计器误差}}
+
\underbrace{\widehat K_{\rm NN-Y}-\widehat K_{\rm formula-Y}}_{\text{额外拟合路径差异}}.
\]

第一项（formula_Y - analytic）如果已失败，不能把总差异直接归咎于 NN。第二个差并不能无条件等同于“NN 真实函数的导数误差”，因为仍经过 surrogate 拟合、正则化和有限采样。

三个不同判据也必须独立输出：

- **A. 轴极小值**：0°/90° 附近是否为 *minimum*，而非 maximum？
- **B. 曲线形状**：analytic–formula、analytic–NN、formula–NN 的相关性有多高？
- **C. 数值导数误差**：局部张量范数误差、符号和方向是否恢复？

曲线高度一致不代表其最小值正确；导数数值不精确，也可能恰好保留某个标量旋转损失曲线。这两种情况都在已运行实验中出现过。

---

## 8. 已有 CR3 实验的实测结果：平均 Hessian 与局部高阶指标

数据：CR3 5000 subjects × 500 fixed stimuli，四真实参数、六 X 特征。先对 \(T\approx A^*Z+b\) 回归，再围绕 6 个参数对做 \(0°,5°,\dots,90°\) 扫描。下面是现有上传结果的复核数据，**不是理论证明**。

### 8.1 第 06 号：平均 Hessian 的丰富稀疏性

来源：rich_sparsity_rescored.zip；实现 [math/06](../../math/06_cr3_hessian_rotation_landscapes/README.md)。

- 262 个被筛选的指标名称 × 3 Y × 6 二维平面 = **11,772** 个指标/响应/平面组合；
- 其中 **50** 个组合满足三路曲线相关、端点极小及非平坦的严格规则；
- **没有任何指标在全部六平面同时三路成功**；
- XZ Top-k、Capped-L2²、penalized support 等有局部成功，不能据此声称可普适恢复所有轴；
- 一个重要反例：某些 α–τ 平面三条曲线一致，却共同在约 40°–45° 有最低复杂度；这说明**损失定义本身**就可能不偏好真轴。

### 8.2 第 07 号：局部二/三/四阶复杂度

来源：monomial_u4_x4_lambu1e-06_lambx1e-06.zip；实现 [math/07](../../math/07_cr3_local_hessian_complexity/README.md)。

- 总计 **995** 个输出的指标/诊断列，其中 **884** 个以 _loss 结尾并参与筛选；
- 6 平面 × 19 角度 × 9 路径 = **1,026** 条函数/路径/旋转分析记录；
- 严格“三路同时恢复”筛选共有 **15,912** 个组合，其中 **1,844** 个组合通过，但**没有同一指标在所有六平面三路都通过**；
- 某些三阶 ZX–X 调制与 ZZ 四阶指标已实现 **5/6** 平面的三路通过；
- 多数三/四阶数据驱动导数仍有明显数值误差，尤其 ZX–X 和高阶 X 导数；“优化角度对”不等于“高阶导数逐元素对”。

### 8.3 只看纯解析 CR3：真轴确实是一些复杂度指标的六平面最小点

仅考虑 logit 和 probability 的**884 个待优化损失**（analytic choice 的期望曲率等于 probability，不能算独立数学证据）：

| 严格度 | Logit 通过 6/6 的指标数 | Probability 通过 6/6 |
|---|---:|---:|
| 恰好 0° 或 90° 为最小 | 1 | 1 |
| 容差 ±5° 且变化幅度 ≥2% | 7 | 3 |
| 容差 ±10° 且变化幅度 ≥2% | **21** | **19** |

故有 **40 个输出—指标组合** 在解析路径得到 6/6（不是 40 种独立数学机制；同类统计及等价表达大量重复）。在这 40 项中，若换成**真实解析 Y 回归**：

| 要求 | 数量 |
|---|---:|
| 解析成功全集 | 40 |
| Formula-Y fit 六平面仍有端点极小值与非平坦性 | **19** |
| 再要求六平面 analytic–formula 曲线 Pearson \(r\geq0.90\) | **4** |
| 至少一个平面没有恢复端点极小值 | **21** |

四项最扎实的**解析+解析 Y 回归**指标：

| Y | 完整指标名 | 六平面 Pearson 最低值 | 主要结构 |
|---|---|---:|---|
| Logit | zz_mixed_z_third_mean_abs_loss | 0.989 | ZZ 多变量混合三阶导数平均绝对值 |
| Logit | zz_diagonal_curvature_between_subject_variance_loss | 0.995 | ZZ 对角曲率随被试变化 |
| Probability | zz_offdiagonal_magnitude_q99_abs_loss | 0.966 | 局部 ZZ 非对角绝对曲率的 99% 分位 |
| Probability | zz_rms_flat_next_k1_loss | 0.943 | ZZ RMS 矩阵向量中次大的相对作用 |

其中 **zz_mixed_z_third_mean_abs_loss（Logit）** 在解析和 formula-Y 两条路径的六个二维平面**都恰好于 0° 最小**，是目前最干净的解析/数据驱动配对成功基准。

然而这四项在 NN-Y 路径尚未达到全部六平面三路通过，不能声称已跨越第三个技术 gap。

### 8.4 40 项解析成功指标按数学类别的可恢复性

| 类别 | 解析 6/6 | Formula-Y 极小值 6/6 | Formula-Y 极小值+曲线 6/6 |
|---|---:|---:|---:|
| ZZ 二阶局部曲率/残余稀疏 | 7 | 6 | 3 |
| ZZ 纯 Z 高阶导数 | 10 | 8 | 1 |
| ZZ 自身曲率受控扫描 | 3 | 3 | 0 |
| ZZ 两维函数不可加性 | 1 | 1 | 0 |
| ZZ–X 调制高阶导数 | 5 | 0 | 0 |
| ZX 稀疏支持、Top-k | 9 | 1 | 0 |
| ZX 高阶混合导数 | 5 | 0 | 0 |
| **合计** | **40** | **19** | **4** |

由此可见：**当前估计器相对擅长 ZZ 纯 Z 的局部结构，不擅长含 X 导数的 ZX 和 ZZ–X 高阶结构**。不能直接上升为“ZX 物理上不重要”或者“所有回归估计器都不行”的一般命题。

Formula-Y 对其中 40 项的逐平面恢复数量为：α–λ 40/40、α–γ 23/40、α–τ 36/40、λ–γ 36/40、λ–τ 29/40、γ–τ 40/40。α–γ 与 λ–τ 尤其值得调查估计偏差。

进一步区分：
- **两条路径均 6/6 且高度相关**：如 ZZ mixed-third mean-abs；
- **端点全部正确但至少一条相关不足**：如 Probability ZZ offdiag energy fraction；可用于寻找方向的证据有限、不能说数值导数重建准确；
- **纯解析 6/6，formula-Y 至少一平面偏离**：多数 ZX Top-k / ZZ–X 调制；需优先检验拟合与求导误差。

### 8.5 第 07 号的代表性“成功但非普适”的指标

\[
L_{\rm ZXX}=\mathbb E_{U,X,i,k}
\left|\frac{\partial^3 f}{\partial U_i\partial X_k^2}\right|
\]

对应 zx_own_X_modulation_third_mean_abs_loss。它在 Logit 与 Probability 的三路对照都可达到 **5/6**；剩余 λ–τ 平面三路共同在约 35°–40° 最小，解析函数也失败。因此，修复估计器**不会自动修复该损失定义的全部平面**。

---

## 9. Toy 和 CR3 共同揭示了“输出变换依赖性”

Toy 的 T11-id = CR3 logit，T11-sig = CR3 sigmoid probability。Toy M006 和第 07 号 zz_offdiag_total_energy_fraction_loss 是**同类型的 ZZ 非对角能量占比**，但不同参数测度、标准化及积分方法意味着其数值不必完全相等。

已保存的解析结果在六平面上的通过数：

| 模型/输出 | Toy M003 local XU L1 | Toy M006 ZZ offdiag | Toy M017 JD-v1 | Toy M018 JD-v2 | Toy M021 second-X JD |
|---|---:|---:|---:|---:|---:|
| T8-id | 6/6 | 0/6 | 5/6 | 6/6 | 5/6 |
| T8-sig | 6/6 | 1/6 | 6/6 | 6/6 | 5/6 |
| T10-sig | 6/6 | 2/6 | 3/6 | 4/6 | 3/6 |
| T11-id | 5/6 | 1/6 | 3/6 | 5/6 | 5/6 |
| T11-sig | 2/6 | **6/6** | 3/6 | 5/6 | 5/6 |

相同类指标 M006 在 T11 的 logit 仅 1/6，而 sigmoid probability 为 6/6；CR3-07 解析路径也呈现 **Logit 1/6、Probability 6/6**。

数学上这是可以发生的，因为

\[
\nabla_U^2p
=\sigma'(\eta)\nabla_U^2\eta+
\sigma''(\eta)(\nabla_U\eta)(\nabla_U\eta)^\top .
\]

外层 sigmoid 不只是给 Hessian 乘常数，而是添加梯度外积项。其影响要独立定量分解，不能仅凭现有相关曲线推断哪个项“导致成功”。

Toy 的解析 T11-sig M006 在已保存的一条四维 joint_ramped 路径上也于 0° 最小；**这不是对整个 SO(4) 全局极小的证明**。

另外，Toy 已保存的 T11 成对扫描主要只测试 M003/M006/M017/M018/M021；不能声称全部 M001–M024 都有相同的六平面结果，更不能把它的解析后端当成已验证的 NN 后端。

---

## 10. 旋转维度：二维切片成功，不等于四维联合优化成功

四维正交旋转群 \(SO(4)\) 有

\[
\dim SO(4)=4(4-1)/2=6
\]

个自由度。两两旋转对应六条一维曲线；固定第四轴、三维子空间的旋转有三个自由度；完整四维联合旋转有六个自由度。

可由一组 Givens 乘积形成

\[
R=R_{12}(\theta_1)R_{13}(\theta_2)R_{14}(\theta_3)
  R_{23}(\theta_4)R_{24}(\theta_5)R_{34}(\theta_6),
\]

也可用 \(R(\theta)=\exp(\theta K)\) 构造固定反对称 \(K\) 的一维联合路径。

严格反例：

\[
L(\theta_1,\theta_2)
=\theta_1^2+\theta_2^2-3\theta_1\theta_2.
\]

每条坐标轴切片于 0 最小，但沿 \(\theta_1=\theta_2=t\) 的联合方向 \(L(t,t)=-t^2<0\)。**因此任何“六平面 6/6”都不能自动推出 \(A^*\) 在整个 \(SO(4)\) 是局部/全局最小值。**

对单一二维正交平面，在**符号/置换不变的 loss** 中 0° 与 90° 常对应等价坐标轴；一般三/四维联合路径的 90° 不保证是带符号置换，不能照搬这种等价规则。

---

## 11. 三个技术 gap（各自必须通过）

1. **Analytic gap（指标定义）**：真实解析函数上，正确 \(A^*\) 是否优于错位旋转？要求真实轴附近是 **minimum**、曲线非平坦、不同角度和不同模型上稳定；最终要扩展到 SO(4)。
2. **Data-only estimation gap（函数估计）**：不知道解析式，只给定 \((U,X,Y_{\rm formula})\)，能否用统一 estimator 恢复同一复杂度指标的角度结构？不能以预测 Y 准确代替高阶导数准确。
3. **Frozen-NN gap（解码器逼近）**：把 Y 换成冻结 NN 输出，使用**同一个** estimator，是否仍恢复同一损失曲线与真实轴？需要单独测量 NN Y 与 Formula Y 的差异。

另外，正式“发现轴”的验证必须考察 \(U\) 与 \(T\) 的**对应高相关、非对应低相关**，不仅是最大化任意某一维的相关。对符号、轴顺序以及必要的尺度约定保持一致。

**选择指标原则**：不要仅因为某个指标在一组数据中使 \(A^*\) 最小，就为单一模型调超参。先区分解析极小值正确性、矩阵推断稳定性、NN Y 路径、不同随机种子和模型外推，再考虑组合 loss。

---

## 12. 下一步最值得做的独立实验

| 优先级 | 研究任务 | 成功与失败都要保存 |
|---|---|---|
| P0 | 将 **Toy M017–M023** 的算子构造及其积分次序，在 C1/C2/C3/CR3 统一复现 | 原算子矩阵、保留阈值、JD 曲线、是否有近重复算子 |
| P0 | 将 CR3-07 四个“解析+Formula-Y 6/6”ZZ 指标移到 Toy T8/T10/T11 | 各模型/输出六平面、完整 SO(4) 路径与数值导数误差 |
| P1 | 对失败的 ZX、ZXX、ZZX 做估计器对照 | Polynomial order、ridge、局部回归、held-out 预测误差、局部二三四阶误差 |
| P1 | 对“多主作用 + 微小残余”做专门控制实验 | Top-k、capped L2²、soft L0、log-sum、row-mean/max 在不同残余幅度上的敏感性 |
| P1 | 评价尺度与测度稳健性 | X 原始/标准化、U SD 标准化、固定积分分布、多个采样区间、不同随机种子 |
| P2 | 从六条二维切片转向 SO(4) 多起点联合优化 | \(L(A)\) 减小、正确轴对应矩阵、off-diagonal 低相关；不让真 T 进入 loss |

避免一次把所有旧 rotation 套餐混在同一个运行器：**math/ 中每个编号文件夹是独立数学实验，rotations/ 中每个 rotation 套餐也应独立实现、独立输出。**

---

## 13. 快速结论

1. **稀疏性不是“L1 越小就越好”**：当前研究需要多主效应容许、对小残余敏感的相对 Top-k、Capped、Soft-L0、Log-sum 与 support-penalty；整体 Frobenius 不提供正交方向信息。
2. **ZX 行和列均有价值，但研究问题不同**：Z→X 的逐行稀疏与 X→Z 的逐列稀疏不可混为一谈；Mean/Max 的鲁棒性也不同。
3. **ZZ 不只非对角性**：ZZ 自身曲率变化、跨 Z 三阶/四阶导数、X 调制与二维非可加性分别对应不同复杂度来源。
4. **“解析成功” ≠ “矩阵成功” ≠ “NN 成功”**：当前 CR3 有 40 个解析六平面成功的输出—指标组合，19 个可保留 Formula-Y 端点极小值，4 个保留六平面形状相关；还没有全部六平面三路同时成功的单指标。
5. **M006 不是通用真理**：T11-sig 与 CR3 probability 均 6/6，但对应 logit 都只有 1/6；输出变换会改变 Hessian 结构。
6. **最有希望的互补方向**：CR3 的 ZZ mixed-third / local curvature 分布指标，以及 Toy 的 M017/M018 功能算子 JD；尚未证实它们能统一 C1/C2/C3/CR3。
7. **二维 6/6 不能代替 SO(4)**：最终需要联合旋转、独立验证和真正的无监督 \(A\) 优化。

### 关联代码和档案

- [原有 Hessian 研究历史](README.md)
- [Hessian 场、非线性、交互与主效应](README_HESSIAN_FUNCTIONAL_METRICS_AND_EFFECTS_20261109.md)
- [06：平均 Hessian 旋转、丰富稀疏与三路误差分解](../../math/06_cr3_hessian_rotation_landscapes/README.md)
- [06：稀疏 loss 的具体实现](../../math/06_cr3_hessian_rotation_landscapes/sparsity.py)
- [07：局部 ZZ/ZX 高阶复杂度、受控扫描](../../math/07_cr3_local_hessian_complexity/README.md)
- [07：局部指标计算](../../math/07_cr3_local_hessian_complexity/local_metrics.py)
- [07：单轴及双轴函数扫描](../../math/07_cr3_local_hessian_complexity/controlled_sweeps.py)
- Toy 参考实验：modlearn-rot-toy / catalog / README_METRICS.md（独立仓库，不与本仓库的 M01–M20 编号合并）。

> **结果溯源**：数值小结依据用户提供的 CR3 第 06 号 rich_sparsity_rescored.zip、第 07 号 monomial_u4_x4_lambu1e-06_lambx1e-06.zip，以及 modlearn-rot-toy.zip 的存档输出。它们是实验快照，不保证在此 GitHub 仓库中包含全部二进制结果。若后续重新采样/更改生成分布、正则化、输出变换或旋转参数化，应新建版本化结果记录，不直接覆盖本历史结论。
