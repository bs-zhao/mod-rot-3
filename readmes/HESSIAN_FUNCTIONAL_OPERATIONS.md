# Hessian 函数的两个正交操作：消除 W 依赖 × 改变输出形状

> 对应 [研究总框架](README.md)；与 [积分和求平均 Q](HESSIAN_INTEGRATION_AND_AVERAGING.md) 和 [Hessian→数值矩阵指标 M01–M20](HESSIAN_TO_MATRIX_METRICS.md) 并列。本文件是**数学分类与研究约定**，不是已经实现的代码接口。
>
> **核心思想**：给定 \(H(W)\)，\(W=(Z,X)\) 或 \((U,X)\)，有两个相互独立的维度：
>
> - **D：消除自变量 W 的依赖**。从函数/矩阵场转成固定数值，或保留部分自变量。
> - **P：改变输出形状（或变换矩阵结构）**。从 n×n 到标量、向量、低维/高维矩阵，或在 n×n 内变换。
>
> 二者可以单独做、一起做，也可以按不同先后顺序组合。已有 M01–M20 可理解为其中的**复合数值矩阵指标**，不应与所有可能的输出形式划等号。

## 1. 四个象限

设 \(f:\Omega\subseteq\mathbb R^n\to\mathbb R\)，\(H(W)=\nabla^2 f(W)\in\mathrm{Sym}(n)\)。在本项目也可以取 \(H_{UU}\)、\(H_{UX}\)、\(H_{XX}\) 子块，尤其 \(H_{UX}\) 可以是矩形。

| 是否还有自由变量 W | 仍是 n×n（或保持原子块形状） | 形状改变：标量、向量、其他尺寸矩阵 |
|---|---|---|
| **仍依赖 W** | \(H(W)\)、\(H(W)^2\)、逐元素 \(|H(W)|\) | \(\mathrm{tr}H(W)\)、\(\lambda_{\max}(H(W))\)、\(\mathrm{diag}H(W)\)、\(H(W)v\)、\(P^\top H(W)P\) |
| **不再依赖 W** | \(H(w_0)\)、\(E_Q H\)、逐元素 RMS/Q99、\(E_Q[HH^\top]\) | \(\mathrm{tr}H(w_0)\)、\(E_Q[\mathrm{tr}H]\)、\(\sup_W\Vert H(W)\Vert_2\)、\(E_Q[\mathrm{diag}H]\)、\(\mathrm{Cov}_Q(\mathrm{vec}H)\) |

**一个仅对 X 的条件积分** \(E_X H(Z,X)\) 仍含 Z，因此不属于“彻底消除 W”象限；可作为中间结果。

## 2. D：消除变量依赖的操作

| ID | 方法 | 典型定义 | 条件/风险 |
|---|---|---|---|
| D01 | **在某点评价** | \(g(w_0)\) | 依赖 w0；无需概率分布 |
| D02 | **有限加权求和** | \(\sum_\ell\omega_\ell g(w_\ell)\) | 样本点、权重、是否为成对设计 |
| D03 | **期望/先验积分** | \(E_Q[g(W)]\) | 必须指定测度 Q，并保证可积 |
| D04 | **条件期望/部分积分** | \(E[g(W)\mid Z]\) | 可能留下自由变量 |
| D05 | **轨迹/时间平均** | \(T^{-1}\sum_tg(W_t)\) | 轨迹分布不一定等于研究先验 |
| D06 | **指数滑动平均** | \(M_t=\rho M_{t-1}+(1-\rho)g(W_t)\) | 给定 t 输出数值，依赖时间序列 |
| D07 | **取极限** | \(\lim_{t\to\infty}g(W_t)\) | 极限未必存在 |
| D08 | **上确界/下确界** | \(\sup_{w\in\Omega}g(w)\)、\(\inf g\) | 对标量或逐元素可定义；矩阵无天然全序 |
| D09 | **分位数/稳健统计** | median、Q90/Q99、trimmed mean | 需先声明元素、谱还是标量作用对象 |
| D10 | **矩阵序上界/下界** | 找 M 满足 \(H(w)\preceq M\)（全部 w） | 半正定 Loewner 序是偏序；一般不存在唯一最小上界，需要另设目标 |

**D 与既有积分维度 Q 的关系**：Q 决定评价点、Z/X 测度、权重、积分计算算法、有效域；D 决定怎样把函数的 W 依赖消除。D02/D03/D04 的具体实施依赖 Q；D01（点评价）或 D08（最大值）不必先有一个概率先验。两者应分别记录，不要简单都叫“平均”。

## 3. P：逐点改变 Hessian 的输出

### P0：输出维数保持 n×n，但修改矩阵含义

- \(H^2=HH\)、\(H^k\)：矩阵乘法，**不同于**逐元素 \(H\odot H\)。
- \(H_+=V\,\mathrm{diag}(\max(\lambda_i,0))V^\top\)：正曲率矩阵。
- \(H_-=V\,\mathrm{diag}(\max(-\lambda_i,0))V^\top\)：负曲率幅度矩阵。
- \(|H|_{\mathrm{op}}=(H^2)^{1/2}\)：对称矩阵的谱绝对值，**不同于**逐元素 \(|H|_{\mathrm{el}}\)。
- threshold、正则化 shrinkage、逐元素绝对值/平方：保持形状但通常不保留原矩阵的谱性质或正交等变性。

### P1：输出变成标量函数

| 算子 | 定义与数学含义 | 注意 |
|---|---|---|
| Trace | \(\mathrm{tr}H(W)=\Delta f(W)\) | 线性；旋转不变 |
| Determinant | \(\det H(W)\) | 特征值乘积，可能为负或零 |
| Logdet | \(\log\det H(W)\) | 实数需 det>0；通常要求 SPD 才有标准曲率解释 |
| 最大/最小特征值 | \(\lambda_{\max},\lambda_{\min}\) | 局部最大正曲率、最小代数特征值 |
| Norm | Frobenius、spectral、nuclear norm | 总曲率强度/最大奇异值/奇异值和 |
| 方向二次型 | \(v^\top H(W)v\) | 固定方向 v 上的曲率；选择方向可能泄漏真 T |
| Rank / Morse index | \(\mathrm{rank}(H)\)、负特征值数量 | 受零特征值阈值影响 |
| Condition number | \(\kappa_2(H)=\sigma_{\max}/\sigma_{\min}\) | 不可逆为∞；**一般不能用 λmax/λmin，除非 SPD** |

### P2：输出变成向量

- \(\mathrm{diag}H(W)\in\mathbb R^n\)：各原坐标轴上的二阶曲率。
- \(\lambda(H(W))\in\mathbb R^n\)：排序后的特征值谱。
- \(H(W)v\in\mathbb R^n\)：给定方向 v 的 Hessian-vector product。
- 最大特征向量等：向量符号不唯一，重根或交叉时方向可能不连续。

### P3：变成不同尺寸矩阵

- **子块** \(H_{SS}(W)\in\mathbb R^{k\times k}\)：固定子集 S。
- **子空间投影** \(P^\top H(W)P\in\mathbb R^{k\times k}\)：固定 \(P\in\mathbb R^{n\times k}\)。
- **方向矩阵作用** \(H(W)P\in\mathbb R^{n\times k}\)。
- **Schur 补** \(H_{AA}-H_{AB}H_{BB}^{-1}H_{BA}\)：需声明分块，且 \(H_{BB}\) 可逆。
- **Kronecker/低秩分解**：得到若干小矩阵因子，但通常是近似且可能不唯一；K-FAC 更多是 Fisher/Gauss–Newton 类型的曲率近似，并非任意 Hessian 的精确分解。

## 4. D 与 P 的两种顺序：不能默认交换

对于某个逐点形状算子 \(\Phi_P\)，以及一个整体消除 W 的算子 \(\mathcal R_{D,Q}\)，可能有两种路径：

\[
\boxed{
P\to D:\quad \mathcal R_{D,Q}[\Phi_P(H(W))]
}
\]

\[
\boxed{
D\to P:\quad\Phi_P(\mathcal R_{D,Q}[H(W)])
}
\]

**固定同一 Q 且 P 是线性算子**时，常有交换性：

\[
E_Q[\mathrm{tr}H]=\mathrm{tr}(E_QH),\qquad
E_Q[\mathrm{diag}H]=\mathrm{diag}(E_QH),
\]

\[
E_Q[P^\top HP]=P^\top(E_QH)P
\quad(P\text{ 固定，不随 W 变化}).
\]

**非线性操作一般不交换**：

\[
E_Q[\lambda_{\max}(H)]\geq\lambda_{\max}(E_QH),\qquad
E_Q[\Vert H\Vert_F]\geq\Vert E_QH\Vert_F,
\]

\[
E_Q[HH^\top]\ne(E_QH)(E_QH)^\top,\qquad
E_Q[\det H]\ne\det(E_QH).
\]

**最小反例**：W 以相等概率取 ±1，令

\[
H(W)=\begin{pmatrix}W&0\\0&-W\end{pmatrix}.
\]

则 \(E_QH=0\)，但 \(E_Q[H^2]=I_2\)，且

\[
\lambda_{\max}(E_QH)=0,\qquad
E_Q[\lambda_{\max}(H)]=1.
\]

同一个完全准确的解析 Hessian，只改变操作先后次序，就得到不同的数值指标；**不涉及神经网络估计误差**。

## 5. 场级联合统计：不一定能拆成一次逐点 P + 一次 D

例如

\[
C_H=\mathrm{Cov}_Q(\mathrm{vec}(H(W)))\in\mathbb R^{n^2\times n^2}
\]

需要同时计算多个矩阵矩与均值；条件 ANOVA、构造多组功能算子后的联合对角化也是类似情况。这些应标注为 **field-joint functional**，不是简单的单次逐点算子。

已有 [矩阵指标 a=M01–M20](HESSIAN_TO_MATRIX_METRICS.md) 中，M15 Gram、M18 Hessian 元素协方差、M19 条件 ANOVA、M20 高阶 Hessian 变化均属于该扩展范围。**a 是具体复合指标；D/P 则是更一般的数学分类，不必推翻原编号。**

## 6. 对 Mod-Rot 的特别意义

- **旋转不变量可能根本无法辨识方向**。在正交 U=RZ 下，局部 ZZ Hessian 满足 \(H_{UU}=RH_{ZZ}R^\top\)。其 trace、det、特征值、Frobenius 范数在固定点/一致推前测度下都不随 R 改变。若把它们单独定义为旋转简单性 loss，通常会得到**平坦角度曲线**，不能恢复真实参数轴。diag、offdiag、逐元素稀疏性等则通常依赖坐标轴，可能提供方向信息。
- C1/C2/C3 Logit 对真 T 线性，所以真实 \(H_{TT}(\eta)=0\)。只用它的 trace/eigenvalue/范数无法产生非平坦旋转区分；需要考虑 TX/ZX 或其它函数作用结构。
- CR3 的共享 500 Xset 与真实 T 先验要分开定义 Q。把先验 T 直接当成 NP 的 Z 分布会改变评价对象。
- 若定义域凸、f 二阶连续可微，\(\sup_\Omega\Vert H(W)\Vert_2\) 在常见正则条件下界定梯度的 Lipschitz 常数；\(\inf_\Omega\lambda_{\min}(H)\) 仅在正时才给出全域强凸保证。不能脱离定义域/正则条件直接解释为优化常数。
- 投影、分块、特征向量方向的选择不能借助真 T/解析 oracle 直接决定最终盲旋转最优方向；需要做独立的 oracle-diagnostic 和真正的 ZXY-only 优化。

## 7. 与既有研究层级和代码的兼容

原有独立因素仍为：

- **h/e**：Hessian 来源与数值估计器；
- **Q**：积分的点位、测度、权重、数值算法；
- **a=M01–M20**：特定的 Hessian→固定数值矩阵统计指标；
- **k**：输入数值矩阵之后的标量简单性。

新增分类让**不必生成固定 n×n 的路径**也能够清楚表达：D=消除 W 的操作、P=输出形状/结构操作、order=P→D / D→P / 多步联合。D 不等于 Q：例如在点评价、求极限、求上界时没有通常意义的概率积分。

建议每次输出记录：

| 字段 | 内容 |
|---|---|
| variable_reduction_id | D01–D10；消除 W 的方式，是否仍有 Z/X 自由变量 |
| integration_id | Q；只有需要时设置被试/X 测度、权重和数值求积 |
| shape_operator_id | P0/P1/P2/P3 与具体 trace/diag/projection 等 |
| operation_order | P_then_D / D_then_P / alternating / field_joint |
| input_hessian_block | UU / UX / XX / full；原始矩阵尺寸 |
| output_shape | n×n / scalar / vector / k×k / n²×n² 等 |
| remaining_variables | none / Z / X / Z_and_X |
| invariance_class | 正交旋转不变、等变、坐标依赖等 |
| scalar_simplicity_metric_id | 只有在另定义需要最小化的 loss 时才填 |

**不要因为某个操作输出了一个数，就自动称其为简单性 loss。** 它可能仅是数学摘要；真正的 loss 还须明确优化方向、归一化、适用条件与真轴可辨识性。现有 rotations/ 仍保持每个旋转套餐独立文件夹的安排。