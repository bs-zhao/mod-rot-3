# 从 Hessian 计算复杂度与稀疏性：全指标归档、Tail-L2 与文献（2026-11-09）

> **本文件性质**：截至本轮聊天的讨论归档，而非数学上已证明的普适最优解，也不是所有候选已经在 C1/C2/C3/CR3 上通过实验的声明。
>
> **相关历史和方法文档**：[历史总纲](README.md)｜[Hessian 数学变换与主效应](README_HESSIAN_FUNCTIONAL_METRICS_AND_EFFECTS_20261109.md)｜[研究总框架](../README.md)｜[20 种矩阵泛函](../HESSIAN_TO_MATRIX_METRICS.md)｜[变量消除与形状变化](../HESSIAN_FUNCTIONAL_OPERATIONS.md)｜[积分方案](../HESSIAN_INTEGRATION_AND_AVERAGING.md)。

## 1. 从什么出发、什么叫“复杂度”？

设已知光滑解析函数 f: Ω⊂Rⁿ→R；W=(W₁,…,Wₙ)，H(W)=∇²f(W)，Hᵢⱼ(W)=∂ᵢ∂ⱼf(W)。

“简单”的至少六种不同含义：① 单维自身效应较线性；② 不同维度之间函数交互较少；③ 一个维度只与少数其他维度相联，**小但非零的剩余作用也必须被惩罚**；④ 多个局部 Hessian 共享同一个简洁的方向基；⑤ 曲率沿 W 改变得较慢（高阶变化小）；⑥ 整个 f 更接近加性/低阶多项式/少交互函数类。**它们不是一个具有统一定义的“复杂度”**。

Hessian 非对角 Hᵢⱼ 描述 Wⱼ 如何影响 Wᵢ 对 f 的边际作用；它**不是** Wᵢ 与 Wⱼ 的统计相关。对角 Hᵢᵢ 是沿 Wᵢ 的曲率，但 Hᵢᵢ 还可能依赖 W₋ᵢ，因此不一定是可独立分离的单维主效应。

固定输出尺度（logit / p）、坐标单位与输入测度 Q 后，才有可比较的数值大小。只加仿射函数不改变 Hessian，因此 Hessian-only 指标无法刻画线性主效应强弱。

### 1.1 三个独立数学层级

1. **积分和测度 Q**：在哪些 W 点、用什么概率/权重、精确求和或 MC/quadrature。先只对部分坐标积分往往仍保留剩余变量。
2. **从 Hessian 场到矩阵/标量的算子 a**：逐元素 RMS、平均、Gram、分位数、谱算子，或直接先求局部标量再平均。
3. **最终标量简单性损失 k**：从数值矩阵/局部场计算 offdiag、Tail-L2、稀疏度、joint-diagonalization、平滑性等。希望**越小越简单**，但不能由此推定真参数轴最小。

另有两个正交操作：D=消除 W 依赖；P=改变输出形状（原 n×n、向量、标量或其他矩阵尺寸）。例如 E[tr H]=tr E[H]，但 E[λmax(H)]≥λmax(E[H])；**先变换再积分**与**先积分再变换**通常不同。

## 2. 保留同形数值矩阵或改变形状：M01–M20 汇总

符号：E=E_Q；⊙ 是逐元素乘法；矩阵乘法 H²=HH；|H|el 是逐元素绝对值。

| ID | 数学运算 | 主要保留什么信息 |
|---|---|---|
| M01 Signed Mean | E[H] | 有符号平均；会正负抵消 |
| M02 Mean Absolute | E[&#124;H&#124;el] | 元素平均曲率绝对幅度 |
| M03 RMS | √E[H⊙H]（逐元素） | 曲率能量、无符号抵消 |
| M04 Mean Square | E[H⊙H] | 每个元素二阶矩 |
| M05 Variance | Var(Hᵢⱼ) 逐元素 | 局部曲率的变异 |
| M06 Standard Deviation | √Var(Hᵢⱼ) | 曲率变异的原单位 |
| M07 General Lp | (E&#124;Hᵢⱼ&#124;ᵖ)^(1/p) | p 不同则对强值/尾部敏感不同 |
| M08 Median Absolute | median(&#124;Hᵢⱼ&#124;) | 典型作用、抗极端值 |
| M09 Absolute Quantile | Q90/Q99(&#124;Hᵢⱼ&#124;) | 尾部强作用 |
| M10 Maximum/Supremum | supΩ &#124;Hᵢⱼ(W)&#124; | 最坏情形 |
| M11 Exceedance | Pr(&#124;Hᵢⱼ&#124;>τ) | 大于阈值的作用比例 |
| M12 Sign Consistency | &#124;E[sign(Hᵢⱼ)]&#124; | 符号稳定性，不是强度 |
| M13 Positive/Negative parts | E[(Hᵢⱼ)₊]、E[(-Hᵢⱼ)₊] | 正负曲率分别留存 |
| M14 Trimmed/Winsorized | 截尾/缩尾后平均 | 稳健性 |
| M15 Row Gram | E[HHᵀ] | 行方向综合作用 |
| M16 Column Gram | E[HᵀH] | 列方向综合作用 |
| M17 Spectral absolute | E[(H²)^(1/2)]，H 对称 | 整体矩阵谱的绝对强度 |
| M18 Covariance of vec H | Cov(vec H) | 各 Hessian 元素共同变化，尺寸改变 |
| M19 Conditional/ANOVA | E[Var(Hᵢⱼ&#124;Wₖ)] 等 | 曲率由哪些输入调节 |
| M20 Higher derivative aggregation | E&#124;∂ₖHᵢⱼ&#124;、RMS(∂ₖHᵢⱼ)、四阶 | 曲率改变与更高阶交互 |

可计算而未必是复杂度损失的其他对象：H(W₀)、时间/路径平均、指数移动平均、极限、矩阵上下界、tr H=Δf、det H、logdet、特征值、条件数、Hessian-vector product、方向二次型、主子块、Schur complement、投影和 Kronecker 近似。

**基本对照**：Hᵢⱼ(W) 取 ±5 各一半时，E Hᵢⱼ=0，Mean Abs=5，RMS=5。因此 M01 的零不表示没有局部交互；RMS²=(E Hᵢⱼ)²+Var(Hᵢⱼ)。同一 f 可对应不同的 M，矩阵摘要本身也是独立研究变量。

## 3. 从 Hessian 计算标量复杂度：所有讨论过的指标族

### 3.1 强度、交互、解耦和自身非线性

| 家族 | 代表损失（越小越简单） | 适用目的与风险 |
|---|---|---|
| 总 Hessian 能量 | E&#124;&#124;H&#124;&#124;F²、&#124;&#124;E H&#124;&#124;F² | 曲率总量小；正交旋转下前者不变 |
| 逐元素 L1 | Σᵢⱼ E&#124;Hᵢⱼ&#124;，或 Σᵢⱼ&#124;Mᵢⱼ&#124; | 小系数较显著地受罚，但易偏向整体缩放小 |
| Off-diagonal L1 | Σᵢ≠ⱼ E&#124;Hᵢⱼ&#124; | 函数交互少；可对局部或汇总矩阵评分 |
| Off-diagonal L2 | Σᵢ<ⱼ E[Hᵢⱼ²] | 二阶交互能量小；单张对称矩阵可对角化 |
| Offdiag / Total | &#124;&#124;offdiag(M)&#124;&#124;F²/(&#124;&#124;M&#124;&#124;F²+ε) | 相对解耦；分母与坐标性质需固定 |
| Offdiag / Diag | &#124;&#124;offdiag(M)&#124;&#124;F²/(&#124;&#124;diag(M)&#124;&#124;²+ε) | 交互相对自曲率少；分母可病态 |
| Diag self-curvature | Σᵢ E[Hᵢᵢ²] 或 Σᵢ E&#124;Hᵢᵢ&#124; | 自身非线性小；可能与少交互冲突 |
| Direction-specific interaction | Σⱼ≠ᵢ E[Hᵢⱼ²] | 第 i 维参与其他维度的交互多少 |
| Row/column sparsity | 对每行/列取 L1、Tail-k、Soft-L0 等 | 一个维度应只影响少数其他维度 |
| Group sparsity | Σᵢ&#124;&#124;rowᵢ(M)&#124;&#124;₂、列组惩罚 | 少数行/列整体激活，与单元素支持不同 |
| Multiple Hessian joint diagonalization | Σ_c&#124;&#124;offdiag(R H_c Rᵀ)&#124;&#124;F² | 同一 R 简化多个局部/条件矩阵 |
| Local nonlinear penalty | E[S(H(W))] | 保留局部结构，未必等于 S(E[H]) |

注意“矩阵 L1”在本文特指逐元素绝对值和；这与数学上诱导 1-范数（最大绝对列和）不同。只优化完整 Hessian Frobenius 范数或 ZZ 谱、trace、det，无法辨别正交旋转方向。

### 3.2 Tail-L2、Tail-L1：特别强调“主作用之后的小残余”

设非负作用向量 v 的绝对值降序为 a₁≥⋯≥a_d≥0，预先指定要保留的主要作用数 k，定义 tailₖ(v)=(a_{k+1},…,a_d)。

\[
\boxed{
\mathrm{TailL1}_k(v)=\sum_{j>k}a_j,\quad
\mathrm{TailL2}_k(v)=\left(\sum_{j>k}a_j^2\right)^{1/2},\quad
\mathrm{TailL2sq}_k(v)=\sum_{j>k}a_j^2.
}
\]

归一化版（ε>0 防止除零）：

\[
\mathrm{RelTailL2}_k=\frac{\mathrm{TailL2}_k}{\|v\|_2+\epsilon},\qquad
\mathrm{RelTailL2sq}_k=\frac{\mathrm{TailL2sq}_k}{\|v\|_2^2+\epsilon}.
\]

它们可理解为**最佳 k-项稀疏近似的残差**。保留 k 个主要作用之后，剩下的“微小但非零”项必须尽量小。本文将这一数学残差应用于 Hessian 行/列/交互指纹作为候选旋转损失，**属于研究目标设计，不代表有文献已证明它能恢复理论轴**。

更严格地，令 \(\Sigma_k=\{u\in\mathbb R^d:\|u\|_0\le k\}\)，则

\[
\boxed{\sigma_k(v)_p=\inf_{u\in\Sigma_k}\|v-u\|_p
=\left(\sum_{j>k}a_j^p\right)^{1/p},\qquad 1\le p<\infty.}
\]

这说明 Tail-L1/Tail-L2 **确实**是成熟的 best k-term approximation error，而不是为特定数值例子发明的公式；**但把它作为某一行 Hessian 的复杂度、让真参数轴恰好使其最小，则是另一件需要验证的研究假设**。非线性近似的经典理论综述见 DeVore (1998)。排序遇到同值时最优 k 项选择可不唯一，但残差值相同。


#### 对用户指定的两组数值实际检验（k=2）

A=[2, 2, 0.01, 0.001]；B=[1, 0.5, 0.01, 0.01]。

| 指标 | A | B | 是否认为 B 更复杂 |
|---|---:|---:|---|
| 总 L1 | 4.011 | 1.520 | **否**（反向） |
| 总 L2 | 2.828445 | 1.118123 | **否**（反向） |
| **Tail-L1 k=2** | **0.011000** | **0.020000** | **是** |
| **Tail-L2 k=2** | **0.0100499** | **0.0141421** | **是** |
| **Tail-L2² k=2** | **0.000101** | **0.000200** | **是** |
| **Relative Tail-L2 k=2** | **0.003553** | **0.012648** | **是** |
| Relative Tail-L2² k=2 | ≈0.00001262 | ≈0.00015997 | **是** |
| Hoyer（大=更稀疏） | ≈0.58191 | ≈0.64058 | **否**，偏 B |
| Shannon entropy of normalized &#124;v&#124;（小=集中） | ≈0.71100 | ≈0.70731 | **否**，略偏 B |
| exp(entropy) 有效支持（小=集中） | ≈2.0360 | ≈2.0285 | **否** |
| Gini（大=更集中） | ≈0.49782 | ≈0.56908 | **否**，偏 B |

**重要的反向结果**：传统 Hoyer/Entropy/Gini 主要度量**整条向量的集中性**，B 的两个强项本身比例更不均匀，因此即便 B 有两条 0.01 的剩余联系，也可能被判为更集中、更“稀疏”。你的定义是“允许两个主作用后额外作用要少”，因此**Top-k residual / tail**比不加限制的全局集中度更直接。

**第二个已讨论例子（k=1）**：C=[2,0.011,0.001]、D=[2,0.001,0.001]，TailL1(C)=0.012>D=0.002；TailL2(C)≈0.0110454>D≈0.0014142；TailL2sq(C)=0.000122>D=0.000002。它们均认为 C 更复杂，但平方损失使小效应的梯度更小。

**关键依赖**：若把第一例 k 改为 1，A 的尾部含有 2、B 的尾部含有 0.5，比较会反转。必须在不使用真参数答案的条件下预先选 k，或对 k∈{1,…,d−1} 做敏感性与自适应复杂度惩罚；不能为了满足例子事后改 k。

建议区分：
- **Raw Tail-L1/L2**：绝对尾部大小，明确量纲，适合惩罚微小但真实的剩余链接；
- **Relative Tail-L2**：移除统一输出缩放影响，可能对微弱总作用过度敏感；
- **Tail-L2sq**：比 Tail-L1 对极微小残余惩罚弱（但这两组 k=2 仍排序正确）；
- **Row-wise / Column-wise / Global**：对每行保留 k 个与全矩阵只保留 k 个不等价；应按科学假设分别定义。

### 3.3 软支撑、L0、非凸惩罚、阈值和自适应 k

令 a_j=|v_j|≥0、τ>0。以下多数越小表示越少有效项：

| 名称 | 典型公式 | 目的与限制 |
|---|---|---|
| Exact L0 | Σ_j 1(a_j≠0) | 数量意义最直接；噪声和微小数值误差会使其失效 |
| Hard support | Σ_j 1(a_j>τ) | 超阈值项数；非连续，结果对 τ 敏感 |
| Rational soft-L0 | Σ_j a_j²/(a_j²+τ²) | 饱和的连续近似支持 |
| Exponential soft-L0 | Σ_j[1−exp(−a_j/τ)] | 近似支持数，零附近保留斜率 |
| Capped-L1 | Σ_j min(a_j/τ,1) | 大项饱和，小项持续被罚 |
| Log-sum | Σ_j log(1+a_j/τ) | 非凸近似 L0，小项有非零斜率 |
| lp quasi-norm | Σ_j a_j^p，0<p<1 | 促进稀疏；零点附近梯度可能奇异 |
| L1 | Σ_j a_j | 稀疏凸惩罚，同时惩罚主项 |
| Reweighted L1 | Σ_j ω_j a_j，ω_j≈1/(a_j_old+τ) | 迭代偏向小支持；权重历史必须固定 |
| Penalized k-support | min_k{TailL2sq_k+λk} | 让多保留一个主项也付成本 |
| Top-k shoulder penalty | 对第 k+1 项和其余尾部不同加权 | 着重压掉临界多余联系；项目候选组合 |
| Row/column group Lasso | Σ_i &#124;&#124;M_i,:&#124;&#124;₂ | 少数整行/列激活，非元素支撑稀疏 |

例：绝对阈值 τ=0.005 时，A 的两条尾项中 0.01 超阈值而 0.001 不超；B 的两条 0.01 都超。若 τ=0.02，二者的小尾项全被忽略。**阈值是定义，不是从数据自动得出的“真复杂度”**。相对阈值 τ∝row RMS 可能改变排序，因为大主项也改变了门槛。

### 3.4 熵、集中度、有效支持与组/谱稀疏

对 p_j=a_j/Σa（全零另定义）：

- **Hoyer**：(\(\sqrt d-\|v\|_1/\|v\|_2\))/(\(\sqrt d-1\))；**高=稀疏**，做 loss 应用 1−Hoyer。
- **Shannon**：−Σ p_j log p_j；**低=集中**；**effective support**=exp(Shannon)。
- **Rényi α**：log(Σp_j^α)/(1−α)；α 改变对主项/尾部的敏感度。
- **Gini**：Σ_{jk}|a_j−a_k|/(2dΣa_j)；**高=集中**，做 loss 用 1−Gini。
- **Participation ratio**：(Σa_j)²/(Σa_j²)；越小通常越集中。
- **Top-k energy share**：(Σ_{j≤k}a_j²)/(Σa_j²)；越高越集中；1−此量即 squared relative Tail-L2。
- **Matrix singular-value tail**：Σ_{j>k}σ_j(M)²；矩阵最佳低秩近似残差，**不等于元素 top-k**。
- **Effective rank / spectrum entropy / nuclear norm / stable rank**：反映有效维度或低秩，未必惩罚特定坐标轴的多余链接。
- **Group support**：按整行/整列、按变量组、按条件组汇总，而非逐元素。
- **Joint diagonalization**：多个条件/局部矩阵共同压低非对角，避免“单点总能对角化”的空泛结论。

熵/Hoyer/Gini 的尺度不变性质与 Tail 原始量纲不同。**没有数学定理保证某种通用集中度指标会给出指定的两组向量排序**；以稀疏支持和尾项残差为科学目标时，必须把它们写进定义，而非单纯期望熵自动产生相同排序。

## 4. 单维非线性、两两交互、整体交互：可计算为数字的体系

### 4.1 自身曲率的总体、主效应和调制（不同概念）

\[
N_i^{total}=\sqrt{E[H_{ii}^2]},\qquad
g_i(w_i)=E[H_{ii}(W)|W_i=w_i].
\]

\[
N_i^{mean}=\sqrt{E[g_i(W_i)^2]},\qquad
N_i^{mod}=\sqrt{E[\operatorname{Var}(H_{ii}|W_i)]}.
\]

**精确的条件方差恒等式**：

\[
(N_i^{total})^2=(N_i^{mean})^2+(N_i^{mod})^2.
\]

在独立输入、积分/求导可交换时，mean 可以解释为 partial dependence 主效应的二阶曲率；**相关输入时不能无条件这样解释**。例如 f=w₁²w₂、独立 E[w₂]=0，整体自曲率可以非零，但平均主效应曲率为零。单维三阶/四阶 ∂ᵢHᵢᵢ、∂ᵢᵢHᵢᵢ 以及沿单轴扫描的 TV/variance，更适合区分二次与三次或更高阶局部形状。

### 4.2 两两与整体交互

\[
I_{ij}^H=\sqrt{E[H_{ij}(W)^2]},\quad i\ne j,\qquad
I_{\rm all}^H=\sum_{i<j}E[H_{ij}^2].
\]

混合二阶导数非零表明函数有交互；例如 f=w₁w₂w₃ 的 H₁₂=w₃，可能有明显的两维**参与高阶交互**，而独立零均值 W 下纯二阶 Sobol S₁₂=0。更高阶分辨可看 ∂ₖHᵢⱼ 与四阶导。

如果定义域为连通的矩形、f 二阶连续可微，所有 i≠j 的 Hᵢⱼ 恒为零等价于 f 在原坐标系可加分离。**只在均值为零不充分**。

### 4.3 函数值层面对照——不是 Hessian 指标

若独立输入且 f 平方可积，Sobol / functional ANOVA 分解可定义：

\[
S_{ij}=\mathrm{Var}(f_{ij}(W_i,W_j))/\mathrm{Var}(f(W)),
\quad
I_{\rm global}^{ANOVA}=1-\sum_iS_i.
\]

前者是**纯二阶**方差贡献；还可构造所有包含 i,j 的更高阶贡献之和。crossed-DGSM 使用 E[Hᵢⱼ²] 并在适当 Poincaré/独立测度条件下与交互 Sobol 上界关联，**两者不是恒等式**。

另有：单变量主效应最佳线性残差、整个 f 最佳加性模型残差、Friedman H-statistic、ALE、controlled 2D additivity gap、三点/四点函数差分、函数类最低阶数、MARS/GAM 复杂度、频域 DCT 高频比例、单调性违例、gradient covariance/Gram、HSIC/dCor/MI/JADE/ICA。它们可辅助研究效应或依赖，但**非全部由 Hessian 唯一决定**；输入统计独立也不等于功能可加。

## 5. 非局部曲率与高阶变化：此前讨论的完整指标族

| 家族 | 示例 | 衡量什么 |
|---|---|---|
| Hessian sign cancellation diagnostic | E Hᵢⱼ vs RMS(Hᵢⱼ)、比值 | 被 signed mean 隐藏的真实作用 |
| Third derivative own vs other | ∂ᵢHᵢᵢ、∂ⱼHᵢᵢ(j≠i)、∂ₖHᵢⱼ | 自身形状变化 / 别的变量调制 / 高阶交互 |
| Fourth derivative | ∂ₖₗHᵢⱼ，RMS/abs | 更高阶曲率变化 |
| Hessian conditional ANOVA | Var conditional、subject/X/interaction | 变化来自哪组输入 |
| Controlled axis curvature | 固定其他 W 扫描 Wᵢ 的 Hᵢᵢ 方差、TV | 避免其他维度同步变化混入 |
| Local linear/quadratic/cubic residual | 受控切片最佳低阶多项式拟合残差 | 非线性多项式阶数 |
| 2D controlled additivity gap | 对二维 f 网格双中心的残差 | 非局部交互、不同于总体 Sobol |
| Derivative smoothness/fingerprint | Hessian 稳定性、Sobol/HSIC 等互补描述 | 函数作用的正则性/轴辨识 |
| Local-to-global consistency | across-subject / across-X / cross-angle stability | 验证指标稳定，**不等于指标本身** |

例：f(w)=w²→H=2；f(w)=w³→H=6w。若 w 分布对称，后者 E H=0，却有更大 RMS 或非零三阶导 6。评价“高次更复杂”时必须声明输入范围、尺度与指标（不能仅凭一次平均 Hessian）。

## 6. 旋转、尺度、角度、源一致性：数学与验证规范

对正交 U=RZ（X 固定时），ZZ 块 H_UU=R H_ZZ Rᵀ；ZX 块 H_UX=R H_ZX。所有表达都在**相同物理点、对应推前测度**上成立。

- \(\mathrm{tr}\)、det、eigenvalues、rank、Frobenius、\(\|H_{ZX}\|_F\) 对此类正交变换不变；单独用它们作为 Loss 会是平坦旋转曲线；
- offdiag、对角能量、rowwise top-k tail、soft-L0、局部元素稀疏等通常随轴改变，**但真轴极小仍需实验验证**；
- 多张局部/条件 Hessian 共同对角化不同于单张可对角化；
- RMS、Mean Abs、Q99 等元素非线性指标必须**先变换每个局部 Hessian，再聚合**，不能把原始 RMS 矩阵直接 RMRᵀ 当作旋转后的 RMS；
- 标准化输入单位与输出量纲（按导数阶数乘以各输入 SD，注意重复索引）；同时保留 raw/standardized 曲线；
- 真实轴的极小值需要非平坦的曲线、允许符号/置换等价；检查六个 2D 平面、3D/4D 联合 SO(d)，并检查 U–T 恢复的非对应 off-diagonal 低相关；
- 解析式上通过并不保证从 WY 估计器/NN 都通过；如果 formula-Y inferred 也失败，**先怀疑 inference gap**，不要全归咎于 decoder；
- logit、sigmoid probability 和 Bernoulli choice 的目标不同：\(H_p=\sigma'(\eta)H_\eta+\sigma''(\eta)\nabla\eta\nabla\eta^\top\)；choice 应把条件均值 p 作为光滑目标，不能将一次 0/1 结果当可导响应；
- 若从 WY 学函数，只凭预测准确不保证导数/Hessian 准确；一些线性平均导数可用 Stein 积分分部直接估计，而 RMS/平方曲率通常更困难。

**最重要的原则**：真生成模型可能本就含有交互。一个完全合理的“最少交互”旋转，不必是生成理论参数轴；真实轴可辨识是需要检验的假设而非定义。

## 7. 和仓库已实现实验的对照（不虚构成功）

- [math/06_cr3_hessian_rotation_landscapes](../../math/06_cr3_hessian_rotation_landscapes/README.md)：旋转损失曲线与稀疏矩阵指标；
- [math/07_cr3_local_hessian_complexity](../../math/07_cr3_local_hessian_complexity/README.md)：**已在代码列入** Tail1/Tail2/Tail2sq k=1–4、rowwise soft-L0/capped/logsum、Hoyer/Shannon/Rényi/Gini、ZZ/ZX 高阶、条件变异与受控 sweep；具体 registry 以生成的 metric_catalog.csv 为准；
- [math/08_cr3_functional_sparsity](../../math/08_cr3_functional_sparsity/README.md)：解析 Hessian 与函数值 3×3 非加性间隙对照；
- [math/09_cr3_three_gap_reaudit](../../math/09_cr3_three_gap_reaudit/README.md)：对先前指标准确划分 analytic / formula-Y fitted / NN-Y fitted 三层差异；
- [Hessian 数值估计方法](../HESSIAN_FROM_ZXY_METHODS.md)：E01–E14 的候选及已实现范围。

这些说明“实现过什么”，**不等于**有跨 C1/C2/C3/CR3 与全部 2D/3D/4D 旋转的通过证据。本 README 不从未读取的运行结果虚构成功率。

## 8. 推荐的可检验候选与验收规则

从真实解析 Hessian 出发先做独立对照（不混合权重），重点测试：**逐行 Tail-L1 k、Tail-L2 k、Tail-L2² k、Relative Tail-L2 k、soft-L0、capped-L1、logsum、lp(0<p<1)、offdiag L1/L2、ZZ 自身 3/4 阶变化、条件 ANOVA、joint diagonalization、controlled additivity gap**。对每个指标保存解析正确坐标轴是否**极小**、曲线平坦性、旋转等变性、输出尺度不变性、k/τ 稳健性。之后再跨越 formula-generated WY 与 NN-generated WY 两道鸿沟。

三个关键禁忌：**不能仅为两个例子挑 k/阈值；不能把 NN 函数预测精度当作 Hessian 准确性；不能把谱不变量的平坦 loss 判作成功。**

## 9. 参考文献（成熟方法 vs 本项目候选要区分）

以下论文支持稀疏优化、函数导数、Sobol/ANOVA、估计器等方法族；**没有哪篇证明本文所有候选的真实理论轴可恢复性**。

1. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS B, 58*, 267–288. [DOI](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). L1/坐标稀疏。
2. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS B, 67*, 301–320. [DOI](https://doi.org/10.1111/j.1467-9868.2005.00503.x). L1/L2 组合。
3. Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008). Enhancing sparsity by reweighted L1 minimization. *JFAA, 14*, 877–905. [DOI](https://doi.org/10.1007/s00041-008-9045-x). 加权稀疏优化。
4. Hoyer, P. O. (2004). Non-negative matrix factorization with sparseness constraints. *JMLR, 5*, 1457–1469. [Original](https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf). Hoyer sparsity。
5. Hurley, N., & Rickard, S. (2009). Comparing measures of sparsity. *IEEE Trans. Information Theory, 55*, 4723–4741. [DOI](https://doi.org/10.1109/TIT.2009.2027527). 比较稀疏指标。
6. Eckart, C., & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika, 1*, 211–218. [DOI](https://doi.org/10.1007/BF02288367). 最佳低秩残差（不同于元素 top-k）。
7. Li, K.-C. (1992). On principal Hessian directions for data visualization and dimension reduction: Another application of Stein's lemma. *JASA, 87*, 1025–1039. [DOI](https://doi.org/10.1080/01621459.1992.10476258). 期望 Hessian。
8. Sobol', I. M., & Kucherenko, S. (2009). Derivative based global sensitivity measures and their link with global sensitivity indices. *Mathematics and Computers in Simulation, 79*, 3009–3017. [DOI](https://doi.org/10.1016/j.matcom.2009.01.023). DGSM。
9. Roustant, O., Fruth, J., Iooss, B., & Kuhnt, S. (2014). Crossed-derivative based sensitivity measures for interaction screening. *Mathematics and Computers in Simulation, 105*, 105–118. [DOI](https://doi.org/10.1016/j.matcom.2014.05.005). 混合二阶导数与交互。
10. Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation, 55*, 271–280. [DOI](https://doi.org/10.1016/S0378-4754(00)00270-6). 功能 ANOVA。
11. Hooker, G. (2007). Generalized functional ANOVA diagnostics for high-dimensional functions of dependent variables. *JCGS, 16*, 709–732. [DOI](https://doi.org/10.1198/106186007X237892). 相关输入。
12. Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. *Annals of Applied Statistics, 2*, 916–954. [DOI](https://doi.org/10.1214/07-AOAS148). Friedman H 与变量交互。
13. Apley, D. W., & Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. *JRSS B, 82*, 1059–1086. [DOI](https://doi.org/10.1111/rssb.12377). ALE。
14. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). [Publisher](https://doi.org/10.1201/9781315370279). GAM。
15. Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science, 11*, 89–102. [DOI](https://doi.org/10.1214/ss/1038425655). P-splines。
16. Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. [Publisher](https://www.routledge.com/Local-Polynomial-Modelling-and-Its-Applications-Monographs-on-Statistics/Fan/p/book/9780203748725). Local Hessian 从 WY 估计。
17. Friedman, J. H. (1991). Multivariate adaptive regression splines. *Annals of Statistics, 19*, 1–67. [DOI](https://doi.org/10.1214/aos/1176347963). MARS/低阶交互函数。
18. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: A survey. *JMLR, 18*(153), 1–43. [Original](https://www.jmlr.org/papers/v18/17-468.html). 已知代理的自动微分，不保证真实导数对齐。
19. DeVore, R. A. (1998). Nonlinear approximation. *Acta Numerica, 7*, 51–150. [DOI](https://doi.org/10.1017/S0962492900002816). best k-term approximation 的标准理论背景；本项目 Hessian tail loss 不是该论文验证的旋转恢复方法。

### 结论性约定

**函数结构的“简单性”与向量“更集中”不同。** 用户希望 A=[2,2,0.01,0.001] 比 B=[1,0.5,0.01,0.01] 更简单，其最直接数学落实是先明确允许 k=2 个主要作用，再用 **Tail-L1、Tail-L2 或 Relative Tail-L2** 惩罚多余效应；而并非整体 L1/L2、Hoyer、Gini 或熵自动给出相同偏好。这个偏好定义合理，但其旋转最小值能否对应理论参数方向仍必须用解析式、WY 推断和 NN-WY 三道检验逐个验证。
