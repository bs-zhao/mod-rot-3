# Mod-Rot：研究框架总 README

> **独立研究维度与方法目录**：[真实生成模型 C1/C2/C3/CR3](TOY_GENERATIVE_MODELS.md) · [从 ZXY 提取 Hessian 的统计估计器 e](HESSIAN_FROM_ZXY_METHODS.md) · [积分/求平均的评价域、测度与权重 Q](HESSIAN_INTEGRATION_AND_AVERAGING.md) · [从 Hessian 函数获得数值矩阵的 20 类指标 a](HESSIAN_TO_MATRIX_METRICS.md) · **[Hessian 两个正交操作：消除 W 依赖 D × 输出形状 P](HESSIAN_FUNCTIONAL_OPERATIONS.md)**。不必强制所有指标先经过固定 n×n 数值矩阵；标量简单性 k 是另一层目标定义。

> **目的**：用一份长期维护的框架文档明确研究中的不同层次、实验因素、比较对象和评价标准。本文是**研究架构说明**，不是新的运行器，也不改变已有 NP、math 或 rotations 项目。  
> **日期**：2026-11-07。当前聚焦 C1、C2、C3、CR3；后续允许增加模型、估计器、积分方式和复杂度指标。  
> **核心分离**：生成数据、NP 表征、Y 的类型与来源、Hessian 数值估计器 e、**积分与平均方案 Q**、**Hessian → 数值矩阵指标 a**、标量简单性 k、旋转搜索和 Hessian 准确性诊断必须分别定义。

## 0. 总览：九个研究层级 + 独立 Hessian 准确性诊断

~~~text
L1 真实生成模型与数据：C1 / C2 / C3 / CR3
   └── 真 T、先验、真实解析式、Xset、真实 choice
                       │
                       ▼
L2 各模型冻结的 NP：Encoder → Z，Decoder 预测 Y
                       │
                       ▼
L3 Y 的通道与来源（两组独立因素）
   ├── 通道：logit / probability / binary choice
   ├── 来源 A：解析式直接求 Hessian（oracle）
   ├── 来源 B：解析式生成数值 Y → 估计器
   └── 来源 C：冻结 Decoder 生成数值 Y → 同一个估计器
                       │
                       ▼
L4 从数值 ZXY 推断 Hessian 的估计器 e
   Polynomial / spline / local regression / KRR / GP / Stein …
   （解析 oracle 不经过此步骤；有些方法直接估计平均 Hessian）
                       │
                       ▼
   Hessian 函数 / 局部导数字段 H(U,X)
                       │
             ┌─────────┴──────────────┐
             │                        ▼
             │             诊断层：Hessian fidelity
             │             解析 / 解析 Y / NN Y 的误差
             ▼
L5 【独立】积分与求平均方案 Q
   对什么求积？Z/X/联合、实际 Xset/真实 prior/经验 Z
   什么权重？被试均等、试次均等、条件/重要性权重
   用何数值规则？有限求和 / MC / Sobol / quadrature
                       │
                       ▼
L6 【更一般：D × P 两个可独立组合的操作】
   D 消除 W 依赖：点评价、期望、轨迹平均、极值、分位数等
   P 改变输出形状：n×n / scalar / vector / smaller/larger matrix
   → 特例 a=M01–M20：Hessian → 固定数值矩阵
                       │
                       ▼
L7 【独立】数值矩阵/向量/标量 → 简单性指标 k
   非对角能量 / L1 / Top-k tail / JD / 指定标量目标等
                       │
                       ▼
L8 旋转自变量 U=AZ+b：两两、3D、4D、SO(d)
   观察因变量 L(φ) = 简单性 Loss 如何随角度变化
                       │
                       ▼
L9 验证：解析真轴最小？数值来源重现？TEST U 与 T 对应？
~~~

**特别区分**：
- **D × P（两个独立数学操作）**：D 决定是否/如何消除 (W=(Z,X)) 的自由变量，P 决定输出仍为 n×n、变为标量/向量还是其他矩阵。它们可以单独使用、同时使用、改变先后顺序；例如 tr H(W) 仍依赖 W，E[H(W)] 则仍是 n×n。完整四象限和数学警告见 [Hessian 函数的二维运算分类](HESSIAN_FUNCTIONAL_OPERATIONS.md)。
- **Q（积分/求平均方案）**：固定测度、变量、样本点、权重及数值求积规则。例如先对 500 个 X 等权，再对每位被试等权；Q 允许只做条件积分而保留某个变量。
- **a（数值矩阵指标）**：固定 Q 后，决定被求期望的是 H、|H|、H²、HHᵀ，或者取分位数、条件方差等。这一步决定数值矩阵的**数学含义**。
- **k（简单性指标）**：对给定数值矩阵计算非对角 L1、Top-k 或其它**标量**损失。不要把 RMS 与 off-diagonal sparsity 混为同一层。
- **特殊直接分支**：可以先对每个局部 Hessian 求标量简单性、再按 Q 积分；此时**没有**先生成一张固定数值矩阵，必须用不同 pipeline_mode 明确标注。
- **部分估计器限制**：Stein 等方法可能只直接输出一个带权均值矩阵，并不产生局部 Hessian 场；它们不能无条件参与要求局部 RMS/Q99/高阶变化的 a 指标比较。

两个最基本、不能混淆的科学问题：

1. **Hessian extraction / fidelity**：给定真实生成函数和同一积分定义，数值 (X,Z,Y) 估计出的 Hessian 是否接近解析真值？
2. **Simplicity-based axis recovery**：给定一个已经计算好的 Hessian 或局部导数字段，无论它来自哪里，某种“简单性”损失是否在理论参数的坐标轴方向达到最小值？

第二个问题可以先使用完全准确的解析 Hessian，不必牵涉 NP 或拟合器。第一种失败不应简单归因于第二种失败，反之亦然。

---

## 1. L1：真实生成模型与数据

每个生成模型 m 已知：

\[
\eta_m(T,X),\quad p_m(T,X)=\sigma(\eta_m(T,X)),
\quad Y\mid T,X\sim\mathrm{Bernoulli}(p_m(T,X)).
\]

同时已知真实参数 T、真实先验 pi_m(T)、X 的产生方式、实际模拟的被试/试次以及观测 choice。这些不由 NP 学习出来，而是 oracle 和模拟数据的源头。

| 模型 | 真参数及维度 | X 维度 | 参数与函数的特点 | 真实 Xset/试次方式 |
|---|---|---:|---|---|
| **C1** | b1–b4，4 维 | 3 | 独立 N(0,0.5²) 参数先验；logit 对 b、X 线性 | 各 X 独立 Uniform(-1,1)，**逐被试逐试次重新抽样** |
| **C2** | b1–b4，4 维 | 4 | 同样参数先验；包含同一 X1 的线性和二次效应，logit 仍对 b 线性 | 四个原始选项属性逐被试逐试次抽样 |
| **C3** | b1–b6，**6 维** | 4 | 同样参数先验；两个二次 X 效应及 X1×X2 效应，logit 对 b 线性 | 四个原始选项属性逐被试逐试次抽样 |
| **CR3** | α、λ、γ、τ，4 维 | 6 | CPT-like 非线性理论函数；非正态先验 | **全体被试共享固定 500 个试次 Xset** |

CR3 参数先验按仓库原始生成器：α ~ Beta(12,3)；λ ~ Gamma(shape=9, scale=0.22)，区间 [0.3,5] 之外重抽；γ ~ Beta(9,4)；τ ~ N(2.2,0.55²)，截断 [0.5,4]。CR3 固定 Xset 的设计类别数分别为 gain-only 60、gamma-neutral 60、loss-only 40、general CR2 340。不要把这个特定的 500 试次设计当成从假想连续 X 分布均匀采样的结果。

真实实现/权威来源：

- C1：np/src/modlearn_rot/generators/choice/c1_linear_logit.py
- C2：np/src/modlearn_rot/generators/choice/c2_shared_x_quadratic.py
- C3：np/src/modlearn_rot/generators/choice/c3_multi_nonlinear_logit.py
- 逐被试 X 抽样循环：np/src/modlearn_rot/generators/dataset.py
- CR3 试次设计与生成：np/reference/cr3/s0_s1_cr3.py
- CR3 函数及先验：np/reference/s0_cr2_design_recovery.py

**冻结的数据要素**：model、公式版本、真实参数、prior、Xset 的生成 seed / design hash、Y 的观测方式、被试与试次 ID。不能在旋转时偷偷改变生成 Y 或 X 的采样规则。

## 2. L2：每个模型自己的、已经训练好的 Neural Process

对于被试 s，由观测的 (X,choice) 集合编码得到潜变量 Z_s；冻结的 decoder 接收 (Z_s,X) 给出二分类 logits / probability。S5 保存的是**使用每位被试全部已观察试次编码的 posterior mean Z**，并非随机抽样的 latent prior，也并非真参数 T。

研究设计在选定的配置中刻意让潜变量维度与真实参数个数相同，方便进行一一轴映射；并已经通过 Z→T 的线性回归检查较好的参数可恢复性：

\[
\widehat T=A_*Z+b.
\]

**两项必要的限定**：

- **按 checkpoint 核实维度。** 当前仓库存在 CR3 L4、L8 等配置；C3 的真实参数是 6 维。不能宣称所有已训练 NP 的 Z 都与 T 等维。
- **线性回归能较好预测 T ≠ 每个原始 z 轴已对应一个 T。** 多维线性混合、符号/排列/尺度歧义仍可能存在。映射 A_* 只有在方阵且满秩时才可逆。

原始七个 NP 步骤：np/s1_generate.py、s2_precheck.py、s3_train.py、s4_evaluate.py、s5_extract_z.py、s6_pca.py、s7_regression.py。

**边界**：训练 NP、提取 Z、拟合最终旋转不应使用真 T；真 T 可用于事后参数恢复与构建**以真轴为锚点的研究性旋转扫描**。如果用真 T 拟合 A_* 来定义 0°，必须诚实标记为 oracle-anchored diagnostic，不能称为完全无监督发现。

## 3. L3-A：Y 类型是独立实验维度

| Y 类型 | 真实生成端 | 冻结 NP 端 | 对 Hessian 的解释 |
|---|---|---|---|
| logit | eta_m(T,X) | class-1 与 class-0 logits 之差 | 连续函数的 Hessian |
| p / probability | sigmoid(eta_m) | 类 1 softmax probability | sigmoid 会改变曲率结构 |
| choice | Bernoulli(p) 的实际二值 Y | 硬分类或指定的 Bernoulli 抽样 | 离散观察本身不具备常规连续 Hessian |

**重要**：解析分支中所谓“choice Hessian”指的是条件均值 E[choice|T,X]=p 的 Hessian；从二值 choice 数值矩阵拟合连续函数再求 Hessian，得到的是**拟合器的平滑延拓导数**，不是实际 0/1 随机选择的二阶导数。若 NN 用 hard choice 而解析端用随机 Bernoulli choice，必须记录这两种不同的 choice protocol。

对 logit 与 p 不能默认得到相同的简单轴：

\[
H_p=\sigma'(\eta)H_\eta+
\sigma''(\eta)\nabla\eta\nabla\eta^\top.
\]

因此输出尺度也属于独立的重要实验维度。

## 4. L3-B：Hessian 的三个计算来源

**三条主要路径固定如下：**

| 标识 | 路径 | 谁使用解析公式 | 实际导数从何而来 |
|---|---|---|---|
| **A / analytic** | 已知真实生成解析式 → 直接求 Hessian | oracle | 原始真实解析函数 |
| **B / formula_y** | 真实解析式 → 生成 numerical (T/X/Y 或对应 U/X/Y) → 数据驱动回归函数 → 求 Hessian | **只有生成 Y 的步骤** | 拟合出的连续函数 |
| **C / nn_y** | 真实 Z、X → **已训练且冻结的 decoder** → numerical Y → **同样**的数值回归/拟合 → 求 Hessian | 不需要 | 拟合出的连续函数；**不对 NN 求导** |

B 与 C 的 Hessian 估计算法应当在相同坐标系、训练数据规模、输入 X 网格、目标类型、超参数和划分下比较。B 的拟合器**不知道**生成 Y 的解析式；否则就不再是数据驱动估计 benchmark。**现在正式将从数值 ZXY 提取 Hessian 的算法定义为独立实验维度 e**。Tensor polynomial、local polynomial、P-spline、KRR、GP、Stein 等不是 data source（公式/NN 来源）的类别，而是作用于给定数值矩阵的可比较估计器；同一 e 必须在 formula-Y 与 nn-Y 路径中按同样协议使用。完整数学解释与原始参考文献见 [方法目录](HESSIAN_FROM_ZXY_METHODS.md)。


### L4：独立的数值 Hessian 估计方法 e

这层的输入固定为数值 \((U,X,Y)\)，输出为一个局部 Hessian 字段、平均导数矩，或其它严格说明的对象：

\[
\widehat{\mathcal H}_{e}
=\operatorname{EstimateHessian}_{e}(\{U_s,X_{st},Y_{st}\}).
\]

| 方法族 | 代表算法 | 当前实现情况 | 核心统计限制 |
|---|---|---|---|
| 全局基函数 | 多项式 sieve、张量乘积、Chebyshev、低秩 | **math/01、04–07 已实现部分** | 拟合函数预测准确不保证 Hessian 准确 |
| 局部多项式 | 局部二次 / 局部高阶 regression、LOESS | 候选 | 维度高，带宽与局部样本不足 |
| 平滑样条 / GAM | B-spline、P-spline、交互样条 | **math/02 T-only 部分已有** | 惩罚/结构先验可能人为压低简单性 |
| 光滑核回归 | Gaussian KRR、Nyström、RBF、GP | **math/02、05 的 KRR/RBF 已有**；GP 候选 | 核长度尺度与正则影响二/三阶导数 |
| 数值微分 | 有限差分、Savitzky–Golay、随机扰动 | 用于部分解析验证；原始稀疏 ZXY 方法待做 | 偏移点若无观测 Y，不能直接差分 |
| Stein / 积分分部 | 二阶 score/weighted Stein | **math/02 已实现带权 T-only 平均** | **平均/加权 Hessian ≠ 局部 Hessian** |
| 其它代理 | 独立平滑 NN + autodiff、Bernoulli GLM | 候选 | 代理的 Hessian 不一定是真函数的 Hessian |
| 计算技巧 | Hessian-vector product | 非独立统计估计器 | 不负责从 ZXY 学习函数 |

**三个必须分开的维度**：来源 d 决定数值 Y 由解析式还是冻结 decoder 产生；估计器 e 决定怎样仅用数值 ZXY 推断导数；积分与平均 q 决定从局部导数得到哪一种统计量。不要把二阶 weighted Stein 的带权均值，与 full-ZX 的局部 10×10 Hessian 直接排名。固定 500 行 X 的连续 X 方向导数也需要光滑函数类假设才能被推断。

**独立详细 README**：[HESSIAN_FROM_ZXY_METHODS.md](HESSIAN_FROM_ZXY_METHODS.md)，共 **E01–E14 14 类**，逐类介绍数学形式、局部/平均目标、优势与局限、仓库已实现/待实现、统一 benchmark 方案，以及 **15 项参考文献**（Fan–Gijbels、Wahba、Eilers–Marx、Wood、KRR/Nyström、GP、Stein、有限差分、autodiff 等）。

### Hessian 的正确坐标与块结构

真 T、NP 原始 Z、旋转后 U 是三个不同坐标：

\[
H_{(U,X)}=
\begin{pmatrix}
H_{UU}&H_{UX}\\H_{XU}&H_{XX}
\end{pmatrix},
\]

其中 H_UU 维度为 d_U×d_U，H_UX 为 d_U×d_X。对于 NP 原始坐标相应称 ZZ/ZX；旧代码有时沿用 ZZ/XZ 名称，应在 metadata 记录坐标系统。

当解析函数在 (T,X) 中计算、而数值回归在 (U,X) 中计算时，必须做正确的 Jacobian 变换。例如若 T=BU+c 是**仿射**变换，则

\[
J=\operatorname{diag}(B,I_{d_X}),\qquad
H_{(U,X)}^{oracle}=J^\top H_{(T,X)}^{oracle}J.
\]

如果 U=AZ+b，则要按反向关系得到相应 Jacobian；如果映射非方阵、退化或非线性，不能直接套用逆矩阵版公式。

**当前 CR3 的 04/06/07 研究**从实际 NP Z 拟合 \(\widehat T=A_*Z+b\)，解析函数在 \(\widehat T\) 处生成 Y 和 Hessian，不能把它误写成使用每位被试原始真实 T 求同一个 Hessian。

**估计器的误差来源**还包括拟合误差、平滑/正则化、X 的离散支持、有限样本、数值微分与模型错设。拟合预测 Y 很准确并**不能推出** Hessian、梯度或高阶导数准确。

## 5. L5：积分与求平均方案 Q（独立研究维度）

**本层回答“对什么积分/平均、按照哪个分布、什么权重、用哪种数值算法”**；它本身**不规定**最终取 Signed Mean、Mean Abs、RMS、Variance、Quantile 还是 Gram。这些由下一层 a 决定。

\[
Q=(\Omega,\mu,\mathcal I,\text{conditioning},\text{mask}),
\qquad
\mathcal I_Q[g]\approx\int_\Omega g(Z,X)\,d\mu(Z,X).
\]

| 子维度 | 可选择的研究条件 | 是否改变统计目标？ |
|---|---|---|
| **积分变量** | X-only、Z-only、Z×X、固定点、分阶段条件平均 | 是；部分积分后仍可能含自由变量 |
| **Z/T 的积分测度** | S5 Z 经验分布、真实 T 经验分布、真实 T 先验 | 是；T 先验不等于 Z 分布 |
| **X 的测度** | CR3 真实 500-X bank、C1–C3 配对的 X、生成器 X 先验、条件子集 | 是 |
| **被试/试次耦合** | 原样配对、经验 Z × X 笛卡儿积、假设独立的乘积分布 | 是 |
| **样本与条件权重** | 每被试等权、每 trial 等权、分组权重、importance、Stein | 是 |
| **积分数值方法** | 固定有限和、MC、Sobol、quadrature | 在同一测度下主要改变近似误差 |
| **可微域/缺失规则** | 无条件严格、定义域内条件、fraction_defined | 是，若筛选改变支持域 |

以 CR3 为例，固定 Xset 时的两种不同联合测度：

\[
\mu_{\mathrm{empirical}}
=\frac1N\sum_s\delta_{Z_s}\otimes\frac1{500}\sum_t\delta_{X_t};
\]

\[
\mu_{\mathrm{prior}}
=\pi_{\mathrm{CR3}}(T)dT\otimes\frac1{500}\sum_t\delta_{X_t}.
\]

两者不仅坐标不同（Z 与 T），被试积分分布也不同。若对 X 先积分，得到的 \(\mathcal I_{Q_X}[H(Z,X)]\) **通常仍是 Z 的矩阵函数**；只有进一步对 Z 进行积分或指定 Z 点，才变成固定数值矩阵。

对于 C1–C3，按每位被试平均与把全部 trial 直接池化，在试次数不同的情况下有不同权重。CR3 的 500 个共享固定 X 不等于连续 X 先验积分。CR3 零金额导致部分金额 X 导数未定义时，必须区分 strict/unconditional、conditional 和 fraction_defined。

此外，\(\nabla_Z^2\mathbb E_X f=\mathbb E_X\nabla_Z^2f\) 只有在允许交换微分和积分、且所用 X 测度满足必要条件时才成立。先对 Y 取 X 均值再求导可能获得平均 ZZ，却不能重建局部 ZX/XX。

**详细子维度、CR3/C1–C3 测度、数值积分及权重约定**：[HESSIAN_INTEGRATION_AND_AVERAGING.md](HESSIAN_INTEGRATION_AND_AVERAGING.md)。

## 6. 诊断层：先评价 Hessian 是否算准，而不是直接评价“简单性”

对于同一个模型、Y、U/X 坐标、积分定义和导数阶数，设

\[
H_A=oracle,\quad H_B=\widehat H_{formulaY},\quad
H_C=\widehat H_{NNY}.
\]

先评价 ZZ/ZX/XX/FULL 以及更高阶导数字段：元素误差、MAE/RMSE、relative Frobenius、余弦/相关性、对角/非对角恢复、局部 RMS/SD、分布及方向稳定性、数值微分收敛等。预测 Y 的误差是**另一项诊断**，不能替代 Hessian accuracy。

严格的误差分解为：

\[
H_C-H_A=(H_B-H_A)+(H_C-H_B).
\]

- **B − A**：真实解析 Y 经数据驱动 Hessian 估计器后的**推断误差**。
- **C − B**：解析 Y 与 decoder Y 两条**拟合结果**的差异，可能包含 decoder 函数差异及估计器对不同 Y 的适应差异；**不能直接解释为 decoder 真 Hessian − 解析真 Hessian**。
- **C − A**：相对于 oracle 的总误差。

这项诊断与下一层的 simplicity 完全分开：一个估计器即使 Hessian 数值偏差较大，也可能保留损失随角度变化的规律；相反，平均 Hessian 接近也不保证更高阶结构一致。

## 7. L6：Hessian 函数的双轴操作 D×P，以及数值矩阵指标 a

**更完整的数学框架**是两个正交操作：**D 去除 W 依赖**（取点、期望、条件平均、极值、极限等）与 **P 变换输出形状**（仍为 n×n、标量、向量或其他维数矩阵）。二者可以独立、组合或按不同顺序使用，不是所有研究结果都必须是固定 n×n。详细分类见 [HESSIAN_FUNCTIONAL_OPERATIONS.md](HESSIAN_FUNCTIONAL_OPERATIONS.md)。

|  | 仍是 n×n（或原块形状） | 形状改变 |
|---|---|---|
| **仍含 W** | \(H(W)\)、\(H(W)^2\) | \(\mathrm{tr}H(W)\)、\(\mathrm{diag}H(W)\)、\(P^\top H(W)P\) |
| **纯数值** | \(H(w_0)\)、\(E_Q H\)、RMS、Q99 | \(\mathrm{tr}H(w_0)\)、\(E_Q[\lambda_{\max}H]\)、\(\mathrm{Cov}_Q(\mathrm{vec}H)\) |

**操作次序不能随意交换**：当 P=Trace（线性）时 \(E_Q[\mathrm{tr}H]=\mathrm{tr}(E_QH)\)；当 P=最大特征值（非线性）时一般有 \(E_Q[\lambda_{\max}H]\geq\lambda_{\max}(E_QH)\)，且常常严格大于。高阶场统计如 Gram/covariance 可能需要多个矩而非一次逐点操作。

**这是新的必要层级**。Hessian 是一个矩阵值函数 \(H(Z,X)\)，可能随 Z、X 改变；即使解析式 Hessian 已完全准确，仍必须指定矩阵统计指标 a，才得到一个固定的数值矩阵 \(M_a\)。

\[
\boxed{M_{q,a}=\operatorname{MatrixStatistic}_a(H;\mu_q).}
\]

同一个积分方案 q 可产生多种不同数值矩阵：

| 指标族 | 指标编号与公式示例 | 核心特点 |
|---|---|---|
| **有符号平均** | M01：\(\mathbb E_Q[H]\) | 正负值可抵消 |
| **幅度/能量** | M02：\(\mathbb E_Q[|H|]\)，M03：\(\sqrt{\mathbb E_Q[H\odot H]}\)，M04：Mean Square，M07：Lp | 强度，不依赖或不同程度依赖符号 |
| **变化/稳健性** | M05：逐元素 Variance，M06：SD，M08：Median，M09：Q90/Q99，M10：Max | 区分稳定与局部极端曲率 |
| **符号/阈值/截尾** | M11–M14：Exceedance、Sign consistency、正/负部分、Winsorized | 衡量出现频率和方向一致性 |
| **整体矩阵算子** | M15：\(\mathbb E_Q[HH^\top]\)；M16：\(\mathbb E_Q[H^\top H]\)；M17：谱矩阵绝对值；M18：\(\mathrm{Cov}(\mathrm{vec}H)\) | 保留方向协变，**输出尺寸/含义可能变化** |
| **条件及高阶变化** | M19：按 Z/X 的条件方差与 ANOVA 分解；M20：\(\partial_ZH,\partial_XH\) 的统计矩阵 | 需要条件测度或三/四阶导数字段 |

**正交旋转不变量应单独排除**：对局部 ZZ Hessian 做 trace、det、特征值或 Frobenius norm 等，若积分测度随坐标一致地变换，这些操作在正交旋转下不变，作为单独的旋转 loss 通常产生平坦曲线，不能定位真参数轴。其他坐标依赖操作如 off-diagonal 或逐元素稀疏性才可能区分轴。

**数值矩阵指标不是标量简单性指标。** 比如 a=M03 得到一张 RMS-ZZ 矩阵，而 k=off-diagonal L1 才将它转成标量。对于 Gram 或 covariance，输出矩阵可能不是原本 ZZ 的维度；必须检查后续简单性损失的合法性及含义。

**例子**：\(f(z_1,z_2,x)=z_1^3+z_1z_2x\)，有 \(H_{ZZ}=\left(\begin{smallmatrix}6z_1&x\\x&0\end{smallmatrix}\right)\)。若 Z1 与 X 都独立标准正态，a=M01 的平均为零矩阵，而 a=M03 的 RMS 是 \(\left(\begin{smallmatrix}6&1\\1&0\end{smallmatrix}\right)\)。**即使使用完全相同的解析 Hessian，矩阵指标不同也会给出截然不同的结构。**

**必须在局部场上先完成旋转，再计算 M02/M03/M05/Q99 等逐元素矩阵统计**；一般不能先把 RMS 矩阵算出再直接对它做正交旋转。最后还需区分先求数值矩阵再计算 S、以及先计算局部 S 再按 Q 平均，两条路径一般不相等。

**扩展四象限**：[HESSIAN_FUNCTIONAL_OPERATIONS.md](HESSIAN_FUNCTIONAL_OPERATIONS.md)；**M01–M20 数值矩阵特例**：[HESSIAN_TO_MATRIX_METRICS.md](HESSIAN_TO_MATRIX_METRICS.md)。

## 8. L7：从固定数值矩阵到标量简单性 S_k

这一层应实现为**与 Hessian 来源和矩阵提取过程无关**的评分器：给定已明确形状/意义的数值矩阵 M，输出标量 Loss。对高阶或非局部指标，另设附加导数字段或局部标量分支：

\[
S_k:M_{q,a}\longrightarrow\mathbb R.
\]

给出**相同的数值矩阵 M**，不论来自 analytic、formula-Y 或 nn-Y，也不论用了哪种积分或估计器，**同一个 k 必须得到相同标量评分**。一些局部/非局部指标不能仅由一个 M 计算，应独立标注附加输入要求或 local-first 分支；不能假称它们全是固定数值矩阵的函数。

| 指标族 | 所需输入 | 衡量什么 |
|---|---|---|
| ZZ/ZX 稀疏度 | H_UU、H_UX 或局部矩阵 | L1/L2、Top-k tail、soft-L0、logsum、capped L2、Hoyer、entropy、Gini |
| 对角/非对角与交互 | ZZ/UU | 非对角能量、曲率分布、最大非主效应 |
| 高阶非线性 | 局部 Hessian 及三/四阶导数字段 | self-curvature、跨 Z 三阶/四阶导数、ZZ 受 X 调制、ZX 随 X/Z 的变化 |
| 功能算子 / 联合对角化 | 多个梯度/Hessian/Fisher-like 算子 | 共同主轴与功能独立性 |
| 非局部函数复杂度 | 函数值、受控单轴/双轴 sweep | 二次拟合残差、矩形交互、加性缺口 |

**最后一类不一定是 Hessian-only**：只给一个平均 H 无法重建原函数或受控扫描。必须注明附加函数输出是输入，不能为了统一术语而隐瞒。

“简单”不是同一个数学概念：稀疏、独立、低阶和可加性可能互相冲突。每个 loss 都需要明确其方向（minimize）、坐标标准化、聚合顺序、可比较的输入域。

## 9. L8：角度是自变量，简单性是因变量

在选定了 A_* 作为研究参考坐标后：

\[
U_\varphi=R(\varphi)(A_*Z)+b,\quad
A(\varphi)=R(\varphi)A_*.
\]

在真实实验中应使用与代码一致的行/列向量方向，明确固定 b、原始 Z/X/Y、NP checkpoint、积分节点和估计器。解析分支可以等变地转换局部导数；数值 B/C 分支在每一个旋转角度下重新以 U,X 拟合指定的数值响应。

定义一条**矩阵优先**的旋转损失路径（所有下标均为独立实验因素）：

\[
\boxed{
L_{m,y,h,e,q,a,k}(\varphi)
=S_k\left(
\mathcal A_a\left[
H_{m,y,h,e}(U_\varphi,X);\;\mu_q
\right]\right).
}
\]

其中 m=生成模型，y=logit/p/choice，h=解析/formula-Y/nn-Y 来源，e=数值 B/C Hessian 估计器（oracle 时不适用），q=积分方案，**a=选择固定数值矩阵这一分支时的 M01–M20 指标**，**k=下游标量简单性指标**，φ=角度，L=最终标量损失。**更一般的输出分支**须另记录 D、P、操作顺序、输出形状及其是否仍依赖 W；不强制所有路径通过 a 或固定 n×n。

**局部标量优先**的替代分支单独写为 \(\mathbb E_{\mu_q}[S_k(H_\varphi(U,X))]\)；它通常不等于 \(S_k(\mathbb E_{\mu_q}H_\varphi)\)，必须明确 pipeline_mode。

旋转族：
- **两两旋转**：4 维时有 6 个平面，0–90° 扫描，通常 0°/90° 为有符号坐标置换等价方向（对于具有相应不变性的损失）。
- **三维联合旋转**：SO(3) 有 3 个自由度；4 维中可选不同三维子空间。
- **四维联合旋转**：SO(4) 有 6 个自由度，允许直接优化整个旋转矩阵。
- 将来可以增加更一般的线性可逆变换，但这不是单纯的正交旋转；需额外解决尺度和退化解。

六条二维切面都在 0° 最小，不证明完整 SO(4) 全局最优。对于一般三维/四维联合路径，90° 也**不一定**代表符号置换等价点。

**成功标准必须分层报告：**

1. **Analytic-only**：指标自身是否偏好已知理论轴？各切面是否在真实轴方向取得**最小值**（不是最大值）？是否排除近乎水平的曲线？
2. **Analytic vs formula-Y**：解析生成 Y → 同一拟合器后，是否仍恢复最小值与曲线形状？解析–拟合 Hessian 是否接近？
3. **Formula-Y vs NN-Y**：冻结 NN 的数值 Y 是否仍保留该最小值？若失败，区分 B−A 估计器误差和 C−B 额外差异。
4. **跨模型**：同一指标定义（不针对测试真 T 调参）是否用于 C1、C2、C3、CR3？在 logit/p/choice 上有什么系统差异？
5. **盲发现**：不让真 T 参与优化，最终 U 与 T 正确配对维度的相关高、错误配对的相关低；评估 held-out TEST 与独立 seed。

**重要**：A_* 由 Z 线性预测 T 得到，只是研究上的**理论轴近似锚点**，不等于模型真实参数已由 NP 完美恢复。若 dim(Z)>dim(T)，还必须明确是否先选择 d_T 子空间、使用 PCA、还是旋转全部 Z；非方阵 A 不能无条件求逆。

## 10. 所有实验因素的总表：后续可以新增轴，而不是重写整个框架

| 层 | 独立因素 | 当前/可选值 |
|---|---|---|
| 1 | 生成模型 | C1 / C2 / C3 / CR3 |
| 1 | 真参数分布、Xset、试次数、seed | 按生成器定义、经验样本或不同固定设计 |
| 2 | NP checkpoint、训练超参、Z 的维数 | beta、latent dim、repeat、context、full-trial S5 |
| 2 | 线性映射与坐标 | 原始 Z、T、T_hat、U，是否 d_Z=d_T |
| 3 | Y 输出 | logit / probability / binary choice（明确采样/阈值） |
| 3 | Hessian 来源 | A 解析 / B 解析 Y 回归 / C NN Y 回归 |
| **3（新增独立轴）** | **Hessian 数值估计器 e** | [E01–E14 方法目录](HESSIAN_FROM_ZXY_METHODS.md)：全局/局部多项式、样条、核/GP、Stein 等；B/C 同配置，明确局部/平均对象 |
| **5（独立 Q）** | **积分、求平均的测度与数值规则** | [Q 方案目录](HESSIAN_INTEGRATION_AND_AVERAGING.md)：Xset、真实 T 先验、经验 Z、配对/乘积分布、权重、有限和/MC/Sobol/quadrature |
| **6（更一般 D×P）** | **去除 W 依赖 × 改变输出形状** | [二维运算分类](HESSIAN_FUNCTIONAL_OPERATIONS.md)：取点/期望/极值、trace/diag/eigen/投影、输出纯数值或仍是函数，先后顺序与不变性 |
| **6（独立 a）** | **Hessian → 数值矩阵指标** | [M01–M20 数值矩阵指标](HESSIAN_TO_MATRIX_METRICS.md)：Mean、Mean Abs、RMS、Variance、Q99、Gram、条件方差、高阶变化 |
| **7（独立 k）** | **固定数值矩阵 → 标量简单性** | 非对角能量、L1、Top-k、JD；单独注明 local-first 例外 |
| 诊断 | Hessian 精度 | A−B、B−C、A−C 的误差与稳定性 |
| **7（补充）** | 附加局部/非局部复杂度指标 | 必须声明高阶导数字段或函数值，而非仅由固定矩阵 M 可推得 |
| 8 | 旋转方式与参数 | 2D、3D、4D、连续 SO(d)、角度步长 |
| 9 | 最终成功标准 | minima、曲线相关、跨模型、独立 TEST 参数匹配 |

**研究策略**：不要立即跑所有维度的巨大笛卡儿积。每次只改变明确的一个因素，其余保持冻结，这样才能定位失败发生在哪里。

## 11. 分层数据接口与 provenance（规范草案，不是现有统一 API）

~~~text
experiment/
  metadata.json          模型版本/参数先验/Xset/数据hash/NP checkpoint/坐标/seed/estimator_id/estimator_hyperparams
  responses/             每种 Y 及其数值预测准确性
  derivatives/           局部 H_UU / H_UX / H_XX；可选 H3/H4
  integration/           Q：积分域/变量/点位/权重/算法/有效域掩码
  functional_outputs/    D×P：去除变量/形状变换后的矩阵、向量或标量函数及数值
  numeric_matrices/      a：M01–M20 固定数值矩阵分支、尺寸、构造方式
  fidelity/              A/B/C 的 Hessian 数值误差及归因
  simplicity/            k：给定数值矩阵或明确补充字段后的标量 loss
  rotation_landscapes/   angle × model × Y × source × estimator × Q × [D,P,order or a] × scalar_simplicity_k
  validation/            TEST 的 U 与 T 对应、非对应相关、稳定性
~~~

**概念上的调用边界**，不代表仓库中已存在这些公共函数：

~~~python
data = load_generator_data(model, data_id)             # L1
z, decoder = load_frozen_np(condition)                # L2
y = produce_numerical_y(source, output_type, data, z) # L3
derivatives = infer_derivatives(estimator, y, coords) # L3-C：B/C 共享估计器
evaluation = choose_points_and_measure(derivatives, Q) # L5：点位/测度/积分规则
accuracy = compare_to_oracle(derivatives, oracle)      # 独立的 Hessian fidelity
functional = reduce_and_shape(evaluation, D, P, order) # L6：可保留变量、改变尺寸
matrix = matrix_metric(evaluation, a)                  # L6：可选的固定矩阵特例 M01–M20
loss = scalar_simplicity(matrix_or_functional, k)      # L7：明确可优化标量定义
landscape[angle] = loss                                # L8：旋转角 → 损失
~~~

**不要因此合并独立旋转套餐。** rotations/ 下每个套餐仍是独立文件夹、独立 run.py；总 README 负责科学架构和接口约定，不负责一键运行所有方法。

## 12. 目前代码分工与既有实验

- **np/**：四个模型的数据生成、NP 训练、验证、Z 提取、PCA、Z→T 线性回归。
- **Hessian 估计器**：[HESSIAN_FROM_ZXY_METHODS.md](HESSIAN_FROM_ZXY_METHODS.md)——从数值 ZXY 恢复 Hessian 的 14 类方法。
- **积分/平均 Q**：[HESSIAN_INTEGRATION_AND_AVERAGING.md](HESSIAN_INTEGRATION_AND_AVERAGING.md)——Z/X 测度、条件、权重、求积算法。
- **函数与形状的两个正交操作 D×P**：[HESSIAN_FUNCTIONAL_OPERATIONS.md](HESSIAN_FUNCTIONAL_OPERATIONS.md)——是否消除 W、矩阵/向量/标量输出、运算顺序与旋转不变性。
- **矩阵提取指标 a**：[HESSIAN_TO_MATRIX_METRICS.md](HESSIAN_TO_MATRIX_METRICS.md)——Hessian 场 → 固定数值矩阵的 M01–M20 类方法。
- **math/01_cr3_hessian_expectation/**：CR3 的解析/数据矩阵 Hessian 与经验、先验积分。
- **math/02_cr3_hessian_estimators/**：不同数据驱动 Hessian 估计器的精度 benchmark。
- **math/03_cr3_hessian_six_y/**、**math/04_cr3_zx_hessian_threeway/**：三种 Y，解析与 NN Y 生成，真实解析对照与统一坐标变换。
- **math/05_cr3_zx_hessian_optimization/**：数值 Hessian 估计方法和超参数探索。
- **math/06_cr3_hessian_rotation_landscapes/**：四维参数的六个两两旋转平面、三个 Y 通道、三条 Hessian 路径、平均矩阵简单性。
- **math/07_cr3_local_hessian_complexity/**：局部二/三/四阶导数、ZZ/ZX 的丰富复杂度损失、受控单轴/双轴扫描。
- **rotations/**：冻结 NP 的各个独立旋转套餐；互不依赖、不设统一批处理入口。
- **metrics/**、**research/**：跨模型的函数指标研究和阶段性审计。

**范围澄清**：math/01–07 主要围绕 CR3；目前不能说 C1/C2/C3 的全部三路 Hessian、先验积分和旋转实验已经按同一标准实施。统一跨模型 benchmark 属于下一阶段扩展。

## 13. 现阶段的核心科学目标与开放问题

**已经明确的研究结构**：已知生成函数/先验/Xset → 训练 NP，验证 Z 包含可线性恢复的 T 信息 → 用三种 Y/三种 Hessian 来源、选择独立数值估计器 e 获得同一坐标的导数字段 → 固定积分方案 Q（测度/样本/权重/求积）→ 按 D×P 分别决定变量依赖与输出形状；常用分支用指标 a 得到固定数值矩阵 M → 用指标 k 将 M 变成标量简单性（或直接使用有明示定义的其他形状与 local-first 分支）→ 考察旋转角度如何影响 loss。Hessian 准确性独立与解析 oracle 比较。

**尚未被证明**：
- 哪个简单性指标能同时在 C1、C2、C3、CR3 和不同 Y 表达中识别理论轴；
- 解析式成功是否必然意味着真实解析 Y 数值回归也成功（实际已知未必）；
- 数据驱动估计器的高阶导数误差能否充分降低；
- 仅二维切面成功能否扩展到完整 SO(d)；
- 在不用真 T 选择指标/角度时，是否能稳定恢复一一对应的理论参数方向。

> **最终目标**：只用冻结 NP 的 Z、真实或预测的 X/Y，构造可估计且可解释的函数复杂度损失，寻找保持预测功能不变的 \(U=AZ+b\)，使 U 的维度与真理论参数 T 尽量一一对应。生成解析式与真实 T 在研究时是验证和误差归因的工具，不能成为最终盲优化的答案。
