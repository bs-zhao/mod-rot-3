# Mod-Rot：研究框架总 README

> **相关独立资料**：[本项目 C1、C2、C3、CR3 的全部真实生成模型、参数先验、Xset 与 logit/p/choice](TOY_GENERATIVE_MODELS.md)。它是本项目生成模型层的目录，区别于 Hessian 提取与简单性指标。

> **目的**：用一份长期维护的框架文档明确研究中的不同层次、实验因素、比较对象和评价标准。本文是**研究架构说明**，不是新的运行器，也不改变已有 NP、math 或 rotations 项目。  
> **日期**：2026-11-07。当前聚焦 C1、C2、C3、CR3；后续允许增加模型、估计器、积分方式和复杂度指标。  
> **核心分离**：生成数据、NP 表征、Y 的类型、Hessian 来源、积分/平均、Hessian 准确性、简单性指标和旋转搜索必须分别定义。

## 0. 总览：七层加一个独立诊断层

~~~text
L1 真实生成模型（C1/C2/C3/CR3）
   ├── 真实生成解析式 eta_m(T,X) → p_m(T,X) → Bernoulli choice
   ├── 已知真实参数 T 和真实参数的先验 pi_m(T)
   └── 已知 Xset 的生成/选择方案及真实 (T,X,Y)
                     │
          ┌──────────┴───────────────┐
          ▼                          ▼
L2 每个模型单独训练的 NP           原始生成解析函数
   ├── encoder: (X,choice) → Z        │
   ├── decoder: (Z,X) → Y             │
   ├── 特定配置令 dim Z = dim T       │
   └── Z→T 线性回归的可恢复性检验      │
          └──────────┬───────────────┘
                     ▼
L3 Y 的表达 / Hessian 计算来源
   × 输出：logit / probability / choice
   A 解析函数直接求导（oracle）
   B 解析函数产生数值 Y → 回归/拟合函数 → 导数/Hessian
   C 冻结 NP decoder 产生数值 Y → 同一拟合器 → 导数/Hessian
                     │
                     ▼
L4 对 X 和 Z/T 进行积分、经验平均，或保留局部 Hessian
   × 固定 Xset / X 分布；经验被试 / 已知 T 先验 / Z 经验分布
   × E[H] / RMS(H) / Var(H) / 局部 H / 三四阶导数字段
                     │
          ┌──────────┴───────────────┐
          ▼                          ▼
诊断层：Hessian fidelity          L5 简单性指标 S(H) → 标量 loss
A 对 B/C 的误差、局部结构等         稀疏性、交互、高阶变化、JD 等
                                     │
                                     ▼
L6 旋转 U=A Z+b；角度是自变量
   两两 2D / 3D、4D 联合旋转 / SO(d) 连续优化
                                     │
                                     ▼
L7 最终验证：loss 是否在真参数轴最小？U 与 T 是否一一对应？
   解析成功？解析 Y 数值估计也成功？NN-Y 也成功？跨四模型成功？
~~~

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

B 与 C 的 Hessian 估计算法应当在相同坐标系、训练数据规模、输入 X 网格、目标类型、超参数和划分下比较。B 的拟合器**不知道**生成 Y 的解析式；否则就不再是数据驱动估计 benchmark。可以将 tensor polynomial、KRR、样条、Stein 等作为 Hessian 估计器的**另一个独立因素**，但不要把“来源”与“估计器种类”写成同一个维度。

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

## 5. L4：积分、求平均与局部保留——另外一组正交维度

明确区分三个选择：

1. **积分测度与网格**：真实 T 先验、模拟被试经验分布、NP Z 经验分布、真实 Xset、或另行说明的设计分布。
2. **积分变量**：仅 X、仅参数/被试、X 与参数联合，或者局部评估点**不积分**。
3. **被积分的量**：H 本身、|H|、H²、局部导数、三阶/四阶张量、甚至局部简单性分数。

CR3 的共享固定 500 试次，例：

\[
\bar H_X(T_s)=\frac1{500}\sum_{t=1}^{500}H(T_s,X_t);
\]

\[
\bar H_{\mathrm{sample}}
=\frac1N\sum_{s=1}^N \bar H_X(T_s);
\]

\[
\bar H_{\mathrm{prior}}
=\int\biggl[\frac1{500}\sum_{t=1}^{500}H(T,X_t)\biggr]
\pi_m(T)\,dT.
\]

其中 prior 积分是**对真实参数 T 积分**，而 X 仍是固定的有限 500 行，不是自动积分某个假想的连续 X 分布。

**C1/C2/C3 的 X 是逐被试随机生成的**，所以对每位被试先平均再对被试平均，和将全部试次直接池化可能产生不同权重；必须说明目标 estimand。

**真实参数的先验 pi(T) ≠ NP Z 的 prior/aggregate posterior。** 如果想在 Z 空间做先验积分，需要显式定义从 T 到 Z 的映射/密度或使用经验 Z 分布，不能直接借用 T 的参数先验。

还要明确：

\[
S(\mathbb E[H])\ne\mathbb E[S(H)],\quad
|\mathbb E[H]|\ne\mathbb E[|H|],\quad
(\mathbb E[H])^2\ne\mathbb E[H^2].
\]

例如三次函数 z³ 在对称 z 分布下平均二阶导数可能为零，却仍有明显三阶复杂度。**为了计算复杂度而选择先平均 Hessian，是方法上的重要选择，不是无关紧要的加速步骤。**

积分实现上可以有精确有限求和、Monte Carlo、Sobol、quadrature 等不同算法。必须保留积分节点数、权重、随机种子、数值收敛诊断。

**CR3 可微域警告**：金额为零时分数次幂的某些 X 导数未定义。必须区分 strict/unconditional mean、定义域条件 mean、fraction_defined；如果为了高阶微分只取正金额 X，则新估计量对应的是**改变后的 X 域**。

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

## 7. L5：由已知 Hessian/导数字段计算简单性 S

这一层应该实现为**与 Hessian 来源无关**的评分器：

\[
S_k:\mathcal H\longrightarrow\mathbb R.
\]

给出相同 H、统计矩阵或局部导数字段，不论它是 oracle 算出来、formula-Y 回归算出来还是 NN-Y 回归算出来，**同一个指标必须得到相同评分**；不应在这里读取真实 T、拟合器类别或网络训练细节。

| 指标族 | 所需输入 | 衡量什么 |
|---|---|---|
| ZZ/ZX 稀疏度 | H_UU、H_UX 或局部矩阵 | L1/L2、Top-k tail、soft-L0、logsum、capped L2、Hoyer、entropy、Gini |
| 对角/非对角与交互 | ZZ/UU | 非对角能量、曲率分布、最大非主效应 |
| 高阶非线性 | 局部 Hessian 及三/四阶导数字段 | self-curvature、跨 Z 三阶/四阶导数、ZZ 受 X 调制、ZX 随 X/Z 的变化 |
| 功能算子 / 联合对角化 | 多个梯度/Hessian/Fisher-like 算子 | 共同主轴与功能独立性 |
| 非局部函数复杂度 | 函数值、受控单轴/双轴 sweep | 二次拟合残差、矩形交互、加性缺口 |

**最后一类不一定是 Hessian-only**：只给一个平均 H 无法重建原函数或受控扫描。必须注明附加函数输出是输入，不能为了统一术语而隐瞒。

“简单”不是同一个数学概念：稀疏、独立、低阶和可加性可能互相冲突。每个 loss 都需要明确其方向（minimize）、坐标标准化、聚合顺序、可比较的输入域。

## 8. L6：角度是自变量，简单性是因变量

在选定了 A_* 作为研究参考坐标后：

\[
U_\varphi=R(\varphi)(A_*Z)+b,\quad
A(\varphi)=R(\varphi)A_*.
\]

在真实实验中应使用与代码一致的行/列向量方向，明确固定 b、原始 Z/X/Y、NP checkpoint、积分节点和估计器。解析分支可以等变地转换局部导数；数值 B/C 分支在每一个旋转角度下重新以 U,X 拟合指定的数值响应。

定义：

\[
L_{m,y,h,q,k}(\varphi)
=S_k\left(\operatorname{Aggregate}_q[
H_{m,y,h}(U_\varphi,X)]\right),
\]

其中 m=生成模型，y=logit/p/choice，h=解析/formula-Y/NN-Y Hessian 来源，q=积分/局部保留方式，k=简单性指标，**旋转角度 φ 是自变量，损失 L 是因变量**。

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

## 9. 所有实验因素的总表：后续可以新增轴，而不是重写整个框架

| 层 | 独立因素 | 当前/可选值 |
|---|---|---|
| 1 | 生成模型 | C1 / C2 / C3 / CR3 |
| 1 | 真参数分布、Xset、试次数、seed | 按生成器定义、经验样本或不同固定设计 |
| 2 | NP checkpoint、训练超参、Z 的维数 | beta、latent dim、repeat、context、full-trial S5 |
| 2 | 线性映射与坐标 | 原始 Z、T、T_hat、U，是否 d_Z=d_T |
| 3 | Y 输出 | logit / probability / binary choice（明确采样/阈值） |
| 3 | Hessian 来源 | A 解析 / B 解析 Y 回归 / C NN Y 回归 |
| 3 | Hessian 估计器 | 多项式、样条、核估计、Stein 等；B/C 同配置 |
| 4 | 积分域与测度 | 真实 Xset、T 先验、经验 Z、局部网格 |
| 4 | 聚合量 | 局部 H、E[H]、RMS、SD、三/四阶导数 |
| 诊断 | Hessian 精度 | A−B、B−C、A−C 的误差与稳定性 |
| 5 | 简单性定义 | ZZ/ZX 稀疏、高阶、JD、函数可加性 |
| 6 | 旋转方式与参数 | 2D、3D、4D、连续 SO(d)、角度步长 |
| 7 | 最终成功标准 | minima、曲线相关、跨模型、独立 TEST 参数匹配 |

**研究策略**：不要立即跑所有维度的巨大笛卡儿积。每次只改变明确的一个因素，其余保持冻结，这样才能定位失败发生在哪里。

## 10. 分层数据接口与 provenance（规范草案，不是现有统一 API）

~~~text
experiment/
  metadata.json          模型版本/参数先验/Xset/数据hash/NP checkpoint/坐标/seed
  responses/             每种 Y 及其数值预测准确性
  derivatives/           局部 H_UU / H_UX / H_XX；可选 H3/H4
  aggregation/           积分定义/权重/平均/RMS/有效域掩码
  fidelity/              A/B/C 的 Hessian 数值误差及归因
  simplicity/            给定导数字段后各指标的 loss 数值
  rotation_landscapes/   angle × model × Y × source × integration × metric
  validation/            TEST 的 U 与 T 对应、非对应相关、稳定性
~~~

**概念上的调用边界**，不代表仓库中已存在这些公共函数：

~~~python
data = load_generator_data(model, data_id)             # L1
z, decoder = load_frozen_np(condition)                # L2
y = produce_numerical_y(source, output_type, data, z) # L3
derivatives = infer_derivatives(source, y, coords)    # L3
field = aggregate_or_retain(derivatives, measure)     # L4
accuracy = compare_to_oracle(field, oracle)            # 诊断层
loss = simplicity_metric(field)                        # L5
landscape[angle] = loss                                # L6
~~~

**不要因此合并独立旋转套餐。** rotations/ 下每个套餐仍是独立文件夹、独立 run.py；总 README 负责科学架构和接口约定，不负责一键运行所有方法。

## 11. 目前代码分工与既有实验

- **np/**：四个模型的数据生成、NP 训练、验证、Z 提取、PCA、Z→T 线性回归。
- **math/01_cr3_hessian_expectation/**：CR3 的解析/数据矩阵 Hessian 与经验、先验积分。
- **math/02_cr3_hessian_estimators/**：不同数据驱动 Hessian 估计器的精度 benchmark。
- **math/03_cr3_hessian_six_y/**、**math/04_cr3_zx_hessian_threeway/**：三种 Y，解析与 NN Y 生成，真实解析对照与统一坐标变换。
- **math/05_cr3_zx_hessian_optimization/**：数值 Hessian 估计方法和超参数探索。
- **math/06_cr3_hessian_rotation_landscapes/**：四维参数的六个两两旋转平面、三个 Y 通道、三条 Hessian 路径、平均矩阵简单性。
- **math/07_cr3_local_hessian_complexity/**：局部二/三/四阶导数、ZZ/ZX 的丰富复杂度损失、受控单轴/双轴扫描。
- **rotations/**：冻结 NP 的各个独立旋转套餐；互不依赖、不设统一批处理入口。
- **metrics/**、**research/**：跨模型的函数指标研究和阶段性审计。

**范围澄清**：math/01–07 主要围绕 CR3；目前不能说 C1/C2/C3 的全部三路 Hessian、先验积分和旋转实验已经按同一标准实施。统一跨模型 benchmark 属于下一阶段扩展。

## 12. 现阶段的核心科学目标与开放问题

**已经明确的研究结构**：已知生成函数/先验/Xset → 训练 NP，验证 Z 包含可线性恢复的 T 信息 → 用三种 Y/三种 Hessian 来源获得同一坐标的导数字段 → 按明确分布积分或保留局部信息 → 先衡量 Hessian 精度，再从该对象独立计算简单性 → 考察旋转角度如何影响 loss。

**尚未被证明**：
- 哪个简单性指标能同时在 C1、C2、C3、CR3 和不同 Y 表达中识别理论轴；
- 解析式成功是否必然意味着真实解析 Y 数值回归也成功（实际已知未必）；
- 数据驱动估计器的高阶导数误差能否充分降低；
- 仅二维切面成功能否扩展到完整 SO(d)；
- 在不用真 T 选择指标/角度时，是否能稳定恢复一一对应的理论参数方向。

> **最终目标**：只用冻结 NP 的 Z、真实或预测的 X/Y，构造可估计且可解释的函数复杂度损失，寻找保持预测功能不变的 \(U=AZ+b\)，使 U 的维度与真理论参数 T 尽量一一对应。生成解析式与真实 T 在研究时是验证和误差归因的工具，不能成为最终盲优化的答案。
