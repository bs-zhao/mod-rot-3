# 02 · 第一波实验：基于 Hessian / 导数结构的无监督潜变量旋转与函数简单性

> **实验设计 README · 2026-11-08**
>
> **上一篇**：[01 · 项目生成模型、NP 与两种变换目标](01_project_overview.md)。
>
> **本篇任务**：在每个已训练、冻结的 NP 模型中，尝试**不使用真实参数 T 作为优化监督**，从各被试的潜变量 Z 和原始行为输入 X 出发，对旋转后的函数 \((U,X)\rightarrow Y\) 构造“简单性”损失，并在正交旋转空间中搜索 R。分别验证：(1) 损失在**真实解析生成函数**上是否偏好正确轴；(2) 仅凭数值 \((W,Y)\) 能否准确估计相同结构；(3) 换成**冻结 decoder 产生的 Y** 后是否依然有效。
>
> **本文性质**：详细研究方案、候选指标目录、模块接口与验证协议，**不是**新算法已经有效的证据；文中“候选”不表示已经实现或取得成功。本次只编写 README，不新增训练/旋转程序。
>
> **三个后续相互独立、可扩展的模块**：
> 1. **M1：Hessian / 导数字段 → 局部、积分及数值结构指标**；
> 2. **M2：只给数值 (W,Y) → 统计推断 Hessian 或目标指标**；
> 3. **M3：Hessian 指标 / 指标矩阵 → 标量函数复杂度（simplicity loss）**。
>
> 这三块彼此正交组合，不能将某一数据源、某一估计器、某一积分方式和某一 loss 捆成唯一方法。

---

## 1. 研究对象和核心科学命题

### 1.1 三个不同的函数与三个不同的参数坐标

给定生成模型 \(m\in\{\mathrm{C1,C2,C3,CR3}\}\)，每位被试 s 有真实参数 \(T_s\)，每个试次 t 有 \(X_{st}\)，真实生成的二元选择满足：

$$
\eta_m(T_s,X_{st})=f_m(T_s,X_{st}),\qquad
p_m(T_s,X_{st})=\sigma(\eta_m(T_s,X_{st})),\qquad
Y_{st}\sim\mathrm{Bernoulli}(p_m).
$$

这里用 \(f_m\) 表示真实的**logit 解析函数**；其 probability 通道为 \(\sigma\circ f_m\)。对 binary choice，解析上有意义的连续目标是条件均值 \(p_m\)，不是某一次随机 0/1 的二阶导数。

对真实 T 的每个维度使用一套固定参考均值、标准差，定义

$$
T_{\mathrm{std}}=S_T^{-1}(T-\mu_T),\qquad
S_T=\mathrm{diag}(\sigma_{T,1},\ldots,\sigma_{T,d}).
$$

完全等价的标准化参数解析函数是：

$$
\boxed{
f_{m,\mathrm{std}}(t,x)=f_m(\mu_T+S_T t,x).
}
$$

这与原 f 是**同一生成机制的重参数化**，在对应参数点产生完全相同的 logit/probability。标准化参考只用于解析 oracle 或事后 TEST 对比，不提供给真正的无监督旋转搜索。

独立训练的 Neural Process 接收每位被试的行为集合 \(\{(X_{st},Y_{st})\}_t\)，冻结后提取后验均值 \(Z_s\)，并由冻结 decoder：

$$
\boxed{
d_\psi(Z_s,X_{st})\;\longrightarrow\;
(\widehat\eta_{st},\widehat p_{st})
}
$$

预测行为。严格来说 decoder 输出的是两个分类 logits \((\ell_0,\ell_1)\)，其**二类对数几率**是 \(\widehat\eta=\ell_1-\ell_0\)，预测概率是 \(\widehat p=\mathrm{softmax}(\ell_0,\ell_1)_1=\sigma(\widehat\eta)\)。不要将 \(\ell_1\) 单独当成可与真实解析 logit 比较的对象；两个 logits 有共同平移的任意性。

这三个核心对象必须始终分清：

| 对象 | 输入坐标 | 能否解析求真导数？ | 在研究中的作用 |
|---|---|---|---|
| 真实生成函数 \(f_m\) | (T,X) | 能 | 原始真值 |
| 标准化生成函数 \(f_{m,\mathrm{std}}\) | (Tstd,X) | 能 | 真参数轴上的理论 oracle |
| 冻结 decoder \(d_\psi\) | (Z,X) | 可以对**网络自身**自动微分；但不等于生成函数真导数 | 现实可访问的预测函数和数值 Y 来源 |

### 1.2 固定 Z，明确“近似标准正态”并不等于“已解耦”

本阶段每个模拟模型使用约 **5,000 位被试**，每位被试从全部已观察 \((X,Y)\) 试次中提取**一个冻结后验均值向量** \(Z_s\)。C1/C2/CR3 取 d=4；C3 取 d=6，和真实参数 T 等维。

神经网络训练中的 Gaussian latent KL 正则化可鼓励潜变量接近指定的标准正态先验，但这**不保证**被试后验均值的 aggregate distribution 恰为 \(\mathcal N(0,I)\)，不保证各原始 Z 维度线性不相关，更不保证各 Z 轴一一对应真实认知参数。应报告实际的 TRAIN Z 均值、协方差和特征值，而不是将“近似正态”“未正交”“独立”“解耦”混用。

当前以 TRAIN 被试均值中心化 \(Z_c=Z-\mu_{Z,\mathrm{TRAIN}}\)。本 README 后续简写为 **Z**，即**固定中心化后的冻结 NP 潜变量**。该固定中心化不使用真实 T。

### 1.3 监督 Oracle：什么叫 \(DRZ=U^*\) “最像” Tstd？

我们需要先给出一个可复现的**监督参照方向** \(R^*_{\mathrm{oracle}}\)，来检验后续无监督 loss 的最低点在哪里。

令

$$
R\in O(d),\qquad R^\top R=I,\qquad
D=\mathrm{diag}(d_1,\ldots,d_d),\quad d_i\ne0,
$$

并允许一个行列重排矩阵 P（输出 U 的顺序和真实 T 的编号不一定一致）。定义：

$$
V_R=RZ,\qquad U_R=DV_R,\qquad
\boxed{P\,D^*R^*_{\mathrm{oracle}}Z\approx T_{\mathrm{std}}.}
$$

**“最像”至少有两种不同的严格定义，不可以混为一谈：**

**Oracle A：逐被试标准化 MSE 最小（建议作为主要参照）**

$$
\boxed{
(R^*_{\mathrm{oracle}},D^*,P^*)
=
\underset{R\in O(d),\,D\mathrm{\ diagonal},\,P\mathrm{\ permutation}}
{\mathrm{argmin}}\;
\frac1{N_{\mathrm{train}}d}
\sum_{s\in\mathrm{TRAIN}}
\left\|PDRZ_s-T_{\mathrm{std},s}\right\|_2^2.
}
$$

所有均值、方差和方向仅在 TRAIN 拟合；VALIDATION 可用于监督 Oracle 的优化/选择起点，**TEST 不参与拟合**。中心化已消除训练平均截距；若采用其他中心化协议，必须显式允许或记录平移。对一个固定 R 和匹配关系，在中心化 TRAIN 样本上，每个 d_i 可直接用单变量最小二乘斜率求解：

$$
d_i^*=
\frac{\mathrm{Cov}_{\mathrm{TRAIN}}((RZ)_i,T_{\mathrm{std},\pi(i)})}
{\mathrm{Var}_{\mathrm{TRAIN}}((RZ)_i)}
$$

（若斜率为零，应作为退化候选处理，而不是偷偷违反 D 非零的定义）。由此能高效剔除 D 的连续搜索，而不改变 Oracle 的最小二乘定义。TRAIN 上若每个 Tstd 方差按同一协方差分母标准化为 1，则最优单维 MSE 与 \(1-r^2\) 相对应。

**Oracle B：相关轴分离最优（可作为独立对照）**

设 \(C_{ij}(R)=|\mathrm{Corr}((RZ)_i,T_{\mathrm{std},j})|\)。以全局一一匹配 \(\pi\) 定义：

$$
\mathrm{MCC}(R)=
\max_{\pi\in S_d}\frac1d\sum_i C_{i,\pi(i)}(R),
$$

$$
\mathrm{Leakage}(R,\pi)=\frac1{d(d-1)}
\sum_i\sum_{j\ne\pi(i)}C_{ij}(R).
$$

这一 Oracle 可选用最大化 \(\mathrm{MCC}\) 或同时惩罚 Leakage；**必须在实验开始前固定具体目标和权重**，因为它不一定等于最小 MSE 的 Oracle A。记录匹配均值、最差匹配、平均/最大错配及 T–T 相关基线。相关性只衡量方向，不衡量数值校准。

**不论哪个 Oracle，都是使用真实 T 的监督上限**，仅用于定位“期望的方向”及画诊断曲线；**不参与**真正无监督 simplicity 目标的训练、调参或旋转选择。

若对应关系近似成立，定义

$$
\boxed{
U_s^*=D^*R^*_{\mathrm{oracle}}Z_s
}
$$

为真实 \(T_{\mathrm{std},s}\) 的**近似恢复坐标**，并允许一个固定 P 将其轴排列到 Tstd 顺序。它不是严格等于 Tstd，也不保证泛化时每一个被试精确恢复；CR3 的非零错配相关尤其需要诚实报告。

### 1.4 为什么实际无监督优化只搜索 R 而不搜索 D？

若 \(V=RZ\)，任意非零逐轴缩放 D 不改变绝对 Pearson U–T 相关：

$$
\boxed{
|\mathrm{Corr}(DV,T_{\mathrm{std}})|
=
|\mathrm{Corr}(V,T_{\mathrm{std}})|.
}
$$

本阶段对函数结构损失还强制施加**旋转后的逐轴缩放不变性**：

$$
\boxed{
L(DR)=L(R)
\quad\text{对全部非零对角 D 成立。}
}
$$

注意这**是候选 loss 必须通过的数学/数值审计，不是任何 Hessian 原始值天然满足的性质**。如果具体 loss 不满足，就需要正确标准化或明确将其排除，不能放任 D 不受约束地放大从而人为减小 Hessian。

四维正交矩阵具有 6 个连续自由度，可用 \(G_{12},G_{13},G_{14},G_{23},G_{24},G_{34}\) 的 Givens 角度积搜索；C3 的六维正交旋转具有 15 个连续自由度。对符号/排列对称的 loss，实际只优化 \(SO(d)\) 旋转角度即可覆盖 \(O(d)\) 的全部**无向轴等价类**。否则必须另行检查反射分支。

**不能混淆**：旋转**之前**对白化/输入尺度做改变，是另一个搜索几何；不由旋转**之后**的 D 不变性自动消去。当前主实验使用固定的中心化 Z；raw-Z、TRAIN marginal-Z-score、whitened-PC 等若要研究应成为独立条件，并重做监督 Oracle，不能混用不同空间的 R*。

### 1.5 实际被优化的是哪一个函数？

每个候选 R 都生成：

$$
U_s=RZ_s,\qquad
W_{st}=[U_s,X_{st}].
$$

但网络**原本接受的是 Z**；对任意 U 查询，必须使用它的逆变换送回冻结 decoder：

$$
\boxed{
d_R(u,x)=d_\psi(R^\top u+\mu_{Z,\mathrm{TRAIN}},x)
}
$$

这里若实现层已经将 Z 视为中心化并包装了 decoder，那么相应地省略加回均值，**不能重复加减**。于是：

$$
\boxed{
d_R(R(Z_s-\mu_Z),X_{st})=d_\psi(Z_s,X_{st})
}
$$

为精确等式。换 R 只改变函数的坐标表示，不重新训练 decoder，也**不会改善或恶化原始行为预测性能**。

如果使用逐轴标准化 \(U^{(z)}=S_R^{-1}(RZ-\mu_R)\)，则相应的逆变换也必须一并更新；不存在“先改变 U 再直接把 U 当成原始 Z 传入网络”的合法捷径。

---

## 2. 旋转实验的 WY 数据流与三种 Y 通道

### 2.1 每个 R 构造同一批观测点的 W

对于每位被试 s 的一个 Z 和该人的 \(\{X_{st}\}_{t=1}^{n_s}\)：

$$
U_s(R)=RZ_s,\quad
W_{st}(R)=[U_s(R),X_{st}]\in\mathbb R^{d+p},
$$

并得到每个旋转候选下的数值表：

$$
\mathcal D_R^{(y)}=\{(W_{st}(R),Y_{st}^{(y)})\}_{s,t}.
$$

总数据量大约 5,000 位被试 × 每人 300–500 试次（CR3 每人固定 500），但**只有约 5,000 个不同的 U**；同一人的所有 trial 共享同一个 U。任何训练、交叉验证、bootstrap 必须按被试分组，不能把百万级 trial 当作百万个独立的潜变量点。

对 CR3，全部被试共享精确的 500×6 X bank；对 C1/C2/C3，各被试/试次原始 X 重新随机产生。两种联合测度的统计意义不同，不能为了简化计算将其无标注混为同一采样设计。

### 2.2 使用冻结 decoder 产生三种 Y

| 通道 | 冻结网络输出 | WY 回归 / 导数估计目标 | 关键区别 |
|---|---|---|---|
| **logit** | \(\widehat\eta=\ell_1-\ell_0\) | 连续函数 \(\widehat\eta(W)\) 的 Hessian | 不受两类 logits 共同平移影响 |
| **probability** | \(\widehat p=\sigma(\widehat\eta)\) | 连续函数 \(\widehat p(W)\) 的 Hessian | sigmoid 链式项可能改变最简单方向 |
| **binary choice** | \(Y^\mathrm{sim}\sim\mathrm{Bernoulli}(\widehat p)\)，并固定随机种子；可另列阈值 hard choice 作为独立条件 | 从 Bernoulli 观察拟合条件概率 \(p(W)\)，必要时再求其导数 | **单次 0/1 不可直接求 Hessian**；抽样不同于阈值输出 |

“最终 binary 输出”需要显式选择**Bernoulli 采样**还是**硬阈值**；二者给不同的数值目标。与原始真实观测 \(Y_{st}^{\mathrm{observed}}\) 比较时，必须将其单独命名，不得把原始真实 Y 冒充 decoder 生成的二元响应。

如有多次抽样，可用重复随机样本降低 Bernoulli 噪声，但必须记录每个 (U,X) 的重复次数、seed 和估计目标。对概率或 logit 使用确定性数值 WY，和 binary 的噪声统计性质不同。

### 2.3 整个方法的分层流水线

~~~text
S1 数据 + 已冻结 NP checkpoint
                  |
                  v
          每位被试 Z_s, X_st
                  |
          候选 R in O(d)
                  |
                  v
    U_s = R(Z_s - mean_Z_train)
    W_st = concat(U_s, X_st)
                  |
       +----------+------------+------------------+
       |                       |                  |
  NN / frozen decoder     analytic formula      original observed
  logit / p / binary      logit / p / binary    binary Y (optional control)
       |                       |                  |
       +-----------------------+------------------+
                  |
                  v
           numerical (W,Y)
                  |
           M2: infer field H / related operators
                  |
           M1: select blocks + aggregate/integrate
                  |
           M3: scalar simplicity L(R)
                  |
      multistart optimize R or sweep angles
                  |
                  v
        R_hat (NO T used in optimization)
                  |
       post-hoc compare to T / R*_oracle
~~~

**注意顺序不是概念上的依赖混乱**：M1 规定想测量的**Hessian 属性与数值 estimand**；M2 选择可否从 WY 估计这些目标；M3 将指标转为复杂度。实际计算时通常先执行 M2 产生局部字段，再由 M1 汇总，最后 M3 打分。若 M2 直接估计加权平均 Hessian，则 M1 只能消费兼容的平均量，不能伪造局部矩阵。

---

## 3. 模块 M1：从 Hessian 函数字段构造结构指标（独立、可扩展）

### 3.1 完整 Hessian 的坐标和分块

记旋转后的连续响应函数为 \(h_y(U,X)\)，联合输入 \(W=[U,X]\)，其中 \(\dim U=d,\ \dim X=p\)。

$$
\boxed{
H_y(W)=\nabla_W^2h_y(W)=
\begin{pmatrix}
H_{UU}&H_{UX}\\
H_{XU}&H_{XX}
\end{pmatrix}.
}
$$

- \(H_{UU}\in\mathbb R^{d\times d}\)：U 与 U 的二阶交互；**对角元素 \(H_{u_i u_i}\) 衡量某个 U 的局部自身曲率/非线性**，非对角元素衡量一对 U 的局部非加性；
- \(H_{UX}\in\mathbb R^{d\times p}\)：U 与某个 X 的混合偏导；可刻画“哪个个体因素调节哪个刺激输入的作用”；
- \(H_{XX}\in\mathbb R^{p\times p}\)：刺激自身的曲率与交互，主要作基准和辅助，**X 的坐标不随 R 旋转**；
- \(H_{XU}=H_{UX}^\top\)（在足够光滑、二阶导可交换处）。

对指定模型维数：C1 的总 W 为 7D；C2 为 8D；C3 为 10D；CR3 为 10D。必须记录具体的 Y 通道与可微定义域。

二阶 Hessian 的一个重要限制是：它测量**局部二阶变化**，不是自动等于变量“效应大小”，也不等于完整函数的最高多项式次数。对非线性函数，某些交互在特定点二阶导可为 0，而其他地方不为 0。

### 3.2 Q：对什么求积分/平均？这也是独立实验维度

统一记 \(Q=(\Omega,\mu,\mathrm{weights},\mathrm{mask},\mathrm{rule})\)。例如：

$$
\mathbb E_Q[g(W)]
=\int_{\Omega}g(w)\,d\mu(w)
\approx\sum_{n=1}^M\omega_n g(w_n).
$$

候选积分方式：

| ID | 域 / 操作 | 含义及风险 |
|---|---|---|
| Q01 | **保留局部字段** \(H(U_s,X_t)\) | 最充分；数据和计算量较大 |
| Q02 | **仅对 X 求平均** \(\mathbb E_{X|U}[H(U,X)]\) | 结果通常仍依赖 U |
| Q03 | **仅对 U 求平均** \(\mathbb E_{U|X}[H(U,X)]\) | 结果通常仍依赖 X |
| Q04 | **联合被试×试次平均** | 固定统计目标；每人等权和每 trial 等权不同 |
| Q05 | **CR3 固定 500 X bank** | 精确覆盖此固定设计，不等于连续 X 总体积分 |
| Q06 | **经验 U × 指定 X bank 的笛卡尔积** | 若允许函数离表查询，可研究反事实空间；改变了原始样本测度 |
| Q07 | **条件/分组平均** | 例如 CR3 gain-only / loss-only / gamma-neutral / general |
| Q08 | **全体/部分可微域** | X 为 0 时 CR3 分数幂金额导数可不存在；必须标记有效比例 |
| Q09 | **Monte Carlo / quasi-MC / quadrature** | 记录采样分布、节点、权重及收敛误差 |
| Q10 | **分层/重要性/Stein 权重** | 可能改变 estimand；不能与无权平均混同 |
| Q11 | **固定点/锚点及条件路径** | 可研究局部几何，不应当作全分布统计 |
| Q12 | **截尾、分位数、局部极值域** | 稳健/极端结构需清楚定义截断及有效支持 |

在 M1 中 **Q 只负责测度、评价点、积分和权重**。是对 Hessian 本身、绝对值、平方、Gram 等求平均，则由下面的指标 A 决定。对 X-only 积分后得到的 \(M(U)\) 仍可能是矩阵**函数**，不能称已获得固定数值矩阵。

### 3.3 A：从每个局部 Hessian 形成数值指标、矩阵或函数

把 \(H(W)\) 暂时看作任意所选 Hessian 块；\(H_{ij}(W)\) 是其一个元素。以下均可按 \(UU\)、\(UX\)、\(XX\) 分块分别计算，不同块维数不必相同。

#### A1 · 一阶/二阶矩、强度和变化（至少 20 类可扩展）

| ID | 名称 | 数学定义（逐元素或注明矩阵） | 理解 |
|---|---|---|---|
| A01 | Signed mean | \(\mathbb E_Q[H_{ij}]\) | 有正负抵消风险 |
| A02 | Mean absolute | \(\mathbb E_Q[|H_{ij}|]\) | **重点：平均绝对交互/曲率** |
| A03 | RMS | \(\sqrt{\mathbb E_Q[H_{ij}^2]}\) | 局部导数的典型能量 |
| A04 | Mean square | \(\mathbb E_Q[H_{ij}^2]\) | 不开方的二阶矩 |
| A05 | Variance | \(\mathrm{Var}_Q(H_{ij})\) | 曲率是否随 W 变化 |
| A06 | SD | \(\sqrt{\mathrm{Var}_Q(H_{ij})}\) | 变化的导数单位 |
| A07 | Mean \(L_p\) | \((\mathbb E_Q |H_{ij}|^p)^{1/p}\) | 调整对大值敏感度 |
| A08 | Median absolute | \(\mathrm{median}_Q|H_{ij}|\) | 稳健典型曲率 |
| A09 | Absolute quantile | \(\mathrm{Quantile}_{.90/.95/.99,Q}|H_{ij}|\) | 尾部很强的作用 |
| A10 | Max absolute | \(\max_{\Omega}|H_{ij}|\) | 极端区域，敏感 |
| A11 | Threshold occupancy | \(\Pr_Q(|H_{ij}|>\tau)\) | 在多大范围内“激活” |
| A12 | Sign consistency | \(|\mathbb E_Q\mathrm{sign}(H_{ij})|\) | 相互作用方向是否稳定 |
| A13 | Positive / negative mean | \(\mathbb E_Q\max(H_{ij},0)\) 及负部 | 不丢失符号信息 |
| A14 | Trimmed mean | 对元素分布先截尾再求均值 | 抗极端值 |
| A15 | Coefficient of variation | \(\mathrm{SD}(H_{ij})/(\mathbb E|H_{ij}|+\epsilon)\) | 相对不稳定性；要处理 0 |
| A16 | Conditional X variance | \(\mathbb E_U\mathrm{Var}_{X|U}(H_{ij}|U)\) | 交互随刺激条件变化 |
| A17 | Conditional U variance | \(\mathbb E_X\mathrm{Var}_{U|X}(H_{ij}|X)\) | 交互随个体位置变化 |
| A18 | Row Gram | \(\mathbb E_Q[HH^\top]\) | 行向量相似性，输出矩阵可变 |
| A19 | Column Gram | \(\mathbb E_Q[H^\top H]\) | 列向量相似性 |
| A20 | Covariance of vectorized H | \(\mathrm{Cov}_Q(\mathrm{vec}H)\) | 各二阶作用共同波动 |
| A21 | Spectral absolute | \(\mathbb E_Q[(H^2)^{1/2}]\) | **仅对对称方阵**成立；与逐元素绝对值不同 |
| A22 | Hessian trace / eigenvalues | \(\mathrm{tr}H(W)\)、\(\lambda_i(H(W))\) 及其统计量 | 坐标谱性质；可能对旋转不敏感 |
| A23 | Conditional / functional ANOVA | 对 \(H_{ij}(U,X)\) 的条件主效应与交互分解 | 需要明示联合测度和独立性假设 |
| A24 | Gradient–Hessian relation | 例如 \(\mathbb E[\nabla h\,(\nabla h)^\top]\) 与 H 对照 | gradient Gram **不是 Hessian**，应标明算子来源 |
| A25 | Hessian variation | \(\mathbb E_Q|\partial_{w_k}H_{ij}(W)|\) | 需要**三阶导数** |
| A26 | Fourth-order variation | \(\mathbb E_Q|\partial_{w_k}\partial_{w_l}H_{ij}(W)|\) | 需要**四阶导数** |

这些指标可以输出单个标量、向量、\(d\times p\)、\(d\times d\)、\(p\times p\)、或者更大的协方差矩阵。后续 M3 必须检查输入形状及含义，不能强制所有方法都先形成同一个固定大小的 Hessian。

最容易混淆的三项：

$$
|\mathbb E H_{ij}|
\quad\ne\quad
\mathbb E|H_{ij}|
\quad\ne\quad
\sqrt{\mathbb E H_{ij}^2}.
$$

另外，**先逐点计算指标再平均**，与**先平均 H 再施加非线性函数**通常不等价。两条路径都允许，但必须分别命名：

$$
L_{\mathrm{local-first}}=\mathbb E_Q[S(H(W))],
\qquad
L_{\mathrm{matrix-first}}=S(\mathbb E_Q[H(W)]).
$$

#### A2 · 对角 Hessian：一个 U 自身的非线性

特别保留：

$$
c_i(U,X)=H_{u_i u_i}(U,X)
=\frac{\partial^2 h_y}{\partial u_i^2}(U,X).
$$

从 \(c_i\) 可以提取：平均绝对曲率、RMS、方差、Q90/Q99、激活比例、符号反转次数、沿 \(u_i\) 方向的总变化、曲率的三阶/四阶变化、对 X 的条件依赖。

可以对不同 \(u_i\) 形成**单轴复杂度向量** \(C^{\mathrm{self}}\in\mathbb R^d\)，而不是只给一个全局数字。其分布、均衡性及是否集中到少量方向，再留给 M3 定义总 loss。

**注意**：\(H_{u_i u_i}=0\) 只说明该输出函数对 \(u_i\) 在该区域局部线性，不说明 \(u_i\) “没有作用”。例如 \(h(u,x)=ux\) 的 \(H_{uu}=0\)，但 \(H_{ux}=1\)。

#### A3 · 非对角/混合 Hessian：两两之间的交互

专门保留：

$$
h_{ij}^{UX}(U,X)=\frac{\partial^2 h_y}{\partial u_i\partial x_j},
\qquad
h_{ik}^{UU}(U,X)=\frac{\partial^2 h_y}{\partial u_i\partial u_k},\quad i\ne k.
$$

它们对应：
- U_i 作用到哪些 X_j（**U–X 二部交互图**）；
- 哪些 U_i 与 U_k 之间存在非加性交互（**U–U 图**）；
- X–X 结构作为固定刺激坐标上的对照；
- 对每个边的 A02/RMS/分位数/条件变化形成多张**结构矩阵**。

但“存在边”与“作用强度大”不是一回事；M3 将重点对**边数、作用集中度、所需交互阶数及结构复杂度**施加惩罚，而不是简单地把所有边的绝对强度求和。

### 3.4 只用二阶导数不足以区分低阶与高阶多项式

这是本实验第二类简单性必须特别处理的关键反例：

$$
h_1(u,x)=ux,\qquad h_2(u,x)=u^3x.
$$

两者都有 \(H_{ux}\)：

$$
\frac{\partial^2 h_1}{\partial u\partial x}=1,
\qquad
\frac{\partial^2 h_2}{\partial u\partial x}=3u^2.
$$

如果只看“是否存在 U–X 作用”，两者都具有一条连接，但 \(u^3x\) 对 u 的作用更非线性、更高阶。可以额外考虑：

$$
\frac{\partial^2 h_1}{\partial u^2}=0,
\qquad
\frac{\partial^2 h_2}{\partial u^2}=6ux,
$$

$$
\frac{\partial^4 h_1}{\partial u^3\partial x}=0,
\qquad
\frac{\partial^4 h_2}{\partial u^3\partial x}=6.
$$

所以候选复杂度需要至少区分：

1. \(H_{uu}\) 自曲率；
2. 混合 Hessian \(H_{ux}\) **随 u 的变化**（例如方差或条件方差）；
3. 三/四阶导数，或受控一维函数效应的曲率/高阶差分。

也要比较外层 sigmoid 对结果的影响：对 logit 近似低阶，不代表对 probability 仍低阶，因为

$$
H_p=\sigma'(\eta)H_\eta+
\sigma''(\eta)\nabla\eta\,\nabla\eta^\top.
$$

这是**不同输出通道**的结构差异，不能不加说明地当成高阶复杂度发现。

---

## 4. 模块 M2：从未知非线性函数的数值 (W,Y) 推断 Hessian（独立统计问题）

### 4.1 严格定义：只给 WY，不知道函数表达式

假设真实连续响应函数 \(h(W)\) 极其复杂、非线性且包含多变量交互，但**估计器看不到**生成该响应的解析表达式，也不能读取 oracle Hessian：

$$
\mathcal D=\{(W_n,Y_n)\}_{n=1}^M,\qquad
Y_n=h(W_n)+\varepsilon_n
$$

（对 binary 使用 Bernoulli likelihood，而不使用不恰当的加性高斯假设）。目标是统计推断以下之一：

$$
\widehat h(W),\qquad
\widehat H(W)=\nabla_W^2\widehat h(W),\qquad
\widehat M_a\approx M_a(H;Q).
$$

同一 estimator 可以针对不同 Y 来源构成**可比较的测试**，但必须固定超参数选择规则、训练样本量、评价域、输出通道、分组划分及正则化。

**数学上的不可识别性**：有限个 (W,Y) 值不能唯一决定离散观测点之外的连续 Hessian；即使两个函数在所有观测点给出完全相同的 Y，也可以有不同的二阶和高阶导数。M2 必须借助平滑性、函数类、分布、可查询的反事实点或积分分部边界等假设。这不是多加几千名被试就必然消失的问题。

### 4.2 候选方法目录（可以持续扩展；不是宣称全部已经实现）

| ID | 方法族 | 从 WY 得到什么？ | 核心假设 / 局限 | 参考 |
|---|---|---|---|---|
| E01 | 全局多项式 sieve / series | 拟合 \(\widehat h\)，解析求全 Hessian / 高阶导数 | 阶数、岭惩罚、基函数振荡和边界偏差 | [R1], [R2] |
| E02 | Tensor-product polynomial / separable basis | \(\widehat h=\phi_U^\top B\phi_X\)，逐块求导 | 张量维数高、rank/正则需审计 | [R1], [R3] |
| E03 | Local polynomial regression / local quadratic | 局部梯度与 Hessian | 8–10 维局部邻域稀疏，带宽关键 | [R4] |
| E04 | Penalized B-splines / P-splines | 可微平滑函数及局部导数 | knots、惩罚强度、曲率偏差 | [R5], [R6] |
| E05 | GAM / tensor-interaction smooths | 主效应与交互的光滑分解及导数 | 若先强制可加，再评价可加性会出现循环论证 | [R7] |
| E06 | Gaussian RBF kernel ridge regression / Nyström | 核展开 \(\widehat h\) 及解析核二阶导数 | lengthscale 影响 H；大样本代价高 | [R8], [R9] |
| E07 | Gaussian process regression | Hessian 后验、导数不确定性 | 协方差核光滑阶数和大样本缩放 | [R10] |
| E08 | Chebyshev / Legendre / spectral series | 基函数展开后的高阶导数 | 基底、标准化域、截断阶数 | [R2] |
| E09 | Finite differences / Savitzky–Golay | 可查询或规则网格上的局部导数 | 普通离散 WY **不自动有**偏移点的 Y | [R11], [R12] |
| E10 | Second-order Stein / integration-by-parts | **条件满足时的平均/加权平均 Hessian** | 要知道/估计输入密度和边界项；通常非局部 H | [R13] |
| E11 | Gaussian smoothing / zeroth-order perturbation | 平滑后的 Hessian | 必须能反事实查询函数，改变了 estimand | [R14] |
| E12 | 独立训练的 smooth neural surrogate + autodiff | surrogate 的 Hessian | **不是原 NP decoder 的 Hessian**；需要审计导数偏差 | [R15] |
| E13 | Logistic GLM / smooth Bernoulli sieve | 对 binary Y 拟合 \(\widehat p\) 或 logit，再求导 | Bernoulli 噪声与输出 link 正确，基函数仍受限 | [R1], [R7] |
| E14 | Hessian-vector products / matrix-free calculation | 计算已有可微函数的 H 或 Hv | **计算技术，不是独立的 WY 统计估计器** | [R16] |
| E15 | 直接估计 function-value 交互对比 | 平滑/受控 2×2 或 3×3 factorial non-additivity | 对反事实网格需要函数查询或明确代理；不自动等于 H | [R17] |

**现有仓库状态**：此前 `math/01, 04, 05, 06, 07` 已有多项式/tensor 基和部分核估计；`math/02` 有 T-only 的加权平均 Stein、样条或 KRR 部分实现。这里列出的全部 15 类是**实验候选目录**，不能声称每个统计估计器已支持完整 \(H_{UU},H_{UX},H_{XX}\) 或高阶导数。详细原目录见 [数值 Hessian 方法](../HESSIAN_FROM_ZXY_METHODS.md)。

### 4.3 M2 输出必须使用明确的能力接口

建议不同估计器报告如下能力，而不是只叫 “get_hessian”：

| 输出能力 | 所需结果 | 可以接哪些 M1 / M3？ |
|---|---|---|
| F0 | 可查询 \(\widehat h(W)\) | 预测、差分、函数值交互 |
| F1 | 可查询 \(\nabla_W\widehat h\) | 单变量效应、gradient 结构 |
| F2 | 可查询 \(\widehat H(W)\) | 全部局部 Hessian 及 A01–A23 的适用子集 |
| F3 | 可查询第三/四阶导数 | 高阶单轴复杂度、A25/A26 |
| AVG | 仅返回某种 \(\mathbb E[\omega H]\) | 仅与同一带权均值相容的矩阵 loss |
| CONTRAST | 受控 factorial 差分 | 非导数形式的局部函数交互对照 |

例如 Stein 的 AVG 不得伪装成 F2 再去求局部 Q99、曲率方差或分组非线性。二维 factorial gap 反映非加性结构，但**不是**严格的 Sobol 主效应/交互方差指数，不能直接改名。

### 4.4 公平统计对照与真实困难

1. **Y 的拟合误差与 Hessian 的拟合误差分开测量**：选择概率预测得准，并不保证二阶导数正确；二阶导数 RMSE 很小，也不保证 M3 的极小旋转方向正确。
2. **原始 5,000 个 U 与每人多个 X**：局部回归依赖被试间 U 的覆盖，重复同一 U 的试次不能视为独立 U 样本；按被试分组训练/验证。
3. **CR3 500 行固定 X**：原始 WY 不能唯一决定这 500 行之外的连续 X 导数；仅从固定网格拟合 X 曲率必须有平滑/外推假设。其他模型虽然 X 连续抽样，但联合维数仍很高。
4. **同一个 R 对应函数等价坐标变化**：估计器如果对坐标变化本身敏感（例如有固定各维长度尺度/不一致的特征归一化），其 \(L(R)\) 曲线可能主要来自估计器偏置，而非真实生成函数的简单性。要用精确解析路径做对照，并报告 estimator-only rotation artifacts。
5. **可查询函数与一次性观测不可混同**：若用 decoder 在新的 U/X 偏移点补出 Y，这是 query-access benchmark；若只用原始固定 WY 记录，就不拥有这些额外点。
6. **选择函数通道独立比较**：对 deterministic logit/p 可用带误差模型的连续响应拟合；binary 用正确的 Bernoulli/probit/logit likelihood，记录估计不确定性。
7. **核/GP 的扩展性**：百万级 trial 不适合直接构造稠密 \(M\times M\) 核矩阵，需 Nyström、inducing points、分块、有限基等；统计复杂度与计算成本同时汇报。

---

## 5. 模块 M3：从 Hessian 结构指标定义“复杂度 / 简单性”标量 loss

### 5.1 不是惩罚影响强度，而是惩罚结构复杂度

M3 输入可以是：M1 的固定数值矩阵、单轴结构向量、条件结构函数 \(M(U)\)、局部 H/W 导数字段或更高阶对比。输出必须是对每个 R 可以计算的**一个标量**：

$$
\boxed{
L_k(R)=S_k\bigl(\mathcal A_a(H_R;Q)\bigr)
}
$$

或 local-first 版本 \(L_k(R)=\mathbb E_Q[S_k(H_R(W))]\)。

我们的主要直觉不是让函数尽可能接近常数、梯度尽可能小或者 Hessian 总能量尽可能小，而是：

> **如果一个 U 只通过少数简单的、可分解的方式作用于少数 X，那么它比一个同时参与许多复杂多元交互、且依赖关系高度非线性的 U 更简单。**

例如 \(f(u,x)=100ux\) 虽然作用强度很大，但结构上只有一个低阶 U–X 交互；其结构复杂度应有机会小于一个由很多弱但不同的高阶混合项构成的函数。**幅度稀疏 \(\sum|M_{ij}|\) 与连接稀疏 / 函数复杂度不是同一个目标**。

### 5.2 K-A：两两交互结构的复杂度（U–X / U–U / 全 W）

定义各类交互 strength matrix（示例为 RMS）：

$$
M^{UX}_{ij}(R)=
\sqrt{\mathbb E_Q[H_{u_i x_j}(W)^2]},
$$

$$
M^{UU}_{ik}(R)=
\sqrt{\mathbb E_Q[H_{u_i u_k}(W)^2]},
\qquad i\ne k.
$$

**这些矩阵的元素绝对强度本身还不是复杂度**。下面列出可独立尝试的**复杂度定义**：

| K 编号 | 复杂度候选 | 示例数学定义 / 思路 | 想惩罚什么 |
|---|---|---|---|
| KA01 | **Hard support count** | \(\sum_{ij}{\bf1}(M^{UX}_{ij}>\tau)\) | 激活的 U–X 边数 |
| KA02 | **Soft-L0 support** | \(\sum_{ij}M_{ij}^2/(M_{ij}^2+\tau^2)\) | 可微近似“连接条数” |
| KA03 | **Row degree penalty** | \(\sum_i \mathrm{softCount}(M^{UX}_{i,:})\) | 一个 U 连接多少 X |
| KA04 | **Per-U concentration / entropy** | 对每行归一化作用权重 \(q_{ij}\)，累加熵 \(-\sum_j q_{ij}\log q_{ij}\) | 作用分散到很多 X |
| KA05 | **Effective number of effects** | \(\exp(\mathrm{Entropy}(q_i))\)，或 \(1/\sum_jq_{ij}^2\) | 有效连接数，不直接依赖总体幅度 |
| KA06 | **Top-k tail** | 每行前 k 个最大元素之外的比例能量 | 能否由少量连接解释 |
| KA07 | **Structured group sparsity** | 对每个 U 的 X/条件函数整体作 support/组惩罚 | 交互是否集中在少数效应 |
| KA08 | **UU off-diagonal support** | 仅对 \(i<k\) 统计非零 \(M^{UU}_{ik}\) | 不同 U 之间的非加性交互个数 |
| KA09 | **Joint UX+UU graph complexity** | 组合两个图的边数、度数分布 | 跨 U 与跨 X 的整体混杂 |
| KA10 | **Full W interaction support** | 对 Hessian 全部非对角 pair 建图，按子块分权 | 变量两两作用结构整体复杂度 |
| KA11 | **Higher-order / graph motifs** | 用三/四阶多变量导数字段统计三元以上超边 | 二阶边无法概括的交互阶数 |
| KA12 | **Low-rank interaction structure** | 对 \(M^{UX}\) 的有效 rank / 奇异值尾部评分 | 是否需要很多独立作用模板 |
| KA13 | **Conditional link instability** | 比较同一 U–X 边在不同 X 子域/U 子域的激活支持 | 连接结构是否频繁变化 |
| KA14 | **Additivity gap** | 在固定其他变量后，对两个变量构建 2×2 或 3×3 对比的非加性残差 | 不要求显式计算二阶导数的交互 |
| KA15 | **Sparse interaction decomposition** | 用若干单变量/两变量作用成分逼近，按非零成分数评分 | 最简加法+低阶交互表达 |

要特别说明以下风险：

- 对 \(M\) 取简单 L1 往往主要惩罚**幅度**，未必反映“连接数量”；
- hard/soft support 依赖阈值 \(\tau\)，阈值必须以**尺度不变的标准化导数**或相对能量设定，并进行敏感性分析；不能调 \(\tau\) 去迎合 TEST T；
- **纯粹惩罚 UU 非对角 Hessian 并不保证选对认知参数轴**。例如真实函数 \(f(t_1,t_2)=t_1t_2\) 在真实轴上有交叉二阶导；45° 旋转后变为 \((u_1^2-u_2^2)/2\)，反而没有 UU 交叉 Hessian。因此 KA08 可能偏好错误方向，必须经过解析 oracle；
- UX 的 \(\|M\|_F^2\) 若只是 \(\mathbb E\|H_{UX}\|_F^2\) 的正交旋转不变量，则整条角度曲线是常数，**不能充当轴搜索 loss**；
- X–X 结构本身不随 U 旋转，单独的 X–X 全局能量可能完全无法识别 R。只有清楚说明某种与 U 的耦合方式，才可纳入目标；
- 独立参数先验不意味着**旋转后的 U 仍独立**（CR3 尤其非高斯），经典 Sobol 分解的独立输入假设需要单独核验；不可将任意相关输入下的 ANOVA 分量当成标准 Sobol 指数。

### 5.3 K-B：单个 U 的作用复杂度、非线性与阶数

这类指标不是“U 连了多少 X”，而是“**某个 U 的效应作为函数有多难描述**”。

例如：

$$
h_1(U,X)=u_i x_j,\qquad
h_2(U,X)=u_i^3 x_j.
$$

二者均有一条 U_i–X_j 边，但 \(h_2\) 对 U_i 的作用非线性更强，需额外的函数结构才能描述。

| K 编号 | 单轴候选 | 所依赖的 M1 / 函数信息 | 希望度量的复杂度 |
|---|---|---|---|
| KB01 | **Mean self-curvature** | \(\mathbb E|H_{u_i u_i}|\) | U 自身的局部非线性 |
| KB02 | **Self-curvature RMS / variance** | A03 / A05 的 UU 对角 | 强曲率及空间变化 |
| KB03 | **Curvature tail / occupancy** | A09 / A11 的 UU 对角 | 少数区域是否高度弯曲 |
| KB04 | **Gradient change along U** | \(\partial h/\partial u_i\) 的受控单轴曲线总变差 | U 效应是否需要频繁变化 |
| KB05 | **Hessian variation** | \(\partial_{u_i}H_{u_i u_i}\) 及相关三阶字段 | 自曲率本身是否变化 |
| KB06 | **Higher-order U–X coupling** | \(\partial_{u_i}H_{u_i x_j}\)、\(\partial_{u_i}^2H_{u_i x_j}\) | 区分 UX 与 \(U^3X\) |
| KB07 | **Local linearity / quadratic approximation error** | 沿 \(u_i\) 扰动，预测 vs 一/二阶 Taylor | 给定局部尺度上是否接近简单低阶函数 |
| KB08 | **Univariate polynomial/spectral tail** | 对 U_i 条件效应作正交基展开，统计高阶尾部 | 所需有效多项式阶数 |
| KB09 | **Partial-dependence curvature** | 在固定 Q 下对 U_i 的平均响应曲线求曲率 | 单轴平均效应非线性；平均可能抵消 |
| KB10 | **Conditional effect complexity** | U_i 效应随 X 子域的局部曲线族 | 同一 U 在不同 X 上作用是否规律 |
| KB11 | **Monotonicity violations / turning points** | 固定 X 下沿 U_i 的可微曲线 | 效应是否反复改变方向 |
| KB12 | **1D functional total variation** | 对二阶/三阶曲率场沿 U_i 积分总变差 | 不仅看强度，也看曲折复杂度 |
| KB13 | **Minimal basis / effective dimension** | 在受控误差下拟合 U_i 的最低复杂度单变量作用 | 单轴函数表达是否紧凑 |
| KB14 | **Higher-order interaction participation** | 指定高阶 \(U_i,U_j,X_k\) 等导数/差分 | U_i 是否参与高阶多元交互 |

**无法只用一个固定的二阶 Hessian 数值矩阵计算所有 KB 指标。** 例如 KB06 必须保留局部 \(H_{UX}(U,X)\) 关于 U 的变化，或直接获得更高阶导数；M1 聚合过早将导致不可逆信息丢失。M2 方法要根据要求明确提供 F2/F3/F0 能力。

### 5.4 合并与模型选择：保持每个候选可单独评价

可探索交互 K-A 与单轴 K-B 的组合：

$$
\boxed{
L_{\mathrm{simplicity}}(R)
=\lambda_A L_{\mathrm{interaction}}(R)
+\lambda_B L_{\mathrm{single}}(R).
}
$$

但这里 \(\lambda_A,\lambda_B\) 是尚待研究的**结构权重**，不能以 TEST 真实 T 的相关最大化来调；如为了验证存在性而用 T 调权重，必须明确标成 Oracle / supervised diagnostic，不能宣称无监督算法。

所有候选都应保存原始量纲与经过固定规则标准化后的损失；不能因两个指标的数值单位不同而任意放大其中一项。对比必须保留单独 A、B 结果而不是只输出总 loss，否则无法判断方向偏好来自哪种结构。

---

## 6. 三条 Y / Hessian 来源路径：同一框架内的对照

你提出的关键想法是：**让同一套 W，在不同的 Y 生成方式下进入同一个 M2 → M1 → M3 流水线**，从而定位问题到底出在函数本身、数值估计器，还是 decoder 的函数学习上。这个方向可行，但**解析函数接收 T 而非 U/Z**，因此必须明确对应关系。

### 6.1 路径 NN-Y：冻结 decoder 生成数值 Y（主要应用场景）

给同一批 \(W_{st}=[RZ_s,X_{st}]\)，通过**逆旋转**访问冻结 decoder：

$$
Y_{st}^{\mathrm{NN},\eta}=d_R^\eta(U_s,X_{st}),
\quad
Y_{st}^{\mathrm{NN},p}=\sigma(Y_{st}^{\mathrm{NN},\eta}),
$$

$$
Y_{st}^{\mathrm{NN},b}\sim\mathrm{Bernoulli}(Y_{st}^{\mathrm{NN},p}).
$$

依次输入 M2 估计、M1 聚合与 M3 简单性 loss。搜索 R 时不能读取真实 T。

还应设置一个辅助参考：**直接对冻结 decoder 使用自动微分**，求该网络本身的局部导数字段，和通过它生成的离散 WY 由 M2 得到的估计比较。这是同一个 decoder 的 direct-derivative benchmark，**并不是**解析真函数的 Hessian。将其独立记录，才能把统计估计误差与 decoder/生成函数差异分开。

### 6.2 路径 Formula-Y：已知解析生成式生成数值 Y，再隐藏公式

当使用**相同 W** 时，解析函数无法无条件直接代入 U，因为：

$$
f_{\mathrm{std}}(T_{\mathrm{std}},X)\quad
\text{与}\quad d_R(U,X)
$$

的参数坐标不同。尤其 \(D^*R^*Z\approx T_{\mathrm{std}}\) 通常只是**近似关系**，不能把任意 U 简单当作 Tstd。

提供两种必须分开命名的 formula-Y 诊断协议：

**F-Paired（配对真实参数 Y）**：同样的 \(U_s=RZ_s\)、\(X_{st}\)，却用该被试**真实 T_s** 生成连续/二元响应：

$$
Y_{st}^{\mathrm{Fpaired},\eta}=f_m(T_s,X_{st}),
\qquad
Y_{st}^{\mathrm{Fpaired},p}=\sigma(f_m(T_s,X_{st})),
$$

$$
Y_{st}^{\mathrm{Fpaired},b}\sim\mathrm{Bernoulli}(Y_{st}^{\mathrm{Fpaired},p}).
$$

这与 NN-Y 使用**完全相同的 W 行**，因此能比较数值估计器接收到不同来源 Y 后的结果。但 F-Paired **使用了 T 生成 Y，属于 oracle 数据源**；若存在 \(T_s|U_s\) 的信息噪声，它未必是 U 的确定性函数，即 M2 实际估计的是 \(E[Y|U,X]\)。**不能**据此把 F-Paired 的估计 Hessian 宣称为 \(f_{\mathrm{std}}(U,X)\) 的解析 Hessian。

**F-Mapped（经已知坐标映射的公式 Y）**：为研究函数坐标结构，使用**事先监督拟合的、锁定的** \(P^*,D^*,R^*_{\mathrm{oracle}}\) 构造近似真参数：

$$
\widehat T_{\mathrm{std}}(u;R)
=P^*D^*R^*_{\mathrm{oracle}}R^\top u,
$$

$$
Y_{st}^{\mathrm{Fmapped},\eta}
=f_{m,\mathrm{std}}(\widehat T_{\mathrm{std}}(U_s;R),X_{st}).
$$

相应 p 与 Bernoulli Y 由 sigmoid 和 Bernoulli 抽样生成。这里的映射为固定参考，不随待扫描 R 重新监督拟合；若使用 R* 来构造 F-Mapped，它**明确是监督锚定的诊断**，不是可直接部署的无监督旋转源。其优点是它提供一个定义明确、可微且能和 M2 比较的“在学习 Z 坐标上的代理真实生成函数”。

若 \(P^*D^*R^*_{\mathrm{oracle}}Z_s\) 与真实 Tstd 的差距不小，F-Mapped 与 F-Paired 产生的 Y 也不同；应单独报告其误差。

**三种输出通道均分别计算**：logit、probability、Bernoulli binary，绝不把单次二元随机值当成连续解析 Hessian。

### 6.3 路径 Analytic：完全不从 WY 统计估计，直接用解析式求导和 loss

这是**最关键的理论识别性基准**。

直接使用 \(f_{m,\mathrm{std}}(t,x)\)，在 Tstd 的真实坐标下对其求 \(\nabla^2\)（必要时三/四阶导），从而取得**无统计估计误差的真实导数字段**。为了比较真实参数轴与混合旋转，令：

$$
v=Qt,\quad Q\in O(d),\qquad
f_{\mathrm{std},Q}(v,x)=f_{m,\mathrm{std}}(Q^\top v,x).
$$

这里 \(Q=I\) 就是**真实标准化参数坐标**，其他 Q 是真参数坐标的受控混合。求出每个 Q 下的 M1、M3，画函数复杂度随 Q 改变的曲线，直接回答：

> **真实参数轴（及其带符号置换）是否是这个简单性指标的最小点？**

对于 UU Hessian，坐标变换可解析计算：

$$
H_{vv}(v,x)=Q H_{tt}(t,x)Q^\top,
\qquad
H_{vx}(v,x)=Q H_{tx}(t,x).
$$

若对真实 T 做了标准化，先使用：

$$
H_{tt}=S_T H_{TT}S_T,\qquad
H_{tx}=S_T H_{TX},
$$

然后再按 Q 旋转。若 X 也标准化，还应右乘 \(S_X\)；任何跨通道比较都应注明量纲。

**重要区分**：在真参数 Tstd 坐标进行的 Q 扫描，其理想方向为 \(Q=I\)，**不是直接把学习 Z 空间的 \(R^*_{\mathrm{oracle}}\) 塞给 fstd**。若希望用学习 Z 的监督 R* 作为公式路径锚点，必须通过 F-Mapped / 显式坐标映射转换，且记录其近似误差。解析真函数可以没有映射误差地测试“真轴是否最简单”，这正是最值得先过的关卡。

此外，C1/C2/C3 的 **logit 对 T 是线性的**，故 \(H_{TT}^\eta=0\) 对**所有正交参数旋转**都为零；单独的 logit UU 自曲率或 offdiag Hessian loss 不可能识别这些模型的真实 T 轴。要关注 UX 结构、函数值效应、高阶 X 依赖及 probability 通道，而不是将平坦 UU 曲线误判为代码故障。CR3 的 logit 对 T 非线性，所以 UU 可非零，但不能因此自动证明真轴最小。

---

## 7. 绕开数值推断误差的角度曲线、全空间搜索与方向诊断

### 7.1 Oracle R* 周围的多平面偏离扫描

先在**学习 Z 坐标**用第 1.3 节的监督方法确定 \(R^*_{\mathrm{oracle}}\)（仅用 TRAIN，固定一次），然后对一个二维坐标平面 (i,j) 施加附加 Givens 偏转：

$$
\boxed{
R_{ij}(\varphi)=G_{ij}(\varphi)R^*_{\mathrm{oracle}}.
}
$$

对所有 \((i,j)\) 平面分别画：

$$
\boxed{
\varphi\ \longmapsto\
L_k(R_{ij}(\varphi)).
}
$$

**横轴**：相对于**固定的**监督 Oracle R* 的旋转角度（度或弧度一致）；**纵轴**：M1+M3 定义的某个 simplicity loss。对于四维 R 有 6 个平面；六维 C3 有 15 个平面。横轴也可采用与带符号置换等价类之间的正交矩阵测地距离，避免将 90° 的轴交换误判为另一组错误的参数方向。

**我们希望看到**：在 0°（或者与 R* 等价的 90°、符号/排列位置）附近 loss 较小；离开这些等价类后，loss 一般上升。**但 R* 是监督预测最优，不代表它是解析真函数上的必然简化最优；这仍然需要实验检验。**

每条曲线至少保存：0°、±小角度、全 0–90°（需要时扩到 0–180°）；最小值角度、0° loss、局部斜率/曲率、全局差距、端点等价比较、flatness 和起点/种子稳定性。数值积分节点和随机种子应在角度间**共享**，不能把抽样噪声画成旋转起伏。

### 7.2 真解析函数中的独立“0° 真轴”扫描

为了去掉学习表征与监督映射近似误差，另在**Tstd 自身坐标**上设置 Q=I 为 0°，旋转 \(Q_{ij}(\varphi)=G_{ij}(\varphi)\)，对真实解析 Hessian 直接评估 \(L(Q_{ij}(\varphi))\)。这与 7.1 的学习 Z 空间扫描是**两个不同实验**，不应仅因为名字都叫“oracle curve”就混为一谈。

**判断顺序建议：先检验真解析轴的可识别性，再检验统计 WY transfer 和冻结 NN transfer**。如果解析真函数都在错误方向取得更小损失，那么继续增加 WY 推断方法或改进 decoder Hessian 拟合，也无法修复该指标的方向识别错误。

### 7.3 两两平面只能是诊断，必须继续做全空间优化

仅在每个二维平面上验证 0° 为局部最小，不能证明六角度（或十五角度）联合旋转的全局最小。需要多初始条件、全角度联合优化与独立数值验证，至少比较：

1. 真 Tstd 轴（analytic Q=I）；
2. 监督 Oracle 学习 Z 方向 R*；
3. 无监督 loss 得到的 \(\widehat R\)；
4. 随机旋转、多起点局部最优；
5. 与前述轴等价的任意 signed permutation。

优选指标应具有足够的“正确方向 margin”，而不是在全部旋转上几乎平坦，也不能只是一个在某个特定平面偶然得到 0° 极小的指标。

---

## 8. 本阶段的硬性数学约束：缩放不变、符号不变、排列不变

### 8.1 对逐轴 D 不变：使用候选坐标 Z-score Hessian

普通 Hessian 会随 U 的单位变化：若 \(v=RZ\)，再变为 \(u=Dv\)，则

$$
H_{uu}=D^{-1}H_{vv}D^{-1},\qquad
H_{ux}=D^{-1}H_{vx}.
$$

所以直接最小化 \(\sum|H_{uu}|\) 或 \(\sum|H_{ux}|\) 会受到 D 单位影响，并可能被无限放大 D 人为降低。

定义**每一个候选旋转后的输出轴**的 TRAIN 标准差：

$$
s_i(R)=\mathrm{SD}_{\mathrm{TRAIN}}((RZ)_i),\qquad
S_R=\mathrm{diag}(s_1,\ldots,s_d).
$$

按候选轴构造逐轴 Z-score：

$$
\boxed{
\widetilde U=S_R^{-1}(RZ-\mu_R).
}
$$

在固定 R 下对应同一个冻结函数的**标准化 Hessian**：

$$
\boxed{
\widetilde H_{UU}
=S_R\,(R H_{ZZ}R^\top)\,S_R,
\qquad
\widetilde H_{UX}
=S_R\,R H_{ZX}.
}
$$

若 X 也按固定 TRAIN 定义标准差 \(S_X\) 转换，则 \(\widetilde H_{UX_s}=S_R R H_{ZX}S_X\)；对 \(XX\) 的标准化同理。对输出 Y 自身是否标准化要另外说明（binary probability 已无量纲；logit 可按固定尺度作对照，但**不能**在各旋转角度重新任意改变 output 单位）。

在任何非零逐轴 \(u'=Du+b\) 之后，同一个候选轴的标准化坐标只变化逐维符号，因而 \(|\widetilde H_{ij}|\)、平方、行/列对称聚合以及支持复杂度指标可构造为对 D 严格不变。要求每一项 K loss **单独通过测试**：

$$
\boxed{
L(DR)=L(R)\quad\text{和}\quad L(PR)=L(R),
}
$$

这里 \(P\) 为带符号置换。**候选坐标 Z-score 是在旋转之后做，不是先将原始 Z 任意各向异性缩放再声称 R 搜索空间没有变化。**

对任意非零 D，标准化系数的符号会影响某些**有符号统计指标**；只有取绝对值/平方或设计成符号对称的 M3 loss，才能保持严格不变。M1 可以保留 signed mean 供诊断，但不能未经处理就宣称它满足所有不变性。

### 8.2 三种不变性之外，还必须证明指标具有方向分辨能力

设计一个对缩放、符号、排列都不变的损失仍不代表能恢复真轴。其最小值的位置是第四个独立的科学问题：

$$
\boxed{
\mathrm{argmin}_{R\in O(d)}L(R)
\ \stackrel{?}{\approx}\
[R^*_{\mathrm{oracle}}]
}
$$

其中方括号表示 signed-permutation 等价类。

例如 \(f(t_1,t_2)=t_1t_2\) 在真坐标有纯交互，但 45° 旋转后变为 \((u_1^2-u_2^2)/2\)；惩罚非对角 UU Hessian 的 loss 会偏好**错误方向**。此外在仅对 U 旋转时 \(\|RH_{UU}R^\top\|_F=\|H_{UU}\|_F\) 严格不变，直接取所有元素 Frobenius 能量也不会辨识 R。**不变性≠可识别性**。

若需要保证整体在任意前置 Z 单位变化下可比，应单独考虑白化后的搜索几何；此处仅证明了**同一候选 R 之后**的 D 缩放不变，并不证明任意 \(Z\mapsto BZ\) 预处理不改变问题。

### 8.3 每个指标必须提供可运行的对称性单元测试规范

- 随机抽样多个 R、正负和悬殊尺度的 D、任意 signed-permutation P；
- 将变换后 WY/decoder 逆坐标正确调整，核验 function values 不变；
- 对局部 H 的导数变换按解析公式检验；核验标准化矩阵指标正确变换；
- 数值验证标量 \(L(DR)\approx L(R)\)、\(L(PR)\approx L(R)\)（严格不变的数值容差必须预设）；
- 验证 loss 在部分非等价 R 上**确实变化**，且变化不是估计器噪声；
- 不满足不变性的候选指标直接标记 `not eligible for R-only search`。

---

## 9. 分阶段实验门槛和建议输出

把问题拆成明确的四个科学门槛：

| 阶段 | 输入 | 不允许使用什么 | 必须回答的问题 |
|---|---|---|---|
| **G0：数学/数据审计** | frozen Z/decoder、固定 X、选定指标 | T 不参与无监督实现 | 逆变换预测是否严格等价？尺度/符号/排列不变吗？ |
| **G1：解析真轴识别** | 已知 \(f_{\mathrm{std}}(Tstd,X)\) 和解析导数 | 不用 M2 推断 | 真参数坐标是否令某个 loss 最小？是否存在错轴更小？ |
| **G2：Formula-Y → WY infer** | 同一声明的 W 和 formula 生成的数值 Y | M2 不准读取公式、真导数 | 不知道函数形式，能否复现 G1 的导数指标与 loss 排序？ |
| **G3：Frozen NN-Y → WY infer** | 相同 W、冻结 decoder 生成三种数值 Y | 优化与调参不得使用 T | 换为神经网络输出，指标和最优 R 是否仍然有效？ |
| **G4：独立 TEST 验证** | 固定的无监督 \(\widehat R\)，仅事后使用真实 T | TEST T 不参与拟合或选参 | U 与 T 是否形成一一高相关、错配低相关？能否接近监督 Oracle？ |

**三个独立模块之间的主要误差路径：**

$$
\text{真实生成函数的结构}
\longrightarrow
\text{M2 估计导数字段}
\longrightarrow
\text{M1 聚合指标}
\longrightarrow
\text{M3 loss / 最优方向}.
$$

再把生成函数换成冻结 decoder，即增加**decoder 函数差异**这第四类偏差。必须同时观察 Y prediction error、Hessian 误差、矩阵指标误差、rotation loss 曲线误差与最终方向误差；单个预测性能数字无法替代后四项。

### 9.1 建议每个模型 / Y 来源 / 估计器 / 指标套餐至少保存

1. **运行元数据**：model、d/p、样本数、trial 数、TRAIN/VAL/TEST IDs、数据/checkpoint hash、Y 通道与来源、binary sampling 协议、R 参数化、D/标准化协议；
2. **M2**：估计器及超参数、函数值误差、局部 gradient/Hessian 各分块误差、是否提供 F2/F3/AVG，以及计算成本；
3. **M1**：每个 R 的 ZZ/ZX/XX 数值矩阵和单轴 self vector，平均/RMS/方差/分位数/条件指标、积分 Q 和有效域比例；
4. **M3**：分别保存交互复杂度 KA 与单轴复杂度 KB 的单项值、组合权重、总 loss 和不变性测试；
5. **旋转图**：六/十五平面的角度曲线、端点/局部/全局 minima、loss 的 flatness、R* 和无监督 Rhat 的相对距离；
6. **TEST 一一对应**：U–T signed/absolute Pearson 全矩阵、Hungarian matched mean、最差 matched、平均/最大 offdiag，且和 T–T 相关及监督 Oracle 对比；
7. **诊断图**：匹配和非匹配的 U–T 散点，U–U 相关，预测 probability/logit 的分布，局部 Hessian 误差，模型间稳定性。

原则：**M1、M2、M3 各自维护可扩展的指标目录与接口**；未来新增估计器不应改写已有损失定义，新增 loss 不应要求硬编码某个估计器。执行仍沿用项目既有约束：每个 rotation 套餐单独文件夹、单独 run.py，不建“一次运行所有旋转方法”的总入口。

---

## 10. 参考文献和当前仓库相关文档

下面是**方法背景文献**，用于说明统计估计、样条、核、导数、高阶交互或函数分解的通行做法；它们**并没有替本项目证明 Hessian 简单性会恢复认知参数轴**。文章链接优先给出版方、学术会议、DOI 或已建立的检索入口。

- **[R1]** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). [DOI](https://doi.org/10.1007/978-0-387-84858-7). 多元回归、basis/核、regularization、GLM。
- **[R2]** Wahba, G. (1990). *Spline Models for Observational Data*. [图书信息](https://search.worldcat.org/title/1120931056). 样条、RKHS、平滑拟合。
- **[R3]** Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). [出版方](https://www.routledge.com/Generalized-Additive-Models-An-Introduction-with-R-Second-Edition/Wood/p/book/9781498728331). Tensor-product smooth / GAM。
- **[R4]** Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. [出版方](https://www.routledge.com/Local-Polynomial-Modelling-and-Its-Applications-Monographs-on-Statistics/Cox-Fan-Gijbels-Reid-Isham-Tibshirani-Louis-Tong-Keiding/p/book/9780412983214). 局部多项式及导数估计。
- **[R5]** Eilers, P. H. C., & Marx, B. D. (1996). Flexible Smoothing with B-splines and Penalties. *Statistical Science*, 11(2), 89–102. [DOI](https://doi.org/10.1214/ss/1038425655). P-splines。
- **[R6]** Wahba (1990)，同 [R2]；penalty spline / smoothing spline 的统计理论背景。
- **[R7]** Wood (2017)，同 [R3]；GAM 与 smooth logistic / binomial model。
- **[R8]** Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to Speed Up Kernel Machines. [会议相关记录](https://www.inf.ed.ac.uk/publications/report/1036.html). 核近似。
- **[R9]** Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. [NeurIPS](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). 大规模核方法。
- **[R10]** Solak, E., Murray-Smith, R., Leithead, W. E., Leith, D. J., & Rasmussen, C. E. (2002). Derivative Observations in Gaussian Process Models of Dynamic Systems. [NeurIPS](https://papers.nips.cc/paper_files/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html). GP 导数推断。
- **[R11]** Fornberg, B. (1988). Generation of Finite Difference Formulas on Arbitrarily Spaced Grids. *Mathematics of Computation*, 51, 699–706. [DOI](https://doi.org/10.1090/S0025-5718-1988-0935077-0).
- **[R12]** Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. *Analytical Chemistry*, 36(8), 1627–1639. [DOI](https://doi.org/10.1021/ac60214a047).
- **[R13]** Janzamin, M., Sedghi, H., & Anandkumar, A. (2014). Score Function Features for Discriminative Learning: Matrix and Tensor Framework. [UC eScholarship](https://escholarship.org/uc/item/60j282mk). 高阶 score / integration-by-parts 思想，**非本项目 weighted Stein 公式的特定实现说明**。
- **[R14]** *On Sharp Stochastic Zeroth-order Hessian Estimators over Riemannian Manifolds* (2023). [期刊](https://academic.oup.com/imaiai/article/12/2/787/6780267). 零阶 Hessian 估计研究背景。
- **[R15]** Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic Differentiation in Machine Learning: a Survey. [JMLR](https://www.jmlr.org/papers/v18/17-468.html).
- **[R16]** Pearlmutter, B. A. (1994). Fast Exact Multiplication by the Hessian. *Neural Computation*, 6(1), 147–160. [DOI](https://doi.org/10.1162/neco.1994.6.1.147).
- **[R17]** Sobol, I. M. (2001). Global Sensitivity Indices for Nonlinear Mathematical Models and Their Monte Carlo Estimates. *Mathematics and Computers in Simulation*, 55, 271–280. [DOI](https://doi.org/10.1016/S0378-4754(00)00270-6). Sobol / functional ANOVA 需要相应输入分布及依赖假设；本 README 的 3×3 local gap 并非直接的 Sobol index。

**与本仓库现有研究框架衔接（不重复另建一个中央运行器）：**

- [上一篇：01 · 整体项目与两种搜索空间](01_project_overview.md)；
- [Hessian-from-ZXY 估计器方法总表 E01–E14](../HESSIAN_FROM_ZXY_METHODS.md)；
- [积分与平均 Q](../HESSIAN_INTEGRATION_AND_AVERAGING.md)；
- [Hessian → 数值矩阵 M01–M20](../HESSIAN_TO_MATRIX_METRICS.md)；
- [Hessian 双轴运算：消除变量依赖 D × 输出形状 P](../HESSIAN_FUNCTIONAL_OPERATIONS.md)；
- [CR3 analytic Hessian vs function-value factorial effects](../../math/08_cr3_functional_sparsity/README.md)；
- [监督 Procrustes / OLS 上限诊断](../../math/09_procrustes_vs_ols/README.md)。

### 本 README 最终一句话

> 以固定被试 Z、各人 X 和冻结 decoder 为基础，对每个正交旋转 R 定义 \(U=RZ\) 与 \(W=[U,X]\)。从 WY 或解析生成函数取得 Hessian/高阶导数字段，经**独立的 M1 结构聚合、M2 统计推断与 M3 简单性定义**生成无监督损失并搜索 R；再使用只在评价阶段可见的真实 T 检查是否找到了与认知参数一一对应的轴。所有纳入 R-only 优化的损失必须对旋转后的逐维缩放 D、轴符号和轴排列不变，同时必须通过解析真轴最小性和 numeric/decoder 迁移三重检验。
