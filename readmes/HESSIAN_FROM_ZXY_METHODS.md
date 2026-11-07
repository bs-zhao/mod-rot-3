# 从数值矩阵 (Z, X, Y) 提取 Hessian：方法目录与文献

> **研究位置**：本模块是 [Mod-Rot 研究总框架](README.md) 中新增的**数值 Hessian 估计器**独立维度，位于 Y 的来源与积分/平均之间。适用于 C1/C2/C3/CR3。
>
> **范围与原则**：仅用数值输入和响应估计函数导数，不允许估计器读取生成公式、真实参数 T 或解析 Hessian 进行拟合。解析 Hessian 可以用于**实验后准确性比较**。下文清楚区分**仓库已有实现**和**研究候选**；列入目录不表示已经实现或验证。
>
> **两个正交因素**：data source（formula-Y 与 NN-Y）决定 *Y 从哪里来*；estimator（本模块 E01–E14）决定 *如何从给定数值 ZXY 恢复 Hessian*。不能混为一个实验因素。

## 1. 估计的数学对象：先确定 estimand

令 W=(U,X)，其中 U 可以是原始 NP 的 Z，也可以是候选旋转 U=AZ+b。

\[
f_y(w)=\mathbb E[Y_y\mid W=w],\qquad
H_y(w)=\nabla_w^2 f_y(w)=
\begin{bmatrix}H_{UU}&H_{UX}\\H_{XU}&H_{XX}\end{bmatrix}.
\]

这里 y 是 logit、probability 或 binary choice 的表达。**如果 y 是观察到的 Bernoulli 0/1 choice，连续目标是其条件均值 p(U,X)，而不是“单次二值选择的二阶导数”。** 如果通过 logistic link 拟合 eta=logit p，则 eta 与 p 的 Hessian 不同。

| 目标量 | 数学定义 | 对简单性指标的支持 |
|---|---|---|
| 局部 Hessian | 每个评价点的 H(U_s,X_t) | 局部 ZZ/ZX/XX、RMS、高阶曲率及逐点稀疏 |
| 平均 Hessian | E_μ[H(U,X)] | 只能评价平均矩阵的结构，**不能推回局部变化** |
| 局部 RMS/方差 | sqrt(E_μ[H²])、Var_μ(H) | 与 abs(E[H]) 不同，可发现均值抵消 |
| 带权平均 | E_μ[ωH]/E_μ[ω] | 与无权均值是不同的统计对象 |
| 拟合函数的 Hessian | ∇² f_hat(U,X) | 即使 Y 预测准确，也不保证接近真实 ∇² f |
| Fisher / NLL Hessian | score 外积或负对数似然的二阶导 | **不是响应函数 f_y 的 Hessian**，需要单列 |

**实验支持差异**：C1/C2/C3 的 X 逐被试/逐试次独立生成；CR3 则是固定 500 行 Xset，全体被试共享。每位被试的不同试次仍共享同一个 Z；训练/验证/测试和 bootstrap 必须按被试分组。对于 CR3，先求每被试平均 Y 再在 Z 上拟合只能直接恢复 X 平均函数的 UU Hessian，不能恢复 UX、XX 或 X 方向的逐点高阶变化。

**识别极限**：有限个散点 (Z,X) 上的 Y 并不唯一确定连续 X/Z 方向的 Hessian。两个连续函数可在全部观察点取相同值，却具有不同导数；任何数值方法都依赖可微性、平滑正则、函数类、采样域和额外外推假设。**这是统计识别限制，不是单纯优化程序缺陷。**

## 2. 估计器类型总表

| ID | 统计/数学方法 | 恢复局部 Hessian？ | 方法特点与局限 | 当前仓库状态 |
|---|---|---|---|---|
| **E01** | Global polynomial sieve / series | 是 | 全局多项式基、可解析求导；需控阶数和振荡 | **已实现** math/01/04/06/07 |
| **E02** | Tensor-product / low-rank separable bases | 是 | f≈φ_ZᵀBφ_X；快速矩阵运算，需限制张量复杂度 | **已实现** math/01/05/06 |
| **E03** | Local polynomial regression / local quadratic | 是 | 点附近核加权二次拟合；高维邻域稀疏 | **候选，未实现独立估计器** |
| **E04** | Penalized B-splines / P-splines | 是 | 样条曲面及二阶惩罚；节点选择重要 | **已有 T-only 平均版本** math/02 |
| **E05** | GAM / ANOVA interaction splines | 是（受模型类限制） | 主效应/指定交互样条；可能预先强制简单性 | **候选** |
| **E06** | Gaussian RBF kernel ridge regression (KRR) | 是 | 核拟合后对核求导；长度尺度影响曲率 | **已有 T-only 及 Z/X 版本** math/02、05 |
| **E07** | Gaussian process regression (GP) | 是 | 核微分得后验 Hessian 与不确定性；计算昂贵 | **候选** |
| **E08** | Orthogonal polynomial / spectral series | 是 | Chebyshev/Legendre/Fourier 等正交展开 | **Chebyshev 部分实现** |
| **E09** | Finite differences / Savitzky–Golay | **需连续查询或密网格** | 原始离散稀疏 ZXY 不能凭空获得偏移点 Y | **解析微分验证已用**，不等于完整 ZXY 估计器 |
| **E10** | Second-order Stein / integration by parts | **通常仅平均** | 不拟合局部函数；需要密度、边界及积分假设 | **已有 weighted T-only mean** math/02 |
| **E11** | Gaussian smoothing / randomized zeroth-order | 对平滑目标是 | 需要可查询函数；估计的是经过平滑的 Hessian | **候选** |
| **E12** | Independently fitted smooth NN surrogate + autodiff | 是（针对代理） | 与原 NP decoder 求导不同；预测好不代表导数好 | **候选** |
| **E13** | Binomial/logistic GLM or smooth logistic sieve | 是（针对拟合函数） | 适合 choice 的 Bernoulli 噪声；基函数仍需选定 | **候选** |
| **E14** | Hessian-vector products / matrix-free | 并非拟合器 | 已有连续函数后的计算技术，不从 ZXY 学函数 | **计算选项，非统计方法** |

**当前代码与状态的精确映射**：

- math/01：在真实 T 和 X 上的 tensor polynomial sieve，包含 X 固定 500 行的经验/先验积分。
- math/02：Nyström Gaussian KRR、penalized tensor cubic B-spline、second-order **weighted** Stein；**输入已对 X 平均的真实 T,Y，仅 4×4 TT 均值 Hessian**，不是完整 ZXY。
- math/04：原始 NP S5 Z、固定 X、六通道数值 formula-Y/NN-Y，通过同一种 full Z/X tensor polynomial 估计 10×10。
- math/05：多项式阶数、ridge、monomial/Chebyshev、full Z/X Gaussian RBF 基函数及 Z-only spline/KRR 对照。
- math/06/07：旋转重新拟合的数值多项式，07 附加局部二/三/四阶导数与受控扫描。**这里没有实现 E03/E05/E07 等全部候选**。

## 3. 方法原理、所需假设和失败原因

### E01 / E02：全局多项式、可分离张量基

\[
\hat f(z,x)=\phi_Z(z)^\top B\phi_X(x),\qquad
\hat H_{z_i x_j}=
(\partial_i\phi_Z)^\top B(\partial_j\phi_X).
\]

ZZ/XX 与三/四阶导数同理对基函数多次求导。ridge、最高阶、monomial/Chebyshev、张量 rank 都是估计器超参数。

- **优势**：固定 Xset 和许多被试时，可利用矩阵乘法与 factored moments，计算快、可重复、能输出所有 Hessian 块。
- **风险**：全局函数类的偏差；高阶多项式的边界振荡/病态；拟合 Y 的交叉验证误差很小，未必保证局部导数准确。
- **低秩变种**：对 B 施加低秩/稀疏限制可降低计算量，但不能未经检验假定真实函数具有同样的低秩作用。[R01][R02]

### E03：局部多项式回归

在每一个目标 w0=(z0,x0) 附近，以 δ_i=W_i-w0 构建局部二次回归：

\[
\hat\beta(w_0)=\arg\min_{a,b,H}
\sum_i K_h(\delta_i)
\left[Y_i-a-b^\top\delta_i-\frac12\delta_i^\top H\delta_i\right]^2.
\]

局部估计出的对称矩阵 H 即该点的 Hessian 估计；需估三/四阶时增加多项式阶数。

- **优势**：不必使用一个全局统一的多项式表达式，可拟合空间中不同区域的曲率。
- **风险**：当 d_Z+d_X 达 8–10 维时，局部邻域有效观测可能极少，估计方差高且边界偏差明显；带宽和局部模型阶数是核心方法维度。
- **参考**：Fan & Gijbels（1996）[R03]。

### E04 / E05：惩罚样条与广义加性模型

\[
\hat f(w)=\sum_j\beta_j B_j(w),\qquad
\hat\beta=\arg\min_\beta
\sum_i(Y_i-\hat f(W_i))^2+\lambda\beta^\top P\beta.
\]

P-spline 使用 B-spline 基与差分惩罚；可对可微的样条基直接求导。GAM 或 tensor ANOVA 可以只保留主效应和有限阶交互以减少参数个数。

- **优势**：粗糙度惩罚清楚，通常比高阶全局多项式更容易做局部平滑度控制。
- **危险的循环论证**：如果在拟合阶段强制函数可加，再在旋转后以「函数可加性」当作真轴指标，可能只是估计器施加的先验。须与未强制该简单性的估计器做对照。
- **参考**：Eilers & Marx（1996）、Wahba（1990）、Wood（2017）[R02][R04][R05]。

### E06 / E07：KRR / GP 与核函数导数

\[
\hat f(w)=\sum_i\alpha_i k(w,W_i),\qquad
\hat H(w)=\sum_i\alpha_i\nabla_w^2k(w,W_i).
\]

Gaussian RBF 核可解析求二阶导。Nyström 与 random Fourier features 可以减小训练开销；但**核函数值近似得好不意味着二阶核导数也近似得好**。X 与 Z 可使用分组长度尺度和不同正则化，必要时尝试 ARD/各维长度尺度。[R06][R07]

GP 对核的相应导数还可以推导**导数的后验不确定性**，但要求核足够可微且计算资源允许。Solak et al.（2002）提供了 GP 与导数观测建模的相关背景；本项目即使没有导数标签也可在条件满足时研究 GP 后验函数的导数。[R08]

### E08：正交基与谱方法

\[
\hat f(w)=\sum_{\alpha}c_\alpha\Psi_\alpha(w),\qquad
\hat H_{ij}(w)=\sum_\alpha c_\alpha\partial_{ij}\Psi_\alpha(w).
\]

Chebyshev/Legendre/Hermite 等谱基有不同定义域和正交测度要求；真实 T 先验不等于 NP Z 经验分布。**换正交基不等于自动学到正确 Hessian**；应单独量化截断偏差和基函数条件数。[R01][R02]

### E09：有限差分与平滑滤波微分

对于能够在任意偏移点查询的连续函数 f，

\[
\partial_{ii}f(w)\approx
\frac{f(w+he_i)-2f(w)+f(w-he_i)}{h^2}.
\]

混合二阶导数使用四个偏移点的中心差分。**原始有限 ZXY 矩阵一般并不包含这些位置的 Y**；不能把连续插值的结果伪装成原始观测。对 noisy choice 直接做差分会有随 \(h^{-2}\) 放大的误差。Savitzky–Golay 为规则或可处理的局部采样提供经典平滑与微分思想；Fornberg 给出了不等距网格的差分权重。[R09][R10]

### E10：二阶 Stein / 积分分部

对于有光滑密度 q、满足消失的积分分部边界项的连续 W，记 s=∇log q，可得

\[
E_q[H_f(W)] =
E_q\left[f(W)\{ss^\top+\nabla^2\log q(W)\}\right].
\]

**等式的边界条件不可省。** CR3 的截断先验存在非零边界密度，直接套用无权恒等式会不正确。项目 math/02 的实现引入在边界及其一阶导数为零的权函数 ω，使用

\[
E_\pi[\omega H_g]=E_\pi[g\Psi_\omega],
\quad\Psi_\omega=\frac{\nabla^2(\omega\pi)}{\pi}.
\]

因此报告的是加权均值 Hessian，**不是原先的无权 prior mean**；通常也不返回局部 H(U,X)。

更重要的是，已知的真实 T 先验不等于冻结 NP 的 Z 密度；CR3 的固定离散 X 也没有通常意义的连续密度 score。使用 Z/X 上的 Stein 估计需另行估计输入密度/score，明确边界假设，不能简单挪用原 T 先验。[R11][R12]

### E11–E14：补充方向

- **随机零阶估计**：在可以重复调用函数的情况下通过 Gaussian perturbation 估计 smoothed Hessian；原始一次性观测矩阵并不提供 off-grid Y，且平滑后的目标量不同。[R13]
- **独立 NN surrogate 与 autodiff**：仅用数值 ZXY 新拟合可微代理，对**该代理函数**求二阶导；不同于直接对原 NP decoder 求导。自动微分的数值精度不能保证对真实生成函数的导数恢复。[R14][R15]
- **Choice 用 logistic GLM / logistic series**：以 Bernoulli likelihood 训练可微 logit，随后根据 \(H_p=\sigma'(\eta)H_\eta+\sigma''(\eta)\nabla\eta\nabla\eta^\top\) 得到概率 Hessian。与数据噪声模型更匹配，但依然受到基函数假设制约。[R01][R05]
- **Hessian-vector product**：在拟合函数已知时更高效计算 Hv；**不是独立的从数值 ZXY 恢复 f 的统计方法**。[R15]

## 4. 方法比较与现有三路研究如何组合

将估计器 e 从数据来源 d 中分离：

\[
\widehat H^{(m,d,y,e)}=
\operatorname{HessianEstimator}_e
\bigl(\operatorname{NumericalData}_{m,d,y}(U,X)\bigr),
\]

\[
L_{m,d,y,e,q,k}(\varphi)=
S_k\!\left(\operatorname{Aggregate}_q
[\widehat H^{(m,d,y,e)}(U_\varphi,X)]\right).
\]

| 独立实验轴 | 含义 |
|---|---|
| m | C1 / C2 / C3 / CR3 |
| d | formula-Y（真实式只用于生成数值 Y）或 nn-Y（冻结 NP decoder 生成数值 Y）；analytic oracle 为真值路径 |
| y | logit / p / choice；对 choice 记录 sampling 与 hard-threshold 的区别 |
| **e** | **本模块估计器 E01–E14 的适用子集** |
| q | 保留局部 Hessian、X 平均、经验被试/真实先验积分，或其它明确的加权统计量 |
| k | 来自 Hessian 或导数字段的简单性指标 |
| φ | 两两或联合旋转角，简单性 loss 是随之变化的因变量 |

**公平对照原则**：

1. 固定 S1 数据 hash、NP checkpoint、S5 Z、X bank、Y 的输出定义、坐标 U、评价点以及**按被试分割**的 TRAIN/VAL/TEST。
2. formula-Y 与 nn-Y 路径使用**同一个 e、同一种调参规则**；拟合器不能访问 oracle 公式、真实 T 或真 Hessian。解析真值只用来计算评估误差。
3. 所有 Hessian 做相同单位/尺度换算；二阶、三阶、四阶**分开**检验 relative Frobenius、局部相关、RMS 和不同矩阵块的误差；Y prediction error 不可替代 Hessian error。
4. 明确比较的是局部、无权均值还是带权均值。**Stein 的加权平均不参与局部 Hessian 或无权均值的错误排名**。
5. 比较算法的角度等变性、数值收敛、sample seed / regularization 稳定性；不能因某一模型某一角度曲线好看就把它宣称为通用。
6. 只有在上述核对后，才应用**来源无关**的简单性函数 S_k；分别报告 analytic、formula-Y、nn-Y 的曲线相关、真实轴是否为**最小值**（非最大值）、平坦曲线占比、计算耗时与内存。

**针对当前项目的实验优先级**：继续保留已跑通的 tensor polynomial 为基线；优先将已存在的 Z/X Gaussian RBF 估计器纳入同一局部 Hessian 对照；第二阶段增加局部 quadratic 与 penalty spline/GAM；GP 可在需要 Hessian 不确定性时投入；Stein 仅在平均导数对象上独立比较。优先检查 CR3 高阶 ZX/ZZ 导数估计器误差，而非马上增加更多简单性指标。

---

## 5. 文献与出处（原文链接）

下面文献说明**一般统计方法**，不代表已在本项目通过 benchmark。编号与上述正文一致。

- **[R01]** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). [Springer DOI](https://doi.org/10.1007/978-0-387-84858-7). 基函数、正则化、核、GLM。
- **[R02]** Wahba, G. (1990). *Spline Models for Observational Data*. [Book](https://search.worldcat.org/title/1120931056). 样条/RKHS。
- **[R03]** Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. [Publisher](https://www.routledge.com/Local-Polynomial-Modelling-and-Its-Applications-Monographs-on-Statistics/Fan/p/book/9780203748725). 局部多项式导数。
- **[R04]** Eilers, P. H. C., & Marx, B. D. (1996). Flexible Smoothing with B-splines and Penalties. *Statistical Science, 11*(2), 89–102. [DOI](https://doi.org/10.1214/ss/1038425655). P-splines。
- **[R05]** Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). [Publisher](https://www.routledge.com/Generalized-Additive-Models-An-lntroduction-with-R-Second-Edition/Wood/p/book/9781498728331). GAM/penalized smoothers。
- **[R06]** Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to Speed Up Kernel Machines. *NeurIPS 13*, 682–688. [Edinburgh record](https://www.inf.ed.ac.uk/publications/report/1036.html). 核低秩。
- **[R07]** Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NeurIPS 20*. [NeurIPS](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). Random features。
- **[R08]** Solak, E., Murray-Smith, R., Leithead, W. E., Leith, D. J., & Rasmussen, C. E. (2002). Derivative Observations in Gaussian Process Models of Dynamic Systems. *NeurIPS 15*. [NeurIPS](https://papers.nips.cc/paper_files/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html). GP 导数模型。
- **[R09]** Fornberg, B. (1988). Generation of Finite Difference Formulas on Arbitrarily Spaced Grids. *Mathematics of Computation, 51*(184), 699–706. [DOI](https://doi.org/10.1090/S0025-5718-1988-0935077-0). 不规则网格差分。
- **[R10]** Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. *Analytical Chemistry, 36*(8), 1627–1639. [DOI](https://doi.org/10.1021/ac60214a047). 数值平滑与微分。
- **[R11]** Janzamin, M., Sedghi, H., & Anandkumar, A. (2014). Score Function Features for Discriminative Learning: Matrix and Tensor Framework. [arXiv](https://arxiv.org/abs/1412.2863). 高阶 score/integration-by-parts 背景；**非当前 weighted Stein 实现的特定算法论文**。
- **[R12]** Li, Y., & Turner, R. E. (2018). Gradient Estimators for Implicit Models. *ICLR*. [OpenReview](https://openreview.net/forum?id=SJi9WOeRb). **估计输入密度 score**，不是直接函数 Hessian。
- **[R13]** *On Sharp Stochastic Zeroth-order Hessian Estimators over Riemannian Manifolds* (2023). [Journal](https://academic.oup.com/imaiai/article/12/2/787/6780267). 随机零阶二阶导数研究。
- **[R14]** Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic Differentiation in Machine Learning: a Survey. *JMLR, 18*(153), 1–43. [JMLR](https://www.jmlr.org/papers/v18/17-468.html). 自动微分。
- **[R15]** Pearlmutter, B. A. (1994). Fast Exact Multiplication by the Hessian. *Neural Computation, 6*(1), 147–160. [DOI](https://doi.org/10.1162/neco.1994.6.1.147). Hessian-vector product。

**总原则**：对已知 (Z,X,Y) 求 Hessian 的统计估计器是独立的研究维度；它不能与真实解析 Hessian、数据源、积分方式、简单性 loss 或旋转参数混为一谈。