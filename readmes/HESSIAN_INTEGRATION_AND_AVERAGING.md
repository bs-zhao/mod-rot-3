# 独立实验轴 Q：积分、采样测度、权重与数值求积

> 本文件与 [研究总框架](README.md)、[Hessian 数值估计器](HESSIAN_FROM_ZXY_METHODS.md)、[Hessian → 数值矩阵指标](HESSIAN_TO_MATRIX_METRICS.md)并列。
>
> **本层只决定：在哪些 Z、X 点上、按照什么联合分布与权重评价导数字段，以及如何数值积分。**
>
> **本层不决定：对 H 做有符号平均、Mean Absolute、RMS、方差、分位数或 Gram；后者属于独立的数值矩阵指标 A。**

## 1. 数学定义：测度 Q 与统计指标 A 分离

设同一函数的 Hessian 场为 H(Z,X)。一个积分方案

\[
Q=(\Omega,\mu,\mathcal I,\mathrm{conditioning},\mathrm{mask})
\]

包含评价域 Ω、联合测度 μ、数值积分器 \(\mathcal I\)、条件及有效域掩码。对给定函数 g，\(\mathcal I_Q[g]\approx\int_\Omega g\,d\mu\)。这里 g 可以是 H、绝对值、平方、外积、指标函数等，由后续指标 A 决定。

**同样的 Q 可以配不同 A**：

\[
M_{\rm mean}=\mathcal I_Q[H],\qquad
M_{\rm RMS}=\sqrt{\mathcal I_Q[H\odot H]},\qquad
M_{\rm Gram}=\mathcal I_Q[HH^\top].
\]

不要把“采用 Monte Carlo 积分”误当成“采用 RMS 矩阵指标”；一个指定数值算法，另一个指定积分的被积函数和后处理。

## 2. Q 的独立子维度

| ID | 子维度 | 主要候选 |
|---|---|---|
| Q1 | **积分变量** | 不积分、X-only、Z/T-only、联合 Z×X、分阶段条件积分 |
| Q2 | **Z 或 T 的分布** | NP S5 的 Z 经验分布、真实 T 的被试经验分布、已知 T 先验、经注明的坐标变换分布 |
| Q3 | **X 的分布** | 真实固定 Xset、真实被试试次、生成器设计分布、指定刺激子集 |
| Q4 | **联合/配对方式** | 原始配对 (Z_s,X_st)、经验 Z × 固定 X bank、指定独立乘积测度、相关联合分布 |
| Q5 | **被试与试次权重** | 每被试等权、每 trial 等权、分组等权、importance weights、Stein 权重 |
| Q6 | **积分计算算法** | 精确有限和、Monte Carlo、Sobol/QMC、数值 quadrature、分层抽样 |
| Q7 | **定义域与条件** | 全部区域、可微区域、按实验条件分组、指定外推域 |
| Q8 | **部分积分的剩余变量** | 积分后是否仍含有 X 或 Z；是否需要第二阶段积分或指定条件取值 |

**仅对 X 取平均**：

\[
G(Z)=\mathbb E_X[H(Z,X)\mid Z]
\]

一般仍是 Z 的矩阵函数，不是一个固定数值矩阵。仅对 Z 积分同理；要么进一步对剩余变量求平均，要么固定它，才能得到单个数值矩阵。

## 3. CR3 与 C1–C3 的不同测度

CR3 每人共享同一个固定 500×6 X bank，给定 N 个 Z_s，可明确定义

\[
\mu^{\rm CR3}_{Z\times X}
=\left(\frac1N\sum_{s=1}^N\delta_{Z_s}\right)
\otimes\left(\frac1{500}\sum_{t=1}^{500}\delta_{X_t}\right).
\]

**已知真实 T 先验的积分是另一种测度**：

\[
\mu^{\rm CR3}_{T\times X}
=\pi_{\rm CR3}(T)\,dT\otimes
\left(\frac1{500}\sum_t\delta_{X_t}\right).
\]

CR3 的连续先验积分只改变 T 维度，**不自动改变固定的 500 个 X**。真实 T 的先验并不是 NP Z 的先验或 aggregate posterior；若由 T 映射到 Z，需明确映射、支持域和误差。

C1/C2/C3 的 X 每被试每试次重新抽取，最自然的配对样本测度是

\[
\mu_{\rm paired}
=\frac1N\sum_{s=1}^N\frac1{n_s}
\sum_{t=1}^{n_s}\delta_{(Z_s,X_{st})}.
\]

按总 trial 池化得到的测度会给试次多的被试更多权重；自行生成新 X bank 并和所有 Z 交叉则属于另一种**反事实评价测度**。不能混用。

## 4. 数值求积算法的区别

- **固定设计有限求和**：对指定 X bank 是精确有限样本平均，但不自动代表总体 X 设计分布。
- **被试经验平均/cluster bootstrap**：同一被试所有试次共享 Z，bootstrap 必须按被试分组。
- **Monte Carlo**：明确抽样分布、样本量、随机种子、估计误差。
- **Sobol / quasi-Monte Carlo**：记录从单位立方体映射到真实参数先验的变换、scrambling 与有效域。
- **Quadrature**：真实先验密度权重、节点选择与阶数收敛。
- **条件/分层平均**：例如 CR3 的 gain-only、gamma-neutral、loss-only、general；每组的归一化与全设计权重不同。
- **Importance/Stein weights**：记录 weight 函数和有效样本量。边界消失的加权 Stein **直接估计加权平均 Hessian**，不是无权局部场。

## 5. 需要额外声明的数学问题

1. 若 X 的测度与 Z 无关且满足充分的可微和可积条件，可以有 \(\nabla_Z^2 E_X f=E_X\nabla_Z^2 f\)。不满足这些条件时不能随意交换。
2. 对局部 \(H_{ZX}\) 或 \(H_{XX}\) 的估计，先将 Y 对 X 平均会丢失 X 信息；不能从仅依赖 Z 的平均响应模型重建这些局部 Hessian。
3. CR3 分数幂在金额为零时可能令部分金额 X 导数未定义。严格无条件统计量、仅可微区域的条件统计量以及 fraction_defined 应分别输出。
4. 完整的 Q 需要明确坐标系：真实 T、原始 NP Z 或旋转 U；对 Z 经验分布求积不能与对 T 先验求积混为同一个对象。

## 6. 建议记录的元信息

- integration_id / integration_axes / remaining_axes;
- coordinate_system / integration_measure / support_domain / x_bank_hash;
- pairing_or_product / subject_weighting / trial_weighting;
- estimator_quadrature_rule / node_count / seed / weights / effective_sample_size;
- masked_domain / fraction_defined / conditional_vs_unconditional;
- convergence_error / standard_error / data_split.

给定相同 Q，下一层可自由替换 [矩阵指标 A](HESSIAN_TO_MATRIX_METRICS.md)，随后再替换标量简单性损失 S。**本文件是研究架构约定，不代表所有积分组合都已实现。**
