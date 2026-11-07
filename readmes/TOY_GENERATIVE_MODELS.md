# Toy 真实生成模型：C1 / C2 / C3 / CR3

> 本文**仅**描述当前 [mod-rot-3](https://github.com/bs-zhao/mod-rot-3) 仓库中的四个真实生成模型 C1、C2、C3、CR3，属于[研究总框架](README.md)的生成模型/数据层。
>
> **注意：这些模型不是另一个 modlearn-rot-toy 仓库的 T1–T11。** 本文的数值定义全部根据当前项目的生成器核对；不是从神经网络预测、旋转结果或简单性指标反推。

## 1. 全局定义

对于模型 \(m\)、被试 \(s\) 和试次 \(t\)，真实生成系统为

\[
T_s\sim\pi_m(T),\qquad
X_{st}\sim\mathcal D_m\ \text{（或者读取固定 Xset）},
\]

\[
\boxed{
\eta_{st}=f_m(T_s,X_{st}),\quad
p_{st}=\sigma(\eta_{st}),\quad
Y_{st}\mid T_s,X_{st}\sim\operatorname{Bernoulli}(p_{st}).
}
\]

其中 \(\sigma(v)=1/(1+e^{-v})\)。三种输出分别是 **Logit（连续 \(\eta\)）**、**Probability（连续 \(p\)）**、**Choice（随机 0/1）**。一组参数和刺激定义一个选择概率，而不是一个确定的二元选择。

**Choice 的 Hessian 约定：**单次随机 0/1 choice 没有通常意义的解析 Hessian。所谓解析 Choice Hessian 指条件期望 \(E[Y\mid T,X]=p(T,X)\) 的 Hessian；若将 Bernoulli 样本用于回归，再对拟合函数求 Hessian，得到的是平滑拟合函数的导数。

| 模型 | 真实参数个数 | X 维度 | 真实参数先验 | 真实 X 产生方式 | Logit 的结构 |
|---|---:|---:|---|---|---|
| **C1** | 4 | 3 | 各 b 独立 N(0,0.5²) | 每被试每试次独立抽样 | 对参数和原始 X 都线性 |
| **C2** | 4 | 4 | 各 b 独立 N(0,0.5²) | 每被试每试次独立抽样 | 对参数线性；X1 有线性及二次作用 |
| **C3** | **6** | 4 | 各 b 独立 N(0,0.5²) | 每被试每试次独立抽样 | 对参数线性；两个二次项与 X1×X2 交互 |
| **CR3** | 4 | 6 | Beta / 截断 Gamma / Beta / 截断 Normal | **所有被试共用同一个固定 500 试次 Xset** | 对参数非线性；CPT-like 彩票效用 |

当前生成器的默认规模为：C1/C2/C3 各 5000 个被试，每位被试 300–500 试次；CR3 默认 5000 个被试，每位被试**恰好 500 试次**。这些是代码默认值，实际运行可能通过命令行指定不同被试数。

---

## 2. C1：线性 Logit

真实参数

\[
T=(b_1,b_2,b_3,b_4),\qquad
b_j\overset{\text{iid}}{\sim}N(0,0.5^2).
\]

真实输入

\[
X=(x_1,x_2,x_3),\qquad
x_j\overset{\text{iid}}{\sim}\operatorname{Uniform}(-1,1).
\]

每个被试的每个试次都重新生成三个 X；并不存在跨被试共享的固定 Xset。

**生成公式**

\[
\boxed{
\eta_{\mathrm{C1}}=
b_1+\sqrt3\,b_2x_1+\sqrt3\,b_3x_2+\sqrt3\,b_4x_3.
}
\]

然后用统一的 sigmoid 得到 \(p\)，再以 Bernoulli(p) 得到 choice。\(b_1\) 控制总体选择倾向，\(b_2,b_3,b_4\) 分别控制三个 X 方向的斜率。由于 Uniform(-1,1) 的方差为 1/3，乘以 \(\sqrt3\) 后相应设计特征的方差为 1。

**解析导数结构**：\(\nabla_T^2\eta=0\)，\(\nabla_X^2\eta=0\)；但 \(\nabla_T\nabla_X\eta\) 不为零。施加外层 sigmoid 后 \(H_{TT}(p)\) 通常不为零。因而只依赖 Logit-方向 ZZ 非零曲率的简单性指标在 C1 上通常缺乏旋转识别能力。

**源代码**：[c1_linear_logit.py](../np/src/modlearn_rot/generators/choice/c1_linear_logit.py)。

---

## 3. C2：共享 X1 的线性与二次作用

真实参数仍为四维：

\[
T=(b_1,b_2,b_3,b_4),\qquad
b_j\overset{\text{iid}}{\sim}N(0,0.5^2).
\]

原始输入的四列顺序**必须保留**：

\[
X=(X_{1A},X_{1B},X_{2A},X_{2B}),
\]

各原始 X 坐标独立服从 Uniform(-1,1)，每被试、每试次重新采样。NP 接收的是**这四个原始输入**，不是 Oracle 的手工设计基函数。

定义：

\[
\begin{aligned}
\Delta X_1&=X_{1A}-X_{1B},\\
\Delta X_2&=X_{2A}-X_{2B},\\
\Delta Q_1&=(3X_{1A}^2-1)-(3X_{1B}^2-1).
\end{aligned}
\]

**生成公式**

\[
\boxed{
\eta_{\mathrm{C2}}=
b_1+\sqrt3\,b_2\Delta X_1
+\frac{\sqrt5}{2}b_3\Delta Q_1
+\sqrt3\,b_4\Delta X_2.
}
\]

这里 \(b_1\) 是选项 A 的截距偏置；\(b_2\) 对应 X1 线性差值；\(b_3\) 对应 X1 二次差值；\(b_4\) 对应 X2 线性差值。**同一个 X1 家族对应不同的参数方向**，是验证函数效应是否可分的一个重要难点。

**解析导数结构**：尽管对 X 二次非线性，Logit 对 \(T\) 仍是严格线性，所以 \(H_{TT}(\eta)=0\)；\(H_{TX}(\eta)\) 随 X 变化，\(H_{XX}(\eta)\) 出现非零二次曲率。外层 sigmoid 使 Probability 对 T 的曲率一般非零。

**源代码**：[c2_shared_x_quadratic.py](../np/src/modlearn_rot/generators/choice/c2_shared_x_quadratic.py)。

---

## 4. C3：六参数的双二次作用与 X1×X2 交互

**真实参数有六个**：

\[
T=(b_1,b_2,b_3,b_4,b_5,b_6),\qquad
b_j\overset{\text{iid}}{\sim}N(0,0.5^2).
\]

仍然使用 C2 的四列原始输入 \(X=(X_{1A},X_{1B},X_{2A},X_{2B})\)，每个坐标独立 Uniform(-1,1)，每位被试的每个试次单独采样。

在 C2 的差值基函数之外，增加：

\[
\Delta Q_2=(3X_{2A}^2-1)-(3X_{2B}^2-1),
\]

\[
\Delta C=X_{1A}X_{2A}-X_{1B}X_{2B}.
\]

**生成公式**

\[
\boxed{
\begin{aligned}
\eta_{\mathrm{C3}}={}&b_1+\sqrt3\,b_2\Delta X_1
+\frac{\sqrt5}{2}b_3\Delta Q_1
+\sqrt3\,b_4\Delta X_2\\
&+\frac{\sqrt5}{2}b_5\Delta Q_2
+3b_6\Delta C.
\end{aligned}
}
\]

| 参数 | 真实功能 |
|---|---|
| \(b_1\) | 整体 choice-bias |
| \(b_2\) | X1 的线性差值 |
| \(b_3\) | X1 的二次差值 |
| \(b_4\) | X2 的线性差值 |
| \(b_5\) | X2 的二次差值 |
| \(b_6\) | 同一选项内 X1×X2 的交互差值 |

C3 是 C2 的嵌套扩展：令 \(b_5=b_6=0\) 即恢复 C2 的 Logit。C3 与 C2 的共同点是对真实 T **仍然线性**；其新的难点来自多个 X 维度共享参数作用、两个二次方向和混合 \(X_1X_2\) 作用。

**解析导数结构**：\(H_{TT}(\eta)=0\)，但 \(H_{TX}\) 的 X 依赖和 \(H_{XX}\) 的混合结构更丰富。因此若希望恢复 C3 的六条参数轴，不能仅依靠 Logit 的 \(H_{TT}\) 稀疏性。

**源代码**：[c3_multi_nonlinear_logit.py](../np/src/modlearn_rot/generators/choice/c3_multi_nonlinear_logit.py)。

---

## 5. CR3：CPT-like 四参数选择模型

真实参数为

\[
T=(\alpha,\lambda,\gamma,\tau).
\]

**真实先验**（四维独立抽样）：

| 参数 | 原始生成器的分布 | 作用 |
|---|---|---|
| \(\alpha\) | Beta(12,3) | 正负金额的效用幂指数 |
| \(\lambda\) | Gamma(shape=9, scale=0.22)，超出 [0.3,5] 时重新抽样 | 损失厌恶 |
| \(\gamma\) | Beta(9,4) | 概率加权形状 |
| \(\tau\) | Normal(2.2,0.55²)，截断到 [0.5,4] | 效用差转化为 Logit 的尺度 |

每个 X 行含两个彩票选项，列顺序为

\[
X=(g_A,\ell_A,p_A,g_B,\ell_B,p_B),
\]

其中 \(g\) 为非负 gain 金额，\(\ell\) 为非负 loss magnitude（损失结果为 \(-\ell\)），\(p\) 是获得 gain 的概率。

首先定义概率加权：

\[
c(p)=\operatorname{clip}(p,10^{-12},1-10^{-12}),\qquad
w(p;\gamma)=\sigma\!\left(\gamma\,\operatorname{logit}(c(p))\right).
\]

对于彩票选项 \(L\in\{A,B\}\)，定义

\[
V_L=w(p_L;\gamma)\left(\frac{g_L}{100}\right)^\alpha
-\lambda\,w(1-p_L;\gamma)\left(\frac{\ell_L}{100}\right)^\alpha.
\]

最终真实生成公式为

\[
\boxed{
\eta_{\mathrm{CR3}}=\tau(V_A-V_B),\quad
p_{\mathrm{CR3}}=\sigma(\eta_{\mathrm{CR3}}),\quad
Y\sim\operatorname{Bernoulli}(p_{\mathrm{CR3}}).
}
\]

**代码忠实性要求**：\(w(p;\gamma)\) 与 \(w(1-p;\gamma)\) 分别调用，并对 \(p\) 和 \(1-p\) **各自独立裁剪**；不要擅自用 \(1-w(p;\gamma)\) 代替损失项加权。

### CR3 固定 500 试次 Xset

**与 C1–C3 不同，CR3 不为每位被试重新采样 X。** 所有被试使用完全同一个固定 \(500\times6\) 的 bank，来自先前 CR2 四参数固定刺激设计，混入指定的特殊条件，并排除严格一阶随机占优（FOSD）的试次：

| 条件类型 | 固定行数 | 结构 |
|---|---:|---|
| gain_only | 60 | 两个选项的 loss magnitude 为 0 |
| gamma_neutral | 60 | 两个选项的 gain probability 为 0.5 |
| loss_only | 40 | 两个选项的 gain 金额为 0 |
| general_CR2 | 340 | 从原 CR2 非支配一般设计中抽样，必要时补充候选 |
| **总计** | **500** | 全体被试共享同一 Xset |

这里**没有重新做 Fisher 优化**。固定 Xset 的文件为：

outputs/choice/CR3/s0_design/CR3_4P_fixed500.npz

原始设计脚本的默认 design seed 为 20261023；默认 data seed 为 20261028。运行元信息中的实际 seed 和 bank hash 应作为可复现研究的最终记录。

**CR3 Hessian 的特殊难点：**

1. 与 C1–C3 相比，Logit \(\eta_{\mathrm{CR3}}\) 对 T 本身**非线性**，因此 \(H_{TT}(\eta)\) 通常非零。
2. 当 gain/loss amount 为 0 且 \(\alpha\in(0,1)\) 时，相关 X 金额导数可能未定义。应区分 ZZ 与包含金额 X 导数的 ZX/XX 有效域。
3. 对 X 求平均时，应明确是使用真实固定 500 行，还是仅使用可微子集；对真实 T 先验积分和对模拟被试的 T 或 NP Z 经验分布求平均是不同实验因素。

**源代码**：[CR3 的固定 Xset 与模拟数据](../np/reference/cr3/s0_s1_cr3.py)、[CR3 的真实先验与效用公式](../np/reference/s0_cr2_design_recovery.py)、[解析 Hessian 实现](../math/04_cr3_zx_hessian_threeway/oracle.py)。

---

## 6. 四个模型在 Hessian 简单性研究中的对照意义

前三个模型都是

\[
\eta(T,X)=T^\top\phi(X)
\]

形式。因此必然有

\[
H_{TT}(\eta)=0,\qquad
H_{TX}(\eta)=J_\phi(X)^\top,\qquad
H_{XX}(\eta)=\sum_j T_j H_{XX}(\phi_j).
\]

但对 \(p=\sigma(\eta)\)，仍有

\[
H_{TT}(p)=\sigma''(\eta)\,\phi(X)\phi(X)^\top,
\]

一般非零。因此**Logit 的 ZZ/TT Hessian 完全为零不是参数轴被成功恢复**：它可能使所有旋转方向的指标都相同，无法辨识正确坐标。此时必须考虑 TX/ZX、梯度作用、X 相关导数或函数算子。

| 解析性质 | C1 | C2 | C3 | CR3 |
|---|---|---|---|---|
| 参数维度 | 4 | 4 | **6** | 4 |
| \(H_{TT}(\eta)\) | 零 | 零 | 零 | 通常非零 |
| \(H_{XX}(\eta)\) | 零 | 有非零二次项 | 有二次和混合项 | 非线性，部分边界未定义 |
| \(H_{TX}(\eta)\) | 简单、对 X 常量 | 随 X 变 | 随 X 变、结构更复杂 | 非线性 |
| \(H_{TT}(p)\) | 通常非零 | 通常非零 | 通常非零 | 通常非零 |
| X 跨被试共享？ | 否 | 否 | 否 | **是** |

### 之后的 NP / Hessian 层不属于真实生成模型本身

每个模型对应独立训练的 NP，但其 latent dimension 要按具体 checkpoint 验证（例如 C3 的真参数为 6，CR3 也存在 L4/L8 的网络配置）。NP 的 \(Z\) **不是**真实参数 \(T\)，可以通过后验线性映射 \(\widehat T=A_*Z+b\) 检验可恢复性。

下游三条 Hessian 计算路径应统一坐标和数值评价点：

- **analytic**：直接对本 README 的 \(\eta\)/\(p\) 求导（在 Z/U 坐标中需相应的 Jacobian 转换）。
- **formula_y**：本 README 的解析式只负责生成数值 (T,X,Y) 或 (U,X,Y)；同一数据驱动估计器通过数值 Y 拟合函数再求导。
- **nn_y**：将冻结 NP 的 Z、X 输入 decoder 生成数值 Y，再用同一数据驱动估计器求导；不对 NN 本身求导。

Hessian 估计方式、Xset 积分、参数先验/样本平均、简单性指标、二维/三维/四维旋转都属于**另一个独立实验维度**，详见[研究总体框架](README.md)。

---

## 7. 实现文件与事实来源

| 模型 | 当前仓库的真实生成实现 |
|---|---|
| C1 | [np/src/modlearn_rot/generators/choice/c1_linear_logit.py](../np/src/modlearn_rot/generators/choice/c1_linear_logit.py) |
| C2 | [np/src/modlearn_rot/generators/choice/c2_shared_x_quadratic.py](../np/src/modlearn_rot/generators/choice/c2_shared_x_quadratic.py) |
| C3 | [np/src/modlearn_rot/generators/choice/c3_multi_nonlinear_logit.py](../np/src/modlearn_rot/generators/choice/c3_multi_nonlinear_logit.py) |
| CR3 | [np/reference/cr3/s0_s1_cr3.py](../np/reference/cr3/s0_s1_cr3.py)、[np/reference/s0_cr2_design_recovery.py](../np/reference/s0_cr2_design_recovery.py) |

实际按被试抽样/保存的通用实现见 [dataset.py](../np/src/modlearn_rot/generators/dataset.py)，生成入口为 [np/s1_generate.py](../np/s1_generate.py)。原始数据存于 outputs/choice/<MODEL>/，不随代码提交到 GitHub；实际随机种子、数据与 checkpoint hash 应以相应运行 manifest 为准。

**本项目只有 C1、C2、C3、CR3 这四个真实 Toy 生成模型；本 README 不再包含 T1–T11。**
