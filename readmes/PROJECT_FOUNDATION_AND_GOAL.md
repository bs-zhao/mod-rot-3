# 项目研究定位、真实生成模型、Neural Process 与潜变量轴恢复目标

> **Mod-Rot-3 — Research Foundation and Transformation Search Spaces（第一、二部分）**  
> 日期：2026-11-08  
> 本文是逐步扩展的项目研究说明。**第一部分**定义真实生成机制、四个模型、冻结 NP/Z 与研究总目标；**第二部分**定义两类待搜索的变换空间：一般仿射映射，以及标准化真参数目标下的正交旋转（加可省略搜索的逐轴缩放）。函数简单性损失、Hessian 指标及具体优化器留待后续部分。
>
> **最重要的边界**：真实参数 \(T\) 在模拟中是已知的，因此可以用于评价或建立监督上限，但本研究最终希望开发的**潜变量变换发现方法，不以 T 为监督信号**。不能把监督回归的成功视为已完成无监督轴发现。

## 1. 项目总体定位：我们知道真实生成机制，但只给神经网络行为数据

本项目研究一个清晰的模拟决策场景：对于每一个被试（subject），存在一套固定但不同于其他被试的**真实认知模型参数**；每个试次（trial）呈现一组输入信息 X；已知的真实解析生成函数将真实参数与试次信息结合，决定一个选择概率；最后从这个概率随机产生实际观察到的二元选择 \(Y\in\{0,1\}\)。

记被试为 \(s=1,\ldots,N\)，试次为 \(t=1,\ldots,n_s\)，生成模型为 \(m\in\{\mathrm{C1,C2,C3,CR3}\}\)。

- **每位被试的一套真实参数**：\(T_s=(T_{s1},\ldots,T_{sd})^\top\sim\pi_m(T)\)。同一被试的所有试次共享相同的 \(T_s\)；不同被试独立抽样自己的参数。
- **每个试次的输入**：\(X_{st}\)，来自模型指定的输入生成规则；有的模型每次重新抽样，有的模型使用所有被试共享的固定刺激集合。
- **已知的真实解析式**：\(\eta_m(T_s,X_{st})\)，表示生成选择的 logit。
- **真实选择概率**：\(p_{st}=\sigma(\eta_m(T_s,X_{st}))\)，其中 \(\sigma(v)=1/(1+\exp(-v))\)。
- **实际观察到的选择**：\(Y_{st}\mid T_s,X_{st}\sim\mathrm{Bernoulli}(p_{st})\)。

统一写作：

$$
\boxed{
\begin{aligned}
T_s&\sim\pi_m(T),\\
X_{st}&\sim\mathcal D_m\quad\text{或来自固定试次集合},\\
\eta_{st}&=\eta_m(T_s,X_{st}),\\
p_{st}&=\sigma(\eta_{st}),\\
Y_{st}\mid T_s,X_{st}&\sim\mathrm{Bernoulli}(p_{st}).
\end{aligned}}
$$

这里务必区分三个对象：

1. **Logit \(\eta\)**：由真实解析函数确定的连续量；
2. **Probability \(p\)**：同一解析函数经 sigmoid 得到的、范围在 \((0,1)\) 内的条件选择概率；
3. **Binary choice \(Y\)**：真正观测到的随机 0/1 回答。

真实生成函数及 T 对研究者是已知的，目的是提供可控、可验证的 benchmark。**NP 的训练输入是行为试次中的 X 和 Y，而不是真实 T，也不需要看到真实解析公式。**

### 1.1 四个真实生成模型概览

| 模型 | 参数 \(T\) 维度 | X 维度 | 参数生成分布 | X 的生成方式 | 解析 logit 的主要性质 |
|---|---:|---:|---|---|---|
| **C1** | 4 | 3 | 四个独立 \(N(0,0.5^2)\) | 每位被试、每个试次重新独立抽样 | 对 T、原始 X 均线性 |
| **C2** | 4 | 4 | 四个独立 \(N(0,0.5^2)\) | 每位被试、每个试次重新独立抽样 | 对 T 线性，包含 X1 的二次作用 |
| **C3** | **6** | 4 | 六个独立 \(N(0,0.5^2)\) | 每位被试、每个试次重新独立抽样 | 对 T 线性，包含两个二次作用和 X1×X2 交互 |
| **CR3** | 4 | 6 | Beta / 截断 Gamma / Beta / 截断 Normal | **所有被试共享同一套固定 500 个 X 试次** | CPT-like 彩票效用；对 T 非线性 |

当前生成器的默认配置：C1/C2/C3 为每模型 5,000 位被试、每人 300–500 个试次；CR3 为 5,000 位被试、每人固定 500 试次。实际数据以相应运行的配置与 manifest 为准。

---

## 2. 四个真实生成例子的精确定义

### 2.1 C1：四参数线性 Logit 选择模型

**真实参数：**

$$
T_s=(b_{1s},b_{2s},b_{3s},b_{4s})^\top,
\qquad b_{js}\overset{\mathrm{iid}}{\sim}\mathcal N(0,0.5^2).
$$

**试次输入：**

$$
X_{st}=(x_{1,st},x_{2,st},x_{3,st}),\qquad
x_{k,st}\overset{\mathrm{iid}}{\sim}\mathrm{Uniform}(-1,1).
$$

每位被试、每个试次的三个 X 都重新抽样，不存在跨被试共享的固定 X 矩阵。

**真实解析函数：**

$$
\boxed{
\eta_{\mathrm{C1}}(T,X)
=b_1+\sqrt{3}\,b_2x_1+\sqrt{3}\,b_3x_2+\sqrt{3}\,b_4x_3
}
$$

$$
p_{\mathrm{C1}}(T,X)=\sigma\!\left(\eta_{\mathrm{C1}}(T,X)\right),
\qquad
Y\sim\mathrm{Bernoulli}(p_{\mathrm{C1}}).
$$

其中 \(b_1\) 控制总体选择偏置，\(b_2,b_3,b_4\) 分别控制三个 X 输入的作用强度。设计式中的 \(\sqrt3\) 是生成器原有的特征尺度因子。

**真实生成实现：** [`c1_linear_logit.py`](../np/src/modlearn_rot/generators/choice/c1_linear_logit.py)。

### 2.2 C2：四参数、同一 X1 同时具有线性与二次作用

**真实参数：**

$$
T_s=(b_{1s},b_{2s},b_{3s},b_{4s})^\top,
\qquad b_{js}\overset{\mathrm{iid}}{\sim}\mathcal N(0,0.5^2).
$$

每个试次呈现选项 A 和 B 各自的两个属性，**原始输入**为：

$$
X=(X_{1A},X_{1B},X_{2A},X_{2B}).
$$

四个原始属性分别独立服从 \(\mathrm{Uniform}(-1,1)\)，并对每个被试的每个试次重新抽样。

定义原始属性的差异特征：

$$
\begin{aligned}
\Delta X_1&=X_{1A}-X_{1B},\\
\Delta X_2&=X_{2A}-X_{2B},\\
\Delta Q_1&=(3X_{1A}^2-1)-(3X_{1B}^2-1).
\end{aligned}
$$

**真实解析函数：**

$$
\boxed{
\eta_{\mathrm{C2}}(T,X)
=b_1+\sqrt3\,b_2\Delta X_1
+\frac{\sqrt5}{2}b_3\Delta Q_1
+\sqrt3\,b_4\Delta X_2
}
$$

$$
p_{\mathrm{C2}}(T,X)=\sigma(\eta_{\mathrm{C2}}(T,X)),
\qquad Y\sim\mathrm{Bernoulli}(p_{\mathrm{C2}}).
$$

四个参数分别控制：总体选择偏置、X1 线性差值、X1 二次差值、X2 线性差值。**NP 只接受四个原始 X 值，不接受专为解析模型构造的差异或二次基函数。**

当 \(Y=1\) 时，表示选择 A；\(Y=0\) 时表示选择 B。

**真实生成实现：** [`c2_shared_x_quadratic.py`](../np/src/modlearn_rot/generators/choice/c2_shared_x_quadratic.py)。

### 2.3 C3：六参数、双二次作用及跨属性交互

**真实参数共有六个：**

$$
T_s=(b_{1s},b_{2s},b_{3s},b_{4s},b_{5s},b_{6s})^\top,
\qquad b_{js}\overset{\mathrm{iid}}{\sim}\mathcal N(0,0.5^2).
$$

与 C2 一样，使用四个原始 X：

$$
X=(X_{1A},X_{1B},X_{2A},X_{2B}),
$$

全部独立服从 \(\mathrm{Uniform}(-1,1)\)，每位被试、每个试次重新抽样。

沿用 C2 的 \(\Delta X_1,\Delta X_2,\Delta Q_1\)，再定义：

$$
\begin{aligned}
\Delta Q_2&=(3X_{2A}^2-1)-(3X_{2B}^2-1),\\
\Delta C&=X_{1A}X_{2A}-X_{1B}X_{2B}.
\end{aligned}
$$

**真实解析函数：**

$$
\boxed{
\begin{aligned}
\eta_{\mathrm{C3}}(T,X)
={}&b_1+\sqrt3\,b_2\Delta X_1
+\frac{\sqrt5}{2}b_3\Delta Q_1
+\sqrt3\,b_4\Delta X_2\\
&+\frac{\sqrt5}{2}b_5\Delta Q_2
+3b_6\Delta C .
\end{aligned}}
$$

$$
p_{\mathrm{C3}}(T,X)=\sigma(\eta_{\mathrm{C3}}(T,X)),
\qquad Y\sim\mathrm{Bernoulli}(p_{\mathrm{C3}}).
$$

六个参数的真实意义：

| 参数 | 对生成行为的作用 |
|---|---|
| \(b_1\) | 总体选择偏置 |
| \(b_2\) | X1 的线性差值效应 |
| \(b_3\) | X1 的二次差值效应 |
| \(b_4\) | X2 的线性差值效应 |
| \(b_5\) | X2 的二次差值效应 |
| \(b_6\) | 同一选项内 X1×X2 的交互效应差值 |

C3 在 C2 的基础上增加两个新参数，设 \(b_5=b_6=0\) 即恢复 C2。虽然输入 X 与生成函数的关系更加复杂，C3 的 **logit 对六个真实参数本身仍为线性**。NP 接受原始四列 X，不直接接收上述手工基函数。

**真实生成实现：** [`c3_multi_nonlinear_logit.py`](../np/src/modlearn_rot/generators/choice/c3_multi_nonlinear_logit.py)。

### 2.4 CR3：四参数 CPT-like 非线性选择模型

**每位被试的真实参数：**

$$
T_s=(\alpha_s,\lambda_s,\gamma_s,\tau_s)^\top.
$$

四个参数按相互独立的先验抽样：

| 参数 | 真实生成分布 | 在函数中的含义 |
|---|---|---|
| \(\alpha\) | \(\mathrm{Beta}(12,3)\) | 金额效用的幂指数 |
| \(\lambda\) | \(\mathrm{Gamma}(\mathrm{shape}=9,\mathrm{scale}=0.22)\)，只保留 \([0.3,5]\) 范围内的值（超出则重抽） | 损失厌恶 |
| \(\gamma\) | \(\mathrm{Beta}(9,4)\) | 概率加权形状 |
| \(\tau\) | 截断正态 \(\mathcal N(2.2,0.55^2)\)，取值限制在 \([0.5,4]\) | 价值差的 logit 尺度 |

**每个试次包含两个彩票选项 A 与 B**。原始 X 的六列顺序为：

$$
X=(g_A,\ell_A,p_A,g_B,\ell_B,p_B).
$$

其中 \(g\ge0\) 是 gain 金额；\(\ell\ge0\) 是 loss magnitude，实际损失金额为 \(-\ell\)；\(p\) 是获得 gain 的概率。

CR3 的 X **不是对每位被试重新随机生成的连续独立输入**。所有被试共享同一个事先构造的 **500×6 固定试次集合**，按当前生成器包括：

| 固定试次类别 | 行数 | 设计特征 |
|---|---:|---|
| `gain_only` | 60 | 两个选项的 loss magnitude 均为 0 |
| `gamma_neutral` | 60 | 两个选项的 gain probability 均为 0.5 |
| `loss_only` | 40 | 两个选项的 gain 金额均为 0 |
| `general_CR2` | 340 | 从既有 CR2 非支配一般彩票设计中抽样并按需要补充 |
| **合计** | **500** | 所有被试共享 |

该设计混合不同条件，并排除严格一阶随机占优的试次；不是通过每人独立均匀采样得到的 500 个 X。

定义概率加权函数（\(\mathrm{clip}\) 防止 logit 在数值端点无定义）：

$$
\begin{aligned}
c(p)&=\mathrm{clip}(p,10^{-12},1-10^{-12}),\\
w(p;\gamma)&=\sigma\bigl(\gamma\,\mathrm{logit}(c(p))\bigr).
\end{aligned}
$$

对任一彩票 \(L\in\{A,B\}\)，定义主观价值：

$$
\boxed{
V_L(T,X)=w(p_L;\gamma)\left(\frac{g_L}{100}\right)^\alpha
-\lambda\,w(1-p_L;\gamma)\left(\frac{\ell_L}{100}\right)^\alpha
}
$$

**真实解析选择函数：**

$$
\boxed{
\begin{aligned}
\eta_{\mathrm{CR3}}(T,X)&=\tau\bigl(V_A(T,X)-V_B(T,X)\bigr),\\
p_{\mathrm{CR3}}(T,X)&=\sigma(\eta_{\mathrm{CR3}}(T,X)),\\
Y&\sim\mathrm{Bernoulli}(p_{\mathrm{CR3}}(T,X)).
\end{aligned}}
$$

真实实现中对 \(w(p;\gamma)\) 和 \(w(1-p;\gamma)\) **分别计算**，不能将损失概率加权简化为 \(1-w(p;\gamma)\)。该函数对 T 是非线性的，与 C1–C3 构成明显的生成结构对照。

**真实生成实现：** [`CR3 生成和固定 X 设计`](../np/reference/cr3/s0_s1_cr3.py) · [`参数先验及价值函数`](../np/reference/s0_cr2_design_recovery.py)。

---

## 3. 使用行为数据训练 Neural Process，从每位被试提取潜变量 Z

### 3.1 网络得到什么信息、不能得到什么信息？

对于被试 s，NP 能看到的是一组已观察到的行为试次：

$$
\mathcal D_s=\{(X_{st},Y_{st}):t=1,\ldots,n_s\}.
$$

这些试次来自上面已知的真实生成机制，但**训练网络时不提供**该被试的真实参数 \(T_s\)，也**不向网络指定**生成公式 \(\eta_m(T,X)\)。

研究者希望通过不同被试在相同/不同 X 条件下表现出的选择规律，让 NP 自己学习一个用于概括个体差异的潜在表示。

### 3.2 Encoder：从一组 (X,Y) 试次压缩出被试表征

当前使用的离散选择 Neural Process（`DiscreteNeuralProcess`）具有集合式 Gaussian encoder。训练时从一个被试的观察试次中抽取 context 子集，经共享嵌入与集合注意力/聚合得到被试层面的潜变量分布参数：

$$
\boxed{
q_\phi(Z_s\mid \mathcal D_{s,\mathrm{context}})
=
\mathcal N\!\left(
\mu_{\phi,s},
\mathrm{diag}(\exp(\ell_{\phi,s}))
\right)
}
$$

其中 \(\mu_{\phi,s}\) 是潜变量后验均值，\(\ell_{\phi,s}\) 是后验 log-variance。编码器对 context 试次的顺序不应敏感。当前架构使用 64 维集合摘要、4 个注意力头；具体训练配置以对应 checkpoint 为准。

### 3.3 Decoder：用被试 Z 和新试次 X 预测二元选择

Decoder 是一个 MLP，接收同一被试的潜变量与某个 target X 拼接后的输入：

$$
\boxed{
\widehat{\boldsymbol\eta}_{st}
=
\mathrm{MLP}_\psi([Z_s,X_{st}])
}
$$

输出两个 choice 类别的 logits，经 softmax 得到

$$
\widehat p_\psi(Y_{st}=1\mid Z_s,X_{st}).
$$

当前离散 NP 的 decoder 默认隐藏层为 128、256、128，激活函数 GELU，最终输出 2 个 logits。这里的网络预测函数与真实解析公式是两种不同对象：

- **真实生成函数**：\(p_m(Y\mid T,X)\)，知道真实 T、数学结构；
- **训练后的预测函数**：\(\widehat p_\psi(Y\mid Z,X)\)，使用学到的 Z、X 预测 Y。

二者希望在行为预测上接近，但不要求 Z 的某一维天然等于某一个真实参数。

### 3.4 训练目标与固定网络

当前 NP 通过试次 choice 的 cross-entropy 学习预测，并可加入高斯潜变量 KL 与潜变量均值稀疏惩罚：

$$
\mathcal L_{\mathrm{NP}}
=
\mathcal L_{\mathrm{choice\,CE}}
+\beta\,\mathcal L_{\mathrm{KL}}
+\lambda_{\mathrm{sparse}}\mathcal L_{\mathrm{latent\,L1}}.
$$

训练时在被试的观察试次中采样 context，典型配置为每人有效试次的 25%–80%；使用独立 TRAIN、VALIDATION、TEST 被试划分。训练完成后**冻结 encoder 和 decoder**，下游潜变量研究不再改变已经学好的预测网络。

### 3.5 最终固定使用每个被试的一个 Z 向量

模型冻结后，对每位被试使用**其全部已观察的 (X,Y) 试次**，调用 encoder，取潜变量后验均值，而不是随机后验样本：

$$
\boxed{
Z_s=\mu_\phi(\mathcal D_s)\in\mathbb R^d.
}
$$

整理成被试×潜变量维度的矩阵：

$$
\mathbf Z=
\begin{pmatrix}
Z_1^\top\\
Z_2^\top\\
\vdots\\
Z_N^\top
\end{pmatrix}
\in\mathbb R^{N\times d}.
$$

本研究**固定选择潜变量 Z 的维度与真实参数 T 的维度相同**：

| 模型 | T 维度 | 本研究固定使用的 Z 维度 |
|---|---:|---:|
| C1 | 4 | 4 |
| C2 | 4 | 4 |
| C3 | **6** | **6** |
| CR3 | 4 | 4 |

这是**本研究接下来分析所采用的固定实验配置**；仓库里历史上可能存在其他 latent_dim 的 checkpoint，但不属于本文定义的等维主要分析设置。

**代码入口：** [`np/s3_train.py`](../np/s3_train.py)（训练） · [`DiscreteNeuralProcess 定义`](../np/src/modlearn_rot/nn/np/point/discrete.py)（网络） · [`np/s5_extract_z.py`](../np/s5_extract_z.py)（所有试次提取后验均值 Z）。

---

### 3.6 Windows CMD：生成数据与训练 NP（两个主要阶段）

本节列出**从仓库根目录直接执行**的命令。研究流程分为“阶段 A：生成数据”和“阶段 B：训练神经网络”；但是现有代码的 `s3_train.py` 会强制检查 `s2_precheck.py` 生成的合格报告，因此两个主要阶段之间还必须完成一次 **S2 数据预检查**。实际执行顺序为 **S1 → S2 → S3**，不能省略 S2 或以“只运行两条 Python 命令”代替。

首次在本机运行（已有环境时忽略安装步骤）：

~~~cmd
conda activate py11
cd /d D:\Projects\mod-rot-3
git pull --ff-only
pip install -e ".[dev]"
~~~

#### 阶段 A：S1 生成真实参数 T、试次 X 和二元选择 Y

~~~cmd
python np/s1_generate.py --model C1 --data-seed 20261006
python np/s1_generate.py --model C2 --data-seed 20261006
python np/s1_generate.py --model C3 --data-seed 20261006
python np/s1_generate.py --model CR3 --subjects 5000 --design-seed 20261023 --data-seed 20261028
~~~

- C1、C2、C3 原始生成器默认各 5,000 位被试、每人 300–500 试次。此版本 `s1_generate.py` 对这三个模型调用原始生成器；**`--subjects` 参数不改变这三个模型类内固定的被试数量**，所以这里不传这个标志。
- CR3 明确设定 5,000 位被试、每人 500 个共享 X 试次；如果没有固定 bank，S1 将先准备相应的 CR2/CR3 设计文件。再次执行已存在数据的 S1 时会保留原有文件，**不会覆盖**。
- 四模型的原始生成数据保存到 `outputs/choice/<MODEL>/s1_generate/s1_generate__dataset.npz`，同时存储数据来源与摘要 manifest。

#### 必需的中间校验：S2 Precheck（训练前必须成功）

~~~cmd
python np/s2_precheck.py --model C1
python np/s2_precheck.py --model C2
python np/s2_precheck.py --model C3
python np/s2_precheck.py --model CR3
~~~

这些命令分别生成 `qualified` 检查报告。CR3 还会验证固定 X bank 的来源、500×6 形状、被试间共享关系及严格一阶随机占优等约束。`s3_train.py` 在缺少合格 S2 报告时会直接报错并拒绝训练。

#### 阶段 B：S3 训练各模型自己的 NP

下面以当前这份 README 所定义的**潜变量维数与真实参数维数相同**的配置为例，按四个模型**分别训练**。β 取之前监督 Oracle 四模型对照所使用的设置（不是声称这些就是全局最优超参数）：C1/C2/C3 用 `0.0005`，CR3 用 `0.00005`。四模型的稀疏惩罚系数设为 0，context 占比 0.25–0.80。

~~~cmd
python np/s3_train.py --model C1 --latent-dim 4 --beta 0.0005 --lambda-sparse 0 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --device auto
python np/s3_train.py --model C2 --latent-dim 4 --beta 0.0005 --lambda-sparse 0 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --device auto
python np/s3_train.py --model C3 --latent-dim 6 --beta 0.0005 --lambda-sparse 0 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --device auto
python np/s3_train.py --model CR3 --latent-dim 4 --beta 0.00005 --lambda-sparse 0 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --device auto
~~~

训练脚本各自独立运行，使用 TRAIN/VALIDATION/TEST 被试划分（默认 70%/15%/15%），在 VALIDATION 上选择早停 checkpoint；**所有 NP 训练均不使用真实 T**。`--device auto` 自动选择可用的 CUDA 或 CPU。

训练后的模型参数及日志保存到：

~~~text
outputs/choice/<MODEL>/s3_train/<CONDITION>/repeat01/
    s3_train__checkpoint.pt
    s3_train__history.csv
    s3_train__summary.json
~~~

**已有 checkpoint 的安全约束**：训练脚本不会静默覆盖已有 checkpoint；如果同一目录已存在与配置一致的 checkpoint，会保留已有文件，配置不一致则报错。要测试新的参数组合或独立重复运行，应合理调整条件参数或 `--repeat`，不要删除此前实验。

> **本文的两阶段命令到此结束**。训练完成后，若需要按第 3.5 节所述提取最终后验均值 Z，应另行执行 `np/s5_extract_z.py`（参见 [NP 流程说明](../np/README.md)）。S5 是训练后的下游数据提取，**不包含在本次要求的“生成数据、训练神经网络”两个主要阶段内**。

---

## 4. 已有事实：全部 Z 包含可线性恢复真实参数 T 的信息

对于每个模拟被试，我们现在有两套**按被试 ID 一一配对**的向量：

$$
\boxed{
(Z_s,T_s),\qquad s=1,\ldots,N.
}
$$

- \(T_s\) 是真实生成机制抽出的**真实参数**；
- \(Z_s\) 是在不知道真实参数的前提下，NP 从该被试实际行为中提取的**学习表征**。

此前的监督诊断已经表明，在本项目所研究的等维配置中，**使用全部 Z 维度进行线性回归，可以较好预测各个真实参数 T**：

$$
\boxed{
\widehat T_s=AZ_s+b.
}
$$

这一发现很重要：它意味着训练后的潜变量空间**已经包含较丰富的、可线性利用的个体参数信息**。因此，研究工作的重点不必先设为“从无到有重新学习参数信息”，而可以转向“如何从已经学得的潜变量中自动找出有理论意义的坐标轴”。

但必须严格区分：

**多元线性可恢复性 ≠ 各个原始 Z 维度已经对应独立的真实参数。**

例如，\(\widehat T_1=a_{11}z_1+\cdots+a_{1d}z_d\) 可以预测得很好，却不能说明 \(z_1\) 本身就是 \(T_1\)。所有 T 也可能依赖多个 Z 的混合。

目前的监督回归只作为**信息是否存在的诊断和可行性上限**；它不是项目希望最终使用的无监督发现方法，而且它的高预测表现并不保证线性关系精确、完美可逆或在所有 checkpoint 上相同。

既有分析实现：[`np/s7_regression.py`](../np/s7_regression.py) · [`监督 Procrustes / OLS 对照`](../math/09_procrustes_vs_ols/README.md)。

---

## 5. 整个项目的最终目标：不使用真实 T，自动把 Z 变换为可解释的 U

### 5.1 想要发现的不是另一个预测 Y 的神经网络，而是一组有意义的潜变量轴

训练 NP 后，我们已经拥有：

1. 所有被试的冻结 Z；
2. 原始训练所使用的行为信息 \((X,Y)\)；
3. 已经训练好的 encoder / decoder 及其可查询的预测函数。

**最终发现算法只允许使用这些信息**；真实 T 不进入发现过程。

目标是学习或确定一个变换：

$$
\boxed{
U_s=g(Z_s),
\qquad U_s\in\mathbb R^d,
}
$$

使新的坐标 \(U=(U_1,\ldots,U_d)\) 比原始 Z 的坐标具有更直接的认知参数解释。

这里先用一般记号 \(g\) 表示待发现的变换；本 README 的**第二部分（第 6 节）**将进一步限定目前要研究的两类候选：一般仿射变换和正交旋转（允许逐轴缩放但无需额外优化 D）。

### 5.2 “理想的 U”应当具备什么性质？

由于 T 是模拟中的真实生成参数，我们希望每个 U 维度只代表某一个真实参数的变化，并且不出现重复占用或遗漏参数。

具体要求：

**要求 A：每个 U 都与某一个 T 高度相关。**

对每一个 \(i\)，存在某个真实参数索引 \(j\)，使：

$$
|\mathrm{Corr}(U_i,T_j)|\text{ 尽可能接近 }1.
$$

绝对相关系数用于允许坐标轴正负号翻转。

**要求 B：同一 U 与其余非对应 T 的相关尽可能低。**

如果 \(U_i\) 对应 \(T_j\)，则希望：

$$
|\mathrm{Corr}(U_i,T_k)|\text{ 尽可能接近 }0,\qquad k\ne j.
$$

这意味着一个 U 轴应尽量只表达一个真实参数，而不是多个参数的混合。

**要求 C：形成真正的一一对应，不允许两个 U 都对应同一个 T。**

四维例子中：

$$
U_1\leftrightarrow T_3,\quad
U_2\leftrightarrow T_4,\quad
U_3\leftrightarrow T_2,\quad
U_4\leftrightarrow T_1
$$

与 \(U_i\leftrightarrow T_i\) **同样理想**。参数编号本身没有意义；真正重要的是对应关系为**双射（bijection）**：每个 U 恰好对应一个不同的 T；每个 T 都被且仅被一个 U 对应。

反之，如果 \(U_1,U_2\) 都高度对应 \(T_1\)，而没有 U 对应 \(T_2\)，即使其他相关性很高，也不算完整的成功恢复。

### 5.3 数学上：绝对相关矩阵应接近某个置换矩阵

令

$$
C_{ij}=|\mathrm{Corr}(U_i,T_j)|,
\qquad C\in[0,1]^{d\times d}.
$$

理想情况是：

$$
\boxed{
C\approx P,
\qquad
P\text{ 为 }d\times d\text{ 置换矩阵}.
}
$$

每一行和每一列的 P 都**恰好有一个 1，其余为 0**。例如：

$$
C\approx
\begin{pmatrix}
0&0&1&0\\
0&0&0&1\\
0&1&0&0\\
1&0&0&0
\end{pmatrix}
$$

表示 \(U_1\) 对应 \(T_3\)、\(U_2\) 对应 \(T_4\)、\(U_3\) 对应 \(T_2\)、\(U_4\) 对应 \(T_1\)。这与单位矩阵形式一样成功。

四个模型均采用对应维数的方阵表达；C3 的 P 是 \(6\times6\)。

这一目标可称为：

- **Axis-wise ground-truth factor recovery**：逐轴真实参数恢复；
- **Signed-permutation alignment**：允许正负号和编号变化的轴对齐；
- **Axis-aligned disentanglement**：轴对齐解耦表示。

上述名字用于表达研究目标，**并不意味着现阶段已经证明唯一识别性或已经设计出成功的无监督算法**。

### 5.4 两种评价/发现阶段必须严格分开

**阶段 1：无监督发现 U。** 算法只能访问冻结 NP、Z、X、Y 等允许的信息，求出 \(g\) 和最终 U；**不能读取 T 以决定变换方向、调超参数或挑选方法**。

**阶段 2：使用隐藏真值 T 进行事后评价。** U 固定以后，研究者才计算 U–T 的 Pearson 相关矩阵，检查每个 U 是否高相关于某一个 T、低相关于其他 T，以及是否形成完整双射。在解释相关矩阵时，还需要检查真实 T 之间的样本相关：即使理想地 \(U=T\)，若 T 本身有关联，非对应 Pearson 相关也未必严格等于 0。

这是本项目的核心研究逻辑：

$$
\boxed{
\underbrace{(X,Y)\xrightarrow{\text{NP training}}Z}_{\text{不使用真实 }T}
\quad\longrightarrow\quad
\underbrace{Z\xrightarrow[\text{可访问冻结 NP / 行为数据}]{\text{automatic discovery}}U}_{\text{不使用真实 }T}
\quad\longrightarrow\quad
\underbrace{U\stackrel{\text{post-hoc only}}{\longleftrightarrow}T}_{\text{仅用于评价}}
}
$$

**整个项目最终要解决的问题：能否仅利用已训练网络的知识及它生成的潜变量表示，在不查看真实认知参数标签的条件下，从潜变量空间中自动恢复与真实参数逐维一一对应的坐标？**

---

# 第二部分：要搜索的变换及目标

## 6. 当前只研究两类 Z → U 变换

**本部分承接第 5 节的科学目标，而不是重新训练 NP。** 对每个已完成 NP 训练的模型，都已获得按被试编号配对的 \(Z_s\) 和真实参数 \(T_s\)，其中二者的维度固定相同，记为 \(d\)（C1/C2/CR3 的 \(d=4\)，C3 的 \(d=6\)）。

我们希望在**不使用真实 T 作为拟合标签**的情况下，从已训练神经网络、冻结的 Z 及原始行为数据 (X,Y) 等信息中，自动找到一种变换 \(g\)，使新坐标 \(U_s=g(Z_s)\) 逐轴对应于真实参数 T。

目前明确规定**两种**变换搜索目标，之后可以添加第三种、第四种；本次只定义搜索空间及目标，不提前规定其对应的无监督 loss 或具体优化过程。

### 6.1 第一种：一般仿射变换（Affine transformation）

这里所说的“每个真实参数由若干 Z 按斜率线性组合，再加截距”在数学上称作**仿射变换**：

$
\boxed{
U_s=AZ_s+b
}
$

其中

$
A\in\mathbb R^{d\times d},\qquad b\in\mathbb R^d.
$

- \(A\) 是各维潜变量的线性组合系数矩阵：每一行都可以同时使用全部 d 个 Z 维度；允许旋转、缩放、非正交混合（shear）等。
- \(b\) 是各个输出 U 维度的**截距向量**：可实现不同参数的平移与均值校准。
- 所以 \(A\) 并不要求正交，也不要求元素为正数；它覆盖比第二种正交旋转更大的线性搜索空间。
- 如果希望保留全部 Z 信息，并且能够将变换精确吸收进冻结 decoder 的坐标重表达，应进一步要求 \(A\) **可逆（满秩）**。单纯用于监督预测 T 时，线性回归并不自动强制此条件。

按列向量约定，逐维写为：

$
U_i=b_i+\sum_{j=1}^{d}A_{ij}Z_j,\qquad i=1,\ldots,d.
$

**理想恢复目标**：找到某个 \(A^*,b^*\)，使 \(U\) 在经过允许的参数顺序重排后，尽可能接近**真实参数的原始数值**：

$
\boxed{
P(A^*Z_s+b^*)\approx T_s
}
$

这里 \(P\) 是一个置换矩阵，仅用于表示“输出 U 的维度顺序不一定和 T 的编号顺序一致”；也可将 P 吸收进 A 和 b 的行排列，因此在数学上不需要额外枚举 P 才能表示目标。

这个目标**比仅仅要求相关性高更强**：若要恢复 T 的原始数值，还需要斜率、截距和尺度的校准。\(U_i=10T_j+5\) 与 \(T_j\) 的 Pearson r 仍为 +1，却不是同一数值；而合适的仿射变换有能力同时恢复均值和尺度。

**参数数目（搜索自由度）**：

$
\underbrace{d^2}_{A}+\underbrace{d}_{b}
=d^2+d.
$

- 四维：\(4^2+4=20\) 个连续系数；
- 六维：\(6^2+6=42\) 个连续系数。

**极其重要的监督/无监督区分**：

$
\widehat T= A_{\mathrm{supervised}}Z+b_{\mathrm{supervised}}
$

是使用真 T 拟合的 OLS/Ridge 等**监督诊断**，已证实这类线性预测具有较好的可恢复性。我们现在想研究的不是重复做一次带标签回归，而是**不查看 T 的情况下自动确定 A 和 b**。因此 \(U\approx T\) 在本节只是最终的**事后成功判据和理想方向**，绝不是优化时可以直接计算的监督损失。

由于不给优化器真实 T，原始 T 的绝对单位和截距是否能仅凭行为与 decoder 信息唯一确定，**目前不能保证**；后面定义具体无监督指标时必须处理这一可识别性问题。

### 6.2 第二种：正交旋转 R + 逐轴缩放 D（实际只搜索 R）

第二种进一步限制线性变换的形状，希望通过**正交旋转（orthogonal rotation）**找到真实参数的方向：

$
\boxed{
U_s=RZ_{c,s},\qquad R^\top R=I_d
}
$

其中 \(Z_{c,s}=Z_s-\mu_{Z,\mathrm{TRAIN}}\) 表示将冻结 Z 用 TRAIN 被试的均值做**中心化**。在表达式 \(DRZ=T_{\mathrm{std}}\) 的简写中，**本节以后以 Z 表示这个已中心化的 Z**，避免原始 Z 具有非零均值时出现“不带截距却要拟合零均值 T”的不必要矛盾。中心化只涉及 Z，不使用真 T。

#### 6.2.1 真实参数先标准化：从 T 变为 Tstd

C1/C2/C3 的参数原本以 0 为中心，而 CR3 的 \(\alpha,\lambda,\gamma,\tau\) 具有不同均值、尺度和非高斯先验。为了比较各参数轴，首先把真实参数定义为**逐维标准化**：

$
\boxed{
T_{\mathrm{std},i}
=\frac{T_i-\mu_{T,i}}{\sigma_{T,i}},
\qquad
T_{\mathrm{std}}=S_T^{-1}(T-\mu_T)
}
$

其中 \(S_T=\mathrm{diag}(\sigma_{T,1},\ldots,\sigma_{T,d})\)。均值和标准差可从**TRAIN 真参数**计算，在 Oracle / 事后评价中固定并应用到 VALIDATION/TEST。这个 T 标准化是**研究目标的坐标定义**，不是说无监督旋转优化器可以读取真 T。

于是第二种方法的理想线性对应为：

$
\boxed{
D^*R^*Z\approx T_{\mathrm{std}}
}
$

这里公式中的 Z 已按上文约定做中心化。若发现的 U 轴排列与 T 的原顺序不同，可用带符号置换 \(P\) 写成 \(PD^*R^*Z\approx T_{\mathrm{std}}\)；等价地，也可以吸收 P 重新选择 R 的行顺序和符号。因此不要求“U1 必须对应 T1”。

- \(R\) 为 \(d\times d\) **正交矩阵**：\(R^\top R=I_d\)，改变方向并保持长度/内积；
- \(D=\mathrm{diag}(d_1,\ldots,d_d)\)，每个 \(d_i\ne0\)，仅对旋转后的**各个输出轴独立缩放**；
- 与第一种任意 A 不同，\(A=DR\) 不能表达任意 shear 或一般线性变换。\(A\) 的不同行向量彼此正交。

#### 6.2.2 为什么这里只需要搜索 R，不搜索 D？

我们的最终轴对齐评价基于 **Pearson 绝对相关系数**。任何非零逐轴缩放都不能改变一个输出 U 维度与真实参数的绝对相关：

$
\mathrm{Corr}(d_iU_i,T_j)
=\mathrm{sign}(d_i)\,\mathrm{Corr}(U_i,T_j),
$

$
\boxed{
|\mathrm{Corr}(DRZ,T_{\mathrm{std}})|
=
|\mathrm{Corr}(RZ,T_{\mathrm{std}})|
}
$

这里矩阵绝对值按元素计算，并且 D 必须是**在 R 后面的逐轴缩放**，不能把这个结论误用到 \(RDZ\)（在旋转前缩放输入 Z）。

因此，**对于“每个 U 是否找到正确 T 方向、一一对应是否清晰”的相关性目标，搜索 D 没有额外帮助**。

更进一步，我们**计划**在后续章节探索对各输出轴的尺度也不敏感的函数结构/复杂度指标，要求对任意非零对角 D：

$
\boxed{
L(DR)=L(R).
}
$

这是一项**待为具体指标证明和验证的设计要求**，而不是断言所有 Hessian / 非线性简单性指标天然具有此性质。只有当后续使用的具体指标满足该尺度不变性时，才有：

$
\boxed{
R^*=\underset{R^\top R=I_d}{\mathrm{argmin}}\ L(R)
}
$

而**不必同时数值优化 D**。R 选好后，如只是为了事后可视化或 Oracle 校准，可以再指定尺度；**不能用真实 T 的数值校准结果反向参与无监督搜索**。

这里也要明确：**省掉搜索 D 并不等于自动得到 \(RZ=T_{\mathrm{std}}\) 的数值相等**。我们希望先恢复正确轴的*方向*；恢复到 Tstd 的精确数值还涉及尺度校准，而仅凭 Pearson 相关无法确定这些尺度。

#### 6.2.3 正交 R 有多少自由度？排列和符号翻转怎么办？

四维时，虽然 R 有 16 个矩阵元素，但约束 \(R^\top R=I_4\) 使独立连续自由度只有：

$
\frac{4(4-1)}2=6.
$

可由六个二维平面旋转的有序乘积构成：

$
\boxed{
R=G_{12}(a)G_{13}(b)G_{14}(c)
G_{23}(d)G_{24}(e)G_{34}(f)
}
$

其中 \(G_{ij}\) 表示只在第 \(i,j\) 维平面进行旋转的 Givens 矩阵，\(a,b,c,d,e,f\) 为六个需要搜索的角度。

- **C1、C2、CR3：d=4，六个连续角度**。
- **C3：d=6，共 \(6(6-1)/2=15\) 个连续旋转自由度**。

正交矩阵可以分为行列式 +1 的严格旋转（\(SO(d)\)）和行列式 −1 的反射分支，二者合起来是 \(O(d)\)。由于目标允许 U 的各维**更换顺序、正负号翻转**，只要后续 loss 对逐轴符号和排列严格不变，单独在 \(SO(d)\) 中搜索连续角度，就能覆盖与 \(O(d)\) 相同的无向参数轴解及最小损失值；否则必须分别处理反射分支。

**本项目追求的是一组正确的轴，而不是唯一的 R 矩阵**：交换 U 的编号或翻转 U_i 的正负号得到的结果，应被看作等价的参数恢复，而不是失败。

### 6.3 两类变换的主要差别

| 比较内容 | 方案 1：一般仿射 | 方案 2：正交 R + D |
|---|---|---|
| 数学形式 | \(U=AZ+b\) | \(U=RZ\)，理想尺度校准为 \(DRZ\approx T_{\mathrm{std}}\) |
| 目标真实参数坐标 | 原始 T | 逐维标准化后的 Tstd |
| 是否允许截距 | 是，直接优化 b | 原始 Z 先固定中心化；不额外搜索截距 |
| 线性映射约束 | A 任意（若需可逆则满秩） | \(R^\top R=I\)，D 为非零逐轴尺度 |
| 搜索系数 | d²+d（四维 20；六维 42） | d(d−1)/2 个连续角度（四维 6；六维 15） |
| 是否搜索尺度 | 是，A 含各维单位转换 | **不搜索 D**，前提是具体 loss 尺度不变 |
| 是否需要用 T 监督搜索 | **不使用** | **不使用** |
| T 在项目中的作用 | 判断是否恢复原参数数值及匹配结构 | 判断是否恢复标准化参数轴及匹配结构 |
| 适用任务 | 更宽松的一般线性/仿射参数恢复 | 约束更强、更可控的正交轴恢复 |

两者是**独立的研究目标/搜索空间**，不能把“方案 1 的监督线性回归已经很准”直接当作“方案 2 的无监督正交旋转已经可被识别”的证明。

当前先确定这两种目标。以后如果需要加入新的第三种方法（例如另一个明确规定的变换族），再单独新增小节；**本次不提前定义第三种搜索空间，不设计具体 Hessian / 简单性 loss，不修改任何训练与旋转代码**。

---

## 本文范围说明

**已完成第一部分（第 1–5 节）**：真实生成模型、X/T/Y 的生成规则、NP encoder/decoder 和冻结 Z、已有全 Z 监督线性可恢复性、无 T 的逐轴一一对应目标，并提供 S1/S2/S3 Windows CMD。

**已完成第二部分（第 6 节）**：目前规定两种待发现的映射类型——**一般仿射变换** \(U=AZ+b\)（对齐原始 T）与**正交旋转加逐轴尺度** \(DRZ\approx T_{\mathrm{std}}\)（实际只搜索 R，前提是后续具体损失尺度不变）。明确了中心化、自由度、符号/排列等价和 T 仅用于事后评价。

**后续部分由下一次明确要求再写。** 目前尚未规定具体无监督 loss、Hessian/Sobol/函数简单性指标、估计方法、优化器或第三种变换族；本 README 不宣称这些尚未验证的方法已成功。
