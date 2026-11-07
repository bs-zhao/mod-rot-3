# Toy 真实生成模型总表（T1–T11）

> **本文放在 \`mod-rot-3/readmes/\`，作为研究框架中「生成模型层」的独立参考。**
> **权威来源**：[modlearn-rot-toy 原始模型目录](https://github.com/bs-zhao/modlearn-rot-toy/blob/main/catalog/README_TOY_MODELS.md)；实际实现是该仓库的 \`src/rottoy/toy_models/\`、\`src/rottoy/models.py\`、\`src/rottoy/transforms.py\`、\`src/rottoy/sampling.py\`。
> **核对日期**：2026-11-07。以下 T1–T11 的公式、变量定义及理论输出范围均按原 Toy 项目的代码/模型目录保留；不是从其他实验结果反推生成式。
> **定位**：与 [Mod-Rot 总研究框架](README.md) 配合阅读。本文件是**真实解析生成函数**的说明，M001–M024 是另一个独立层面的简单性指标，不能混为一谈。

## 阅读前必须区分的三件事

**（1）真实 Toy 模型 ≠ C1/C2/C3/CR3 的原始模拟数据集。** Toy 将所有基础参数和 X 坐标定义在 \([0,1]\)，由解析公式生成连续数值 Y。即使 T9 复用了 C2 的数学式、T11 复用了 CR3 的彩票模型数学式，也**没有**复用原来的参数先验、实际被试、试次设计或二元观察标签。

**（2）真实 Toy 函数 ≠ 神经网络近似函数。** 原始模型为 \(\eta_T(\theta,X)\)（analytic oracle）；\`-id/-sig/-sq/-tanh/-sp/-exp\` 只是同一基础函数的六种输出包装。Toy 项目如在 \`--backend both\` 下训练连续输出 MLP，它拟合的 \(\widehat F\) 不是新真实生成模型，也不是 Mod-Rot NP 的训练 checkpoint。

**（3）\([0,1]\) 是解析定义域及数值积分参考区域，不应擅自称作所有模型的实际被试“先验”。** Toy 代码 \`sampling.py\` 用 scrambled Sobol 在单位立方体内部取积分节点（默认端点余量 \`1e-4\`），通常按单位立方体上的均匀参考测度数值积分。它**没有定义与 C1–CR3 真实参数先验相同的概率分布**，也没有在这些 Toy 指标实验中进行 Bernoulli choice 抽样。

---

# Toy 模型目录：基础函数、参数与连续输出范围

本文件是 `src/rottoy/toy_models/` 与 `src/rottoy/transforms.py` 的数学说明。基础函数统一用 **T1、T2、…** 编号。每个基础函数可以使用全部输出变换，版本名为 **`T编号-后缀`**，例如 `T9-sig`。任何输出都属于**连续数值**，`sig` 只是连续 Sigmoid 函数，不采样二分类 choice。

## 1. 统一约定

对 T 模型，令真实参数 $\theta\in[0,1]^K$，刺激输入 $X\in[0,1]^{D_X}$，基础函数为 $\eta_T(X,\theta)\in\mathbb R$。所有参数坐标和所有输入坐标**分别**在闭区间 $[0,1]$。计算时默认使用区间内部的 Sobol 节点，因此有限采样通常不能达到下表的理论端点；幂函数在 $x=0,b=0$ 等位置可能需要单独约定极限，数值实现避开这些边界。

$$
Y_{T,w}(X,\theta)=g_w\!\left(\eta_T(X,\theta)\right),
\qquad u=R\theta,\qquad
\eta'_{T,w,R}(X,u)=g_w\!\left(\eta_T(X,R^\top u)\right).
$$

$R$ 是正交旋转。**旋转后 $u$ 不必仍位于 $[0,1]^K$**，但相应的真实参数 $\theta=R^\top u$ 必须处于原物理定义域。范围与模型参数化无关：正交重参数化不改变在相同物理域上的 $Y$ 理论上下界。

### 全部变换：每个 T 模型自动拥有以下六个后缀

| 版本后缀 | 连续输出公式 $g_w(t)$ | 固有值域 | 可逆性 | 结构作用 |
|---|---|---|---|---|
| **`-id`** | $t$ | $\mathbb R$ | 是 | 保留原函数 |
| **`-sig`** | $\sigma(t)=1/(1+e^{-t})$ | $(0,1)$ | 是；logit | 有界饱和，但**不是 Bernoulli** |
| **`-sq`** | $t^2$ | $[0,\infty)$ | 否 | 损失符号，引入平方和乘积 |
| **`-tanh`** | $\tanh(t)$ | $(-1,1)$ | 是；atanh | 对称饱和 |
| **`-sp`** | $\mathrm{sp}(t)=\log(1+e^t)$ | $(0,\infty)$ | 是 | 正值且光滑，近指数/线性 |
| **`-exp`** | $e^t$ | $(0,\infty)$ | 是；log | 正值和乘性效应 |

上表的“固有值域”是变换作用于全部实数时的值域；**具体 toy 的 $Y$ 有更窄的实际理论上下界**，详见下方第 3 节。

**T11 特别说明：** T11 与其他 toy 一样使用单位立方体数学输入；其四个归一化参数分别还原成 CR3 物理参数，X 中四个金额坐标还原到美元单位。这样既保留原 CR3 效用公式，也维持统一 Toy 数值积分接口。

## 2. 模型公式速查

**每个 Toy 一个完整的基础生成公式**；不同输出版本统一由第 1 节的后缀函数作用于该式，不重复定义模型。

**T1**

$$
\eta_1=ax^b
$$

**T2**

$$
\eta_2=ax_1+bx_2+c
$$

**T3**

$$
\eta_3=ax_1+abx_2
$$

**T4**

$$
\eta_4=ax_1^b+cdx_2^b
$$

**T5**

$$
\eta_5=ax_1+bx_1^2+cx_2+d
$$

**T6**

$$
\eta_6=ax_1+bx_2+cx_1x_2+d
$$

**T7**

$$
\eta_7=ae^{bx}
$$

**T8**

$$
\eta_8=ax_1^b+cdx_2^b+cx_3
$$

**T9**

$$
\eta_9=b_1+\sqrt3\,b_2(X_{1A}-X_{1B})+\frac{3\sqrt5}{2}b_3(X_{1A}^2-X_{1B}^2)+\sqrt3\,b_4(X_{2A}-X_{2B})
$$

**T10**

$$
\eta_{10}=ax_1^b+acx_2^b+d
$$

**T11**

$$
\begin{aligned}
\eta_{11}=\tau\Big[
&\sigma\!\left(\gamma\log\frac{q_A}{1-q_A}\right)
 \left(\frac{g_A}{100}\right)^\alpha
-\lambda\sigma\!\left(\gamma\log\frac{r_A}{1-r_A}\right)
 \left(\frac{\ell_A}{100}\right)^\alpha\\
&-\sigma\!\left(\gamma\log\frac{q_B}{1-q_B}\right)
 \left(\frac{g_B}{100}\right)^\alpha
+\lambda\sigma\!\left(\gamma\log\frac{r_B}{1-r_B}\right)
 \left(\frac{\ell_B}{100}\right)^\alpha
\Big].
\end{aligned}
$$

其中 $L\in\{A,B\}$，概率与补概率按原 CR3 代码分别裁剪：

$$
q_L=\min(1-\varepsilon,\max(\varepsilon,p_L)),\qquad
r_L=\min(1-\varepsilon,\max(\varepsilon,1-p_L)),\qquad
\varepsilon=10^{-12}.
$$

**T11-id** 是上述基础 logit；**T11-sig** 再套最外层 sigmoid，得到原 CR3 的连续概率函数。公式内部用于概率加权的 sigmoid 始终保留。

### 参数、输入与输出范围

各 Toy 的**每个参数和 X 坐标**都在 $[0,1]$。T11 使用下方定义的物理单位映射。

| ID | 参数 | X | 基础 $\eta_T$ 理论边界 |
|---|---|---|---|
| **T1** | $a,b$ | $x$ | $[0,1]$ |
| **T2** | $a,b,c$ | $x_1,x_2$ | $[0,3]$ |
| **T3** | $a,b$ | $x_1,x_2$ | $[0,2]$ |
| **T4** | $a,b,c,d$ | $x_1,x_2$ | $[0,2]$ |
| **T5** | $a,b,c,d$ | $x_1,x_2$ | $[0,4]$ |
| **T6** | $a,b,c,d$ | $x_1,x_2$ | $[0,4]$ |
| **T7** | $a,b$ | $x$ | $[0,e]$ |
| **T8** | $a,b,c,d$ | $x_1,x_2,x_3$ | $[0,3]$ |
| **T9** | $b_1,b_2,b_3,b_4$ | $X_{1A},X_{1B},X_{2A},X_{2B}$ | $[-A,1+A]$ |
| **T10** | $a,b,c,d$ | $x_1,x_2$ | $[0,3]$ |
| **T11** | $\theta^*_\alpha,\theta^*_\lambda,\theta^*_\gamma,\theta^*_\tau$ | $x_1,\ldots,x_6$ | $\subseteq[-B,B]$ |

其中 $A=2\sqrt3+3\sqrt5/2\approx6.8182$，$B=24\cdot2^{0.999}\approx47.96674$。T11 的 $[-B,B]$ 是**保守输出界**（保证覆盖输出，未必是最窄值域）；T1–T10 给出解析边界。

**T11 的单位映射：** $(g_A,\ell_A,p_A,g_B,\ell_B,p_B)=(200x_1,200x_2,x_3,200x_4,200x_5,x_6)$，其中 $g,\ell$ 是非负美元金额；$\alpha=0.02+0.979\theta^*_\alpha$、$\lambda=0.3+4.7\theta^*_\lambda$、$\gamma=0.02+0.979\theta^*_\gamma$、$\tau=0.5+3.5\theta^*_\tau$。它保留 [原 CR3 数学公式](https://github.com/Cognition-Decision-Modeling-Lab/modlearn-rot-2/blob/main/experiment/choice/research_based/s0_cr2_design_recovery.py)，但使用 Toy 的统一数值求积节点，并非原数据集的 500 个刺激或其参数先验。

**结构速记：** T1/T7 幅度与形状；T2 线性效应；T3 $ab$ 乘积；T4 $cd$ 不可识别；T5 同一 X 的线性与二次效应；T6 $x_1x_2$ 项；T8 为 T4 加入 $cx_3$；T9 原 C2 函数；T10 共享指数及 $ac$ 项；T11 原 CR3 效用差函数。

## 3. 每个 toy 的六种 $Y$ 理论取值范围

令 $S(t)=\sigma(t)$、$Q(t)=\tanh(t)$、$P(t)=\log(1+e^t)$，以及上面定义的 $A=2\sqrt3+3\sqrt5/2$ 和 $B=24\cdot2^{0.999}$。因为 $S,Q,P,\exp$ 都严格单调，所以它们的输出边界直接由 $\eta$ 的边界映射得到。平方版本在跨越零点时要特别处理。

| ID | `-id`：$Y=\eta$ | `-sig`：$Y=S(\eta)$ | `-sq`：$Y=\eta^2$ |
|---|---|---|---|
| **T1** | $[0,1]$ | $[\tfrac12,S(1)]$ | $[0,1]$ |
| **T2** | $[0,3]$ | $[\tfrac12,S(3)]$ | $[0,9]$ |
| **T3** | $[0,2]$ | $[\tfrac12,S(2)]$ | $[0,4]$ |
| **T4** | $[0,2]$ | $[\tfrac12,S(2)]$ | $[0,4]$ |
| **T5** | $[0,4]$ | $[\tfrac12,S(4)]$ | $[0,16]$ |
| **T6** | $[0,4]$ | $[\tfrac12,S(4)]$ | $[0,16]$ |
| **T7** | $[0,e]$ | $[\tfrac12,S(e)]$ | $[0,e^2]$ |
| **T8** | $[0,3]$ | $[\tfrac12,S(3)]$ | $[0,9]$ |
| **T9** | $[-A,1+A]$ | $[S(-A),S(1+A)]$ | $[0,(1+A)^2]$ |
| **T10** | $[0,3]$ | $[\tfrac12,S(3)]$ | $[0,9]$ |
| **T11** | $[-B,B]$ | $[S(-B),S(B)]$ | $[0,B^2]$ |

| ID | `-tanh`：$Y=Q(\eta)$ | `-sp`：$Y=P(\eta)$ | `-exp`：$Y=e^\eta$ |
|---|---|---|---|
| **T1** | $[0,Q(1)]$ | $[\log2,P(1)]$ | $[1,e]$ |
| **T2** | $[0,Q(3)]$ | $[\log2,P(3)]$ | $[1,e^3]$ |
| **T3** | $[0,Q(2)]$ | $[\log2,P(2)]$ | $[1,e^2]$ |
| **T4** | $[0,Q(2)]$ | $[\log2,P(2)]$ | $[1,e^2]$ |
| **T5** | $[0,Q(4)]$ | $[\log2,P(4)]$ | $[1,e^4]$ |
| **T6** | $[0,Q(4)]$ | $[\log2,P(4)]$ | $[1,e^4]$ |
| **T7** | $[0,Q(e)]$ | $[\log2,P(e)]$ | $[1,e^e]$ |
| **T8** | $[0,Q(3)]$ | $[\log2,P(3)]$ | $[1,e^3]$ |
| **T9** | $[Q(-A),Q(1+A)]$ | $[P(-A),P(1+A)]$ | $[e^{-A},e^{1+A}]$ |
| **T10** | $[0,Q(3)]$ | $[\log2,P(3)]$ | $[1,e^3]$ |
| **T11** | $[Q(-B),Q(B)]$ | $[P(-B),P(B)]$ | $[e^{-B},e^B]$ |

**区分理论输出范围和有限数据观察范围：** T1–T10 为解析边界（幂函数端点按极限理解）。T11 六种输出范围由保守界 $[-B,B]$ 映射得到，未必是精确最窄区间。`--u-points` / `--x-points` 只改变数值求积精度；不应该根据样本最小值/最大值重写这里的理论上下界。

另外，以上是**解析生成函数**的严格范围。神经近似 $F$ 的输出头会约束到每种变换本身的值域，但不一定严格满足特定 T 模型更窄的上下界。要检查 $F$ 是否越界，应将神经预测与这些理论范围对照。

## 4. 运算、数值取点和新增模型

- `--models T3-id T3-sig T9-sq`：一次选择多个函数版本；裸 `T3` 是 `T3-id` 的简写。
- `--u-points 100 --x-points 100`：指定 U/θ 与 X 的数学求积节点数；不是被试或试次。
- `--x-mode axis_sweeps`：逐一遍历 X 维，每个取固定网格，其他 X 维度固定 0.5。
- `--pairs all --angles 181`：遍历全部参数轴对的平面旋转。
- `--analysis-scale raw|log|inverse`：普通指标分析尺度；平方包装不可逆。
- `--operator-scale raw|inverse`：功能算子（M017–M024）的独立分析尺度。

新增下一项 T12 时，在 `src/rottoy/toy_models/t12.py` 增加一次 `@register_model("T12", k=..., dx=..., description=..., eta_bounds=(L,H))`，无需修改其他模型、输出变换、求积或绘图代码。**必须同步维护本文件：参数/X 取值范围、基础函数、$\eta$ 精确取值范围和六个 $Y$ 版本的输出上下界。**



---

## 5. 原始生成模型参数的角色速查

本节是根据以上公式的结构标注，**不是新增模型定义**。所有参数和 X 在 Toy 中均定义于 \([0,1]\)，T11 是先转换回 CR3 物理单位后计算。

| 模型 | 参数含义 | 模型结构要点 |
|---|---|---|
| T1 | \(a\)：幅度；\(b\)：幂指数 | 幅度与函数形状共同决定 \(a x^b\) |
| T2 | \(a,b\)：不同 X 的线性系数；\(c\)：截距 | 完全线性的加性对照 |
| T3 | \(a\)：整体幅度；\(b\)：仅影响第二个 X 作用的相对系数 | \((a,b)\) 通过乘积 \(ab\) 耦合 |
| T4 | \(a\)：第一项幅度；\(b\)：共享幂指数；\(c,d\)：第二项幅度的乘积分量 | **原代码显式标记 \`identifiable=False\`**：仅有 \(cd\)，不能把 c/d 单独识别出来 |
| T5 | \(a\)：\(x_1\) 线性；\(b\)：\(x_1^2\)；\(c\)：\(x_2\)；\(d\)：截距 | 同一 X 上线性与二次共存 |
| T6 | \(a,b\)：两个 X 主效应；\(c\)：\(x_1x_2\) 交互；\(d\)：截距 | X×X 交互对照 |
| T7 | \(a\)：幅度；\(b\)：指数速率 | 指数型响应变化 |
| T8 | 继承 T4 的 \(a,b,c,d\)，额外加入 \(c x_3\) | 给 c 增加独立可观察作用，是原 T4 的可识别性对照 |
| T9 | \(b_1\)：截距；\(b_2,b_3\)：X1 差值的线性/二次系数；\(b_4\)：X2 差值斜率 | C2 原数学结构的单位立方体 Toy 版本 |
| T10 | \(a\)：共同幅度；\(b\)：共享幂指数；\(c\)：第二项的相对幅度；\(d\)：截距 | 两项共用 a/b，包含 \(ac\) |
| T11 | 归一化的 \(\alpha,\lambda,\gamma,\tau\) 分别对应效用幂指数、损失厌恶、概率加权、logit 缩放 | CR3 四参数函数结构、但 Toy 的单位立方体积分/输入设计 |

### T9 与 C2 的精确关系

- **相同**：\(\eta\) 的四参数函数形式，以及原始 X 列顺序 \((X_{1A},X_{1B},X_{2A},X_{2B})\)。
- **不同**：原 C2 的 \(b_j\) 是独立 \(N(0,0.5^2)\)，原 X 列独立 \(\mathrm{Uniform}(-1,1)\)，且按被试/试次重新抽取；Toy T9 的参数及 X 在单位立方体上取值并由 Sobol 网格积分。
- **因此**：即使函数公式完全相同，Hessian 积分所得均值、稀疏性、Fisher 或旋转损失也可能改变。对比必须控制积分测度。

### T11 与 CR3 的精确关系

- **相同**：原 CR3 的 \(\eta=\tau(V_A-V_B)\) 与 \(w(p,\gamma)=\sigma(\gamma\,\mathrm{logit}(\mathrm{clip}(p)))\)，包括对 \(p\) 和 \(1-p\) **分别裁剪**；\`T11-sig\` 即该函数外套选择概率 sigmoid。
- **不同**：Toy 的归一化四参数分别**仿射**变到 \(\alpha\in[.02,.999]\)、\(\lambda\in[.3,5]\)、\(\gamma\in[.02,.999]\)、\(\tau\in[.5,4]\)，X 的金额列乘以 200 美元。原 CR3 的参数先验是 Beta / Gamma / Beta / truncated-Normal，X 为固定 500 个经过设计的彩票比较试次；Toy 在单位立方体上使用 Sobol/轴扫描点，不复制真实 500 Xset。
- **关键结果**：T11 与 CR3 的数学函数结构可同源，但**积分统计量与旋转最小值不必完全相同**。在比较复杂度指标前，最好统一物理坐标、参数积分权重和 X bank。

## 6. 放入整体研究框架时的约定

1. **L1 生成函数**：本文件给出 T1–T11 的基础 \(\eta_T\)，以及六种确定性连续 \(Y=g(\eta)\)。
2. **L3 Hessian 来源**：对 Toy 解析函数直接求导是 oracle；如果另行训练代理函数，则须区分 \(\eta\) / \(\widehat F\) 的预测与导数误差。原 Toy 的神经端是连续输出 MLP；**不可将其称为已经训练的 NP decoder**。
3. **L4 积分**：记录 \(n_\theta,n_X\)、Sobol seed、X mode、单位立方体坐标与是否做物理尺度映射；Sobol 积分点不是原数据中的“500 次试次”。
4. **L5 简单性指标**：M001–M024 是另外的指标目录，可参考 [Toy 原始指标 README](https://github.com/bs-zhao/modlearn-rot-toy/blob/main/catalog/README_METRICS.md)；指标成功与否不是生成模型的一部分。
5. **L6 旋转**：\(\theta\mapsto u=R\theta\)，实际评估 \(\eta_T(R^\top u,X)\)；旋转后 u 不一定在单位立方体内，原坐标 \(\theta\) 仍要处于定义域。两两扫描和三/四维联合旋转只是不同的坐标变换，不是新的 Toy 生成函数。
6. **未来扩展**：新增 T12 时先在 Toy 源码注册 \(\eta_{12}\) 与理论 \(\eta\) 范围，再同步更新原 Toy 的模型目录和本研究总表；不要仅靠实验输出 CSV 来推断生成公式。

### 溯源链接

- [原 Toy T1–T11 数学目录](https://github.com/bs-zhao/modlearn-rot-toy/blob/main/catalog/README_TOY_MODELS.md)
- [Toy 真实生成器实现目录](https://github.com/bs-zhao/modlearn-rot-toy/tree/main/src/rottoy/toy_models)
- [Toy 的统一输出变换](https://github.com/bs-zhao/modlearn-rot-toy/blob/main/src/rottoy/transforms.py)
- [Toy 的参数及 X 数值积分节点](https://github.com/bs-zhao/modlearn-rot-toy/blob/main/src/rottoy/sampling.py)
- [Toy 中 T11 与 CR3 数学公式一致性的测试](https://github.com/bs-zhao/modlearn-rot-toy/blob/main/tests/test_t11_cr3.py)

> **一句话**：这 11 个真实解析生成模型、六种输出包装及其定义域/物理映射是 Toy 的 **ground truth**；以后无论换 Hessian 提取器、积分方案、简单性指标还是 SO(d) 旋转，都不应默默改动这层。
