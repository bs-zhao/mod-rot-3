# EXP24 — 稀疏性 / 简单性度量目录：替换冻结 EXP13 的 UX 项与 UU 项

**结论先行**：在 594 个候选（29 个 UX 度量、16 个 UU 度量、6 个联合度量及其组合）里，**没有任何一个在 CR3 上同时优于冻结 EXP13（UX_neff + UUd_share）又不伤 C1–C3**；留出的 NP 重复 2 上也一致。EXP13 未被修改，仍是当前推荐方案。

设置（与冻结 EXP13 一致）：p 通道；历史 300 被试（A0）二阶矩；每项独立 Haar-200（seed 1）标准化后等权相加；Haar 初始化的 SO(d) Adam 搜索。T / Oracle 只用于事后评价。代码：`catalog.py`（度量目录）、`audit.py`（示例向量审计）、`screen.py`（筛选 / 确认）、`analyze.py`（排名）；数据在 `results/`。

## 1. 文献与方法调研（只列实际检索到并可核对的来源）
- **稀疏性度量比较**：Hurley & Rickard, *Comparing Measures of Sparsity*, IEEE Trans. Inf. Theory 2009（[arXiv:0811.4706](https://arxiv.org/abs/0811.4706)）比较 15 个度量与 6 条准则（Robin Hood、Scaling、Rising Tide、Cloning、Bill Gates、Babies）；arXiv 版结论是只有 Gini 指数满足全部六条，IEEE 目录记录称 Gini 与 p≤1<q 的 pq-mean 满足，两个版本的说法不一致，需查原文，**本文只把 Gini 当候选，不引用其“唯一满足”的结论**。
- **L1/L2 稀疏度**：Hoyer, *Non-negative matrix factorization with sparseness constraints*, JMLR 5 (2004)（[页面](https://jmlr.org/papers/v5/hoyer04a.html)）：sp(x) = (√n − ‖x‖₁/‖x‖₂)/(√n − 1)，尺度不变，与 L1/L2 比值单调相关（目录中的 `l1l2`）。
- **旋转的简单性准则**：Browne, *An overview of analytic rotation in exploratory factor analysis*, Multivariate Behav. Res. 36 (2001)；Crawford–Ferguson 族（行 / 列简约度加权和）；quartimax / varimax 属 orthomax 族（参数 0 与 1）；Thurstone 的 simple structure 以行 / 列非零元个数衡量复杂度。对应目录中的 `cf0 / cf0.5 / cf1`、`varimax`、`rowneff / colneff`。
- **对数和惩罚**：Candès, Wakin & Boyd, *Enhancing sparsity by reweighted ℓ1 minimization*, J. Fourier Anal. Appl. 14 (2008)（[页面](https://web.stanford.edu/~boyd/papers/rwl1.html)；arXiv:0711.1612）：重加权 ℓ1 是对 Σ log(|x|+ε) 的 majorization-minimization；log-sum 并非该文首创。对应 `logsum`。
- **有效数 / 集中度**：Hill 数 = Rényi 熵的指数；q=0 为丰富度，q=1 为 exp(Shannon 熵)，q=2 为逆 Simpson / 逆参与比（inverse participation ratio）。对应 `neff`、`shannon`、`renyi*`、`maxshare`。检索没有找到把参与比用于神经网络 Hessian 谱的直接来源，未引用。
- 项目内沿用的度量：相对 L2 尾部（tail-L2，区分 [2,.011,.001] 与 [2,.001,.001]）、分数有效支持（覆盖 99% / 90% 能量所需条目数）。

## 2. 度量目录（`catalog.py`；越小越简单；均对整体缩放不变）
- **UX 部分（对 e_UX，d×p）29 个**：neff、Shannon、Rényi(0.5 / 2 / 4)、最大份额、tail-L2(k=1,2,4)、L1/L2、Gini、Lp 比(p=.5,.25)、log-sum(ε=.01,.1)、soft-L0(τ=.03,.1,.3)、有效支持(99% / 90%)、相对阈值软计数(1% / 10%)、行 / 列 neff、Crawford–Ferguson(κ=0,.5,1)、varimax、轴 neff、轴内最大份额均值。
- **UU 部分（对 e_UU，d×d）16 个**：自曲率占比（冻结）及其平方根 / 对数、非对角占比、非对角稀疏度（neff、L1/L2、tail1、Gini、Lp、soft-L0）、对角向量稀疏度（neff、L1/L2、Gini、tail1、soft-L0）、全部上三角含对角的稀疏度（neff、Gini、L1/L2）。
- **联合度量 6 个**（同时替换两部分）：对拼接向量的 neff、L1/L2、Gini、tail2、soft-L0，以及 UU 占二阶总能量份额。
- 组合 = UX 度量 × UU 度量（29 × 16 = 464）+ 单独 UX（29）+ 单独 UU（16）+ 联合（6）+ 冻结基线所在组合，共 **594 个**（含对 `no_signal` 的检查：无）。

## 3. 审计：用户给出的例子（`audit.py`、`results/audit.md`）
把 [3,2,.01,.01] 与 [1,1,.5,.1] 分别按振幅或能量解释：**所有 22 个条目级度量都判定前者更稀疏**，且对整体缩放和排列不变。tail2 / tail4 对 [2,.001,.001] 与 [2,.011,.001] 这类尾部极小差异不敏感（k 大于有效项数时），count / support 类对 [4,1,1,1] 与 [2,2,2,1] 的能量版判断不一致——是这些度量的已知局限，不是实现错误。

## 4. 筛选结果（`results/ranking_rep1.csv`，NP 重复 1，缩小预算 8 起点 × 150 步；`analysis_rep1*.txt`）
- 594 个候选在四个模型上全部有结果。冻结基线 `UX:neff + UU:diagshare`：CR3 MCC .8173、泄漏 .1646、最大泄漏 .368。
- **按预先写好的综合分数（四模型“泄漏相对 Oracle 与不旋转之间的位置”均值）排名会失真**：C2 的不旋转泄漏本来就接近 Oracle（.081 对 .026），分母很小，使 C2 的微小差异主导排名（排第一的 `renyi0.5 + diagshare` 在 CR3 上反而更差：MCC .804、泄漏 .190）。因此**看到排名后**改用“以 CR3 为主、C1–C3 的 MCC 不低于基线 .01”的口径（事后修改，已标注）。
- CR3 上：MCC 高于基线 +.002 的只有 3 个（均为 `tail4 + diagshare` 族，+.004 到 +.007），代价是 CR3 泄漏升到 .183–.188、C2 / C3 的 MCC 掉到 .87–.89；平均泄漏低于基线 .005 以上的：**0 个**；最大泄漏低于基线 .03 以上的：**0 个**（最低 .338 同样来自 `tail4` 族，同样伤 C2 / C3）。
- **UU 部分**：只有“自曲率占比”族（占比、平方根、对数，彼此是单调变换）有效；所有其他 UU 度量（非对角稀疏、对角集中、UU 占比）在组合中的平均排名远差，单独使用时 CR3 MCC 仅 .60–.76、C3 低至 .67–.73。UU 单独（`diagshare only`）：CR3 .759、C2 .940、C3 .730。
- **UX 部分**：neff / Shannon / colneff 与基线几乎相同（Shannon = log neff，二者在数学上同一排序）；l1l2、Rényi-0.5、log-sum、soft-L0、Gini、Lp 等让 C2 / C3 的泄漏略降（C2 .060 → .037–.044）但 CR3 明显变差（MCC .765–.808、泄漏 .185–.225）；varimax、Crawford–Ferguson 类在 CR3 的最大泄漏达 .47–.49。
- **联合度量**全部更差（`J_neff` CR3 MCC .733，`J_l1l2` .646，`UUtotal_share` .615）。单独 UX 稀疏（含 neff）：CR3 ≤ .733。

## 5. 确认（`results/confirm_table.txt`；冻结预算 24 × 250；19 个预先列好的候选；重复 1 与**留出**的重复 2）
表中数字 = MCC / 泄漏均值 / 最大泄漏 / 与 Oracle 的最大轴角度（诊断）。

| 候选 | CR3 重复 1 | CR3 重复 2（留出） |
|---|---|---|
| 冻结 `neff + diagshare` | .8173 / .1646 / .368 / 26.2° | .8122 / .1832 / .378 / 28.9° |
| `Shannon + diagshare`（neff 的单调变换） | .8176 / .1642 / .382 / 27.1° | .8123 / .1816 / .369 / 29.4° |
| `Shannon + sqrt_diagshare` | .8176 / .1634 / .377 / 26.8° | .8123 / .1825 / .374 / 29.1° |
| `l1l2 + log_diagshare` | .8081 / .1851 / .354 / 27.9° | .8011 / .1890 / .400 / 30.6° |
| `renyi0.5 + diagshare` | .8037 / .1896 / .365 / 28.8° | .8003 / .1856 / .389 / 31.2° |
| `tail4 + diagshare` | .8214 / .1878 / .342 / 26.7° | .7980 / .1981 / .412 / 31.1° |
| `UX:neff only` | .7301 / .2415 / .674 / 45.5° | .7328 / .2419 / .663 / 44.7° |

- 留出重复 2 上没有任何候选在 CR3 上超过冻结基线（最好的 `Shannon + diagshare` 是 neff 的单调变换，.8123 / .1816，与基线差异在噪声内）；重复 1 上唯一的 MCC 提高（`tail4`）在重复 2 **不复现**（.8214 → .7980），所以是偶然。
- C1–C3：l1l2 / Rényi / soft-L0 / log-sum 一类在 C2、C3 上略好且在重复 2 复现（C2 泄漏 .043 → .038–.039，C3 .048 → .039），但它们在 CR3 上更差，不能作为 EXP13 的替代。

## 6. 局限
- 筛选预算小于冻结预算（8 × 150）；排名只用重复 1，确认用冻结预算并有留出重复，但同一个 S1 数据集、只有两个 NP；仍属探索性。
- 只替换了 UX / UU 两部分的标量汇总，没有改变所用信息（decoder 二阶矩）；Haar 标准化等权不变。未做权重搜索（x6_plan06 的 E4 / EXP20–23 已显示等权之外的组合方式也无增益）。
- 度量目录并不穷尽：没有包含对三阶导、上下文细分（EXP16 方向）或 WY-only surrogate 的度量。

## 7. 复现
```cmd
conda activate py11
cd /d D:\Projects\rotation\mod-rot-3
python three_gaps/exp24_sparsity_catalog/audit.py > three_gaps/exp24_sparsity_catalog/results/audit.md
:: 筛选（8 个并行分片，各占 2 线程；需要 exp20 的逐被试矩缓存，见 three_gaps/exp20_latent_integration/）
python three_gaps/exp24_sparsity_catalog/screen.py 0 8 1 CR3,C1,C2,C3     :: 分片 0..7
python three_gaps/exp24_sparsity_catalog/analyze.py 1
:: 确认：设置 EXP24_TAG=confirm 与 EXP24_CANDS="候选1;候选2;..."，再以 24 250 运行 screen.py
```
