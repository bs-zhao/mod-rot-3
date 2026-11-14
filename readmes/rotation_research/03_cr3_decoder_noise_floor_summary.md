# 03 · CR3 的 decoder 失真诊断与“扣噪声底 + minimax 增益轴”W–Y 简单性旋转损失

> **存档总结 · 2026-11-14**
>
> **上一篇**：[01 · 项目生成模型、NP 与两种变换目标](01_project_overview.md)；[02 · Hessian / 导数结构实验设计](02_hessian_simplicity_experiment.md)。
> 相关实验日志：[`three_gaps/EXPERIMENT_LOG.md`](../../three_gaps/EXPERIMENT_LOG.md)（EXP00–EXP08 为本篇的起点）。
>
> **本篇性质**：对 11/14 这一轮的结果存档。所有数字来自本机 `outputs/three_gaps/`（git 忽略），关键表格和图已复制到 [`three_gaps/exp11_cr3_search/results/`](../../three_gaps/exp11_cr3_search/results/)。

---

## 1. 这一轮要解决什么

目标（概览 §1）：不使用真实 T 作为监督，从冻结 NP 的潜变量 Z 和行为数据 (X, Y) 出发，用 **W–Y 简单性** 损失搜索正交旋转 R，使 U = RZ 的各轴与真实参数 T 一一对应。

两条约束（用户 2026-11-12 / 2026-11-14 明确）：

1. **最终成功只能在以下 Y 上成立**：真实观测 choice，或神经网络输出的 logit / p / choice。真实解析式 logit / p 只可辅助找指标，**不计成功**。
2. **损失必须基于 W–Y 简单性**（函数 (U,X)→Y 的结构）。ICA / 非高斯性这类基于分布的损失不要。

起点（EXP07/EXP08）：冻结的两阶段方法在 C1–C3 上达到监督 Oracle，在 CR3 上只有 TEST MCC ≈ 0.80（Oracle 0.872）。本轮只攻 CR3，并要求不破坏 C1–C3。

评价口径：除 MCC（匹配后平均 |corr|）外，**同时报告泄漏**（匹配后非对角 |corr| 的均值 / 最大值）。MCC 对混合不敏感，用户判断 U–T 热图是否“干净”看的是泄漏。

---

## 2. 起点：混合到底有多重（三鸿沟热图，EXP10）

TEST 被试上的 |Corr(U, Tstd)|，Gap3 = decoder 的 WY → sieve → 旋转（NN logit）：

| 模型 | 不旋转 MCC / 泄漏 | 冻结方法 Gap3 logit | 监督 Oracle |
|---|---|---|---|
| C1 | .668 / .365 | .969 / .022 | .969 / .018 |
| C2 | .956 / .081 | .971 / .029 | .971 / .026 |
| C3 | .588 / .297 | .962 / .038 | .963 / .036 |
| **CR3** | .573 / .340 | **.805 / .175** | .872 / .089 |

CR3 的混合有固定结构：α↔λ 互漏（约 0.37 / 0.35），γ↔τ 互漏（约 0.46 / 0.53）。解析函数上同一方法几乎干净（泄漏 .060），所以问题出在 decoder。

> 注意：`supervised_abs_rotation/step5` 的 “best available / 网格” 结果以监督 R 为 0° 基准，不是盲搜，不能当成功证据；其盲搜的 CR3 轴偏差曾达 20–50°。

---

## 3. 诊断：decoder 为什么在 CR3 上失真（EXP12 + 梯度场分析）

### 3.1 被否定的假设：后验平均造成的平滑

假设 decoder ≈ E[f(T,x)|z]。用 TRAIN 回归 Tstd ≈ A V + b，以其残差分布给解析 logit 加后验平滑，强度 s = 0 / 0.5 / 1 / 1.5，再对每个场跑冻结方法：

| 场（CR3，硬阶段 A） | absL1 MCC / 泄漏 | quartimax MCC / 泄漏 |
|---|---|---|
| 解析，线性估计 T 的插入式（s=0） | .860 / .115 | .848 / .136 |
| 解析，加平滑 s = 0.5 / 1 / 1.5 | .861 / .861 / .862 | .848 / .849 / .849 |
| NN decoder | .803 / .177 | .833 / .151 |

平滑强度几乎不影响输出 ⇒ **后验平均不是主因**。线性估计 T 本身（A 非正交）也只把 .872 降到 .860。差距在 .860 → .803，即 NN 函数本身相对解析式的偏差。

### 3.2 真正的原因：decoder 梯度里的各向同性噪声底

把 decoder 梯度（对 V）与插入式解析梯度比较，映射回 T 空间（g_t = g_v A⁻¹）：

- 整体相对误差只有 0.151；全局线性映射 g_dec ≈ 0.86·g_plug 的 R² = 0.851，映射矩阵近似 0.86·I ⇒ **没有轴间混合**，只有整体收缩。
- 但误差集中在解析上严格为零的格子：

| 试次类别 | 参数 | 解析梯度能量 | decoder 梯度能量 |
|---|---|---|---|
| 只有收益 | λ | **0.000** | **0.009** |
| 概率 0.5 | γ | **0.000** | **0.007** |
| 只有收益 | α / γ / τ | .005 / .006 / .015 | .010 / .011 / .023 |
| 其余类别 | 各参数 | — | 约为解析的 0.76–0.96 倍 |

所以 decoder = 略微收缩的真梯度 + 约 0.007–0.01 的各向同性噪声底。噪声底恰好抹平了识别 λ（只有收益试次中为零）和 γ（概率 0.5 试次中为零）的零结构，这就是 α–λ、γ–τ 分不开的原因。

### 3.3 第二个原因：阶段 A 的增益轴被 λ 污染

CR3 中 τ 在所有试次里都是增益型（梯度 ∝ η），而 λ 只在“只有损失”的试次里也是增益型（η ∝ τλ）。池化检测出的增益轴因此混入 λ：

| 检测所用试次（logit） | 轴与 λ 的相关 | 轴与 τ 的相关 |
|---|---|---|
| 全部（原阶段 A） | 0.37 | 0.83 |
| 只有收益 | 0.01 | 0.85 |
| 只有损失 | 0.73 | 0.62 |

硬约束固定这根轴，等于把污染锁进整个旋转：此时 quartimax 在 Oracle 处的损失比硬约束下的最优点低 12.7%。

---

## 4. 方法（EXP11 final pipeline）

全部只用 NN decoder 的 W–Y，T 只在事后评价。入口：`three_gaps/exp11_cr3_search/final_pipeline.py`。

1. **上下文**：连续 X → 没有高频取值，只有一个上下文（Legendre 格子照旧）；CR3 固定 bank → 按 X 各列“是否取其高频众数值（如金额 0、概率 0.5）”的模式划分（W-only，`harness.mode_contexts`），自动还原出 只有收益 / 只有损失 / 概率 0.5 三个关键类别（52/60、35/40、60/60 与设计类别重合）。
2. **输出等变轴（阶段 A）**：对每个上下文 c 计算增益 / 平移轮廓投影掉之后的相对残差 e_c(w)；CR3 取 **minimax**：argmin_w max_c e_c(w)。接受门槛 e < 0.5 且 e < 0.6·e₂（比例门槛见 §7）。C1–C3 沿用池化检测（单个上下文，minimax 退化为池化）。
3. **扣噪声底**：每个格子的能量矩阵 E_c = mean g gᵀ，底 σ² = min_c λ_min(E_c)（T-free），E′_c = U max(Λ − σ², 0) Uᵀ；轴 i 的格子能量 e[i,c] = (A E′_c Aᵀ)_ii。
4. **损失**：colLog3（`common/losses.py`，对各格子内各轴能量份额取 log），在固定了阶段 A 行之后的正交补上做 SO(d−1) 多起点（12 起点）搜索。CR3 的损失格子用设计类别（见 §7）。

附带收益：改成直接用格子能量矩阵后，一次搜索从约 1 分钟降到约 6 秒。

---

## 5. 结果

### 5.1 最终流水线（TEST，NN decoder，两个独立训练的 NP，logit 与 p 两个通道）

| 模型 | rep | 通道 | MCC | 泄漏均值 / 最大 | Oracle MCC | Oracle 泄漏均值 / 最大 |
|---|---|---|---|---|---|---|
| CR3 | 1 | logit | .853 | .118 / .313 | .872 | .089 / .227 |
| CR3 | 1 | prob | .856 | .115 / .295 | .872 | .089 / .227 |
| CR3 | 2 | logit | .829 | .143 / .378 | .869 | .105 / .223 |
| CR3 | 2 | prob | .856 | .112 / .316 | .869 | .105 / .223 |
| C1 | 1 / 2 | logit / prob | .969 | .017–.023 / ≤.054 | .969–.970 | .015–.018 / ≤.046 |
| C2 | 1 / 2 | logit / prob | .970–.971 | .026–.030 / ≤.134 | .971 | .026–.027 / ≤.101 |
| C3 | 1 / 2 | logit / prob | .962–.965 | .036–.038 / ≤.132 | .963–.965 | .035–.036 / ≤.107 |

对比 EXP08 的冻结方法（CR3 TEST MCC：rep1 logit .805 / prob .792；rep2 logit .807 / prob .803）：重复 2 上提高 0.02–0.05，泄漏 .115–.143 接近该次训练 Oracle 的 .105。C1–C3 与 Oracle 无差别。
热图：`results/final_heatmaps_rep1.png`、`results/final_heatmaps_rep2.png`（含不旋转、旧冻结方法、新方法两通道、Oracle）。

### 5.2 CR3 上各步骤的贡献（rep 1，TEST MCC）

| 做法 | logit | p |
|---|---|---|
| 冻结方法：硬阶段 A + absL1 | .803 | .793 |
| 14 损失 × 2 视图筛选中最一致者 rowNeff（GCat） | .816 | .816 |
| quartimax / colLog3（未扣底，硬阶段 A） | .833 / .830 | .791 / .744 |
| **colLog3 + 扣底（硬阶段 A，池化轴）** | .845 | .860 |
| **colLog3 + 扣底 + minimax 轴（最终）** | **.853** | **.856** |
| 监督 Oracle | .872 | .872 |

### 5.3 没有帮助或不采纳的做法（均在 rep 1 的 CR3 decoder 上）

| 做法 | 结果 |
|---|---|
| 广义线性族 GL(d)（无阶段 A，absL1） | .622 / 泄漏 .319，更差 |
| 起点数 6 → 48（quartimax、rowNeff、cellNeff） | 所有起点收敛到同一点，数值不变 ⇒ 瓶颈不在优化 |
| 完全不用阶段 A | quartimax .740、rowNeff .744、absL1 .598 ⇒ 增益信息有用 |
| 阶段 A 改软惩罚（权重 0.3 / 1 / 3） | 最好 quartimax 权重 3：logit .843，p 只有 .806，不稳定 |
| logit + p 通道、多损失组合 | .745–.817，无增益 |
| 硬阶段 A 解上放开约束精修 | 损失更低但 TEST 降到 .62–.75（无约束损失的全局最小不在真轴） |
| 非高斯性（ICA 类）项 | CR3 有小幅提升，C1–C3 在大权重下明显退化；**用户明确不要，不采纳** |
| 直接用 9 个纯 W 上下文当损失格子 | CR3 降到 .78–.80（格子过细，切碎了区分 λ、γ 的关键格子）；上下文仅用于检测增益轴则恢复 .853 / .856 |

---

## 6. 结论

1. CR3 失败的主因是 **NN decoder 梯度中的各向同性噪声底**（填满 λ 在只有收益试次、γ 在概率 0.5 试次中的解析零格子），其次是 **增益轴被 λ 污染**。后验平均（平滑）不是主因。
2. 修复只用 W–Y：扣噪声底 + minimax 增益轴 + colLog3。CR3 的 TEST MCC 从 ≈.80 提到 .83–.86（Oracle .87），在**从未参与调参的第二个 NP** 上同样成立；C1–C3 保持 Oracle 水平。
3. 该损失对 C1–C3 无退化；CR3 仍未追平 Oracle。

---

## 7. 限制与注意事项（务必保留）

- **CR3 未追平 Oracle**：MCC 平均差 0.022，泄漏最大值仍有 0.30–0.38；热图中 α–λ（约 0.32）与 γ–τ（约 0.38）还有残余混合。
- **一处事后改动**：minimax 轴的比例门槛由 0.5 放宽到 0.6，因为 rep 1 / p 通道 e₁/e₂ = 0.51 以 0.01 之差被拒。rep 2 是在放宽后的规则下直接运行、未再调参，所以 rep 2 对整条流水线是独立检验，但对这个门槛本身不是。
- **损失格子仍依赖设计类别**：CR3 的损失格子用 bank 设计类别（gain-only / loss-only / p=0.5 / other）。纯 W 的 9 上下文划分还原了关键类别，但直接当损失格子会变差。尚无不依赖设计标记的通用格子划分。
- **噪声底规则**是本轮自行设定的（各格子 λ_min 的最小值），系数 1–4 倍结果稳定，但没有理论保证。
- 验证范围：同一 S1 数据集与同一 train/test 划分；只有 2 个训练的 NP。**未在新数据集上复现。**
- CR3 的 binary 通道（NN 生成 choice）本轮没有重跑；Gap1/Gap2（解析 / 公式 Y）只作辅助，不计成功。
- 筛选和超参选择（损失名、起点数、软权重）在 rep 1 上用 TEST 指标观察过，rep 2 仅用于确认。

---

## 8. 复现

```cmd
conda activate py11
cd /d D:\Projects\rotation\mod-rot-3

:: 三鸿沟热图（需先用 --save-q 重跑 exp06）
python three_gaps/exp06_two_stage/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --view GCat --losses absL1 --variants AB --starts 12 --steps 250 --tag hm --save-q
python three_gaps/exp10_heatmaps/run.py

:: 缓存 decoder 梯度场（四个模型、两个通道、两个重复）后，跑最终流水线
python three_gaps/exp11_cr3_search/cache.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --channel logit
python three_gaps/exp11_cr3_search/final_pipeline.py
python three_gaps/exp10_heatmaps/run_final.py

:: decoder 失真诊断
python three_gaps/exp12_decoder_distortion/run.py
```

`final_pipeline.py` 需要环境变量 `GENERIC_CTX=1`（脚本默认已设置）。

---

## 9. 文件清单

| 路径 | 内容 |
|---|---|
| `three_gaps/exp10_heatmaps/run.py`、`run_final.py` | 三鸿沟热图、最终流水线热图 |
| `three_gaps/exp11_cr3_search/cache.py` | 缓存 decoder / 解析梯度场 |
| `…/harness.py` | SO(d) / GL(d) 优化、评估、纯 W 上下文划分 |
| `…/screen_losses.py`、`multistart.py`、`softA.py`、`ensemble.py`、`polish.py` | 筛选 / 多起点 / 软阶段 A / 组合 / 精修 |
| `…/floor.py` | 扣噪声底的能量矩阵损失 |
| `…/minimaxA.py` | minimax 增益轴 |
| `…/final_pipeline.py` | 最终流水线（四模型 × 两重复 × 两通道） |
| `three_gaps/exp12_decoder_distortion/run.py` | 平滑 / 插入式 / decoder 三类场的对照诊断 |
| `three_gaps/exp06_two_stage/run.py` | 新增 `--save-q`，只多存旋转，不改搜索 |
| `three_gaps/exp11_cr3_search/results/` | 汇总 CSV、热图 PNG、各脚本日志 |
