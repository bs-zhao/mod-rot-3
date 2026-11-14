# 04 · 无角色检测的 W–Y 简单性损失（UX / UU 稀疏性与 U 平滑性）

> **存档总结 · 2026-11-14（第二阶段，进行中）**
>
> **上一篇**：[03 · CR3 的 decoder 失真诊断与扣噪声底 + minimax 增益轴损失](03_cr3_decoder_noise_floor_summary.md)。
> 相关实验日志：[`three_gaps/EXPERIMENT_LOG.md`](../../three_gaps/EXPERIMENT_LOG.md)。
>
> **本篇性质**：对 03 之后改走“纯简单性损失”路线的阶段性存档。所有数字来自本机 `outputs/three_gaps/exp13/`（git 忽略），表格、热图和日志已复制到 [`three_gaps/exp13_pure_simplicity/results/`](../../three_gaps/exp13_pure_simplicity/results/)。**当前结论是：还没有一个对四个模型都成立的无角色检测损失。**

---

## 1. 为什么改路线

用户在 03 之后明确提出（2026-11-14）：

- **不要“找角色”类操作**：例如检测哪个方向对输出起整体增益或整体平移作用（03 中的阶段 A / minimax 轴）。它只对这批生成器有意义，缺乏普遍价值。
- **要用“简单性”本身定义损失**：U–X 作用的稀疏性、U–U 作用的稀疏性、U 的非线性 / 平滑性等。
- 此前已有的约束继续有效：不要 ICA 类分布损失；成功只能在观测 choice 或 NN 的 logit / p / choice 上判定，解析 Y 只用于辅助。

所以本篇的所有损失只由 NN decoder 在 (U, X) 上的二阶导数能量构成，不使用阶段 A、设计类别、噪声底扣除或任何参数角色的先验。

---

## 2. 损失定义（`three_gaps/exp13_pure_simplicity/losses2.py`）

旋转后 u = Qv，逐轴 z-score（`effective_A`）。对 NN decoder 的 logit 或 p 做自动微分，得到 g、H_UU = ∂²h/∂u∂u、H_UX = ∂²h/∂u∂x；取 300 个 TRAIN 被试 × 全部试次，计算二阶矩后只做小矩阵运算：

| 能量 | 定义 | 含义 |
|---|---|---|
| e_UX[i,j] | E[(∂²h/∂u_i∂x_j)²] | 第 i 根 U 轴与第 j 个 X 输入的混合作用强度 |
| e_UU[i,k] | E[(∂²h/∂u_i∂u_k)²]，i≠k | U 轴之间的交互（非加性） |
| e_UU[i,i] | E[(∂²h/∂u_i²)²] | 沿第 i 根轴的自曲率（非线性） |

所有损失都是这些能量的尺度无关统计量，对逐轴尺度、符号、排列不变：

| 家族 | 损失 | 定义 |
|---|---|---|
| UX 稀疏 | `UX_quartimax / UX_l1l2 / UX_neff / UX_softL0 / UX_rowNeff / UX_colNeff` | 对 e_UX 的集中度、Σ√e/√Σe、有效活跃数 exp(熵)、软 L0、按行 / 列的有效数 |
| UU 稀疏 | `UUo_quartimax / UUo_l1l2 / UUo_neff / UUo_share` | 对 e_UU 非对角元素同样的统计量；`UUo_share` = 非对角能量占 e_UU 总能量比 |
| U 曲率 | `UUd_quartimax / UUd_neff / UUd_share / UUd_sum / UUd_l1l2 / UUd_max` | 对 e_UU 对角（自曲率）的集中度；`UUd_share` = 自曲率能量占 e_UU 总能量的份额；`UUd_sum` = 自曲率总和 |

组合方式（`combos.py`）：各项在 200 个 Haar 随机旋转上算标准差，用它标准化后**等权相加**，权重事先固定，不看 T。

---

## 3. 结果（NN decoder，NP 重复 1，TEST 被试；T 只用于事后评价）

### 3.1 CR3 上的第一轮筛选（无任何额外结构，24 个起点）

| 损失 | logit MCC | p MCC |
|---|---|---|
| UX 稀疏（quartimax / l1l2 / neff / softL0 / 行 / 列） | .611–.695 | .598–.748 |
| UU 非对角稀疏（quartimax / l1l2 / neff / 占比） | .636–.658 | .599–.626 |
| U 曲率集中（`UUd_quartimax` / `UUd_neff`） | .691 / .696 | .739 / .743 |
| **`UUd_share`（自曲率占比，越小越好）** | **.821（泄漏 .179）** | .759（.231） |
| 不旋转 / 监督 Oracle | .573 / .872 | — |

CR3 里 τ、λ 各自线性（η 对 τ、对 λ 的自二阶导为零），只有 α、γ 非线性，所以“沿每根轴越线性越简单”确实对应真轴；这也是唯一明显有效的纯二阶项。

### 3.2 组合与四个模型（`results/heatmaps_simplicity_rep1.csv`）

| 损失 | 通道 | CR3 | C1 | C2 | C3 |
|---|---|---|---|---|---|
| 不旋转 | — | .573 | .668 | .956 | .588 |
| UX 稀疏 `UX_neff` | logit | .695 / 泄漏 .257 | **.969 / .022** | **.969 / .034** | **.962 / .039** |
| | prob | .730 / .241 | **.969 / .022** | **.969 / .036** | **.960 / .046** |
| 自曲率占比 `UUd_share` | logit | .821 / .179 | .833 / .232 | .782 / .261 | .770 / .189 |
| | prob | .759 / .231 | .968 / .024 | .940 / .122 | .730 / .231 |
| **`UX_neff` + `UUd_share`（等权）** | logit | **.831 / .160** | .925 / .132 | .912 / .136 | .919 / .107 |
| | prob | **.817 / .165** | .969 / .021 | .965 / .060 | .960 / .046 |
| 监督 Oracle | — | .872 / .089 | .969 / .018 | .971 / .026 | .963 / .036 |

热图：`results/heatmaps_simplicity_logit_rep1.png`、`results/heatmaps_simplicity_prob_rep1.png`（行 = 模型，列 = 不旋转 / UX 稀疏 / 自曲率占比 / 组合 / Oracle）。

### 3.3 要点

1. **UX 稀疏单独**就把 C1–C3 恢复到监督 Oracle 水平（两个通道、泄漏与 Oracle 相当），没有任何角色检测——这是本阶段最干净的结果。
2. **CR3 上它不够**（.695 / .730）。直接量到（X 已标准化）：CR3 里 `UX_neff` 在 Oracle 处的 log-有效数 2.36，随机旋转均值 2.31，即二阶 U–X 能量对真轴没有偏好；C1 则是 1.17 对 1.98。
3. **自曲率项**让 CR3 到 .82–.83，但在 C1–C3 的 logit 通道上会引入噪声：这些模型对参数线性，真实 H_UU 本来为零，Haar 离散度标准化把它放大到与 UX 项等权。
4. **组合**是目前 CR3 上最好的纯简单性损失（.831 / .817），但 C1–C3 的 logit 通道出现可见混合（泄漏 .107–.136，Oracle .018–.036），CR3 的泄漏（.160 / .165）也仍明显高于 Oracle（.089）。

### 3.4 没有成功的权重方式

| 做法 | 结果 |
|---|---|
| 自然尺度：`log neff(UX) + μ·曲率占总二阶能量的份额`（X 已标准化），μ = 0 / 1 / 3 / 10 | μ 完全不起作用（CR3 logit .724–.726，p .668–.669）：CR3 里二阶总能量几乎全在 UX（H_UU 仅占 ~10⁻³），曲率份额在旋转间只变化 ~10⁻⁴ |
| 按“各项能达到的简单性 z 分数”加权（z = (Haar 均值 − 本项最优值) / Haar 标准差） | CR3 .755 / .770，C1 logit .923，C2 logit .909，C3 logit .921；更差（线性模型里 NN 的曲率也有可达简单性，z 并不小） |
| 把一阶 UX 格子（`cells`）加进来、或三项一起 | CR3 .637–.781，变差 |

---

## 4. 限制与注意事项

- **只在 NP 重复 1 上跑了**，没有重复 2 的确认；组合项的选择（`UX_neff + UUd_share`）是在 CR3 重复 1 的 TEST 指标上看出来的，因此不是独立验证。
- 第一轮筛选只在 CR3 上做；C1–C3 只验证了 `UX_neff`、`UUd_share` 及其组合。
- 损失是由二阶矩的统计量构成，没有使用 |·| 型逐点稀疏度（需要原始导数场，开销大），未评估。
- Haar 标准化用 200 个随机旋转（种子 1），不同种子会有小幅差异；起点数 24，种子 7。
- 二阶导数来自 NN decoder 自动微分（只用 NN 输出，符合 Y 规则）；没有做 decoder WY → sieve 的 Gap3 版本。

---

## 5. 复现

```cmd
conda activate py11
cd /d D:\Projects\rotation\mod-rot-3

:: 缓存 decoder 二阶导数场（每个模型、通道一次）
python three_gaps/exp13_pure_simplicity/cache2.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --channel logit
python three_gaps/exp13_pure_simplicity/cache2.py --model C1  --beta 0.0005  --zdim 4 --pca 0 --channel logit
:: （其余模型 / 通道同理，C3 用 --zdim 6）

:: 第一轮筛选、组合、权重试验
python three_gaps/exp13_pure_simplicity/screen.py CR3 logit 1 all 24
python three_gaps/exp13_pure_simplicity/combos.py CR3 logit 1 "UX_neff+UUd_share"
python three_gaps/exp13_pure_simplicity/natural.py CR3 logit 1 0,1,3,10
python three_gaps/exp13_pure_simplicity/weighted.py CR3 logit 1 16

:: 热图
python three_gaps/exp13_pure_simplicity/heatmaps.py 1
```

---

## 6. 文件清单

| 路径 | 内容 |
|---|---|
| `three_gaps/exp13_pure_simplicity/cache2.py` | 缓存 decoder 的 g、H_UU、H_UX |
| `…/losses2.py` | 二阶矩、旋转后能量、全部简单性损失 |
| `…/screen.py` | 单个损失的多起点搜索与 TEST 评估 |
| `…/combos.py` | Haar 标准化的等权组合 |
| `…/natural.py`、`weighted.py` | 自然尺度与 z 加权两种权重试验（均未成功） |
| `…/heatmaps.py` | U–T 热图 |
| `…/results/` | 热图 PNG、汇总 CSV、CR3 筛选 CSV、各脚本日志 |

---

## 7. 诊断：这批通用损失在真实函数上是强信号，decoder 才是瓶颈（`oracle_scan.py`）

对一批通用的简单性损失，算 **监督 Oracle 处的值比 150 个随机旋转的均值低多少个标准差**：z = (Haar 均值 − L(Oracle)) / Haar 标准差。z ≤ 0 的损失不可能在 Oracle 处被最小化。这里 Oracle / T 只用作**诊断**（用来找哪些损失有信号），不用于搜索；解析场只作诊断，不计成功。结果全表见 `results/oracle_scan.csv`。

目录覆盖：UX(i,j)、UX(i,j,上下文)、一阶 G(i,上下文)、UU 非对角、UU 自曲率、UU(i,k,上下文)、UX+UU 联合，各配 neff / quartimax / l1l2 / 软 L0；以及 colLog 家族、各块占比。CR3 上下文 = W-only 格子（`pure_cells`），连续 X 的一阶格子 = Legendre 基函数投影。

### 7.1 主要发现

| 损失 | CR3 解析场（线性估计 T 代入） | CR3 decoder logit / p | C1–C3 decoder |
|---|---|---|---|
| UU 非对角，软 L0 | **7.8** | −0.77 / −0.76 | −1.1 到 +8.1 |
| UU(i,k,上下文)，软 L0 | **7.8** | −0.74 / −0.68 | — |
| 一阶 U×上下文，软 L0 | **5.5** | 0.17 / 0.40 | 5.5 到 14.5（强，扣底版） |
| UX(i,j,上下文)，软 L0 | **5.3** | −0.76 / −0.83 | 4.7 到 24.5（强） |
| 自曲率占 UU 能量的份额 | 1.95 | 0.88 / 0.74 | 0.5 到 3.9 |

1. **C1–C3 的 decoder 上，UX 与一阶 U×上下文类损失 z 常为 +3 到 +25（强信号）**，与 §3.2 中 `UX_neff` 达到 Oracle 一致。
2. **CR3 的 decoder 上几乎所有二阶稀疏性度量 z 为负**（Oracle 处比随机旋转还差），只有自曲率份额 (0.88 / 0.74) 和少数一阶度量有微弱正信号。
3. **CR3 的真实（解析）函数上这些度量是强信号**（软 L0 类 z = 5–8）。所以不是“CR3 本身不简单”，而是 NN decoder 把简单性弄丢了；受损最重的是奖励“精确为零”的计数型统计（软 L0），因为 decoder 的梯度和二阶导中有一层噪声底。
4. **二阶部分按“各块最小特征值”扣底后仍不恢复**（`results/logs/exp13_scan_floor2.log`：CR3 的 UU / UX 扣底版 z 仍为 −0.3 到 −1.0）：decoder 的二阶结构比加性噪声底扭曲得更多（见 03 §3.2：二阶零结构在 decoder 中整体消失）。

### 7.2 一阶 U×上下文软 L0 对噪声底扣除倍数的响应（CR3，`floor_multiple_scan.py`）

| 扣除倍数（相对各格子最小特征值的最小值） | decoder logit | decoder p | 解析场 |
|---|---|---|---|
| ×0 | 0.17 | 0.40 | 5.52 |
| ×1 | 0.27 | 2.00 | 5.52 |
| **×2** | **3.54** | **3.00** | 5.52 |
| ×3 | 3.19 | 2.98 | 5.52 |
| ×5 / ×8 | 2.32 / 0.83 | 1.29 / 0.62 | 5.52 |

含真零方向的格子在 decoder 中的最小特征值为 0.0014–0.0025（logit），即噪声底；取最小值的最小值会偏低，扣 2–3 倍后 Oracle 处 z 升到 3.0–3.5。**注意**：“×2”是看着 Oracle 处的 z 挑的（用了 T 诊断），不是 T-free 推出来的。

### 7.3 但 Oracle 处 z 高并不等于能被优化找到（`cells_opt.py`，CR3 重复 1，无阶段 A）

| 扣除倍数 | 软 L0 logit MCC / 泄漏 | 软 L0 p MCC / 泄漏 |
|---|---|---|
| ×1 | .632 / .290 | .639 / .283 |
| ×2 | .661 / .275 | .721 / .233 |
| ×3 | .707 / .237 | .745 / .220 |

优化找到的最优点并不在 Oracle 附近：随机旋转的最小损失比 Oracle 处低得多，所以 z = 3 的“必要条件”并不够。**当前仍没有一个无角色、对四个模型都成立的损失。**
