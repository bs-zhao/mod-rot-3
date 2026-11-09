# three_gaps experiment log

所有数字来自本机 `outputs/three_gaps/`（git 忽略）。条件：C1/C2 β=0.0005 zdim4，C3 β=0.0005 zdim6，
CR3 β=0.00005 zdim4，均 `--pca 0`。

## EXP00 — 监督 Oracle 上限（只用于事后参照）

| model | OLS R² (TEST, 各参数) | 仅旋转 Oracle MCC | worst | off-max | 恒等(不旋转) MCC |
|---|---|---|---|---|---|
| C1 | .94 .94 .94 .94 | 0.969 | 0.967 | 0.046 | 0.668 |
| C2 | .92 .95 .95 .95 | 0.971 | 0.961 | 0.060 | **0.956** |
| C3 | .91 .94 .93 .93 .93 .93 | 0.963 | 0.953 | 0.088 | 0.588 |
| CR3 | .75 .84 .72 .75 | 0.872 | 0.847 | 0.227 | 0.573 |

- 原始 Z 上仅用正交旋转即可接近 OLS 上限；ZCA 白化不改变 MCC（CR3 off-max 0.23→0.12）。
- **C2 的原始 Z 已经几乎对齐**（恒等 0.956），因此 C2 对“无监督恢复”检验信息量低。

## EXP01 — Gap 1 第一轮：14 个导数视图 × 12 个复杂度损失 × 2 通道

- G0：所有损失 L(PDQ)=L(Q) 误差 ≤ 4e-6（尺度/符号/置换不变性成立）。
- C1：43/228 通过，**全部基于混合 Hessian H_UX**（logit/prob 都过）；梯度、UU Hessian 类全失败。
  与理论一致：C1 logit 中 H_UX 为常数矩阵，列 L1 只在带符号置换处最小（Birkhoff 凹性）。
- C2：2/240（logit H_UX rmsL1、colL1L2）。
- C3：0/240。原因：b2/b3（b4/b5）作用于同一 X 坐标，只靠 RMS 幅度区分，真轴只是**很浅的二次极小**，
  有限 X 样本的奇数阶交叉矩把极小推离 2.5°。⇒ 只靠幅度的识别在 decoder 阶段必然脆弱。
- CR3：1/336（prob HUU/trial softL0）。HUUd rmsL1 在含 λ/τ 的 5 个平面上是尖点（logit 对 λ、τ 线性
  ⇒ 自曲率恒为 0），α–γ 平面失败。

## EXP03 — Gap 1 第二轮：刺激基投影视图

把导数场投影到正交刺激基 Ψ 上（连续 X：Legendre 张量基，总次数 ≤3；CR3 固定 bank：试次指示基），
“单元格”= 基函数。C1–C3 中每个真实参数的梯度剖面在 Legendre 基上的支撑**互不重叠**，
RMS→L1 产生尖点，并且按 Birkhoff 凹性论证真轴是全局极小。

- C3：**GP·rmsL1**（梯度投影）logit/prob 都通过，随机旋转裕度 0.84 / 0.23；HUXP·rmsL1 logit 通过。
- CR3：只有 HUUP·softL0（逐试次 Hessian 支撑计数）通过；GP·softL0_03 在含 λ 或 γ 的平面上是干净尖点，
  但 **α–τ 平面完全平坦**（α、τ 没有可区分的试次零模式）。
- 修复：C1 的投影梯度有精确零单元，√0 处梯度无穷导致 NaN，已改为相对 ε 平滑平方根。

## EXP04 — Gap 3（冻结 decoder）

- **C3 · GP · rmsL1**：decoder autodiff 与 decoder-WY→sieve（logit / prob / binary）**全部**得到
  TEST MCC 0.963 = Oracle（恒等 0.588），与 Oracle 方向相差 1.6–3.2°。
- CR3：所有 Gap-1 候选在 decoder 上失败（MCC 0.58–0.72），且 L(oracle) > L(found)：失败不是优化问题。

## EXP05 + 诊断 — 为什么 CR3 在 decoder 上失败

- Z→T 基本线性（poly1..4 的 TEST R² 几乎相同），CR3 的信息损失是后验收缩噪声，不是非线性扭曲。
- 在 Oracle 轴下 decoder 的**二阶零结构全部消失**（λ、τ 自曲率占比 0.77/0.67，解析为 0）。
  解释：decoder ≈ E[f(T,x)|z]，后验平均破坏了“对 λ/τ 线性”这类二阶性质。
- **一阶试次零结构以相对对比的形式保留**：gain-only 试次上相对梯度能量最小的方向与 λ 的 TEST
  |corr| = 0.84；p=0.5 试次上最小方向与 γ 0.77–0.80（Oracle ≈ 0.87）。
- **输出等变轴**：decoder 中存在显著增益方向（∂η/∂w ∝ η），残差 0.107（次小 0.966），与 τ 相关 0.83；
  C 模型中存在显著平移方向（∂η/∂w ∝ 1），残差 0.02/0.99，与 b1 相关 0.95–0.97。

## EXP06 — 两阶段方法（进行中）

阶段 A：显著的输出等变轴（平移/增益，阈值 e1<0.3 且 e1<0.3·e2，预先固定）；
阶段 B：在正交补上最小化刺激基单元格复杂度。
- CR3 解析：阶段 A 精确找到 τ（e1=0.000）；阶段 B 用逐试次单元格时只有 prob·softL0_03 恢复（4.4°）。
- CR3 decoder：阶段 A 找到增益轴（0.11–0.19），整体 MCC 约 0.74（B-only 约 0.63）。
- 正在测试：CR3 的情境单元格改为**设计类别**（gain-only / loss-only / p=0.5 / general）。

### EXP06 补充测试（CR3）

- 设计类别单元格（GCat）+ 阶段 A：解析 softL0_10 完美（1.2°/0.4°），但 decoder 仅 0.74；
  colLog3 decoder logit 0.830 但 prob 0.744；absL1 两通道最一致（0.80/0.79）。
- 白化坐标（`--whiten 1`）：0.76–0.79，不优于原始坐标。
- 另外两个 CR3 checkpoint（β=1e-5 / 1e-4，仅报告不选择）：0.68–0.78，Oracle 0.866 / 0.773。
- 广义特征“类别零轴”逐个/贪心检测：解析 MCC 1.000，decoder 0.67–0.78（出现解析式中不存在的伪零轴）。
- binary 通道：逐试次 sieve 的梯度噪声给单元格加正能量下限，抹掉相对零结构；du=2/3、
  重复抽样 R=10/50 均无法救回（R=50 时 Gap 2 仍偏 38°）。

## EXP07 — 冻结方法的最终三鸿沟表（AB 两阶段，阶段 B 损失 absL1，GCat 视图）

判据（预先固定）：Gap 1/2 与真 Tstd 轴最差夹角 < 5°；Gap 3 TEST 匹配 |corr| ≥ Oracle − 0.02。

| model, channel | Gap1 解析 | Gap2 公式WY→sieve | Gap3 decoder autodiff (诊断) | Gap3 decoder WY→sieve |
|---|---|---|---|---|
| C1 logit | PASS 0.7° | PASS 0.7° | PASS 0.969/0.969 | PASS 0.969/0.969 |
| C1 prob | PASS 0.0° | PASS 0.1° | PASS 0.969/0.969 | PASS 0.969/0.969 |
| C1 binary | – | PASS 0.7° | – | PASS 0.969/0.969 |
| C2 logit | PASS 0.3° | PASS 0.4° | PASS 0.971/0.971 | PASS 0.971/0.971 |
| C2 prob | PASS 0.4° | PASS 1.0° | PASS 0.971/0.971 | PASS 0.971/0.971 |
| C2 binary | – | PASS 0.9° | – | PASS 0.971/0.971 |
| C3 logit | PASS 1.6° | PASS 1.3° | PASS 0.962/0.963 | PASS 0.962/0.963 |
| C3 prob | PASS 0.1° | PASS 0.8° | PASS 0.963/0.963 | PASS 0.963/0.963 |
| C3 binary | – | PASS 0.8° | – | PASS 0.963/0.963 |
| CR3 logit | fail 10.2° | fail 10.0° | fail 0.803/0.872 | fail 0.805/0.872 |
| CR3 prob | fail 5.4° | fail 5.7° | fail 0.793/0.872 | fail 0.792/0.872 |
| CR3 binary | – | fail 33.8° | – | fail 0.708/0.872 |

（恒等变换 TEST MCC：C1 0.668，C2 0.956，C3 0.588，CR3 0.573。）

- rmsL1 代替 absL1：C1–C3 结论相同；CR3 更差（logit 0.771，prob 0.742）。
- 不用阶段 A（B-only）：C1–C3 结论相同（阶段 A 对 C 模型非必需）；CR3 降到 0.60/0.66。
  ⇒ 阶段 A 的增益轴是 CR3 从 ~0.60 提升到 ~0.80 的主要来源。

## EXP08 — 新训练 NP 上的独立确认（repeat 2）

方法完全冻结（GCat 视图 / absL1 / 阶段 A 阈值 / 种子、起点、步数均与 EXP07 相同）；只换 NP：
`np/s3_train.py --repeat 2`（新训练种子，其余超参与 repeat 1 相同）→ S5 全试次 Z → EXP00 → EXP06。
Gap 1/2 不依赖 NP（在真 Tstd 坐标中），只重跑 decoder 来源。判据不变：TEST MCC ≥ Oracle − 0.02。
（注：4 个训练并行会让笔记本卡死；`exp08_fresh_np/pipeline.sh` 改为顺序执行、每个进程 4 线程。）

| model, channel | Gap3 rep1 | **Gap3 rep2（新 NP）** | 与 Oracle 夹角 rep1 / rep2 |
|---|---|---|---|
| C1 logit / prob / binary | PASS 0.969/0.969 | **PASS 0.969/0.970** | 1.9–2.1° / 1.5° |
| C2 logit / prob / binary | PASS 0.971/0.971 | **PASS 0.970/0.971** | 2.2–2.4° / 2.4–2.6° |
| C3 logit / prob / binary | PASS 0.962–0.963/0.963 | **PASS 0.965/0.965** | 3.5–3.6° / 2.5–3.4° |
| CR3 logit | fail 0.805/0.872 | fail 0.807/0.869 | 30.4° / 29.8° |
| CR3 prob | fail 0.792/0.872 | fail 0.803/0.869 | 34.5° / 31.3° |
| CR3 binary | fail 0.708/0.872 | fail 0.700/0.869 | 48.7° / 48.8° |

- 新 NP 的恒等 MCC：C1 0.686，C2 0.919（仍近对齐），C3 0.664，CR3 0.709 ⇒ C1、C3 仍是真正的旋转恢复检验。
- 阶段 A 在两次训练中检测到同样的轴（C：平移 e1 ≤ 0.09；CR3 logit/prob：增益 e1 ≈ 0.11/0.18；
  CR3 binary：无），decoder autodiff 与 decoder-WY→sieve 再次几乎一致。
- **结论被复现**：C1–C3 在新 NP 上三通道全部达到 Oracle；CR3 在两次训练中都稳定停在 ≈0.80（Oracle 的 92%），
  失败模式相同 ⇒ CR3 的瓶颈是系统性的 decoder 失真，不是某一次训练的偶然。

## 结论（当前）

1. **C1、C2、C3：同一个冻结指标（刺激基投影梯度 × absL1，可选阶段 A）连续跨过三个鸿沟**，
   logit / prob / binary 三通道都在 TEST 上达到监督 Oracle（与 Oracle 方向相差 ≤ 3.6°）。
   理论依据：真实参数的梯度剖面在正交刺激基上支撑互不重叠，RMS/均值→L1 在带符号置换处形成尖点，
   Birkhoff 凹性论证给出全局极小。C2 的原始 Z 已近对齐（恒等 0.956），对它的检验信息量低。
2. **CR3 未通过**。解析式上需要三种结构：增益轴（τ）+ 设计类别零模式（λ: gain-only；γ: p=0.5）
   + 正交补（α）；只有组合起来才能在解析式上恢复。decoder 上一阶零模式只以相对对比形式保留，
   二阶零模式（对 λ/τ 线性）完全消失；最好结果约 0.80（Oracle 0.872 的 92%，恒等 0.573）。
   decoder-WY→sieve 与 decoder autodiff 几乎一致 ⇒ 瓶颈是 **decoder 函数失真（Gap 3）**，不是 M2 估计。
3. ~~不是独立验证~~ → **EXP08 已在新训练的 NP（repeat 2）上原样复现**：C1–C3 全部 PASS，CR3 同样 ≈0.80 失败。
   仍然共享同一个生成数据集（S1）与 train/test 划分；未在新数据集上复现。
