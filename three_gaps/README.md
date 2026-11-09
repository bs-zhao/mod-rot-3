# three_gaps — 依次跨越三大鸿沟的简单性指标搜索

本文件夹是一个**独立**实验系列（不导入其他 rotations/math 套餐），目标见
[`readmes/rotation_research/02_hessian_simplicity_experiment.md`](../readmes/rotation_research/02_hessian_simplicity_experiment.md) §9.0：
找到一个 (导数视图 M1 × 估计器 M2 × 复杂度 M3) 组合，使其

1. **Gap 1**：在精确解析生成函数上，真实 Tstd 轴（带符号置换）是全 SO(d) 的极小；
2. **Gap 2**：只给数值 (W, Y=formula) 时仍复现；
3. **Gap 3**：Y 改为冻结 NP decoder 输出时仍复现，且搜索不读 T。

所有脚本都接受 `--model --beta --zdim --pca`（README §11.2）。输出全部写在被 git 忽略的
`outputs/three_gaps/` 下。

## 结构

```
three_gaps/
  common/analytic.py   四个生成器的 torch 解析 logit（与 S1 predictive_true 校验到 1e-16）
  common/derivs.py     torch.func 精确导数场 g, Htt, Htx, Tttx, Ttxx, Tttt
  common/rotlib.py     视图 (block, 聚合方式)、坐标变换 A = S inv(M)^T、旋转与轴误差
  common/losses.py     M3 复杂度目录（全部对符号/置换/逐轴尺度严格不变）
  common/io.py         读取 S1/S5/S3（--beta --zdim --pca），T 单独放在 Truth 对象
  expNN_*/run.py       每个实验一个入口
  EXPERIMENT_LOG.md    每轮实验的结果与决策
```

## 当前最佳方法：两阶段“输出等变轴 + 设计情境关联稀疏性”（EXP06）

对每个候选 U = R V（V = 以 TRAIN 均值中心化的冻结 Z），只使用函数 h(V, X)（解析式 / sieve
拟合的 WY / decoder）的**一阶梯度场**：

1. **阶段 A — 输出等变轴**。若某个位移方向 w 的梯度剖面 ∂h/∂w(·) 与“平移剖面”
   （logit: 1；prob: p(1−p)）或“增益剖面”（logit: η；prob: p(1−p)·logit p）共线，
   则该方向是平移轴（C 模型的 b1）或增益轴（CR3 的 τ）。用广义特征问题求最小相对残差 e1，
   **仅当 e1 < 0.3 且 e1 < 0.3·e2 时固定该轴**（阈值预先固定）。
2. **阶段 B — 情境关联稀疏性**。把梯度场投影到刺激设计的“情境”上：连续 X 用正交
   Legendre 张量基（总次数 ≤ 3）的基函数，CR3 固定 bank 用设计类别（gain-only / loss-only /
   p=0.5 / general）。单元格 = (轴, 情境)，对被试求平均绝对值，损失
   `absL1 = Σ M / ‖M‖_F`（M 为逐轴 z-score 后的单元格均值绝对梯度），在阶段 A 轴的正交补上
   用 SO(d−m) 多起点 Adam 最小化。

所有损失对逐轴尺度 D、符号、置换严格不变（G0 审计误差 ≤ 4e-6）。T 只在 R 固定后用于 TEST 评价。

```cmd
python three_gaps/exp00_oracle/run.py    --model CR3 --beta 0.00005 --zdim 4 --pca 0
python three_gaps/exp06_two_stage/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --view GCat --losses absL1,rmsL1 --tag final
python three_gaps/exp07_summary/run.py   --tag final --variant AB --loss absL1
```

## 运行（Windows CMD）

```cmd
conda activate py11
cd /d D:\Projects\mod-rot-3
git pull --ff-only
python three_gaps/exp01_gap1_analytic/run.py --model C1  --beta 0.0005  --zdim 4 --pca 0
python three_gaps/exp01_gap1_analytic/run.py --model C2  --beta 0.0005  --zdim 4 --pca 0
python three_gaps/exp01_gap1_analytic/run.py --model C3  --beta 0.0005  --zdim 6 --pca 0
python three_gaps/exp01_gap1_analytic/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0
```
