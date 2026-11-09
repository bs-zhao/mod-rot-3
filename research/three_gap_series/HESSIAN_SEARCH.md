# 统一 Hessian 简单性指标：重新限定范围后的连续搜索

2026-11-09。用户纠正后，之前 transport/utility 模型结构方案仅保留为范围外对照，
不再作为本任务成功。研究未完成。

## 严格边界

- 同一 candidate ID 的 M1/M3 公式、归一化、阈值、k、组合权重在四模型完全相同。
- loss 仅由 Hessian 场及其三/四阶变化构造；不加入梯度身份、参数反演、ICA 或模型专属 flow。
- 模型差异仅限生成/decoder 的数据源、原始 X 维数、zdim 与明确的可微域。
- M2 只拿数值 WY、TRAIN/VALIDATION 划分和数值超参数；不拿模型名称、生成表达式、T 标签、真导数。
- G1 用解析字段；G2 用 formula WY，和同坐标/同评价点的解析字段比较；G3 换冻结 decoder WY。
- G1 失败：记录 FAILED_G1，换指标，不迁移 decoder。G1 通过后 G2 失败：改 inference；
  仍失败则记录 FAILED_G2 并回 G1。G1/G2 成功而 G3 失败：单独记录 FAILED_G3。
- 同一个指标四模型 G1 均通过才进入 G2，四模型 G2 均通过才进入 G3。
- logit、native probability、Bernoulli 条件均值分别记录；link 变换必须单独标记。
- 不能以二维切片代替全维多起点，不以预测 RMSE 代替 Hessian 或 loss 方向准确性。
- 原 TEST 已用于历史探索；最终方案若出现，再冻结并生成新的独立确认 cohort。
- 所有代码、日志、结果只在 `research/three_gap_series/`，代码先 push，D 盘 pull 后用 py11 跑。

## 指标从 history 得到的实验轴

`strict_hessian.py` 是不读取模型名称的注册表及 M1/M3。
区别 UU diag/offdiag 与 UX；行/列 top-k tails；L1/L2²、soft-L0、logsum、capped、lp；
signed mean/RMS/SD/local-first；通用 X 多项式投影的 Hessian 指纹；Hessian 的三阶变化。
通用投影字典对每个模型都枚举同阶全部单项式，不使用 option 差分、货币参考或参数式。
微弱效应默认不逐列单位归一化，另行记录固定块尺度的作用。

Q 的对照：`--uu-full-domain` 在同一公式下对 UU 使用全部原始 TRAIN 刺激，
UX 仍只在 X 可微域计算。零金额时 UU 可以存在而金额 UX 不存在，不能因此丢弃全部 UU 信息。
这只是导数存在域的区分，不使用 gain-only / gamma-neutral 的模型条件分组。
该 Q 对照目前先用于解析 G1，若指标通过，再实现相同域上的 WY inference。

每个 candidate 一次 `run.py` 调用，显式 `--model --beta --zdim --pca`。
保留随机旋转、全部平面、实际 G1 counterexample 和全维优化状态。
G1 先在压力模型 CR3 筛选，失败即可否决“通用”候选，其他模型记未运行而非通过。
单平面发现严格更低 loss 足以证明 G1 失败；平面未发现反例则必须继续全维多起点。
近轴但严格更低的反例仍保留，不为得到通过而放宽判据；必要时扩大 Q、换 seed 检查采样效应。

## 文献调研与 inference 路线

以下是方法来源，不是本项目恢复理论轴的保证：

1. Ruppert & Wand (1994), *Multivariate Locally Weighted Least Squares Regression*,
   [原文 DOI](https://doi.org/10.1214/aos/1176325632)：局部加权多项式，可估计局部导数。
   实验 E03 将比较带宽、邻居数、二/三次阶数与局部条件数；高维邻域稀疏是主要风险。
2. Solak et al. (2002), *Derivative observations in Gaussian Process models of dynamic systems*,
   [NeurIPS 原文](https://papers.nips.cc/paper_files/paper/2002/file/5b8e4fd39d9786228649a8a8bec4e008-Paper.pdf)：
   可微核的函数/导数联合建模背景。没有导数标签时，只由 WY 拟合核函数并微分，不能假装观察了真导数。
3. Zhu (2021), *Hessian Estimation via Stein's Identity in Black-Box Problems*,
   [原文](https://arxiv.org/abs/2104.01317)：随机零阶查询估计 Hessian。
   必须区分 QUERY_ACCESS、平滑目标与原始固定 WY；平均 Hessian 不能直接替代 RMS/quantile。
4. Peebles et al. (2020), *The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement*,
   [ECCV 原文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510579.pdf)：
   非对角 Hessian 惩罚及随机估计，仅作为候选，交互可变成自身非线性的反例仍需 G1 排除。
5. Baydin et al. (2018), *Automatic Differentiation in Machine Learning: a Survey*,
   [JMLR 原文](https://www.jmlr.org/papers/v18/17-468.html)：对拟合代理求导，不等于原真实函数导数。

拟比较的通用 M2：E01 完整多项式、E03 局部多项式、E06 RBF/Nyström、E07 GP、
E08 正交谱、E10 适用时直接平均泛函、E12 独立平滑 NN。binary 使用 Bernoulli 似然。
超参数只能用 TRAIN/VALIDATION、模拟控制的数值误差与预先固定规则选择，不能用 decoder TEST T。
E03 已实现 `local_wy.py`：完整局部二/三次多项式、同一欧氏邻域、multinomial 权重 ridge、
二元响应局部 Bernoulli IRLS。局部三次 Hessian/三阶导数和隐藏旋转等变性有独立控制测试。
这仅说明数值实现可用，不表示有候选已经通过 G2；严格按上面的先后门槛运行真实实验。
