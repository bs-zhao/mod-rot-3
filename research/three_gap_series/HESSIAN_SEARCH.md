# 统一 Hessian 简单性指标：重新限定范围后的连续搜索

2026-11-09。用户纠正后，之前 transport/utility 模型结构方案仅保留为范围外对照，
不再作为本任务成功。研究未完成。

## 严格边界

- 同一 candidate ID 的 M1/M3 公式、归一化、阈值、k、组合权重在四模型完全相同。
- 用户明确允许几个 loss 相加；允许固定组合，禁止不同数据库各自选择分量或权重。
- 用户进一步限定：先观察单项互补才组合，禁止无依据的权重扫参。此前 h01–h03 的组合探索
  保留为历史，不作为确认性选择依据。新组合必须引用单项运行、说明互补轴/平面及共同权重规则，
  用 `--combination-rationale` 读取本目录内的证据 JSON，否则运行器拒绝。
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
用户随后明确允许解析式旋转误差≤5°，故 `gap1_angle_pass` 为当前 G1 判据：
所有平面最优轴误差≤5°、所有平面非平坦、全维多起点的最差轴误差≤5°。
`gap1_strict_pass` 继续保留为更严格的历史诊断。2.5° 处低于恒等轴不再自动淘汰。
旧 h01/h02 的 exact-zero 预筛结果不能直接视作新规则下失败或成功，需新轮重跑全维搜索。
有争议的近轴结果扩大 Q、换 seed 确认，误差按符号/排列匹配后的最差轴计，不只看某平面。

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

## h04 单项互补证据与 h05 组合假设

同一 Q、seed、24 起点下，`hs_jd_profile1_strength0_selfl1_w0` 在 C1–C3 通过，
CR3 六平面最小角分别 17.5、0、15、72.5、55、20 度，只有 alpha–gamma 正确。
`hs_uu_self_local_l1` 在 C1–C3 恒零；CR3 除 alpha–gamma（42.5°）外五个平面均 0°。
这是实际单项失败模式的互补，不是假设任意两项相加都更好。

从四模型单项 plane 曲线算出同一个权重区间：使真轴优于所有偏离>5°网格点的
充分区间约 [6.2952,9.7272]；只要求最小点在5°内时可行区间约 [4.9545,15.3109]。
取较强充分区间的几何中心并取整，得到 **所有模型固定 weight=8**。
这只利用解析 G1，不读 decoder TEST T。h03 更早的权重探索不算确认。
将这个具体假设提交为 `hs_jd_profile1_strength0_selfl1_w8`，先用新 seed、更大 Q、64 起点
确认四模型 G1，再测试 generic TensorWY 的 G2，不跳关。

用户随后明确“5.8°也行啦，先宽松一些”。当前初筛以 **6°** 为验收容差，原5°诊断保留。
h05 的全维最大角度为 C1≈0、C2≈0.038、C3≈0.040、CR3≈5.796°，故可进入 G2。
不覆盖原 summary 的5°失败记录，台账按新用户规则重新判定并注明。
此前准备的 weight=10 跟进暂缓，不为0.8°继续调权重；**冻结 weight=8 做 inference**。

`tensor_wy.py` 是通用 E02 inference：全部≤2次原始 X 单项式 + 同一各向同性 Gaussian RBF
字典，以及完整 latent polynomial。模型名称、option 差分、金额参考、真参数式均不进入拟合器。
是否共享 X bank 仅从数值输入检测，以优化同一回归式的计算。它不是先前的 utility/profile 方法。
G2 在相同 W、相同 M1/M3、相同 UU/UX 域比较：最差轴≤5°、所有平面最小相关≥0.90、
loss 曲线相对 L2 误差≤0.10；局部字段误差另行完整报告，不能拿预测误差替代。

## h07：先记录 inference 失败，再改变 inference

固定 weight=8 的首个 generic TensorWY（64 RBF、length=1、latent degree=3）在 C2/C3
复现解析方向，但 CR3 轴误差约10.435°；loss 曲线相对误差0.03515、最差平面相关0.92190，
仍不能算 G2 通过。与解析最优 R 本身的最大轴差约8.26°，不是只因解析本身偏5.8°而误判。
CR3 prediction validation RMSE≈0.10764；整体 Hessian 相对误差≈0.816，需拆 UU/UX/XX。
不进入 decoder，不换 loss；下一轮只增加通用 RBF 分辨率与调整长度尺度，去重 TRAIN 中心。

补充 inference 文献：

- Larsson & Schaback, *Scaling of Radial Basis Functions*,
  [原文](https://arxiv.org/abs/2210.05617)：尺度选择影响插值及其误差，不能把核宽度当作无关实现参数。
- *Extending error bounds for radial basis function interpolation to measuring the error in higher order Sobolev norms*,
  [原文](https://arxiv.org/abs/2302.07306)：函数值与高阶导数的误差度量不同。
- Rasmussen & Williams, *Gaussian Processes for Machine Learning*,
  [第9章原文](https://www.gaussianprocess.org/gpml/chapters/RW9.pdf)：可微核与导数建模的背景。

这些文献为数值方法和误差分析提供依据，不证明本项目某一 loss 的参数语义可恢复。
