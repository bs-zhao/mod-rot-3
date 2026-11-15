# 08 · Derivative-aware decoder engineering：二阶函数结构的训练、集成与数据设计

> Mod-Rot-3 / 2026-11-14 / 研究综述与 Codex 实施路线  
> 状态：新增实验尚未执行。本文是基于已有结果的定向领域调研和可证伪实验计划，不是系统综述或成功报告。  
> 背景：[01 项目总览](01_project_overview.md)、[02 Hessian 研究](02_hessian_simplicity_experiment.md)、[03 噪声诊断](03_cr3_decoder_noise_floor_summary.md)、[04 EXP13](04_role_free_simplicity_losses.md)、[05 总路线](05_next_steps_after_exp13.md)、[06 补充实验](06_complementary_ideas_and_experiment_cards.md)、[07 积分与多尺度](07_latent_integration_and_multiscale_hessian.md)。  
> 已有实测：[post-EXP13 RESULTS](../../three_gaps/post_exp13_20261114/RESULTS.md)。

## 0. 总任务与不可修改的基线

前一波 EXP20–23 研究固定 decoder 的 Hessian 怎样积分、平滑、取有限尺度。本轮 EXP24–31 研究怎样训练 decoder，使其在预测行为的同时保留更可靠的一阶、二阶结构。两条路线互补，不能混在同一个实验中以致无法定位改进来源。

永久保留 [EXP13](../../three_gaps/exp13_pure_simplicity/) 及其[冻结标签](https://github.com/bs-zhao/mod-rot-3/tree/exp13-frozen-doc05-20261114)、[复现归档](../../three_gaps/post_exp13_20261114/baseline/README.md)。历史方法：p 输出，UX_neff + UUd_share，原 X 单位，Haar 200 seed 1 等权标准化，24 起点 seed 7，Adam 250 步 lr=.05。不得覆盖原 checkpoint、Z、缓存、代码、结果或公共依赖兼容性。新 decoder 使用新 checkpoint 与独立实验目录。本文编号若已占用则顺延并说明。所有运行前检查实际 repo 状态、manifest、数据与 checkpoint hash；大文件留在本机 outputs，代码及小型结果提交 main。

| 模型 | 冻结 p MCC rep1 / rep2 | 平均 leakage rep1 / rep2 | 正交 Oracle MCC |
|---|---|---|---|
| C1 | .9688/.9689 | .0212/.0202 | .969/.970 |
| C2 | .9645/.9677 | .0599/.0427 | .971/.971 |
| C3 | .9600/.9615 | .0464/.0478 | .963/.965 |
| CR3 | .8173/.8122 | .1646/.1832 | .872/.869 |

约束：训练与盲旋转不得输入真实 T、解析式、参数名称/角色或设计类别，不用 ICA/独立性目标，不按 TEST MCC 选网络、权重、带宽、激活函数、早停或最终旋转；不要求所有 UU 交互消失。真实 T、解析式和监督 Oracle 仅用于事后诊断；单列 Oracle-supervised 正对照，绝不能混入可用方法。四模型既有 TEST 已被反复查看，都是探索性验证，最终冻结规则后用新 S1、NP repeat 3–5、未见模型族评价。先固定 Z 再重训整个 NP；无额外真实标签的虚拟扰动不是新的行为数据。

## 1. 科学动机：预测正确、导数正确、旋转正确是三件事

ModLearn 从被试行为 (X,Y) 经 NP encoder 学习 Z_s，decoder 预测 p(Y=1|Z,X)。真实模拟有 T_s，但真实应用中没有。预测在可逆重参数化下不变；正交旋转 U=RZ 是寻找可解释参数轴的一个受限变换族。C1/C2/C3 的 logit 对 T 线性，真实 UU 二阶导数为零；CR3 的 CPT-like logit 是 tau[G(alpha,gamma,X)-lambda D(alpha,gamma,X)]，既有必要交互又有真实自曲率。CR3 的 5000 被试共享 500 个刺激，因此 Z 覆盖和 X 覆盖是不同问题。

普通 CE 只监督函数值。f_hat(x)=f(x)+eps sin(omega x) 即使 eps 很小，二阶误差可达 eps omega²。更严格地，有限 X bank 上可以构造一个光滑函数，它在所有观测 X 点取零、但在这些点附近有非零导数；把它加到 decoder 上不改变观测预测，却改变 UX。因此从有限函数值并不能无条件识别导数。Hornik 的逼近定理只说明足够光滑的网络具有同时逼近函数及其导数的能力，不说明普通 CE 的优化必然找到这种逼近。[L1,L2]

还存在三种不同误差来源：
- decoder 间随机误差：初始化、抽样和优化轨迹不同；
- 所有 decoder 共享的偏差：固定 X bank、网络结构、训练目标、posterior mean Z 的信息损失；
- 旋转目标偏差：即便真实 Hessian 完全正确，简单性 loss 的全局最小也未必是真实参数轴。UUd_share 偏好低轴向自曲率占比，并不是最小化 UU 交互；a²-b² 旋转成 2uv 是反例。

已有实测：EXP13 p 组合四模型较稳；CR3 logit UUd_share 单项 .821、组合 .831，但 logit 组合损害 C 模型。EXP14 p Hessian 的梯度外积 B-only CR3 rep1 .8057，完整 p .8173；单靠梯度不能解释全部成功。EXP17 子集可靠性加权未稳定改善；EXP18 小规模 surrogate 多数失败。这些结果不是充分的独立 decoder 工程对照。本轮不重复小规模拟合并宣称新突破。

一个重要理论区别：固定 Z 时，decoder 的统计目标通常接近 p(Y|Z,X)，未必等于 p(Y|T,X)。若 Z 含噪，条件期望对 T 的混合会改变导数。用 TRAIN 拟合线性 T_hat(V) 代入解析式，只是一个事后 plug-in 参考，不是真实条件期望的导数。若需要更严格的模拟诊断，应单列已知合成坐标实验、条件期望参考估计、以及真实 NP 的旋转恢复；不能直接把 H_TT 与 H_ZZ 相减。

## 2. 领域调研地图（现有证据 vs 本项目扩展）

以下是面向输入微分结构的定向文献检索，关键词包括 Sobolev training, Jacobian matching, input Hessian regularization, deep ensembles, mutual learning, curvature regularization, spectral bias, derivative observations, smooth function approximation。主要依据原始会议论文、作者论文页面或 DOI。未将一般预测、鲁棒性或图像生成实验的结果视为 CR3 参数恢复证据。参考文献编号在末尾。

| 类别 | 现有技术 / 文献 | 可尝试的 ModLearn 分支 | 主要限制 |
|---|---|---|---|
| A 集成与不确定性 | Bagging [L3]、Deep Ensembles [L4] | 独立 decoder；函数平均、Hessian 平均、交叉矩去噪；异构 ensemble | 共同偏差不消失；多样性与一致性冲突 |
| B 共同学习 | Deep Mutual Learning [L5] | 预测/梯度/Hessian 一致性；leave-one-out teacher；独立预训练后协同 | 原论文主要匹配类别概率；二阶一致性是新扩展 |
| C 教师与轨迹平均 | Mean Teacher [L6]、SWA [L7]、Knowledge Distillation [L8] | EMA/SWA decoder；导数 Mean Teacher；ensemble student | 权重空间平坦≠输入空间曲率准确 |
| D 导数监督与蒸馏 | Sobolev Training [L9]、Jacobian Matching [L10] | 值+梯度+Hessian 蒸馏；平滑教师；有限尺度教师 | 教师可能学错；Oracle 导数不可用于盲方法 |
| E 输入正则 | Double Backprop [L11]、CURE [L12]、Input Hessian Regularization [L13]、VAT [L14] | 一阶/二阶能量；Hessian 波动；尺度一致性；随机方向 | 鲁棒性常鼓励局部不变性，可能抹掉真实心理效应 |
| F 高效高阶计算 | Hessian Penalty [L15]、随机 trace/HVP [L16,L17] | HVP、双线性随机估计、有限差分、分块 | 不照搬 off-diagonal=0 的结构目标 |
| G 架构与频谱 | Hornik [L1,L2]、Spectral Bias [L18]、Spectral Normalization [L19]、SIREN [L20]、Fourier Features [L21] | 光滑激活、频率控制、容量、早停、残差结构 | 表达能力不等于学到正确导数；高频也可能是真信号 |
| H 数据与局部训练 | Mixup [L22]、VAT [L14]、实验设计/空间填充 [L23] | 真实连续 X、锚点+邻域、教师软标签、局部 Taylor 拟合 | 不能随意复制扰动前 choice 或假定线性插值正确 |
| I 函数空间估计 | GP derivatives [L24,L25]、Sparse GP [L26]、样条 [L27]、多项式/sieve | GP、KRR、PCE、样条、基函数+NN | kernel/基函数是先验；百万行数据需扩展算法 |
| J 评估与边界 | 不可识别性与光滑逼近 [L1,L9] | 新 S1、解析正对照、负对照、未知模型族 | 一致/平滑不是正确；有限数据不存在无条件保证 |

A–J 并非要求一次全部运行。优先用低成本实验建立误差来源，再依结果扩展。以下每个分支的具体损失、权重和预算是本文的研究提案，而非文献已证明有效的 ModLearn 方法。

## 3. 共同坐标与训练协议

阶段 I（decoder-only）：从同一原 NP checkpoint 提取全观测 posterior mean Z，冻结 encoder、V=Z-mu_train、TRAIN/VAL/TEST、刺激及 Oracle。所有 decoder 使用相同物理 [X,Z]，禁止独立编码器造成的坐标差异。若标准化网络输入，保留原坐标逆映射，并用链式法则换回原 VV/VX 导数。避免把内部标准化的导数直接与历史 Hessian 比较。

训练样本按被试划分，CEWithLogits 用真实 Y；另可训练 teacher soft-target 模型，必须标明标签来源。一个细节：全观测 Z 包含该被试的 target choice；因此该协议用于同一 frozen representation 的 decoder 工程对照，不能把这种重建 CE 称作独立预测。另设置 held-out trial/context 编码评估：用不含 target 的 context 提取 Z 并预测 target；报告性能下降或分布偏移。固定 Z 的各方法必须使用完全相同评估协议。原 NP 训练时随机后验样本 vs decoder-only 后验均值是不同训练分布，应分别对照 mean-only、posterior-sample（可获取时，固定 encoder 输出），不可把差异全归因于正则。

原 NP decoder 与同架构、同固定 Z、仅 CE 重新训练的 decoder 均为基线。先从 config 核实实际 GELU、层宽、训练预算；不能只比较新网络与原 NP 然后宣称导数正则有效。默认 CR3 beta=.00005,d=4；C1/C2 beta=.0005,d=4，C3 d=6；pca=0，rep1/2。提供可指定 beta/zdim/PCA/repeat 的接口，高维 PCA residual 现有 io 限制须明确。

固定无 T 训练选择：VAL CE/Brier 为首要约束；预先声明容许 CE 相对基线下降不超过某阈值（例如 +1% 作为探索性非劣界，报告绝对差和不确定性），再用导数稳定性与支持域选择，不能用 TEST T。对所有正则以同一固定度量计算：例如 V 用 TRAIN 协方差定义白化度量或一个全局标量尺度，X 用固定单位换算；正交不变性仅在固定正交度量下成立。保留原始 X 单位 EXP13 评价，与训练内部度量明确区分。记录系数、各项原始值与梯度范数，避免 λ 受量纲支配。初始正则强度可按独立 warmup 批次 loss/梯度量级校准，再测 0,.01,.1,1 的相对强度；固定校准规则，不看 T。

阶段 II（full NP）：只有阶段 I 的稳定方案才迁移到联合 encoder/decoder。新 encoder 可能改变 Z 的信息量与几何，必须重新报告全 Z 回归、监督 Oracle、latent covariance、预测及旋转；分别比较“Oracle 提升”和“盲旋转接近 Oracle 的比例/差距”，不能把两种贡献混在一起。

## 4. EXP24 — 独立 decoder 集成与 shared Hessian（首要）

### 4.1 设计

固定 Z 训练 B=5 个 decoder，预算允许扩展 B=10；同架构不同 init 为主，另做 subject bootstrap、不同被试子集、不同架构的异构 ensemble。每个成员必须独立训练并保存 checkpoint，保留原始 CE-only ensemble 不参与之后一致性训练。比较相同每成员训练预算和相同总预算两种设置；B 增大不能用最优 MCC 成员作为结果。一个小成本版本：同一 decoder 多个训练阶段的输出 ensemble，但高度相关，单列。

对 h_b=p_b 或 eta_b（分别计算，不能平均 logit 后 sigmoid 与平均 probability 混为一谈）：
$$
\bar h=\frac1B\sum_b h_b,\quad \bar H=\frac1B\sum_b H_b,\quad
E_{total}=\mathbb E_W\frac1B\sum_b H_b^{\circ2},\quad
E_{shared}=\mathbb E_W\bar H^{\circ2}.
$$
$$
E_{total}=E_{shared}+E_{disagreement},\quad
E_{disagreement}=\mathbb E_W\frac1B\sum_b(H_b-\bar H)^{\circ2}.
$$
这是精确恒等式，适用于任何共同线性坐标变换。误差独立同分布时平均随机误差方差降至 1/B；若 pairwise correlation rho，则等方差下为 sigma²[rho+(1-rho)/B]，共同偏差仍在。计算 ensemble size 1/2/3/5/10 的曲线及预设子集重复。

### 4.2 四种集成目标

E1：先平均 H 再平方（平均函数真 Hessian）；E2：先平方再平均；E3：每成员独立计算 Haar-normalized EXP13 loss 后平均；E4：一致性加权 loss（参照 EXP17），权重按共同 Haar landscape 的 split/decoder 相关与近零 SD 规则。Ablation：仅 disagreement 能量能否产生同样低谷？它不代表真实信号，结果仅诊断。

额外研究交叉矩：
$$
M_{cross}=\frac1{B(B-1)}\sum_{b\ne c}\mathbb E[H_b\otimes H_c].
$$
独立零均值误差且同一真实场时，该项在成员随机性期望下去掉自噪声方差；共享偏差及相关误差仍在。有限 B 的 cross moment 可不半正定、能量可负，不能直接喂给 neff；报告负特征值、与 PSD projection 或 shrinkage 的对照，主方法仍用合法非负的 shared energy。此分支为可选诊断。

### 4.3 交付

同一 W 上 p、gradient、H_VV/H_VX 的相对误差与 disagreement；共同 SX/T4、共同 Haar landscape、低 loss 区域排序、最优 R signed-permutation 距离；每成员及 aggregate U–T 图；预测指标与计算成本。先 CR3 rep1/2，再四模型。若平均 H 改善而 E2 不改善，支持符号平均消噪；若所有方法相同，可能是共同偏差或目标偏差；若 ensemble 更差，检查真实信号抵消、坐标对齐与预测拟合。

## 5. EXP25 — Deep Mutual Learning 的导数扩展

原始 DML [L5] 是 peer 网络相互匹配预测分布，不是 Hessian 恢复。本项目扩展为 3–5 个共享 Z 的 decoder，先 CE warmup，随后逐步增加一致性：
$$
L=\frac1B\sum_b CE(Y,p_b)+\lambda_0 C_0+\lambda_1 C_1+\lambda_2 C_2.
$$
C0=E_b,W ||p_b-p_teacher||² 或 Bernoulli KL；C1=E||g_b-g_teacher||²；C2=E[||H_VV,b-H_VV,t||²+c||H_VX,b-H_VX,t||²]。teacher 可为共同平均（全梯度对称方差）或 leave-one-out stopgrad（两种实现分开），同一 W、同一输出通道。梯度/曲率监督只约束一致，不强制为零。训练含真实 CE，虚拟 Z/邻域只用于 consistency，不附伪真实 choice。

分支：25A 仅 CE 独立；25B C0；25C C0+C1；25D C0+C2；25E C0+C1+C2；25F 独立 warmup 后固定/EMA peer teacher；25G 多尺度 finite response 替代精确 H；25H VV-only、VX-only、both。p 为主，logit 另做输出分解。控制相同初始化、成员数、数据、每成员更新数；额外报告总 FLOPs/wall time。

防塌缩：记录 ||H||²、||g||²、p 变化范围、各成员误差与 disagreement，保留无一致性对照；如果 disagreement 下降但全部 H 接近零或预测失去真实结构，判失败。不要为了提高一致性而惩罚真实交互。联合训练成员不再独立，不能用 1/B 噪声方差定律解释结果。可测独立 holdout decoder 的结构一致性，不能以训练时被直接惩罚的 disagreement 作为唯一胜利标准。

## 6. EXP26 — EMA/SWA/Derivative Mean Teacher

Mean Teacher [L6] 用权重 EMA 构造预测一致性 target；SWA [L7] 平均同一训练轨迹后期权重。它们在预测任务有证据，但不能推出输入二阶导数改善。不要平均不同独立初始化网络的权重（神经元排列等价造成问题），EMA/SWA 在同一轨迹中比较。

$$
\bar\theta_{t+1}=\rho\bar\theta_t+(1-\rho)\theta_{t+1}.
$$
rho=.99,.999（记录更新频率与有效时间窗口）；SWA 从预设训练进度如 75% 开始，每固定步数保存权重，并用一致的 LR schedule。分支：原 CE checkpoint；EMA evaluation-only；SWA evaluation-only；checkpoint prediction ensemble；EMA teacher 的 C0、C0+C1、C0+C2、全部。Teacher 对参数 stopgrad，但对输入必须允许求导（不能在整个 teacher forward 使用 no_grad 以致 H 消失）。对小 batch 预先计算并 detach teacher 导数；权重/Buffer 同步，eval mode，BN 若存在需正确统计。

比较固定训练预算以及 VAL CE 匹配 checkpoint。记录训练 epoch→预测、Hessian disagreement、moment、旋转 loss、MCC（MCC 仅事后）；研究是否存在预测继续改善而结构退化的阶段。结合 spectral bias 假说但不预设“早停一定好”。

## 7. EXP27 — 导数正则、高阶平滑与数据尺度一致性

### 7.1 正则家族

CURE [L12] 等主要控制输入 loss 曲率及鲁棒性；本项目关心输出 eta/p 的曲率，必须区分 CE Hessian、参数 Hessian、输出 Hessian。Double Backprop [L11]、VAT [L14]、输入 Hessian 正则 [L13] 提供工程先例。

| 分支 | 正则 | 含义 |
|---|---|---|
| 27A | E||g||² | 限制一阶敏感度；可能抹掉真实效应 |
| 27B | E||H_VV||F²、E||H_VX||F² | 总曲率能量；保留坐标旋转不变度量，但压缩真实曲率 |
| 27C | E||H(W+delta)-H(W)||F² | 惩罚局部变化，而不是恒定非零 Hessian |
| 27D | E||∇H||F² | 三阶 Sobolev 半范数；小尺度 27C 极限 |
| 27E | E||H_AD-H_smooth||² | 微观与固定数据尺度的曲率一致 |
| 27F | E[D²_h1 f-D²_h2 f]² | 仅前向的随机方向多尺度响应一致 |
| 27G | 局部 Taylor residual | 惩罚二次展开无法解释的局部残差 |
| 27H | VAT/局部输出 consistency | 弱平滑 baseline，不能把 X 真实效应当应不变 |

在固定 X、各向同性小 delta 下，E||H(V+delta)-H(V)||²/sigma² → ||∇_V H||²（足够光滑、可积）。对 V 二次函数 H 恒定，该正则不惩罚真实非零 H；若对 X 也扰动，abx 的 H_VV 随 X 变，不能宣称不惩罚。27C–G 可能保留真实二阶结构但也会抑制真实三阶变化；CR3 必须实测。

随机方向二阶差分 D²_h f=[f(W+hv)-2f(W)+f(W-hv)]/h²；比较 h1=.05/.1 与 h2=.25/.5/1 个参考 SD。步长是半宽。使用 doc07 的实际 X 间距、kNN 和支持域规则；先 V-only，再真实合法 X chord 或连续支持充分的 X。对有限差分尺度一致性，使用同一个中心与方向，单位一致，避免 float32 小步长消减。Taylor residual 用 f(W+δ)-f(W)-g^Tδ-δ^THδ/2，可用正负对称点及随机方向；拟合二次近似不是强制 f 全局二次。

训练目标 CE+λ R；每个分支独立 λ 网格和 warmup，同样预设校准规则。保留 0 正则；报告 VAL CE、H 总能量、三阶变化、函数值差、支持与旋转。若出现 H 稳定但 near-zero 且 MCC 下降，说明正则过强或结构先验不合适。不得直接将 EXP13 simplicity loss 加入 decoder 训练并宣称独立发现（可能制造偏好轴）；若研究 joint rotation learning，必须另设实验、明确循环风险。

## 8. EXP28 — Sobolev distillation / ensemble + smooth teacher

Sobolev Training [L9] 利用已有目标导数提高函数与导数逼近；Jacobian Matching [L10] 研究导数匹配与输入扰动蒸馏关系。它们不从无标签导数中凭空创造真值。本项目三条证据线：

28A Oracle 正对照：在已知合成坐标中，真实函数值/梯度/Hessian 作为标签。比较 value-only、value+g、value+g+H；用相同采样及架构判断导数监督技术潜力和 loss 上限。真实 NP 的 T 与 Z 不等价，若用 T_hat(V) plug-in 必须标为近似参考；不可进入最终方法。
28B teacher：用 EXP24 的独立 ensemble 平均 p 或 eta，训练 Student 以真实 Y CE 加 soft value、g、H 匹配：
$$
L=CE(Y,p_s)+\lambda_0||p_s-\bar p||²+\lambda_1||g_s-\bar g||²+\lambda_2||H_s-\bar H||².
$$
分别比较单 teacher、ensemble teacher、value-only、g-only 补充、H-only 补充、全部、随机投影版本。若采用平均 eta 作为导数目标，不能把平均 p 的导数混入同一项。
28C smooth teacher：固定原 V 核，f_smooth=E_delta[bar f(V+delta,X)]，依 doc07 平滑 sigma=.05,.1,.25,.5（0 为无平滑）；平均 teacher 函数/导数再蒸馏。内层积分和外层中心固定，避免 TEST。另比较 finite-response distillation，用教师真实四角输出构造多尺度目标；可使用 QMC 虚拟 Z，但只用教师软标签，不是真实行为。

必须比较 Student 与直接 ensemble、直接 smooth teacher；若 Student 不更准确但同等结构且更省查询，仍可记为压缩收益，而非恢复突破。教师输出/导数按 checkpoint hash 缓存并 detach；小批次先核验精确 H 与 HVP。教师 uncertainty 可用 inverse disagreement 权重（设上限、方差下限；同时 report 低置信区覆盖）；不可无条件忽略困难区域。训练中保留真实 CE 防止过度跟随教师。最终同一 EXP13 评价，不以拟合教师 Hessian 好作为科学成功。

## 9. EXP29 — 光滑架构、频谱和训练轨迹

Hornik [L1,L2] 的存在性定理说明光滑网络可表达导数；Rahaman [L18] 指出频率依赖学习速度；SIREN [L20] 和 Fourier features [L21] 在连续信号的细节表示中有优势，但高频表达会放大二阶误差。目标不是一律低频，而是与真实行为分辨率匹配。

| 子分支 | 候选 | 控制/解释 |
|---|---|---|
| 29A activation | GELU、SiLU、tanh、softplus beta=.5/1/2/5 | 同层宽深度、相近参数预算；记录饱和、激活曲率；PyTorch softplus threshold 分支需审计 |
| 29B negative control | ReLU | 普通仿射-ReLU logit 几乎处处 H=0；p Hessian 可由 sigmoid 外积非零，不能误判真实 logit 曲率 |
| 29C capacity | depth 2/3/4、width 64/128/256 或实际 baseline 的比例 | 等参数量与等算力比较；欠拟合与过拟合都可能 |
| 29D weight control | weight decay、spectral norm [L19]、weight norm、gradient clipping | 一阶 Lipschitz 控制不保证二阶准确；报告真实输入导数 |
| 29E frequency | 低带宽 Fourier features、SIREN 频率与初始化、progressive bandwidth | 从低频到高频逐步学习，限制无数据支持的频率；SIREN 不是默认去噪 |
| 29F trajectory | CE early stopping、EMA/SWA、epoch ensemble | 同一轨迹看预测与微分结构随时间变化，VAL 选停止 |
| 29G structure | smooth residual MLP、低阶 basis+NN residual | 通用 base 学低频，残差保留复杂非线性；不预设 CPT |

Softplus_beta''(x)=beta sigmoid(beta*x)(1-sigmoid(beta*x))；beta 较大曲率更集中，较小较平滑。这里 beta 是激活参数，不是 NP KL beta。对 spectral norm，区分训练时约束的层谱范数与输入 Hessian，不能声称其必然降低所有曲率误差。SIREN/Fourier 高频候选只在能改善真实预测与结构的条件下保留。

## 10. EXP30 — 数据设计与真实局部覆盖（长期高优先级）

这是从根本上增加真实信息的路线。当前 5000×500 观测并不等于 2.5M 个不同 X，固定 bank 的 X 导数缺乏连续约束。保持真实 T 分布、总被试/试次数、合法刺激约束、NP 架构、训练预算，重新生成 S1：
- A 全共享固定 bank（历史对照）。
- B 每人独立连续 X（合法生成域内）。
- C 50% 共享/50% 独立，另 25/75% 作预设对照。
- D 共享锚点+连续局部邻域，固定半宽/空间填充设计；保证跨被试可比与局部方向覆盖。
- E 可选：跨被试分块刺激 bank/随机 Latin hypercube 或 Sobol 设计，比较覆盖而非只增加不同点数。

本分支必须区分重新生成真实模拟行为与工程数据增强。生成标签使用 S1 真函数属于模拟环境设计，不允许训练网络读 T 或解析导数。真实数据中不能把 X 扰动后原 Y 直接复制，也不能假定 mixup 软标签是真实概率。Mixup [L22] 是线性插值先验，VAT [L14] 是局部不变性先验，两者可作为有限正则对照但可能破坏真实决策效应。更合理的 teacher augmentation 用独立/ensemble 教师产生软 p，仍不是新证据；只用观测数据时局部回归/近邻拟合应按实际方向与误差报告。

评估：各设计的唯一 X 数、kNN 距离、局部方向秩/条件数、Z-X 联合覆盖、合法性、held-out 新刺激 CE、导数/有限响应误差、Oracle、MCC/泄漏。分开测试固定训练 Z 与 full NP 的影响（不同数据集无法直接比较 R；用 T 匹配或合成共同坐标）。选择不同 X bank 做训练/验证/测试，另保留共享锚点；不把在同一个 X bank 上 held-out subjects 当作新刺激泛化。若 B/D 显著改善 UX 导数和旋转，则有证据表明实验刺激设计影响函数结构识别；若只改善 Oracle，说明表示信息增加而非 decoder Hessian 单独改善。

## 11. EXP31 — 非 MLP 与混合函数空间

EXP18 的小规模 surrogate 已有负结果。本轮扩大实际 TRAIN 预算、直接拟合固定 Z 的真实 choice 或 teacher soft p，使用显式平滑与 held-out 选择，不再只重复 64×64。

31A GP：RBF、Matérn nu=5/2（允许二阶均方导数；更低 nu 不满足同样要求）、可选复合 kernel；比较核长度尺度、导数后验不确定性。GP derivative observations [L24,L25] 可联合教师导数；不需要真实导数时从值拟合再求后验均值导数。对二元 choice 用 Bernoulli likelihood 的近似推断/variational GP；对 soft probability/eta 可回归但需明确输出范围与变换。百万行不能用 exact GP 全矩阵，先 inducing 256/512/1024/2048，按预算使用 sparse GP [L26]，报告近似误差与耗时。GP posterior mean 的 Hessian 不等于随机函数 Hessian 能量的期望；后者包含 posterior variance，单列。

31B KRR：RBF/Matérn，Nyström/随机特征近似；核梯度/Hessian 解析可得。长度尺度由 VAL 预测与无 T 稳定性选。31C polynomial/sieve/PCE：总阶数 2–5、ridge、正交基；对于 Gaussian V 可用 Hermite 基、连续 X 用 Legendre 或适当标准化基；full interaction 候选避免先验禁止真实交互，高维组合数显式计算。31D spline：cubic/quintic tensor 或低秩平滑基、thin-plate spline；惩罚二阶/三阶粗糙度，确保所需导数连续；纯 additive GAM 只能作为结构受限负/对照，不能预设真实参数作用可加。31E hybrid：
$$
eta(V,X)=eta_{base}(V,X)+eta_{residual,NN}(V,X).
$$
基函数负责低复杂度结构，NN 拟合残差，正则残差 Hessian 波动而非全部真曲率。比较 base-only、NN-only、joint、先 base 后 residual、teacher distill。用 logit 相加后 sigmoid，不能随意相加 probability 产生超范围输出。

所有 surrogate 的坐标与尺度换回原 V,X，再用同一个 EXP13 evaluator。依据 TRAIN/VAL 选择复杂度，测试真实行为预测、函数值、导数、旋转成本。严格 WY-only 分支只有矩阵输入输出，不查询原 decoder；teacher 生成额外节点/导数是 decoder-query/supervised distillation，不能宣称跨过 Gap2/3 的矩阵限制。

## 12. 高阶训练的计算技巧与可靠性

训练 H loss 对参数反传需要输入二阶与参数一阶的混合导数，计算比单纯评价 H 昂贵。d=4,p=6 时完整块本身很小，先以精确小 batch 实现，再优化：
- HVP：对随机 v、E[vv^T]=I，E||Hv||²=||H||F²；矩形 H_VX 同样适用。用 torch.func jvp/vjp 或 autograd grad(create_graph=True)；教师 target detach，student 不能 detach。
- 随机双线性 u^T H v，独立零均值单位协方差 u,v，E[(u^T H v)²]=||H||F²；可分别对 VV、VX 使用，少数方向 1/2/4/8 并记录估计方差。
- Gaussian 二次型 E[(v^T H v)²]=2||H||F²+tr(H)²（对称 H）；不能当作纯 Frobenius。Hessian Penalty [L15] 的 Rademacher 方差针对 offdiag，其高效技巧可借鉴，目标不能直接照搬。
- finite HVP: [g(W+hv)-g(W-hv)]/(2h)；随机双线性四角 difference 只需要 forward，适用于多尺度一致性；精确与有限尺度是不同目标。
- Hutchinson [L16] 与 Pearlmutter [L17] 是计算基础；记录随机方向分布、种子及单位度量。先对解析二次 toy 验证估计无偏/尺度；对 Student 正则验证参数梯度非零、与直接完整矩阵一致。
- 稀疏更新：每 K=1/4/8/16 个 CE step 做一次 derivative batch（预先固定权重与有效频率），小 batch 16/32/64、梯度累积；同样记录总 updates、查询、峰值显存及训练时长。不能用便宜近似的高一致性冒充精确 Hessian 的准确性。
- float32 为历史基线，必要时关闭 mixed precision 的高阶部分或用 float64 小批数值核对；小 h 差分有消减误差。确保 dropout/eval 语义，不能把随机 dropout mask 产生的差异误当导数噪声。

## 13. 通用验收与 Codex 执行顺序

建议独立目录 three_gaps/exp24_decoder_ensemble、exp25_mutual_derivatives、exp26_ema_teacher、exp27_curvature_regularization、exp28_sobolev_distill、exp29_smooth_architecture、exp30_stimulus_design、exp31_function_approximators。每套独立 README/run.py/train.py/evaluate.py/tests/results，公共只复用数据 IO 和冻结 evaluator。实际脚本写好后再给真实 CMD，不把建议接口当已经实现。

本机 py11，读取实际 checkout，git pull --ff-only；GPU 训练顺序执行，checkpoint/large cache 输出本机，日志和小型表图提交 GitHub main。建议 CLI: --model --beta --zdim --pca --repeat --seed --tag --stage --device --batch-size --epochs --member-count --reg-type --lambda --channel --h-u --h-x --sigma --teacher --resume；配置 hash、原子 COMPLETE 标记、失败 traceback。新训练 checkpoint 不覆盖 NP 原 checkpoint。

| 顺序 | 批次 | 最低交付 |
|---|---|---|
| 0 | 冻结历史 + CE-only decoder | 复现表、共同 Z、checkpoint manifest、原 NP vs refit 对照 |
| 1 | EXP24 独立 ensemble B=5 | 两 CR3 repeat；shared/total/disagreement、三种 loss、B 曲线 |
| 2 | EXP27C/F + 27A/B 对照 | H 波动/多尺度 vs 总曲率；相同预测预算及精确小批测试 |
| 3 | EXP25 0/1/2阶一致性 | 独立预训练、消融、塌缩审计、固定 Z |
| 4 | EXP26 EMA/SWA/teacher | 低成本轨迹比较、同验证预测水平 |
| 5 | EXP28 distillation | 直接 ensemble/平滑 teacher vs student；oracle 正对照隔离 |
| 6 | EXP29/31 | 光滑激活、容量、核/基函数/残差模型的公平比较 |
| 7 | EXP30 | 新 S1 数据设计，固定/连续/锚点对照及 full NP |
| 8 | 联合 doc07 与四模型确认 | 先固定旧 EXP13 300 测度评价，再比较 Gaussian/平滑；避免多因素混淆 |

每阶段先 CR3 rep1/2，p 为主，logit 为机制对照；有希望的规则冻结后 C1/C2/C3；最终新数据及未见公式。优先 EXP24 和 EXP27；EXP25/26 可以并行设计但实际 GPU 顺序。完整方案不用一次运行无穷网格；按预设 VAL/稳定性与预算筛选，保留所有负结果。

四类评估缺一不可：
1. Prediction: TRAIN/VAL/TEST CE、Brier、accuracy、校准、p 对真实生成概率的误差（仅模拟事后）、新 X 与 context-held-out；不能只报 accuracy。
2. Derivatives: g、VV/VX 分块相对误差、cosine、能量、跨成员 disagreement、不同步长、尾部支持、p Hessian 的 A+B 分解与交叉项；真实参考坐标说明。若真实 H=0，不使用除以真范数的相对误差，改绝对误差及预设尺度。
3. Rotation: 同一个 EXP13 cache/evaluator，固定物理积分点和 C_ref；Haar 200、24×250、共同 seed；SX/T4、完整 plane curves、共同 Haar landscape、低谷 signed-permutation distance、全部起点终点。仅最低 loss R 用于结果，真实 T 不选解。新 decoder 与历史 checkpoint 都在旧 300 及 doc07 的全 TRAIN/收敛 P1 测度比较，先单因素再联合。
4. Science: 原始 U–T 相关矩阵、有符号匹配热图、MCC、worst axis、平均与最大 leakage、监督 Oracle 差距；固定 Z 的 Oracle 必须不变。重新训练 NP 时重新计算，区分 Z 信息改善和旋转改善。

阶段目标 CR3 两重复稳定超过 .817/.812，争取 .85+ 且 leakage≤.12，C1–C3 不劣于基线 .01 以上且泄漏不明显恶化；仅为建议，不是普适定理。保留原始预测与计算资源约束。若无方法提升，负结论仍有价值：一致性下降但真导数无改善→共同偏差；Hessian 真值改善但 R 不改善→loss/可识别性；固定 X 失败而新 X 成功→数据支持；Student 不如直接 ensemble→蒸馏未保留结构；只有调 TEST 超参有效→未验证。最终不能将 decoder-query 成功写成严格 WY-only 成功。

输出 config.json（数据/模型 hash、训练/积分/旋转配置、版本）、train_history.csv、checkpoint_manifest.json、prediction_metrics.csv、derivative_metrics.csv、moments.npz、rotations.npz、plane_curves.csv、random_landscapes.npz、UT_heatmaps.png、runtime.csv、RESULTS.md、logs、COMPLETE.json。训练和搜索路径禁止加载 T；评价单独加载。测试：共同 Z、数据 split、teacher stopgrad 但输入导数可算、HVP 精确性、0 正则复现、signed 平均恒等式、p/eta 区别、sigma=0、随机种子、历史未覆盖。保存所有实际结果而非虚构。

## 14. 参考文献与进一步阅读

以下原始文献提供理论或工程方法的依据；每条均说明与本项目的边界。本轮阅读范围以原始摘要及公开论文页面为主，非逐篇全文系统审查。

- [L1] Hornik, K. (1991). [Approximation capabilities of multilayer feedforward networks](https://doi.org/10.1016/0893-6080%2891%2990009-T). Neural Networks 4, 251–257. 光滑网络近似能力；不保证 CE 训练导数收敛。
- [L2] Hornik, Stinchcombe & White (1990). [Universal approximation of an unknown mapping and its derivatives using multilayer feedforward networks](https://doi.org/10.1016/0893-6080%2890%2990005-6). Neural Networks 3, 551–560. 函数和导数同时逼近的理论基础。
- [L3] Breiman (1996). [Bagging predictors](https://doi.org/10.1007/BF00058655). Machine Learning 24, 123–140. 重采样集成；本项目按被试而非试次独立行 bootstrap。
- [L4] Lakshminarayanan, Pritzel & Blundell (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). NeurIPS. 独立网络预测不确定性；扩展到共同 Hessian。
- [L5] Zhang, Xiang, Hospedales & Lu (2018). [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html). CVPR. 多学生预测协同；二阶 peer loss 是本文新提案。
- [L6] Tarvainen & Valpola (2017). [Mean Teachers are Better Role Models](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html). NeurIPS. EMA teacher；输入导数版本需验证。
- [L7] Izmailov et al. (2018). [Averaging Weights Leads to Wider Optima and Better Generalization](https://proceedings.mlr.press/r16/izmailov18a.html). UAI. SWA；权重空间平坦性不能等同输入曲率。
- [L8] Hinton, Vinyals & Dean (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). 软标签蒸馏基础；教师不是真实生成函数。
- [L9] Czarnecki et al. (2017). [Sobolev Training for Neural Networks](https://papers.nips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html). NeurIPS. 显式目标导数；最直接相关，真实应用需构造 teacher target。
- [L10] Srinivas & Fleuret (2018). [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html). ICML. 一阶匹配与噪声蒸馏联系；不等于已证实 Hessian 恢复。
- [L11] Drucker & Le Cun (1992). [Improving generalization performance using double backpropagation](https://doi.org/10.1109/72.165600). IEEE TNN. 输入梯度正则的早期工作。
- [L12] Moosavi-Dezfooli et al. (2019). [Robustness via Curvature Regularization, and Vice Versa](https://openaccess.thecvf.com/content_CVPR_2019/html/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper.html). CVPR. 输入 loss 曲率与鲁棒性，非认知输出 H 精度。
- [L13] Mustafa, Vandermeulen & Kloft (2020). [Input Hessian Regularization of Neural Networks](https://arxiv.org/abs/2009.06571). 预印本版本。输入 Hessian operator norm 高效正则；目标与本项目不同。
- [L14] Miyato et al. (2019). [Virtual Adversarial Training](https://doi.org/10.1109/TPAMI.2018.2858821). IEEE TPAMI. 局部预测平滑；可能压制真实刺激效应。
- [L15] Peebles et al. (2020). [The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/7231_ECCV_2020_paper.php). ECCV. 随机有限差分估计 offdiag penalty；借计算技巧，不照搬无交互假设。
- [L16] Hutchinson (1989). [A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines](https://doi.org/10.1080/03610918908812806). 随机 trace 估计基础。
- [L17] Pearlmutter (1994). [Fast exact multiplication by the Hessian](https://doi.org/10.1162/neco.1994.6.1.147). Neural Computation. 高效 HVP；本项目针对输入 H，而非训练参数 Hessian。
- [L18] Rahaman et al. (2019). [On the Spectral Bias of Neural Networks](https://proceedings.mlr.press/v97/rahaman19a.html). ICML. 频率依赖学习速度，支持检查训练轨迹而非证明早停必然有效。
- [L19] Miyato et al. (2018). [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957). ICLR. 层谱范数控制；不能保证二阶精度。
- [L20] Sitzmann et al. (2020). [Implicit Neural Representations with Periodic Activation Functions](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html). NeurIPS. SIREN 对连续信号与导数表示，需防不必要高频。
- [L21] Tancik et al. (2020). [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739). NeurIPS. 带宽与频谱表达；本项目比较数据尺度匹配。
- [L22] Zhang, Cisse, Dauphin & Lopez-Paz (2018). [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). ICLR. 线性插值正则；不是新刺激的真实标签。
- [L23] McKay, Beckman & Conover (1979). [A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code](https://doi.org/10.1080/00401706.1979.10489755). Technometrics. Latin hypercube 空间填充；用于合法刺激设计。
- [L24] Solak et al. (2002). [Derivative Observations in Gaussian Process Models of Dynamic Systems](https://papers.nips.cc/paper_files/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html). NeurIPS 15（会议2002，出版记录亦见2003）。函数与导数联合建模。
- [L25] Rasmussen & Williams (2006). [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/). MIT Press. 核、导数、回归/分类与不确定性。
- [L26] Titsias (2009). [Variational Learning of Inducing Variables in Sparse Gaussian Processes](https://proceedings.mlr.press/v5/titsias09a.html). AISTATS. 大数据稀疏 GP 基础。
- [L27] Wood (2003). [Thin plate regression splines](https://doi.org/10.1111/1467-9868.00374). JRSS B. 平滑基与粗糙度控制；需保留必要交互。

**总结假说**：通过真实数据覆盖、多个独立 decoder 的共同结构、导数感知蒸馏与数据尺度高阶平滑，可以改善冻结 NP decoder 的有效函数结构；但预测、平滑、一致性和真实参数恢复之间没有自动等价关系。每个实验都必须以同一冻结 EXP13 进行独立、可解释的对照，并保留历史最佳方法。
