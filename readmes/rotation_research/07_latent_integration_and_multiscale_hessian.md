# 07 · 大样本、Gaussian latent 积分与数据尺度 Hessian：下一波实验执行手册

> Mod-Rot-3 / 2026-11-14 / 给 Codex 的实施说明  
> 状态：计划，尚未执行本文新增实验。  
> 前置阅读：[01 项目总览](01_project_overview.md)、[02 Hessian 实验](02_hessian_simplicity_experiment.md)、[03 decoder 噪声诊断](03_cr3_decoder_noise_floor_summary.md)、[04 role-free loss](04_role_free_simplicity_losses.md)、[05 总路线](05_next_steps_after_exp13.md)、[06 补充实验卡](06_complementary_ideas_and_experiment_cards.md)。  
> 已执行的后续实验：[post-EXP13 RESULTS](../../three_gaps/post_exp13_20261114/RESULTS.md)、[EXP15 有限响应](../../three_gaps/post_exp13_20261114/exp15_finite_scale/README.md)。

## 0. 给 Codex 的最高优先级要求

本轮不是再发明大量新 loss，而是**固定已成功的 EXP13 简单性损失，改变 Hessian 的积分测度、样本数量、尺度和估计方法**，回答它为何有效以及能否更稳健地恢复 CR3 参数轴。

必须冻结历史最佳方法：probability 输出、UX_neff + UUd_share、Haar 标准化等权、原始 X 单位、200 Haar seed=1、24 起点 seed=7、Adam 250 steps、lr=.05。原始代码 [EXP13](../../three_gaps/exp13_pure_simplicity/) 与历史提交 a515a45 不得修改；已有 [冻结标签](https://github.com/bs-zhao/mod-rot-3/tree/exp13-frozen-doc05-20261114) 与 [归档/复现说明](../../three_gaps/post_exp13_20261114/baseline/README.md) 必须保留。先核对本机 checkpoint、Z、缓存及 manifest 的 SHA256；归档缺失则报告并从历史版本复现，不覆盖旧文件。历史复现记录称归档约 .69 GB、92 个文件；不把 Git tag 误当本机大文件备份。

新建独立目录，建议命名 three_gaps/exp20_latent_integration、exp21_data_scale_hessian、exp22_smoothed_hessian、exp23_multiscale_joint；若仓库最新版本已占用这些编号，换未占用编号并在 README 写明。每套可独立运行、独立日志/输出。不修改现有 EXP13、post-EXP13 结果，公共依赖若需改动必须保持历史兼容。默认在 main 提交代码和小型结果，大文件放 outputs/three_gaps/ 对应新目录并记录路径，不提交巨型缓存。本文的编号、预算与阈值是预设实验建议，不是已有结果。

不能使用模型名称、真实参数角色、gain/loss 设计类别、真实 T 或 Oracle 来决定 loss、权重、步长、分布、搜索起点或停止条件。T 仅事后诊断。所有四模型历史 TEST 已被查看，属于探索性比较；最终方法冻结后用新 S1 数据、NP repeat 3–5 及未见函数族检验。不得以最高 MCC 选旋转，必须保存最低 loss 解及所有候选。禁止 ICA/独立性目标；Gaussian 分布仅作为积分测度。第一轮冻结 encoder/decoder，不重新训练 NP。若需要新 NP 或独立 decoder，明确列为后续独立实验。

## 1. 背景、问题及现有证据

传统认知模型预设生成式，再拟合个体参数。ModLearn 反过来，从行为数据学习被试表示 Z，再发现具有认知意义的坐标。对被试 s、试次 t，真实 T_s 生成 eta(T_s,X_st)，p=sigmoid(eta)，Y~Bernoulli(p)。NP encoder 只见 (X,Y)，提取每位被试后验均值 Z_s；冻结 MLP decoder 预测 p(Z,X)。预测不区分可逆重参数化，因为 f_R(RZ,X)=f(Z,X)。旋转目的是把个体差异拆成一一对应的参数轴，而不是提高预测。

C1/C2/C3 的参数维数 4/4/6、X 维数 3/4/4；logit 对 T 线性但 C2/C3 对原始 X 有二次和交互。CR3 四参数 alpha/lambda/gamma/tau，六维 X=(gA,lA,pA,gB,lB,pB)，eta=tau[G(alpha,gamma,X)-lambda D(alpha,gamma,X)]，对参数非线性且有必要 UU 交互；5,000 被试共享固定 500 个真实彩票刺激。完整公式见 doc 01。CR3 的固定 X 设计使连续 X 导数尤其容易出现未受数据约束的外推。

历史最优 role-free 结果（冻结 p 组合）：

| model | rep1 MCC / leakage / max leakage | rep2 MCC / leakage / max leakage |
|---|---|---|
| C1 | .9688/.0212/.0576 | .9689/.0202/.0611 |
| C2 | .9645/.0599/.1523 | .9677/.0427/.1450 |
| C3 | .9600/.0464/.1143 | .9615/.0478/.1525 |
| CR3 | .8173/.1646/.3677 | .8122/.1832/.3781 |

CR3 Oracle MCC .872/.869（监督参考，不是理论上界）。logit UUd_share 单项 rep1 .821，UX_neff+UUd_share logit .831，但 C1–C3 退化；p 版本整体更稳。EXP14 的一阶外积 B-only rep1 CR3 .8057，完整 p .8173；不能把成功完全归因于一阶梯度。EXP17 子集可靠性加权没有稳定提升。EXP15 首轮中心点 8×8，最近邻 X chord 投影不是完整 ambient UX，CR3 p rep1 多尺度 .7112，匹配预算 ambient autograd .7640，pair autograd .7100；因此有限尺度并未证实有效，但现有试验不能隔离尺度效应、采样效应和 UX 定义变化。最新报告还指出未完成更大有限扰动预算。不要重复低预算扫描并称新发现。

本轮核心区分：
1. Monte Carlo/经验积分误差：有限 Z 节点是否覆盖足够？
2. 测度误差：真实后验均值分布与 Gaussian/球形参考分布是否不同？
3. 函数结构偏差：decoder 在微观尺度的曲率是否不可靠，平滑或有限差分能否改善？
4. loss 本身偏差：即使估计精确，真实轴是否真的最简单？EXP13 UUd_share 偏好低自曲率占比，不是惩罚非对角交互；a²-b² 旋转成 2uv 是反例。不能假定任何积分/平滑必定恢复真实轴。

## 2. 当前 EXP13 的精确数学与实现（必须复现）

代码：common/decoder.py、common/derivs.py、common/io.py、common/rotlib.py、exp13_pure_simplicity/cache2.py、losses2.py、combos.py。

在当前 pca=0 条件，V=Z-mu_train，只中心化，不预先逐轴 zscore 或 whitening。decoder 输入实际为 [X,Z]，Z=V+mu_train；eta=logit1-logit0，h(V,X)=sigmoid(eta)。冻结权重，torch.func.jacrev(h,0) 得 g_V，jacfwd(g,0) 得 H_VV，jacfwd(g,1) 得 H_VX。自动微分没有步长；decoder 内部 float32，缓存 float32，moment 累积转换 double，历史优化 float32。不得把 float64 输入误报为全程双精度。

CR3：TRAIN 随机 300 人（cache seed 20261108），每人与同一 500 个 X 交叉，150,000 点。C1–C3：300 人与新生成的 200 个 uniform(-1,1) X 及其负值组成的 400 点共享网格交叉，120,000 点；不是每人原始全部试次。H_VV (S,K,d,d)，H_VX (S,K,d,p)，CR3 d=4,p=6。只需 VV/VX，不计算 XX。缓存每批 2048 点。

旋转 R∈SO(d)，用参考 Z 协方差 C_ref 定义 D_R=diag(sqrt(diag(R C_ref R^T)))，U=D_R^{-1}R V（平移不影响导数），A=D_R R。历史 C_ref 是同一抽取 300 TRAIN V 的总体协方差（ddof=0）。H_UX=A H_VX，H_UU=A H_VV A^T。一般 M 可逆时 A=D_M M^{-T}。不得把前置标准化/白化当作原始正交搜索的同一空间。

定义加权期望 E_{P_Z,P_X}，每个积分节点 z_n 权重 w_n，X 权重 v_t，均归一化。原 EXP13 是等权离散经验测度。固定物理评价点，预计算：
$$
S^X_{j,ab}=\mathbb E[H_{VX,aj}H_{VX,bj}],\quad
T_{abcd}=\mathbb E[H_{VV,ab}H_{VV,cd}].
$$
$$
E^{UX}_{ij}=\sum_{ab}A_{ia}A_{ib}S^X_{j,ab},\quad
E^{UU}_{ik}=\sum_{abcd}A_{ia}A_{kb}A_{ic}A_{kd}T_{abcd}.
$$
CR3 SX 6×4×4、T4 4×4×4×4，共 352 个元素。先平方再平均，不是平均 Hessian 再平方；moment contraction 与直接旋转全部 Hessian 等价，浮点误差除外。

$$
q_{ij}=\frac{E^{UX}_{ij}}{\sum_{ij}E^{UX}_{ij}+\epsilon},\quad
L_{UX}=\exp(-\sum q_{ij}\log(q_{ij}+\epsilon)),
$$
$$
L_{UU}=\frac{\sum_i E^{UU}_{ii}}{\sum_{ik}E^{UU}_{ik}+\epsilon},\quad
L=\frac{L_{UX}}{s_{UX}}+\frac{L_{UU}}{s_{UU}}.
$$
精确实现参照 losses2.py（EPS=1e-12，neff 代码在归一化后加 EPS）。Haar 200 共同旋转估计每项 SD；对每个新统计量重新估计 s，另做固定历史 s 的敏感性对照，避免把权重变化误认为结构改善。近零 SD 报告无信号并按预设规则处理，不盲目除零。保持原 24×250 搜索设置及起点，先复现历史结果（允许事先声明的数值容差），再改变一个因素。

## 3. EXP20A：经验分布大样本积分

### 3.1 数学问题

对固定 decoder，H_hat=H_true+E（需在共同坐标和可比物理点定义）。E[H_hat²]=E[H_true²]+2E[H_true E]+E[E²]。增加被试只减少样本矩估计误差，不能自动去除噪声能量或系统偏差。固定 X 也意味着增加 Z 不增加刺激支持域。若样本扩大后 loss landscape 更稳定但 MCC 不变，则说明样本估计已不是主要瓶颈。

### 3.2 主实验

先 CR3 rep1/rep2，prob；logit 为预设机制对照。N=100,300,500,1000,2000,全 TRAIN（实际 N 由 split 读取，禁止假定 5000 全为 TRAIN）。保留历史 300 子集另列 A0；5 个预设种子生成嵌套无放回子集（同一 seed 大集合包含小集合），全 TRAIN 只跑一次。所有 X 固定完整 bank。C1/C2/C3 验证使用冻结 400 点网格，避免同时换 X。可额外做 5000 全样本 transductive 版本，明确 TEST Z 参与测度与优化，不能声称独立 TEST；另保留真正 TRAIN-only 结果。

用全部 TRAIN 一次流式计算每位被试的 SX_s、T4_s（对 X 平均）和 Z 的一二阶矩，保存 per-subject moment 小缓存，再按子集 ID 求加权平均。无需重复 autograd，也无需保存全部原始 Hessian。分块建议 16–64 被试，decoder 点 chunk=2048（显存不足降级并记录），累积 float64。只要 Hessian 和 X 不变，任意子集都可重用。全 TRAIN 规模若 3500 人则 1.75M 点；真实 N 以读取为准；全 5000 为 2.5M 点、约原始 16.7 倍 Hessian 查询。后续张量优化成本几乎不变。

两种尺度控制均运行：
- A-native: D_R 按每个子集的 C_ref 计算，保持 EXP13 语义。
- A-fixed: D_R 固定全 TRAIN C_ref，仅改变积分测度；另有历史 300 C_ref 的复现基准。分开报告二者，以区分 moment 抽样误差和轴尺度估计误差。

输出 N→MCC、leakage、最差轴、相对全 TRAIN SX/T4 误差、共同 Haar landscape Pearson/Spearman、最低点 R 的有符号排列距离、各项 loss SD；5 seeds 均值/SD/范围及逐次结果。比较同一 R 的 loss 误差，不直接比不同标准化后的绝对 loss。记录高曲率被试对能量的贡献分位数与影响诊断（trim/winsor 仅额外预设候选，不隐式删样本）。

## 4. EXP20B：连续 latent 分布积分

### 4.1 分布假设与测度

NP 的 KL 约束 q(Z|D_s) 不等于提取的后验均值 mu_s 必服从 N(0,I)。aggregate posterior 的协方差 = Cov_s(mu_s)+E_s[Cov(Z|D_s)]；因此 posterior samples 与 posterior means 尺度不同。边际正态也不等于联合正态/独立。先只用 TRAIN Z 拟合和诊断：各维 histogram、QQ、均值/SD、相关矩阵、协方差特征值、Mahalanobis 距离与 chi-square_d 比较、随机方向投影 QQ、经验与 Gaussian 的近邻距离/分位数。不要用 TEST T 选分布，也不要把正态检验显著性单独当成选择标准。

主测度 P0=经验 TRAIN；P1=N(mu_train,Sigma_train) 完整协方差；P2=N(mu_train,diag(Sigma_train))；P3=N(0,I) 原 NP prior（在原始 Z 坐标，不是中心化 V 后误加均值）；P4=球形 N(mu_train,trace(Sigma)/d I)；P5=截断 Gaussian（Mahalanobis 半径平方 chi-square_d 的 95%/99% 分位点）；P6=N(mu_train,c² Sigma_train)，c=.5,.75,1,1.25,1.5。P1 为第一推荐，P3/P4 是参考测度实验，不宣称真实人群分布。P5 用条件分布归一化，报告有效节点数、权重及截断率；不得按 R 过滤节点。

P_Z 固定于原物理 Z 空间，所有 R 用同一积分节点。对一般 Gaussian，Z=mu+L eps、LL^T=Sigma；到 decoder 用 V=Z-mu_train。若 Sigma 近奇异，记录 eigenvalues 和数值稳定化，不默默白化或改变搜索空间。P1 完整协方差保留相关，P2 丢弃相关，P4 等方差但不是白化后的同一物理分布。对于 P3 标准正态，R 后分布仍标准正态，分布本身不能识别 R；函数结构仍可识别，例如 z1*x1+z2*x2 的 UX_neff 从 2 变为 4（45°）。

### 4.2 三种积分器

对任意 integrand F（SX/T4 每个元素）：
$$
I=\frac1K\sum_{t=1}^K\int_{\mathbb R^d} F(z,X_t)\,dP_Z(z).
$$
MC：独立 Gaussian 样本，N=256,512,1024,2048,4096,8192；另 N=300 对照历史。误差一般 O(N^{-1/2})（有限方差条件）。每 N 至少 5 seeds，使用嵌套序列。

RQMC：torch SobolEngine(scramble=True) 或 scipy qmc.Sobol，使用 2^m 节点并用 inverse normal CDF，避免 0/1 端点（记录 clipping）；至少 4–8 独立 scrambling，嵌套前缀，计算跨 scramble 误差。这里 Sobol sequence 是积分方法，不是 Sobol 敏感性指数。相同 N 比 MC；不能保证对所有 Hessian integrand 有更快收敛。

Gauss–Hermite：一维 physicists Hermite 节点 a_i、权重 w_i，以 sqrt(2)a_i、w_i/sqrt(pi) 近似标准正态期望；d 维张量积权重相乘，Z=mu+L eps。CR3 d=4，m=3,5,7,9 得 81/625/2401/6561 节点；C3 d=6 同阶 729/15625/117649/531441，先低阶验证，不盲目运行高阶，优先 RQMC 或 sparse-grid（若实现需验证权重可能为负及能量稳定性）。用相邻阶数、不同积分器、共同 R landscape 检查收敛，非多项式 MLP 不存在有限阶精确保证。权重求和、常数/一二四阶高斯矩要单元测试。

### 4.3 分离积分误差和测度效应

第一阶段 P1 的 MC/QMC/GH 各预算，固定 X 和全 TRAIN C_ref；比较同一积分测度内收敛。第二阶段在已收敛预算比较 P0/P1/P2/P3/P4/P5/P6；主比较保持 C_ref=全 TRAIN，让唯一变化是积分测度。另运行 measure-native C_ref=该测度理论/条件协方差，比较轴标准化的影响。标准正态 native D_R=I。Haar SD 分别估计并附固定 s 对照。

所有虚拟节点不需要真实选择标签，也不生成虚拟 T；直接 query frozen decoder。不能用虚拟节点计算 U–T MCC，最终 R 只能应用到原始 held-out Z/T 做事后评价。Gaussian 积分是 decoder-query/autograd 证据，不是严格 WY-only。与严格 WY-only 的区别必须写清。

支持域：Gaussian 尾部并非真实 Z 支持；报告节点 Mahalanobis 分位数、经验 kNN 距离、低密度节点权重、各区域贡献的 Hessian 能量。按固定半径分层（例如经验 Mahalanobis 50/90/95/99 分位区间），计算每层贡献及去掉尾部后的诊断。截断测度是新目标，不是无偏积分；全部节点的权重不能因旋转改变。若 QMC 收敛但 P1 比 P0 差，说明测度改变而非积分器失败；若 P3/P4 好但 P1 差，应解释参考空间的作用而不是声称 prior 更真实。

第一轮先 CR3 完整 P1、P0、P3，QMC 与 GH；其余测度作为第二轮。通过无 T 的积分收敛判据扩展节点数（例如相邻预算相对 moment 误差 <1%、共同 landscape Spearman>.99；建议阈值，报告实际），不按 MCC 选预算。四模型确认至少用相同 P1/P0/P3 规则及同样的积分收敛标准。

## 5. EXP21：真实数据分辨率校准的有限尺度 Hessian

### 5.1 为什么需要步长

自动微分计算点上无限小尺度导数，没有 h。若 f_hat(x)=x²+eps sin(omega x)，二阶误差幅度 eps omega²；eps=.01,omega=20 时函数误差 .01、二阶误差 4。中心二阶差分噪声幅度 4eps sin²(omega h/2)/h²，整体大尺度抑制高频但不严格单调。有限差分是另一个目标：邻域平均曲率，不只是自动微分的低精度近似。

$$
D_{ii}^{h_i}f=\frac{f(u+h_ie_i)-2f(u)+f(u-h_ie_i)}{h_i^2}
=\int_{-1}^1(1-|t|)f_{ii}(u+t h_ie_i)dt.
$$
$$
D_{ik}^{h_i,h_k}f=\frac{f(++ )-f(+-)-f(-+)+f(--)}{4h_ih_k}.
$$
后者是矩形邻域内混合二阶导数的均匀平均。UX 用 u_i 和 x_j 的步长 h_U,h_Xj。若函数误差一致界 eps，混合差分误差≤eps/(h_i h_k)，轴向二阶≤4eps/h_i²；仅是理论界，不是实际 decoder 已满足。步长 h 是半宽，中心差分两端距离 2h；若典型观测点距 1，则 h=.5 是相应的中心差分半宽，不能混淆。

### 5.2 U 与 X 的尺度定义

U 逐轴单位 SD，预设 h_U=.05,.1,.25,.5,1（另 h→0 数值测试）；实际原 V 扰动向量为 h_U R^T D_R e_i。U 相关时沿轴移动仍可能进入低密度区域，按固定 TRAIN Mahalanobis/kNN 诊断，不随 R 剔除点。

X 用训练/固定真实刺激 bank 建尺度，不按模型角色：
(a) 每列排序去重相邻正间距的中位数/分位数；适于有明显离散步距的列，连续随机值的最小间隔不能代表六维分辨率。
(b) 标准化 X 后计算 kNN 距离（k=1,3,5,10），中位数与局部分位数；这是整体距离，不是单轴步长。
(c) 局部可行方向：在邻域内估计方向覆盖、位移矩阵的奇异值/条件数；列方向有足够支持才解释其曲率。
(d) 统一参考 h_Xj=c * SD(X_j)，c=.05,.1,.25,.5,1；另按实际间距校准 c=.5,1,2,4。将 h_Xj 的实际原始单位数值与典型两端距离同时写表。X 标准化仅用于选择邻域/步长时，差分仍除以原始单位 h_Xj 以复现 EXP13 的 UX 单位；另列单位 SD 导数版本，不能混称同一 loss。

CR3 500 点是固定刺激 bank，金额/概率及非支配条件有约束。沿单列扰动未必合法，即使合法也可能未被观测支持。不要把同列边际间距很小等同于该方向上可可靠估计连续导数。

### 5.3 三种 UX 方案（必须分开）

F1 ambient：在原始 X 坐标用四角混合差分，明确是 decoder 外推查询。对每个中心和候选 R 审计各查询点合法性、近邻距离、支持覆盖；不 clipping 后仍除以原步长。边界用合法非对称差分/局部拟合须显式改变公式与标记。若无法构造共同可行区域，报告该方案仅为外推敏感性，不当作数据支持证据。

F2 hybrid：U 用有限差分，X 仍用自动微分。H_UX^{h_U}= [grad_X f(U+h e_i,X)-grad_X f(U-h e_i,X)]/(2h)；UU 有限差分。只查询真实 X，可单独检验 U 曲率尺度；X 导数仍是 ambient、不能声称解决 X 微观噪声。反向 hybrid（U autograd、X finite）作为补充。

F3 bank chord：选择真实可行 X_a,X_b，v=X_b-X_a（距离度量用标准化坐标但输出单位需明确），
$$
I_i=\frac{f(U+h e_i,X_b)-f(U-h e_i,X_b)-f(U+h e_i,X_a)+f(U-h e_i,X_a)}{2h\,||v||}.
$$
它是沿 v 的方向性混合响应，不等于某一 H_UX,ij；将 I 投影到 X 列不等于恢复完整 UX。若需要原矩阵，在局部使用多个方向回归 I≈row(H_UX)·v/||v||，报告 rank、condition number、ridge、残差与可识别方向；若方向欠秩，拒绝完整矩阵恢复。可另定义 chord 空间的新 loss，并与 matched chord autograd 极限对照，而不是直接同 ambient EXP13 比较。真实配对用 kNN/不同距离分位组，不使用设计标签；配对和权重固定，避免 R 选择有利样本。

### 5.4 实验运行顺序和计算预算

必须先复用既有 EXP15 的结果及代码；新实验不重复 8×8 并宣称发现。先统一中心、参考协方差、X 权重、起点、优化器；每种 UX 定义对应 matched h→0 baseline。阶段 F0：解析 toy（线性/双线性/二次函数，差分精确；高频扰动例）；小步长与 autograd 对比（float32 差分消减误差，测试 float64 decoder copy 与原 float32 baseline，不修改原 checkpoint）；检查预测不变、单位换算与支持域。F1：固定共同 64×64 中心，CR3 两 repeat prob，单尺度及 hybrid；先六平面扫描、共同 Haar 子集，记录查询数。F2：128×128 和 300×完整 X 的升级（先对筛选后的少数预设候选，预算不足要报告）；F3：数据校准 X 尺度和多方向 chord 重建；F4：C1–C3 全部验证，C3 15 平面。完整 X 版应与同样中心的 autograd 基线配对，不能拿小预算差分直接比较全 300×500。

重要：有限尺度坐标差分通常**不是二阶张量**，R 改变时四角查询位置改变，必须重新计算；不能预先计算 H^h_ZZ 后用 R H^h R^T 代替。若改用固定平滑函数的真 Hessian，才可一次缓存张量（EXP22）。

优化可用既有 Givens 搜索（固定提案与 seed）或可微 query 下 Adam；两者与各自 matched autograd baseline 同预算比较。有限差分每次 R 查询成本较高，报告 wall time、前向次数、autograd 次数、峰值内存。先做少量角度扫描与候选筛选，仍须最终盲多起点搜索；禁止从 Oracle 起点当作盲搜索。不得以同一 TEST MCC 筛尺度。

## 6. EXP22：先平滑函数/Hessian，再平方积分（优先实施）

与 EXP21 的坐标差分不同，本实验固定原物理坐标的平滑核，定义
$$
f_\sigma(V,X)=\mathbb E_{\delta}[f(V+\delta_V,X+\delta_X)].
$$
在可交换求导与积分条件下，
$$
Hf_\sigma(W)=\mathbb E_\delta[Hf(W+\delta)],\quad
E_\sigma=\mathbb E_W[(\mathbb E_\delta Hf(W+\delta))^{\circ2}].
$$
注意不是 E_{W,delta}[H²]。后者只是扩张采样分布，保留局部噪声能量；前者先有符号平均，可能抵消局部高频变化。Gaussian 卷积对频率 omega 的 Fourier 分量乘 exp(-sigma²||omega||²/2)，但也抹去真实高频结构。Gaussian 平滑可能导致边界外推，尤其 X。

### 6.1 固定平滑核与计算方法

第一阶段只平滑 V，X 保持真实 bank/冻结网格。delta_V~N(0,sigma² C_ref) 或 N(0,sigma² trace(C_ref)/d I)，sigma=0,.05,.1,.25,.5,1。两种核为不同预设版本，C_ref 固定全部 TRAIN，不随 R 改变。主用 covariance-scaled 核，球形核作为对照；不是每次 R 在标准化 U 中重新平滑。固定核在原始坐标定义同一个 f_sigma，因此 H_UU=A H_VV,sigma A^T，H_UX=A H_VX,sigma，仍可沿用 EXP13 的高效 moment contraction。

S1：在中心点 z_n,X_t 周围用固定 Gaussian RQMC/antithetic 积分偏移，内层节点 8,16,32,64，先小预算收敛后扩展。每个中心先累积有符号 Hessian 平均，再外层计算 SX/T4；不能把偏移直接当作独立外层样本。共享内层偏移、成对 ±，保持可复现。外层先用经验 300，随后全 TRAIN 或收敛 P1 QMC。sigma=0 必须精确复现对应同测度、同 C_ref autograd 结果。可选：直接平均函数输出再自动微分，做小规模与平均 Hessian 的恒等式校验。

S2：增加 X 平滑但分三类报告：
- 固定连续核（以原 X 单位的协方差设尺度），仅在支持充分的区域解释，CR3 外推风险单列；
- bank 邻域 f_s(V,X_t)=sum_b w_tb f(V,X_b)，权重基于真实标准化距离、固定邻居。该离散算子可平滑 UU 或 g_U 的刺激剖面，**不能直接把 H_VX 在 bank 上的加权和称为该离散平滑函数的 X Hessian**；如需 UX，必须定义连续可微权重 w_b(x) 并包含其导数，或使用 chord/局部回归版本；
- 只对已有 H_VX 场按固定 bank 邻域平均，作为独立的导数场平滑估计器（不是必然某个函数的 Hessian），保留 U-index 变换，明确命名。

S3：同一中心/偏移比较 (i) H-smooth-then-square、(ii) square-then-neighborhood-average、(iii) sigma=0、(iv) 扩大 Z 积分测度但不平滑。若 i 改善而 ii 不改善，才支持局部有符号消噪机制。另记录平滑前后 p/eta 的函数值变化、选择预测 CE/Brier、饱和比例、能量分布与支持覆盖；本轮不是重新训练，不把 f_sigma 的预测替换原 NP 后声称预测严格不变。最终输出 R 仍应用原 Z/原 decoder，旋转保持原预测不变。

### 6.2 预算

先 CR3 64×64 中心、内层 8/16/32，对 sigma 扫描；随后 300×500 或全 TRAIN/QMC 外层、收敛内层节点。每一组合成本约外层点数×内层节点数的 Hessian 查询，流式计算、分块释放，不存四维大缓存。可将 sigma 的共同偏移标准化样本复用，但不同 sigma 物理点不同，不能直接复用 Hessian；缓存按 sigma、核、节点、checkpoint hash 区分。验证多尺度后再扩展 C1/C2/C3。

## 7. EXP23：多尺度聚合与联合实验

对固定中心 W、候选 R，H_k 表示尺度 k 的差分或平滑 Hessian，w_k≥0,sum=1。三种聚合必须分别实现：

A: signed field averaging（先平均导数）：
$$
E_A=\mathbb E[(\sum_k w_k H_k)^{\circ2}].
$$
B: energy averaging（先平方）：
$$
E_B=\sum_k w_k\mathbb E[H_k^{\circ2}].
$$
C: loss averaging（每尺度独立 Haar 标准化）：
$$
L_C=\sum_k w_k[L_{UX,k}/s_{UX,k}+L_{UU,k}/s_{UU,k}].
$$
A 保留跨尺度交叉项，可能消噪也可能抵消真实信号。B 保留各尺度能量，不抵消；C 让不同尺度的结构排序参与同一优化，不等于 A/B。对固定平滑核版本 A/B 可预缓存 moment（A 必须计算交叉矩或先平均字段），C 可缓存各尺度矩；对 R 依赖差分，仍需每 R 重新 query。不要仅平均已优化的 R（符号/排列且非线性目标），在共同坐标中组合目标并重新搜索。

权重先等权；预设少量对照：按 log-scale 区间等权（说明尺度集合及归一化）、中尺度优先固定权重、跨 scramble/子集/独立 decoder landscape 一致性权重（参考 EXP17，rho 正部或预设方差下限）。不按真实 T 或 TEST MCC 选 w。Haar 标准化可能放大无信号尺度，记录 SD 与近零退化。跨尺度 A/B 比较原能量与每尺度能量归一化版本，明确后者改变了相对幅度含义。

### 7.1 联合设计（分阶段，不做巨大无解释网格）

| 组 | 外层积分 | Hessian 结构 | 目的 |
|---|---|---|---|
| J0 | 历史300经验 | 原 autograd | 冻结基线 |
| J1 | 全TRAIN经验 | 原 autograd | 样本量影响 |
| J2 | 收敛P1 Gaussian QMC/GH | 原 autograd | 测度影响 |
| J3 | 历史300经验 | U 平滑 Hessian | 平滑单独影响 |
| J4 | 全TRAIN经验 | U 平滑 Hessian | 样本×平滑 |
| J5 | 收敛P1 Gaussian | U 平滑 Hessian | 连续积分×平滑 |
| J6 | 同J1/J2中心 | 数据尺度有限差分 | 与 matched autograd 对照 |
| J7 | 同J4/J5中心 | 多尺度 A/B/C | 跨尺度组合 |
| J8 | 固定经验或P1 | 真实bank chord/邻域 | X支持域敏感性 |

先 J0–J5，sigma 以预设网格或无 T 的函数值偏差/稳定性规则选；J6/J8 计算成本高，先 matched 小预算再扩展。所有方法固定相同 X 权重、相同 C_ref（另列 native），必要时比较固定历史 Haar SD 与新 Haar SD。主结果用 p，logit 做机制对照，不事后按模型选输出。最终在四模型、两 repeat、同规则运行；有希望的方法再冻结后上新 S1/NP repeat 3–5。不要求每个候选都跑昂贵全网格，要求每个宣称有效的候选有完整、可核查的公平对照。

## 8. 实现清单、代码接口与测试（Codex 必须落实）

建议目录结构（新文件，不是已存在命令）：
- exp20_latent_integration/{README.md,run.py,nodes.py,stream_moments.py,diagnostics.py,tests/,results/}
- exp21_data_scale_hessian/{README.md,run.py,scales.py,stencils.py,bank_pairs.py,tests/,results/}
- exp22_smoothed_hessian/{README.md,run.py,kernels.py,smooth_fields.py,tests/,results/}
- exp23_multiscale_joint/{README.md,run.py,combine.py,report.py,tests/,results/}

每套可独立运行。共同 CLI 建议：--model --beta --zdim --pca --repeat --channel --seed --tag --stage {selftest,cache,scan,optimize,evaluate,report} --device --chunk --starts --steps --subjects --trials；积分加 --measure --integrator --nodes --quadrature-order --scrambles --cov-ref；有限尺度加 --h-u --h-x-mode --h-x --ux-mode；平滑加 --sigma --kernel --inner-nodes；组合加 --combine --weights。现有 cache2.py build 的 S 参数没有独立 CLI，且硬编码 pca=0；不要直接改旧脚本。新入口支持预设 beta/d/PCA 参数，读取真实 checkpoint/latent 形状；当前 io 不支持高维 PCA 残差逆映射，明确报错或在独立后续实现残差协议，不能假装支持。

建议本机 Windows CMD（以实际 checkout 为准）：
~~~cmd
conda activate py11
cd /d D:\Projects\rotation\mod-rot-3
git pull --ff-only
~~~
若目录不同先定位仓库，不硬编码覆盖。新脚本实际写好并 selftest 后，再在各 README 写真实可执行命令。运行顺序先 CR3 rep1/2，后 C1–C3；GPU 任务顺序执行，避免多个训练/大导数任务并行导致机器卡死；记录实际 GPU、RAM、版本，支持 --resume 与原子 COMPLETE.json。若失败保存 stderr/traceback，修复后重跑，不用虚构结果填空。

必需单元测试：
1. 旧 300 缓存与新 per-subject streaming 聚合、直接全矩阵聚合一致；检查权重归一化、float64 累积、分块边界。
2. MC/QMC/GH 常数、一二四阶 Gaussian 矩；相关 Gaussian covariance；prior 原 Z 与 V 的均值映射；截断权重。
3. 同一物理点旋转预测不变、D_R 计算、直接 autograd 与 contraction 一致；所有 R 使用相同中心与积分测度。
4. finite difference 对 bilinear/quadratic 精确（容差考虑 dtype），h→0 与 autograd 一致，单位变换与半宽正确；R 变化时重新 query；X chord 欠秩明确拒绝。
5. sigma=0 平滑退化；常数/线性/二次函数的 Gaussian 平滑 Hessian 不变；平均函数后求导与平均 Hessian 一致；A/B/C 在单尺度退化一致；A/B 差异与交叉项手算对照。
6. 优化路径不加载 T 或 Oracle；优化完成保存 R 后才评价；seed 和缓存配置 hash 一致；重新运行不覆盖历史。

## 9. 输出与判读标准

每组保存 config.json（代码 commit、checkpoint/latent/cache SHA256、model/beta/d/PCA/repeat、dtype、device、TRAIN IDs、X 来源、测度参数、积分节点权重、C_ref、Haar SD、优化器预算）、moments.npz（SX/T4 及必要 per-subject moments）、rotations.npz（全部起点与最终 R、D_R）、metrics.csv、plane_curves.csv、random_landscapes.npz、diagnostics.json、UT_heatmaps.png、runtime.csv、logs、COMPLETE.json。大文件不进 Git；小型汇总、图、README、实际命令进 Git。若旧结果不包含某项，不能声称已经归档。

报告至少包含：
- 经验样本 N→moment误差、MCC/leakage/最差轴/最大leakage、R一致性；
- Gaussian 积分节点 N/阶数→收敛、MC/RQMC/GH 同测度比较、测度之间比较、尾部能量与支持；
- h_U/h_X/sigma→loss landscape、恢复指标、支持率、预测差异、耗时；A/B/C 对比；
- 所有两两平面 0–90°（四维6、六维15），共同 Haar 全局样本（建议独立1000），24起点终点及最低 loss；Oracle 方向仅诊断，不能以 Oracle 初始化冒充盲优化；
- 原始有符号 U–T 相关矩阵与匹配后的矩阵；MCC=Hungarian 最大绝对匹配的均值，leakage=非匹配绝对相关均值，另报每轴与最大；不同 NP 坐标不能直接比 R，必须在共同 Z 或事后 T 匹配后比较结构。

阶段性成功：相对于冻结 p 基线，CR3 两 repeat 均改善，争取 MCC≥.85、leakage≤.12；C1–C3 MCC 不低于冻结基线超过 .01，且不明显恶化最大 leakage。数值是研究目标不是通用理论保证。若新方法只在事后挑选 h、sigma、分布或 seed 后有效，不算通用成功。最终还须新数据/未见模型验证。

判读决策：
- N 增大而 moment/landscape 收敛但 MCC 不变：停止盲目增 N，进入平滑或目标诊断。
- 同 P1 的 QMC/GH 收敛相同但与 P0 不同：测度效应，不是数值积分误差。
- P1/P3 差异大：检查真实 Z 联合分布、prior 尺度与尾部外推。
- 平滑后 MCC 提高且多尺度稳定、预测与支持域损失小：支持有效尺度假说；若 square-then-average 同样提升，可能主要是测度扩张而非消噪。
- 有限差分不超过 matched autograd：报告负结果；若只超过小预算 baseline，不声称超过冻结 EXP13。
- 所有稳定积分仍失败：loss 本身或 decoder 共同结构偏差可能是瓶颈。参考 EXP14/17 和 doc06 的 shared Hessian / 独立 decoder 诊断。若需要跨 decoder，固定同一 Z，区分 E_b[H_b²] 与 (E_b[H_b])²；共同偏差仍不消失。
- 解析场失败：首先检查目标与可识别性；解析场成功、decoder 失败：结构估计/模型偏差；不把 autograd/query 成功当严格 WY-only。后续可将同一测度/平滑思路迁移到只由 WY 拟合的 surrogate，分别验证三个 Gap。

## 10. 最终执行顺序（给 Codex 的任务卡）

| 顺序 | 任务 | 退出条件 / 交付 |
|---|---|---|
| 0 | 核对冻结 EXP13、归档与历史 300 复现 | 复现表、manifest、无历史文件改动 |
| 1 | EXP20A per-subject streaming moments | CR3 两重复、100→全TRAIN、5 seeds、样本曲线 |
| 2 | EXP20B P1 MC/RQMC/GH + P0/P3 | 收敛图、测度对照、尾部诊断 |
| 3 | EXP22 U 平滑 sigma 网格 | 先平均H vs 先平方、sigma=0 对照、CR3 两重复 |
| 4 | EXP21 数据尺度有限差分 | matched预算、U/X半宽、hybrid/chord/ambient 分离、支持审计 |
| 5 | EXP23 A/B/C 多尺度与 J0–J8 | 可解释的单因素/联合对照，保留全部负结果 |
| 6 | 四模型验证、冻结新候选、独立数据 | 所有轴热图、MCC/leakage、独立repeat及新模型验证计划/结果 |

推荐先完成 1–3，再投入高成本的每 R 有限差分。每阶段生成独立 RESULTS.md，清楚标记已完成、部分完成、失败、尚未执行；只有实际跑出的数值才写入结果。若计算资源不足，优先交付可复现 baseline、完整经验积分和 P1 Gaussian 对照，其次 U 平滑；不要为了覆盖更多候选牺牲 matched baseline 与完整记录。

中心研究假说：**参数可解释性可能是冻结 decoder 在数据可支持的尺度和连续 latent 测度下的稳定函数结构，而不是无限小尺度 Hessian 或有限被试抽样的偶然性质。** 本轮的目的就是把这一假说拆成可证伪的独立实验，同时永久保留 EXP13 已有成果。
