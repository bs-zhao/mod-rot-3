# 历史旋转方法完整移植：C1 / C2 / C3 / CR3

已实现并实际运行一个独立项目 `rotations/historical_transfer`，不调用其他旋转项目。冻结现有 NP，未重训。固定协议下完成 8 组“模型×坐标表示”，每组保存 34 个标签，共 272 项。每组含 identity、22 个回归尝试、9 个不同 DDM 尝试以及 2 个精确 joint-Fisher 别名；别名不算独立证据。

## 本轮发现

回归方法的部分结构能够迁移，DDM observed-X Fisher 的成功没有直接迁移到这四个选择模型。以下固定列出同一组方法，均为 PCA 白化坐标下的 TEST 平均 r²：

| 方法 | C1 | C2 | C3 | CR3 |
|---|---:|---:|---:|---:|
| Identity | .7010 | .6293 | .6865 | .4842 |
| logit + level/linear XZ Geomin | .9421 | .9514 | .8946 | .4793 |
| logit + level-anchored entropy/coverage | .9412 | .8835 | .8643 | .4349 |
| observed-X Fisher Geomin | .4640 | .4893 | .2826 | .3446 |

CR3 的一项改善来自另一个坐标/变换族：原始高方差坐标标准化下，原版斜交 JVolmax 从 .4409 提高到 .6241，最差参数 r²=.5908，最大非目标 |r|=.3655。该结果不能与正交 PCA 旋转混为同一可行族，也不是四模型统一获胜方法。PCA 坐标下的 logit basis-joint Geomin 对 CR3 为 .5577，但 alpha 从 .6283 降到 .5387，不能只凭均值上升称为全面改善。

所有 8 组设置、两种输出尺度的曲率细分均未检出共享 X 组，因此最终等同于 anchored_entropy_coverage。这里没有复现原 M4 的曲率增益；也没有降低门槛或手动指定组来制造增益。

这些数字是本次固定预算的探索性结果，不证明任何目标的全局最优或普遍不可识别。完整表格和非目标相关见下方及附带文件。

## 评估协议

默认条件 beta0p0005_lambda0_ctx0p25to0p8_L8，repeat1；固定 K 为 4/4/6/4。128 名 TRAIN 被试用于旋转拟合。选维、PCA 和尺度只用 TRAIN。回归签名每人取 40 个实际 X，observed-X Fisher 使用这些被试全部有效试次。所有请求的拟合和 U 持久化后才读取真实参数。

主指标：TRAIN 相关矩阵上一对一 Hungarian 匹配，固定此匹配后计算 TEST Pearson r²。下面是目标 r² 均值，所有方法均按原清单顺序展示，没有挑选最优方法作为新候选。完整指标含各参数、最差参数、全部有符号相关矩阵和非目标相关。

## 必须说明的适配

- 四个模型只有二元选择，没有 RT；因此没有虚构 RT 通道。choice-only joint Fisher 与 channelwise Fisher 数学上相同，单独标注别名。
- 概率 p 是回归均值的直接对应，logit 是链接尺度对照，两套均运行。
- 比较 PCA 白化及原始高方差坐标标准化；两者定义不同的旋转可行族。
- 原版 JVolmax 允许斜交，予以保留并明确标注；另做正交约束对照。两者都用真正逆变换维持解码等价。
- C1/C2/C3 用 FIT 范围内的多维 Gauss–Legendre 产品均匀积分；CR3 的观测设计是有约束的离散共享 bank，使用完整联合观测行的有限求和，不能称为连续矩形积分。
- CR3 使用相同完整 bank、每人相同试次数，因此 integrated-X 与 observed-X 也定义同一个经验目标；分别优化可反映局部最优差异，不能算独立信息证据。
- 曲率细分只在原算法自动检测到的共享 X 组内执行；未检出时保留原变换，不强行旋转。
- 8 次初始化、500 次正交优化迭代；JVolmax 8 次初始化、最多 3000 Adam 步。与历史项目预算不同，所有收敛记录和失败/弱结果都保留。

## PCA 白化坐标

| 方法 | C1 | C2 | C3 | CR3 |
|---|---:|---:|---:|---:|
| identity | 0.7010 | 0.6293 | 0.6865 | 0.4842 |
| probability__gradient_geomin | 0.4680 | 0.5074 | 0.3852 | 0.3480 |
| probability__linear_xz_geomin | 0.9418 | 0.8839 | 0.6725 | 0.3653 |
| probability__level_linear_xz_geomin | 0.9421 | 0.9514 | 0.7462 | 0.4549 |
| probability__nonlinear_basis_xz_geomin | 0.9139 | 0.9056 | 0.6912 | 0.4887 |
| probability__nonparametric_contrast_geomin | 0.9401 | 0.8483 | 0.8805 | 0.4813 |
| probability__joint_signature_geomin | 0.7256 | 0.8721 | 0.5985 | 0.3322 |
| probability__basis_geomin | 0.9351 | 0.8352 | 0.6823 | 0.4288 |
| probability__basis_joint_geomin | 0.7054 | 0.7788 | 0.4745 | 0.4443 |
| probability__anchored_entropy | 0.9394 | 0.8858 | 0.8632 | 0.4198 |
| probability__anchored_entropy_coverage | 0.9394 | 0.8858 | 0.8632 | 0.4198 |
| probability__functional_shape | 0.9394 | 0.8858 | 0.8632 | 0.4198 |
| logit__gradient_geomin | 0.4748 | 0.5005 | 0.4242 | 0.3592 |
| logit__linear_xz_geomin | 0.9421 | 0.9467 | 0.8752 | 0.3597 |
| logit__level_linear_xz_geomin | 0.9421 | 0.9514 | 0.8946 | 0.4793 |
| logit__nonlinear_basis_xz_geomin | 0.6986 | 0.9187 | 0.8484 | 0.3679 |
| logit__nonparametric_contrast_geomin | 0.9423 | 0.8415 | 0.7576 | 0.3644 |
| logit__joint_signature_geomin | 0.7658 | 0.7321 | 0.6613 | 0.3820 |
| logit__basis_geomin | 0.8880 | 0.8885 | 0.8361 | 0.3639 |
| logit__basis_joint_geomin | 0.5369 | 0.6992 | 0.5686 | 0.5577 |
| logit__anchored_entropy | 0.9412 | 0.8799 | 0.8577 | 0.4349 |
| logit__anchored_entropy_coverage | 0.9412 | 0.8835 | 0.8643 | 0.4349 |
| logit__functional_shape | 0.9412 | 0.8835 | 0.8643 | 0.4349 |
| ddm__fingerprint_svd_varimax | 0.4436 | 0.4809 | 0.3372 | 0.3932 |
| ddm__jvolmax_oblique | 0.5901 | 0.5092 | 0.5063 | 0.2995 |
| ddm__jvolmax_orthogonal | 0.4018 | 0.5027 | 0.3614 | 0.4201 |
| ddm__fixed_common_x_fisher_geomin | 0.4235 | 0.5112 | 0.2866 | 0.3766 |
| ddm__fixed_common_x_fisher_quartimin | 0.4524 | 0.4836 | 0.3199 | 0.3813 |
| ddm__integrated_x_fisher_geomin | 0.4684 | 0.4854 | 0.2781 | 0.3446 |
| ddm__integrated_x_joint_fisher_geomin | 0.4684 | 0.4854 | 0.2781 | 0.3446 |
| ddm__integrated_x_fisher_quartimin | 0.2516 | 0.4888 | 0.4432 | 0.3662 |
| ddm__integrated_x_joint_fisher_quartimin | 0.2516 | 0.4888 | 0.4432 | 0.3662 |
| ddm__observed_x_fisher_geomin | 0.4640 | 0.4893 | 0.2826 | 0.3446 |
| ddm__observed_x_fisher_quartimin | 0.2506 | 0.4893 | 0.2762 | 0.3662 |

## 原始高方差坐标标准化

| 方法 | C1 | C2 | C3 | CR3 |
|---|---:|---:|---:|---:|
| identity | 0.5852 | 0.6245 | 0.5980 | 0.4409 |
| probability__gradient_geomin | 0.4847 | 0.5205 | 0.3354 | 0.3818 |
| probability__linear_xz_geomin | 0.9419 | 0.8773 | 0.6925 | 0.3350 |
| probability__level_linear_xz_geomin | 0.9427 | 0.9492 | 0.7498 | 0.4363 |
| probability__nonlinear_basis_xz_geomin | 0.9164 | 0.8928 | 0.7279 | 0.3659 |
| probability__nonparametric_contrast_geomin | 0.8202 | 0.7269 | 0.7753 | 0.5287 |
| probability__joint_signature_geomin | 0.7198 | 0.8126 | 0.7982 | 0.4078 |
| probability__basis_geomin | 0.9152 | 0.9240 | 0.5805 | 0.3277 |
| probability__basis_joint_geomin | 0.6913 | 0.6946 | 0.6452 | 0.3964 |
| probability__anchored_entropy | 0.9398 | 0.8849 | 0.8622 | 0.3799 |
| probability__anchored_entropy_coverage | 0.9398 | 0.8849 | 0.8625 | 0.3799 |
| probability__functional_shape | 0.9398 | 0.8849 | 0.8625 | 0.3799 |
| logit__gradient_geomin | 0.4856 | 0.5205 | 0.3573 | 0.3818 |
| logit__linear_xz_geomin | 0.9417 | 0.9437 | 0.8572 | 0.4009 |
| logit__level_linear_xz_geomin | 0.9428 | 0.9498 | 0.8949 | 0.3681 |
| logit__nonlinear_basis_xz_geomin | 0.8815 | 0.8161 | 0.7188 | 0.3444 |
| logit__nonparametric_contrast_geomin | 0.9417 | 0.8700 | 0.7941 | 0.3976 |
| logit__joint_signature_geomin | 0.7686 | 0.8058 | 0.8505 | 0.4820 |
| logit__basis_geomin | 0.9151 | 0.9002 | 0.7031 | 0.4304 |
| logit__basis_joint_geomin | 0.7196 | 0.7800 | 0.5939 | 0.3741 |
| logit__anchored_entropy | 0.9418 | 0.8786 | 0.8579 | 0.4333 |
| logit__anchored_entropy_coverage | 0.9418 | 0.8819 | 0.8643 | 0.4333 |
| logit__functional_shape | 0.9418 | 0.8819 | 0.8643 | 0.4333 |
| ddm__fingerprint_svd_varimax | 0.4318 | 0.4783 | 0.3380 | 0.4176 |
| ddm__jvolmax_oblique | 0.7780 | 0.5843 | 0.5521 | 0.6241 |
| ddm__jvolmax_orthogonal | 0.3935 | 0.5087 | 0.3380 | 0.4689 |
| ddm__fixed_common_x_fisher_geomin | 0.4267 | 0.5117 | 0.2901 | 0.3560 |
| ddm__fixed_common_x_fisher_quartimin | 0.4559 | 0.4793 | 0.3284 | 0.4076 |
| ddm__integrated_x_fisher_geomin | 0.4690 | 0.4858 | 0.2506 | 0.3724 |
| ddm__integrated_x_joint_fisher_geomin | 0.4690 | 0.4858 | 0.2506 | 0.3724 |
| ddm__integrated_x_fisher_quartimin | 0.2297 | 0.4863 | 0.4565 | 0.3850 |
| ddm__integrated_x_joint_fisher_quartimin | 0.2297 | 0.4863 | 0.4565 | 0.3850 |
| ddm__observed_x_fisher_geomin | 0.4669 | 0.4903 | 0.2755 | 0.3724 |
| ddm__observed_x_fisher_quartimin | 0.2500 | 0.4884 | 0.2709 | 0.3850 |

## 数值与可复核性

- 最大潜变量往返误差：8.88e-16。
- 抽样审计中的最大预测概率往返误差：5.55e-16（每个变换检查 8 个真实 FIT 状态）。
- 非斜交方法最大正交误差：1.44e-14。
- 生成数据、checkpoint、S5 潜变量文件在导出时重新校验 SHA256，全部与拟合协议一致；源代码哈希一致。
- 正交优化共 1728 次启动，其中 47 次达到 500 次迭代上限，另有 1 次未报告收敛；每项方法按预定无监督目标取预算内最优解，不保证全局最优。完整记录见优化启动诊断和各 fit.json。
- 7 项测试通过：二元 Fisher 恒等式、熵/覆盖与原代码一致、JVolmax 尺度不变性、完整残差逆变换、简单结构恢复、共享组曲率及自动微分/数值导数核对；另覆盖 TRAIN-only 原始坐标选维与尺度。

完整原始 gradient fields、所有 U 和重启矩阵仍在本机各模型的 `outputs/choice/<MODEL>/rotations/historical_transfer/`。本交付包附代码、全部结果、变换矩阵、协议与审计；不重复打包原始训练数据或 checkpoint。`运行来源.json` 给出每一组原始目录。

本轮是已有冻结模型上的方法移植比较，不是独立训练重复，也未因观察恢复结果而调参或挑选新主方法。跨方法的差异是探索性比较；具体结论应结合单参数与非目标相关。
