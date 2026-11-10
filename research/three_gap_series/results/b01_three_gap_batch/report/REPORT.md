# 批量单指标的三鸿沟结果

已完成 385 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。

所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。
初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。

| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |
|---|---:|---:|---:|---:|
| anova2 / 8 | 77 | 10 | 0 | 0 |
| gaussian256 / 8 | 308 | 194 | 161 | 82 |

## 同一单指标在四模型的结果

| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |
|---|---|---:|---|---|---|---:|---|
| hs_b01_ux_rms_soft_t0.07_reference | gaussian256 | 8 | 是 | 否 | 否 | 0.14229 | FAILED_G2 |
| hs_b01_ux_rms_soft_t0.07_ux | gaussian256 | 8 | 是 | 否 | 否 | 0.14248 | FAILED_G2 |
| hs_b01_ux_rms_soft_t0.01_column | gaussian256 | 8 | 是 | 否 | 否 | 0.14603 | FAILED_G2 |
| hs_b01_ux_rms_soft_t0.02_column | gaussian256 | 8 | 是 | 否 | 否 | 0.14609 | FAILED_G2 |
| hs_b01_ux_rms_soft_t0.05_reference | gaussian256 | 8 | 是 | 否 | 否 | 0.16734 | FAILED_G2 |
| hs_b01_ux_rms_soft_t0.05_ux | gaussian256 | 8 | 是 | 否 | 否 | 0.16772 | FAILED_G2 |
| hs_b01_ux_rms_soft_t0.03_reference | gaussian256 | 8 | 是 | 否 | 否 | 0.17515 | FAILED_G2 |
| hs_b01_ux_rms_soft_t0.03_ux | gaussian256 | 8 | 是 | 否 | 否 | 0.17526 | FAILED_G2 |
| hs_b01_ux_rms_global_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.10941 | FAILED_G1 |
| hs_b01_full_rms_global_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.11032 | FAILED_G1 |
| hs_b01_functional1_global_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.11412 | FAILED_G1 |
| hs_b01_functional2_column_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.11731 | FAILED_G1 |
| hs_b01_full_rms_global_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.13098 | FAILED_G1 |
| hs_b01_ux_rms_column_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.13152 | FAILED_G1 |
| hs_b01_ux_rms_global_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.13229 | FAILED_G1 |
| hs_b01_functional1_column_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.13554 | FAILED_G1 |
| hs_b01_functional1_global_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.13747 | FAILED_G1 |
| hs_b01_functional2_column_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.13766 | FAILED_G1 |
| hs_b01_functional2_global_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.13820 | FAILED_G1 |
| hs_b01_ux_rms_global_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.13825 | FAILED_G1 |
| hs_b01_ux_rms_column_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.13883 | FAILED_G1 |
| hs_b01_functional2_column_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.13963 | FAILED_G1 |
| hs_b01_functional1_row_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.14185 | FAILED_G1 |
| hs_b01_functional2_column_soft003 | gaussian256 | 8 | 否 | 否 | 否 | 0.14299 | FAILED_G1 |
| hs_b01_ux_rms_column_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.14470 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.003_column | gaussian256 | 8 | 否 | 否 | 否 | 0.14601 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.03_column | gaussian256 | 8 | 否 | 否 | 否 | 0.14618 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.05_column | gaussian256 | 8 | 否 | 否 | 否 | 0.14636 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.07_column | gaussian256 | 8 | 否 | 否 | 否 | 0.14640 | FAILED_G1 |
| hs_b01_functional2_column_lp05 | gaussian256 | 8 | 否 | 否 | 否 | 0.14653 | FAILED_G1 |
| hs_b01_functional1_column_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.14661 | FAILED_G1 |
| hs_b01_full_rms_column_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.15080 | FAILED_G1 |
| hs_b01_functional1_column_lp05 | gaussian256 | 8 | 否 | 否 | 否 | 0.15105 | FAILED_G1 |
| hs_b01_functional2_column_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.15155 | FAILED_G1 |
| hs_b01_full_rms_column_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.15312 | FAILED_G1 |
| hs_b01_functional1_column_soft003 | gaussian256 | 8 | 否 | 否 | 否 | 0.15526 | FAILED_G1 |
| hs_b01_functional1_column_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.15709 | FAILED_G1 |
| hs_b01_ux_gram_rms_corr0 | gaussian256 | 8 | 否 | 否 | 否 | 0.15774 | FAILED_G1 |
| hs_b01_full_rms_global_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.16290 | FAILED_G1 |
| hs_b01_functional2_global_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.16347 | FAILED_G1 |
| hs_b01_full_rms_column_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.16439 | FAILED_G1 |
| hs_b01_functional1_column_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.17196 | FAILED_G1 |
| hs_b01_functional2_row_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.17544 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.02_reference | gaussian256 | 8 | 否 | 否 | 否 | 0.17740 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.02_ux | gaussian256 | 8 | 否 | 否 | 否 | 0.17745 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.01_reference | gaussian256 | 8 | 否 | 否 | 否 | 0.17876 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.01_ux | gaussian256 | 8 | 否 | 否 | 否 | 0.17877 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.003_reference | gaussian256 | 8 | 否 | 否 | 否 | 0.17918 | FAILED_G1 |
| hs_b01_ux_rms_soft_t0.003_ux | gaussian256 | 8 | 否 | 否 | 否 | 0.17918 | FAILED_G1 |
| hs_b01_functional1_global_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.18152 | FAILED_G1 |
| hs_b01_functional1_row_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.18384 | FAILED_G1 |
| hs_b01_functional2_row_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.19573 | FAILED_G1 |
| hs_b01_functional2_global_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.20685 | FAILED_G1 |
| hs_b01_ux_rms_row_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.24600 | FAILED_G1 |
| hs_b01_full_rms_row_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.24694 | FAILED_G1 |
| hs_b01_functional1_row_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.24936 | FAILED_G1 |
| hs_b01_full_rms_row_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.25781 | FAILED_G1 |
| hs_b01_ux_gram_signed_corr0 | gaussian256 | 8 | 否 | 否 | 否 | 0.26162 | FAILED_G1 |
| hs_b01_ux_gram_rms_corr1 | gaussian256 | 8 | 否 | 否 | 否 | 0.28613 | FAILED_G1 |
| hs_b01_ux_rms_row_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.28733 | FAILED_G1 |
| hs_b01_functional2_row_gini | gaussian256 | 8 | 否 | 否 | 否 | 0.29173 | FAILED_G1 |
| hs_b01_ux_rms_row_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.29193 | FAILED_G1 |
| hs_b01_full_rms_row_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.29810 | FAILED_G1 |
| hs_b01_ux_gram_signed_corr1 | gaussian256 | 8 | 否 | 否 | 否 | 0.32651 | FAILED_G1 |
| hs_b01_ux_gram_signed_corr0 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.11722 | PENDING |
| hs_b01_ux_gram_signed_corr1 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.11722 | PENDING |
| hs_b01_full_rms_global_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.12269 | PENDING |
| hs_b01_ux_rms_global_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.12510 | PENDING |
| hs_b01_ux_rms_global_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.12517 | PENDING |
| hs_b01_full_rms_global_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.12640 | PENDING |
| hs_b01_functional1_global_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.13185 | PENDING |
| hs_b01_functional1_global_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.13297 | PENDING |
| hs_b01_functional2_global_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.13490 | PENDING |
| hs_b01_functional1_column_l1 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14031 | PENDING |
| hs_b01_ux_gram_rms_corr1 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14102 | PENDING |
| hs_b01_ux_gram_rms_corr0 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14102 | PENDING |
| hs_b01_ux_rms_column_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14230 | PENDING |
| hs_b01_functional2_column_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14269 | PENDING |
| hs_b01_ux_rms_column_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14616 | PENDING |
| hs_b01_ux_rms_soft_t0.003_column | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14892 | PENDING |
| hs_b01_ux_rms_soft_t0.01_column | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14896 | PENDING |
| hs_b01_ux_rms_soft_t0.02_column | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14905 | PENDING |
| hs_b01_ux_rms_soft_t0.03_column | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14918 | PENDING |
| hs_b01_ux_rms_soft_t0.05_column | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14946 | PENDING |
| hs_b01_ux_rms_soft_t0.07_column | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.14959 | PENDING |
| hs_b01_functional2_column_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15001 | PENDING |
| hs_b01_full_rms_global_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15213 | PENDING |
| hs_b01_functional1_column_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15339 | PENDING |
| hs_b01_ux_rms_soft_t0.07_reference | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15386 | PENDING |
| hs_b01_functional2_column_soft003 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15397 | PENDING |
| hs_b01_ux_rms_soft_t0.07_ux | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15446 | PENDING |
| hs_b01_ux_rms_global_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15472 | PENDING |
| hs_b01_ux_rms_column_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15790 | PENDING |
| hs_b01_full_rms_column_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15800 | PENDING |
| hs_b01_functional2_column_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15896 | PENDING |
| hs_b01_functional1_column_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15925 | PENDING |
| hs_b01_full_rms_column_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.15942 | PENDING |
| hs_b01_functional1_column_lp05 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.16223 | PENDING |
| hs_b01_functional2_column_lp05 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.16337 | PENDING |
| hs_b01_functional1_column_soft003 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.17303 | PENDING |
| hs_b01_ux_rms_soft_t0.05_reference | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.17676 | PENDING |
| hs_b01_ux_rms_soft_t0.05_ux | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.17702 | PENDING |
| hs_b01_functional1_global_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.17942 | PENDING |
| hs_b01_ux_rms_soft_t0.03_reference | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18264 | PENDING |
| hs_b01_ux_rms_soft_t0.03_ux | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18266 | PENDING |
| hs_b01_ux_rms_soft_t0.02_reference | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18321 | PENDING |
| hs_b01_ux_rms_soft_t0.02_ux | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18322 | PENDING |
| hs_b01_ux_rms_soft_t0.01_reference | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18371 | PENDING |
| hs_b01_ux_rms_soft_t0.01_ux | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18371 | PENDING |
| hs_b01_ux_rms_soft_t0.003_reference | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18389 | PENDING |
| hs_b01_ux_rms_soft_t0.003_ux | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18389 | PENDING |
| hs_b01_functional2_row_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18767 | PENDING |
| hs_b01_functional1_row_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.18881 | PENDING |
| hs_b01_functional2_column_l1 | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.19213 | PENDING |
| hs_b01_functional1_column_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.19841 | PENDING |
| hs_b01_ux_rms_row_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.20007 | PENDING |
| hs_b01_full_rms_row_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.20136 | PENDING |
| hs_b01_ux_rms_row_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.20594 | PENDING |
| hs_b01_functional2_global_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.20971 | PENDING |
| hs_b01_full_rms_row_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.21287 | PENDING |
| hs_b01_ux_rms_row_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.22084 | PENDING |
| hs_b01_full_rms_row_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.22627 | PENDING |
| hs_b01_functional2_global_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.22818 | PENDING |
| hs_b01_functional1_row_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.22900 | PENDING |
| hs_b01_functional2_row_hoyer | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.23262 | PENDING |
| hs_b01_full_rms_column_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.23294 | PENDING |
| hs_b01_functional1_row_gini | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.23770 | PENDING |
| hs_b01_functional2_row_kurtosis | anova2 | 8 | 未齐 | 未齐 | 未齐 | 0.27520 | PENDING |

## 证据与限制

- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。
- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。
- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。
- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。
- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。

精确别名 13 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。
