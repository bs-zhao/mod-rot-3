# 批量单指标的三鸿沟结果

已完成 148 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。

所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。
初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。

| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |
|---|---:|---:|---:|---:|
| gaussian256 / 8 | 148 | 118 | 102 | 35 |

## 同一单指标在四模型的结果

| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |
|---|---|---:|---|---|---|---:|---|
| hs_b08_b06_rank_degree1_edges_schatten05_conditional | gaussian256 | 8 | 是 | 是 | 否 | 0.13261 | FAILED_G3 |
| hs_b08_b06_rank_degree1_edges_schatten05_geometric | gaussian256 | 8 | 是 | 是 | 否 | 0.13311 | FAILED_G3 |
| hs_b08_b06_rank_degree1_edges_schatten05_isotropic | gaussian256 | 8 | 是 | 是 | 否 | 0.13343 | FAILED_G3 |
| hs_b08_b06_rank_degree2_edges_soft_t0.03_geometric | gaussian256 | 8 | 是 | 是 | 否 | 0.15240 | FAILED_G3 |
| hs_b08_b06_rank_degree2_edges_soft_t0.03_isotropic | gaussian256 | 8 | 是 | 是 | 否 | 0.15467 | FAILED_G3 |
| hs_b08_b06_rank_degree2_edges_soft_t0.03_conditional | gaussian256 | 8 | 是 | 是 | 否 | 0.16108 | FAILED_G3 |
| hs_b08_b02_unique_edges_soft_t0.02_geometric | gaussian256 | 8 | 是 | 是 | 否 | 0.21477 | FAILED_G3 |
| hs_b08_b02_unique_edges_soft_t0.02_isotropic | gaussian256 | 8 | 是 | 是 | 否 | 0.21982 | FAILED_G3 |
| hs_b08_b02_unique_edges_soft_t0.02_conditional | gaussian256 | 8 | 是 | 是 | 否 | 0.22075 | FAILED_G3 |
| hs_b08_b02_unique_edges_lp05_conditional | gaussian256 | 8 | 是 | 否 | 否 | 0.13794 | FAILED_G2 |
| hs_b08_b02_unique_edges_lp05_geometric | gaussian256 | 8 | 是 | 否 | 否 | 0.13954 | FAILED_G2 |
| hs_b08_b01_ux_rms_soft_t0.07_reference_conditional | gaussian256 | 8 | 是 | 否 | 否 | 0.13997 | FAILED_G2 |
| hs_b08_b01_ux_rms_soft_t0.07_reference_geometric | gaussian256 | 8 | 是 | 否 | 否 | 0.14115 | FAILED_G2 |
| hs_b08_b02_unique_edges_lp05_isotropic | gaussian256 | 8 | 是 | 否 | 否 | 0.14148 | FAILED_G2 |
| hs_b08_b01_ux_rms_soft_t0.07_reference_isotropic | gaussian256 | 8 | 是 | 否 | 否 | 0.14264 | FAILED_G2 |
| hs_b08_b01_ux_rms_soft_t0.05_reference_conditional | gaussian256 | 8 | 是 | 否 | 否 | 0.16667 | FAILED_G2 |
| hs_b08_b01_ux_rms_soft_t0.05_reference_geometric | gaussian256 | 8 | 是 | 否 | 否 | 0.16724 | FAILED_G2 |
| hs_b08_b01_ux_rms_soft_t0.05_reference_isotropic | gaussian256 | 8 | 是 | 否 | 否 | 0.16864 | FAILED_G2 |
| hs_b08_ux_rms_soft003_isotropic | gaussian256 | 8 | 是 | 否 | 否 | 0.17197 | FAILED_G2 |
| hs_b08_ux_rms_soft003_conditional | gaussian256 | 8 | 是 | 否 | 否 | 0.17443 | FAILED_G2 |
| hs_b08_ux_rms_soft003_geometric | gaussian256 | 8 | 是 | 否 | 否 | 0.17489 | FAILED_G2 |
| hs_ux_rms_soft003 | gaussian256 | 8 | 是 | 否 | 否 | 0.17515 | FAILED_G2 |
| hs_b08_b02_unique_edges_soft_t0.03_geometric | gaussian256 | 8 | 是 | 否 | 否 | 0.20076 | FAILED_G2 |
| hs_b08_b02_unique_edges_soft_t0.03_isotropic | gaussian256 | 8 | 是 | 否 | 否 | 0.20706 | FAILED_G2 |
| hs_b08_b02_unique_edges_soft_t0.03_conditional | gaussian256 | 8 | 是 | 否 | 否 | 0.21406 | FAILED_G2 |
| hs_b08_b05_taylor_fit2_order3_lp05_geometric | gaussian256 | 8 | 否 | 否 | 否 | 0.13332 | FAILED_G1 |
| hs_b08_b05_taylor_fit2_order3_lp05_isotropic | gaussian256 | 8 | 否 | 否 | 否 | 0.13372 | FAILED_G1 |
| hs_b08_b07_nonlinear_axes_global_log0.3_geometric | gaussian256 | 8 | 否 | 否 | 否 | 0.13391 | FAILED_G1 |
| hs_b08_b05_taylor_fit2_order3_lp05_conditional | gaussian256 | 8 | 否 | 否 | 否 | 0.13452 | FAILED_G1 |
| hs_b08_b07_nonlinear_axes_global_log0.3_conditional | gaussian256 | 8 | 否 | 否 | 否 | 0.14327 | FAILED_G1 |
| hs_b08_b07_nonlinear_axes_global_log0.3_isotropic | gaussian256 | 8 | 否 | 否 | 否 | 0.14691 | FAILED_G1 |
| hs_b08_b01_functional1_column_hoyer_conditional | gaussian256 | 8 | 否 | 否 | 否 | 0.15439 | FAILED_G1 |
| hs_b08_b01_functional1_column_hoyer_geometric | gaussian256 | 8 | 否 | 否 | 否 | 0.15571 | FAILED_G1 |
| hs_b08_b01_functional1_column_hoyer_isotropic | gaussian256 | 8 | 否 | 否 | 否 | 0.16705 | FAILED_G1 |
| hs_b08_ux_rms_l2sq_conditional | gaussian256 | 8 | 否 | 否 | 否 | 0.31094 | ENERGY_CONTROL |
| hs_b08_ux_rms_l2sq_geometric | gaussian256 | 8 | 否 | 否 | 否 | 0.31306 | ENERGY_CONTROL |
| hs_b08_ux_rms_l2sq_isotropic | gaussian256 | 8 | 否 | 否 | 否 | 0.32249 | ENERGY_CONTROL |

## 证据与限制

- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。
- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。
- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。
- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。
- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。

精确别名 0 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。
