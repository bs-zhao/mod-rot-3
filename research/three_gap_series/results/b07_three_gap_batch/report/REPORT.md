# 批量单指标的三鸿沟结果

已完成 124 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。

所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。
初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。

| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |
|---|---:|---:|---:|---:|
| gaussian256 / 8 | 124 | 4 | 3 | 1 |

## 同一单指标在四模型的结果

| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |
|---|---|---:|---|---|---|---:|---|
| hs_ux_rms_soft003 | gaussian256 | 8 | 是 | 否 | 否 | 0.17515 | FAILED_G2 |
| hs_b07_nonlinear_axes_global_soft0.3 | gaussian256 | 8 | 否 | 否 | 否 | 0.09979 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_log0.3 | gaussian256 | 8 | 否 | 否 | 否 | 0.12100 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_log0.2 | gaussian256 | 8 | 否 | 否 | 否 | 0.14375 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_log0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.14504 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_log0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.14733 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_log0.5 | gaussian256 | 8 | 否 | 否 | 否 | 0.14759 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_log0.5 | gaussian256 | 8 | 否 | 否 | 否 | 0.14940 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_log0.3 | gaussian256 | 8 | 否 | 否 | 否 | 0.14992 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_log0.2 | gaussian256 | 8 | 否 | 否 | 否 | 0.15029 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_log0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.15077 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_soft0.5 | gaussian256 | 8 | 否 | 否 | 否 | 0.15111 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_log0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.15120 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_soft0.3 | gaussian256 | 8 | 否 | 否 | 否 | 0.15403 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_soft0.2 | gaussian256 | 8 | 否 | 否 | 否 | 0.15501 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_soft0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.15544 | FAILED_G1 |
| hs_b07_nonlinear_axes_row_soft0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.15551 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_soft0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.18004 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_soft0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.18554 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_soft0.3 | gaussian256 | 8 | 否 | 否 | 否 | 0.19587 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_soft0.5 | gaussian256 | 8 | 否 | 否 | 否 | 0.19653 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_log0.5 | gaussian256 | 8 | 否 | 否 | 否 | 0.19734 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_soft0.2 | gaussian256 | 8 | 否 | 否 | 否 | 0.19871 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_log0.3 | gaussian256 | 8 | 否 | 否 | 否 | 0.19875 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_log0.2 | gaussian256 | 8 | 否 | 否 | 否 | 0.20030 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_soft0.5 | gaussian256 | 8 | 否 | 否 | 否 | 0.20290 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_log0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.20371 | FAILED_G1 |
| hs_b07_nonlinear_axes_global_soft0.2 | gaussian256 | 8 | 否 | 否 | 否 | 0.20458 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_log0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.20963 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_soft0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.22973 | FAILED_G1 |
| hs_b07_nonlinear_axes_ux_soft0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.23738 | FAILED_G1 |

## 证据与限制

- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。
- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。
- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。
- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。
- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。

精确别名 0 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。
