# 批量单指标的三鸿沟结果

已完成 132 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。

所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。
初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。

| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |
|---|---:|---:|---:|---:|
| gaussian256 / 8 | 132 | 110 | 97 | 35 |

## 同一单指标在四模型的结果

| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |
|---|---|---:|---|---|---|---:|---|
| hs_b04_subject_edges_global_soft0.03 | gaussian256 | 8 | 是 | 是 | 否 | 0.18990 | FAILED_G3 |
| hs_b04_central0.5_edges_soft0.02 | gaussian256 | 8 | 是 | 是 | 否 | 0.20456 | FAILED_G3 |
| hs_b04_central0.9_edges_soft0.02 | gaussian256 | 8 | 是 | 是 | 否 | 0.20545 | FAILED_G3 |
| hs_b04_central0.75_edges_soft0.02 | gaussian256 | 8 | 是 | 是 | 否 | 0.20685 | FAILED_G3 |
| hs_b04_central0.25_edges_soft0.02 | gaussian256 | 8 | 是 | 是 | 否 | 0.20810 | FAILED_G3 |
| hs_b04_subject_edges_global_soft0.01 | gaussian256 | 8 | 是 | 是 | 否 | 0.26751 | FAILED_G3 |
| hs_b04_central0.5_ux_soft0.05 | gaussian256 | 8 | 是 | 否 | 否 | 0.16450 | FAILED_G2 |
| hs_b04_central0.25_ux_soft0.05 | gaussian256 | 8 | 是 | 否 | 否 | 0.16459 | FAILED_G2 |
| hs_b04_central0.75_ux_soft0.05 | gaussian256 | 8 | 是 | 否 | 否 | 0.16689 | FAILED_G2 |
| hs_b04_central0.9_ux_soft0.05 | gaussian256 | 8 | 是 | 否 | 否 | 0.16897 | FAILED_G2 |
| hs_b04_subject_ux_global_soft0.03 | gaussian256 | 8 | 是 | 否 | 否 | 0.17365 | FAILED_G2 |
| hs_b04_subject_ux_subject_soft0.03 | gaussian256 | 8 | 是 | 否 | 否 | 0.17447 | FAILED_G2 |
| hs_ux_rms_soft003 | gaussian256 | 8 | 是 | 否 | 否 | 0.17515 | FAILED_G2 |
| hs_b04_subject_ux_global_soft0.01 | gaussian256 | 8 | 是 | 否 | 否 | 0.17811 | FAILED_G2 |
| hs_b04_subject_ux_subject_soft0.01 | gaussian256 | 8 | 是 | 否 | 否 | 0.17820 | FAILED_G2 |
| hs_b04_subject_edges_subject_soft0.03 | gaussian256 | 8 | 是 | 否 | 否 | 0.18989 | FAILED_G2 |
| hs_b04_subject_edges_subject_soft0.01 | gaussian256 | 8 | 是 | 否 | 否 | 0.26528 | FAILED_G2 |
| hs_b04_subject_ux_global_soft0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13504 | FAILED_G1 |
| hs_b04_subject_edges_global_soft0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13510 | FAILED_G1 |
| hs_b04_subject_edges_subject_soft0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13547 | FAILED_G1 |
| hs_b04_subject_ux_subject_soft0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13559 | FAILED_G1 |
| hs_b04_central0.5_ux_soft0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.17200 | FAILED_G1 |
| hs_b04_central0.25_ux_soft0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.17290 | FAILED_G1 |
| hs_b04_central0.75_ux_soft0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.17420 | FAILED_G1 |
| hs_b04_central0.9_ux_soft0.03 | gaussian256 | 8 | 否 | 否 | 否 | 0.17577 | FAILED_G1 |
| hs_b04_central0.5_ux_soft0.01 | gaussian256 | 8 | 否 | 否 | 否 | 0.17640 | FAILED_G1 |
| hs_b04_central0.25_ux_soft0.01 | gaussian256 | 8 | 否 | 否 | 否 | 0.17762 | FAILED_G1 |
| hs_b04_central0.75_ux_soft0.01 | gaussian256 | 8 | 否 | 否 | 否 | 0.17831 | FAILED_G1 |
| hs_b04_subject_ux_subject_soft0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.17865 | FAILED_G1 |
| hs_b04_subject_ux_global_soft0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.17876 | FAILED_G1 |
| hs_b04_central0.9_ux_soft0.01 | gaussian256 | 8 | 否 | 否 | 否 | 0.17942 | FAILED_G1 |
| hs_b04_subject_edges_subject_soft0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.25600 | FAILED_G1 |
| hs_b04_subject_edges_global_soft0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.25607 | FAILED_G1 |

## 证据与限制

- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。
- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。
- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。
- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。
- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。

精确别名 0 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。
