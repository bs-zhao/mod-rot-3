# 批量单指标的三鸿沟结果

已完成 100 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。

所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。
初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。

| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |
|---|---:|---:|---:|---:|
| gaussian256 / 8 | 100 | 72 | 71 | 29 |

## 同一单指标在四模型的结果

| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |
|---|---|---:|---|---|---|---:|---|
| hs_ux_rms_soft003 | gaussian256 | 8 | 是 | 否 | 否 | 0.17515 | FAILED_G2 |
| hs_b03_operator_functional1_center1_raw_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.13418 | FAILED_G1 |
| hs_b03_operator_functional1_center0_raw_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.13429 | FAILED_G1 |
| hs_b03_operator_functional1_center0_trace_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.13492 | FAILED_G1 |
| hs_b03_operator_xcolumns_center0_raw_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13612 | FAILED_G1 |
| hs_b03_operator_functional1_center1_raw_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.13672 | FAILED_G1 |
| hs_b03_operator_functional1_center0_raw_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.13681 | FAILED_G1 |
| hs_b03_operator_xcolumns_center1_raw_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13713 | FAILED_G1 |
| hs_b03_operator_functional1_center1_trace_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.14188 | FAILED_G1 |
| hs_b03_operator_functional1_center0_trace_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.14250 | FAILED_G1 |
| hs_b03_operator_functional1_center0_trace_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.14866 | FAILED_G1 |
| hs_b03_operator_functional1_center0_raw_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.14884 | FAILED_G1 |
| hs_b03_operator_xcolumns_center1_raw_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.15219 | FAILED_G1 |
| hs_b03_operator_xcolumns_center1_raw_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.15323 | FAILED_G1 |
| hs_b03_operator_xcolumns_center0_raw_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.15327 | FAILED_G1 |
| hs_b03_operator_functional1_center1_trace_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.15346 | FAILED_G1 |
| hs_b03_operator_xcolumns_center0_raw_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.15426 | FAILED_G1 |
| hs_b03_operator_functional1_center1_raw_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.15731 | FAILED_G1 |
| hs_b03_operator_xcolumns_center0_trace_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.16101 | FAILED_G1 |
| hs_b03_operator_xcolumns_center1_trace_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.16411 | FAILED_G1 |
| hs_b03_operator_functional1_center1_trace_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.17305 | FAILED_G1 |
| hs_b03_operator_xcolumns_center1_trace_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.18722 | FAILED_G1 |
| hs_b03_operator_xcolumns_center0_trace_relative | gaussian256 | 8 | 否 | 否 | 否 | 0.18963 | FAILED_G1 |
| hs_b03_operator_xcolumns_center1_trace_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.19247 | FAILED_G1 |
| hs_b03_operator_xcolumns_center0_trace_l2sq | gaussian256 | 8 | 否 | 否 | 否 | 0.19367 | FAILED_G1 |

## 证据与限制

- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。
- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。
- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。
- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。
- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。

精确别名 0 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。
