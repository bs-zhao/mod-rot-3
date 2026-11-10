# 批量单指标的三鸿沟结果

已完成 124 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。

所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。
初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。

| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |
|---|---:|---:|---:|---:|
| gaussian256 / 8 | 80 | 63 | 53 | 22 |
| gaussian256 / 64 | 4 | 4 | 4 | 1 |
| gaussian512d5 / 8 | 20 | 11 | 8 | 0 |
| zrbf / 8 | 20 | 11 | 1 | 0 |

## 同一单指标在四模型的结果

| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |
|---|---|---:|---|---|---|---:|---|
| hs_b02_unique_edges_soft_t0.02 | gaussian256 | 64 | 是 | 是 | 否 | 0.20992 | FAILED_G3 |
| hs_b02_unique_edges_soft_t0.02 | gaussian256 | 8 | 是 | 是 | 否 | 0.20992 | FAILED_G3 |
| hs_b02_unique_edges_entropy | gaussian256 | 8 | 是 | 否 | 否 | 0.13866 | FAILED_G2 |
| hs_b02_unique_edges_lp05 | gaussian256 | 8 | 是 | 否 | 否 | 0.14233 | FAILED_G2 |
| hs_b02_unique_edges_soft_t0.07 | gaussian256 | 8 | 是 | 否 | 否 | 0.14415 | FAILED_G2 |
| hs_b02_unique_edges_gini | gaussian256 | 8 | 是 | 否 | 否 | 0.14645 | FAILED_G2 |
| hs_b02_unique_edges_lp_t0.25 | gaussian256 | 8 | 是 | 否 | 否 | 0.14711 | FAILED_G2 |
| hs_b02_unique_edges_soft_t0.05 | gaussian256 | 8 | 是 | 否 | 否 | 0.15465 | FAILED_G2 |
| hs_ux_rms_soft003 | gaussian256 | 8 | 是 | 否 | 否 | 0.17515 | FAILED_G2 |
| hs_b02_unique_edges_soft_t0.03 | gaussian256 | 8 | 是 | 否 | 否 | 0.18970 | FAILED_G2 |
| hs_b02_unique_edges_soft_t0.01 | gaussian256 | 8 | 是 | 否 | 否 | 0.27636 | FAILED_G2 |
| hs_b02_unique_edges_kurtosis | gaussian256 | 8 | 否 | 否 | 否 | 0.11005 | FAILED_G1 |
| hs_b02_unique_edges_soft_t0.2 | gaussian256 | 8 | 否 | 否 | 否 | 0.12995 | FAILED_G1 |
| hs_b02_unique_edges_hoyer | gaussian256 | 8 | 否 | 否 | 否 | 0.13146 | FAILED_G1 |
| hs_b02_unique_edges_soft_t0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13724 | FAILED_G1 |
| hs_b02_unique_edges_log01 | gaussian256 | 8 | 否 | 否 | 否 | 0.13941 | FAILED_G1 |
| hs_b02_unique_edges_lp_t0.75 | gaussian256 | 8 | 否 | 否 | 否 | 0.14019 | FAILED_G1 |
| hs_b02_unique_edges_l1 | gaussian256 | 8 | 否 | 否 | 否 | 0.14082 | FAILED_G1 |
| hs_b02_unique_edges_cap01 | gaussian256 | 8 | 否 | 否 | 否 | 0.16074 | FAILED_G1 |
| hs_b02_unique_edges_soft_t0.001 | gaussian256 | 8 | 否 | 否 | 否 | 0.26816 | FAILED_G1 |
| hs_b02_unique_edges_soft_t0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.26946 | FAILED_G1 |
| hs_b02_unique_edges_kurtosis | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.10980 | PENDING |
| hs_b02_unique_edges_kurtosis | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.12025 | PENDING |
| hs_b02_unique_edges_soft_t0.2 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.12046 | PENDING |
| hs_b02_unique_edges_hoyer | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.12164 | PENDING |
| hs_b02_unique_edges_soft_t0.2 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.13045 | PENDING |
| hs_b02_unique_edges_hoyer | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.13269 | PENDING |
| hs_b02_unique_edges_l1 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.13323 | PENDING |
| hs_b02_unique_edges_lp_t0.75 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.13411 | PENDING |
| hs_b02_unique_edges_log01 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.13432 | PENDING |
| hs_b02_unique_edges_soft_t0.1 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.13437 | PENDING |
| hs_b02_unique_edges_entropy | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.13500 | PENDING |
| hs_b02_unique_edges_lp05 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.13876 | PENDING |
| hs_b02_unique_edges_soft_t0.1 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.13946 | PENDING |
| hs_b02_unique_edges_entropy | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.14089 | PENDING |
| hs_b02_unique_edges_log01 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.14190 | PENDING |
| hs_b02_unique_edges_lp_t0.75 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.14263 | PENDING |
| hs_b02_unique_edges_l1 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.14310 | PENDING |
| hs_b02_unique_edges_lp05 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.14501 | PENDING |
| hs_b02_unique_edges_lp_t0.25 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.14607 | PENDING |
| hs_b02_unique_edges_lp_t0.25 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.14943 | PENDING |
| hs_b02_unique_edges_soft_t0.07 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.14957 | PENDING |
| hs_b02_unique_edges_gini | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.15007 | PENDING |
| hs_b02_unique_edges_soft_t0.07 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.15069 | PENDING |
| hs_b02_unique_edges_gini | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.15727 | PENDING |
| hs_b02_unique_edges_cap01 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.15936 | PENDING |
| hs_b02_unique_edges_cap01 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.16127 | PENDING |
| hs_ux_rms_soft003 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.16400 | PENDING |
| hs_b02_unique_edges_soft_t0.05 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.16448 | PENDING |
| hs_b02_unique_edges_soft_t0.05 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.16799 | PENDING |
| hs_ux_rms_soft003 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.17517 | PENDING |
| hs_b02_unique_edges_soft_t0.03 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.18162 | PENDING |
| hs_b02_unique_edges_soft_t0.03 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.19639 | PENDING |
| hs_b02_unique_edges_soft_t0.02 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.22293 | PENDING |
| hs_b02_unique_edges_soft_t0.01 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.24183 | PENDING |
| hs_b02_unique_edges_soft_t0.02 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.24574 | PENDING |
| hs_b02_unique_edges_soft_t0.01 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.24615 | PENDING |
| hs_b02_unique_edges_soft_t0.003 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.24698 | PENDING |
| hs_b02_unique_edges_soft_t0.001 | gaussian512d5 | 8 | 未齐 | 未齐 | 未齐 | 0.24829 | PENDING |
| hs_b02_unique_edges_soft_t0.003 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.25192 | PENDING |
| hs_b02_unique_edges_soft_t0.001 | zrbf | 8 | 未齐 | 未齐 | 未齐 | 0.25265 | PENDING |

## 证据与限制

- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。
- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。
- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。
- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。
- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。

精确别名 0 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。
