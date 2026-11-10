# 批量单指标的三鸿沟结果

已完成 116 个“指标 × 模型 × 推断 × 优化设置”案例；候选 ID 中的精确别名另列，不能作为独立发现计数。

所有下游测量均保留；上游失败时，下游结果只作诊断。每次更换 inference 都以独立分支记录，禁止拼接四模型结果。
初筛多起点成功需要 64 起点复核和新样本确认，才能声明最终成功。

| 推断 / 起点 | 案例数 | G1 通过 | G2 通过 | G3 数值达标 |
|---|---:|---:|---:|---:|
| gaussian256 / 8 | 116 | 95 | 85 | 23 |

## 同一单指标在四模型的结果

| 指标 | 推断 | 起点 | 四模型 G1 | 四模型 G2 | 四模型 G3 | 最大均值差 | 状态 |
|---|---|---:|---|---|---|---:|---|
| hs_b06_rank_degree1_edges_schatten05 | gaussian256 | 8 | 是 | 是 | 否 | 0.13317 | FAILED_G3 |
| hs_b06_rank_degree2_edges_soft_t0.03 | gaussian256 | 8 | 是 | 是 | 否 | 0.15303 | FAILED_G3 |
| hs_b06_rank_degree2_edges_schatten05 | gaussian256 | 8 | 是 | 是 | 否 | 0.15335 | FAILED_G3 |
| hs_b06_rank_degree2_ux_soft_t0.03 | gaussian256 | 8 | 是 | 是 | 否 | 0.17017 | FAILED_G3 |
| hs_b06_rank_degree1_edges_soft_t0.01 | gaussian256 | 8 | 是 | 是 | 否 | 0.18564 | FAILED_G3 |
| hs_b06_rank_degree2_edges_soft_t0.01 | gaussian256 | 8 | 是 | 是 | 否 | 0.21539 | FAILED_G3 |
| hs_b06_rank_degree1_edges_soft_t0.03 | gaussian256 | 8 | 是 | 否 | 否 | 0.14102 | FAILED_G2 |
| hs_b06_rank_degree1_ux_soft_t0.03 | gaussian256 | 8 | 是 | 否 | 否 | 0.14257 | FAILED_G2 |
| hs_b06_rank_degree2_ux_schatten05 | gaussian256 | 8 | 是 | 否 | 否 | 0.14760 | FAILED_G2 |
| hs_ux_rms_soft003 | gaussian256 | 8 | 是 | 否 | 否 | 0.17515 | FAILED_G2 |
| hs_b06_rank_degree2_ux_soft_t0.01 | gaussian256 | 8 | 是 | 否 | 否 | 0.18592 | FAILED_G2 |
| hs_b06_rank_degree1_ux_soft_t0.01 | gaussian256 | 8 | 是 | 否 | 否 | 0.23801 | FAILED_G2 |
| hs_b06_rank_degree1_ux_log01 | gaussian256 | 8 | 否 | 否 | 否 | 0.12846 | FAILED_G1 |
| hs_b06_rank_degree2_ux_log01 | gaussian256 | 8 | 否 | 否 | 否 | 0.12892 | FAILED_G1 |
| hs_b06_rank_degree2_ux_soft_t0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13009 | FAILED_G1 |
| hs_b06_rank_degree1_edges_log01 | gaussian256 | 8 | 否 | 否 | 否 | 0.13052 | FAILED_G1 |
| hs_b06_rank_degree1_ux_schatten1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13082 | FAILED_G1 |
| hs_b06_rank_degree2_edges_soft_t0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13087 | FAILED_G1 |
| hs_b06_rank_degree1_ux_soft_t0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13089 | FAILED_G1 |
| hs_b06_rank_degree1_edges_soft_t0.1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13147 | FAILED_G1 |
| hs_b06_rank_degree1_ux_schatten05 | gaussian256 | 8 | 否 | 否 | 否 | 0.13200 | FAILED_G1 |
| hs_b06_rank_degree2_edges_log01 | gaussian256 | 8 | 否 | 否 | 否 | 0.13206 | FAILED_G1 |
| hs_b06_rank_degree1_edges_schatten1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13226 | FAILED_G1 |
| hs_b06_rank_degree2_ux_schatten1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13526 | FAILED_G1 |
| hs_b06_rank_degree2_edges_schatten1 | gaussian256 | 8 | 否 | 否 | 否 | 0.13665 | FAILED_G1 |
| hs_b06_rank_degree2_ux_soft_t0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.25966 | FAILED_G1 |
| hs_b06_rank_degree2_edges_soft_t0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.26690 | FAILED_G1 |
| hs_b06_rank_degree1_ux_soft_t0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.27232 | FAILED_G1 |
| hs_b06_rank_degree1_edges_soft_t0.003 | gaussian256 | 8 | 否 | 否 | 否 | 0.28344 | FAILED_G1 |

## 证据与限制

- G1/G2 的 PLANE_COUNTEREXAMPLE 表示已有平面反例，没有将该角度冒充全空间最优误差。FLAT_UNRESOLVED 表示不可辨识方向。
- G2 还检查相同 W 上的 loss 曲线误差与最低平面相关；预测拟合好不等于指标推断好。
- G3 每个旋转先写入独立锁定文件，随后才读取真实参数作 TEST 评价。
- G3 的通过要求相对各自监督旋转参考的均值差≤.01、最大错配≤参考+.02，四模型一致。
- TEST 已用于探索性比较，最终候选仍需冻结方案后用独立新样本确认。

精确别名 0 个，见 candidates.json。全部模型级数据见 [cases.csv](cases.csv)。
