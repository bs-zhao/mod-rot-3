# 统一 Hessian 指标：逐关研究台账

当前研究未完成。旧 transport 结果不在本任务范围。用户随后接受约5.8°，当前 G1 初筛按最差轴误差≤6°；5°原始诊断保留。组合必须先有单项互补证据。
完整条目见 [runs.csv](runs.csv) 和 [gates.json](gates.json)。旧 exact-zero 预筛与无依据组合单列，不计入当前通过证据。

G2 固定门槛：所有平面 loss 曲线相对 L2 误差≤0.10、每平面相关≥0.90、最差轴误差≤5°；字段误差单独报告。
G3 预定比较：平均匹配相关与监督参考差≤0.01，最大错配不超过对应参考+0.02；原 TEST 为探索性，最终必须另做冻结后的新被试确认。

## 状态计数

- EXPLORATORY_UNMOTIVATED_COMBINATION: 10
- FAILED_G1: 98
- FAILED_G2: 5
- FAILED_G3: 3
- NOT_ASSESSED: 1
- PASSED_G1: 33
- PASSED_G2: 18
- PASSED_G3: 9
- SUPERSEDED_EXACT_AXIS_RULE: 296

## 当前候选

| 指标 | 通道 | seed | 状态 | G1 已通过模型 | G2 已通过模型 | G3 已通过模型 |
|---|---|---:|---|---|---|---|
| hs_full_rms_soft003 | logit | 20261128 | STOPPED_G1 | C1, C2, C3 |  |  |
| hs_jd_mean_profile1_strength0 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_jd_profile1_strength0_repeat_w90 | logit | 20261128 | STOPPED_G1 | C1, C2, C3 |  |  |
| hs_jd_profile1_strength0_selfcap03_w20 | logit | 20261128 | STOPPED_G3 | C1, C2, C3, CR3 | C1, C2, C3, CR3 | C1, C2, C3 |
| hs_jd_profile1_strength0_selfcap03_w9 | logit | 20261128 | STOPPED_G1 | C1, C2, C3 |  |  |
| hs_jd_profile1_strength0_selfl1_w0 | logit | 20261127 | STOPPED_G1 | C1, C2, C3 |  |  |
| hs_jd_profile1_strength0_selfl1_w0 | probability | 20261129 | STOPPED_G1 | C1, C2, C3 |  |  |
| hs_jd_profile1_strength0_selfl1_w8 | logit | 20261128 | STOPPED_G2 | C1, C2, C3, CR3 | C1, C2, C3 |  |
| hs_jd_profile1_strength0_selfl1_w8 | logit | 20261128 | STOPPED_G3 | C1, C2, C3, CR3 | C1, C2, C3, CR3 | C1, C2, C3 |
| hs_jd_profile1_strength0_selfl1_w8 | logit | 20261128 | STOPPED_G3 | C1, C2, C3, CR3 | C1, C2, C3, CR3 | C1, C2, C3 |
| hs_jd_profile1_strength0_selfsoft003_w0 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_modulation_cap_t0.03 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_modulation_cap_t0.1 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_modulation_log_t0.03 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_modulation_soft_t0.03 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_self_cap_t0.03 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_self_cap_t0.1 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_self_log_t0.03 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_nonlinearity_self_soft_t0.03 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_uu_graph_entries_cap_t0.03 | logit | 20261128 | STOPPED_G1 |  |  |  |
| hs_uu_graph_entries_log_t0.03 | logit | 20261128 | STOPPED_G1 |  |  |  |
| hs_uu_graph_entries_soft_t0.03 | logit | 20261128 | STOPPED_G1 |  |  |  |
| hs_uu_graph_rows_cap_t0.03 | logit | 20261128 | STOPPED_G1 |  |  |  |
| hs_uu_graph_rows_log_t0.03 | logit | 20261128 | STOPPED_G1 |  |  |  |
| hs_uu_graph_rows_soft_t0.03 | logit | 20261128 | STOPPED_G1 |  |  |  |
| hs_uu_off_rms_l1 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_uu_off_rms_l1 | probability | 20261129 | STOPPED_G1 |  |  |  |
| hs_uu_off_rms_soft01 | probability | 20261129 | STOPPED_G1 |  |  |  |
| hs_uu_self_local_l1 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_uu_self_local_l1 | probability | 20261129 | STOPPED_G1 |  |  |  |
| hs_uu_third_mixed_l1 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_uu_third_repeat_l1 | logit | 20261127 | STOPPED_G1 |  |  |  |
| hs_uu_third_repeat_l1 | probability | 20261128 | STOPPED_G1 |  |  |  |
| hs_ux_profile1_l1 | logit | 20261127 | STOPPED_G1 | C1, C2, C3 |  |  |
| hs_ux_rms_l1 | logit | 20261127 | STOPPED_G1 | C1, C2, C3 |  |  |
| hs_ux_rms_soft003 | logit | 20261128 | STOPPED_G2 | C1, C2, C3, CR3 | C1, C2, C3 |  |
| hs_ux_rms_soft003 | logit | 20261128 | STOPPED_G2 | C1, C2, C3, CR3 |  |  |
| hs_ux_rms_soft003 | logit | 20261128 | STOPPED_G2 | C1, C2, C3, CR3 |  |  |
| hs_ux_rms_soft003 | logit | 20261128 | STOPPED_G2 | C1, C2, C3, CR3 |  |  |
