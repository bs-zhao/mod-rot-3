# 实测结果与实施边界

本页由 `report.py` 从实际 CSV/完成标记生成。`pilot` 仅检查代码通路，排除正式汇总。所有四模型 TEST 都已被历史工作查看，本页属于探索性结果。

## EXP13 冻结与原入口复现

| model | repeat | mcc | leakage | max_leakage |
|---|---|---|---|---|
| C1 | 1 | 0.9688 | 0.0212 | 0.0576 |
| C1 | 2 | 0.9689 | 0.0202 | 0.0611 |
| C2 | 1 | 0.9645 | 0.0599 | 0.1523 |
| C2 | 2 | 0.9677 | 0.0427 | 0.1450 |
| C3 | 1 | 0.9600 | 0.0464 | 0.1143 |
| C3 | 2 | 0.9615 | 0.0478 | 0.1525 |
| CR3 | 1 | 0.8173 | 0.1646 | 0.3677 |
| CR3 | 2 | 0.8122 | 0.1832 | 0.3781 |

原 float32 优化器、24×250、Haar 200 seed 1、起点 seed 7；归档 manifest 见 `baseline/results/archive_manifest.json`。

## 六条路线的实际覆盖

| experiment | tag | completed_configs | in_progress_configs | candidate_rows |
|---|---|---|---|---|
| exp14_probability | full | 1 | 0 | 3 |
| exp14_probability | mechanism_v1 | 16 | 0 | 544 |
| exp15_finite_scale | matched_limit | 16 | 0 | 64 |
| exp15_finite_scale | round1 | 14 | 1 | 72 |
| exp16_local_context | context_full_v2 | 32 | 0 | 352 |
| exp16_local_context | local_v1 | 1 | 0 | 11 |
| exp16_local_context | round1 | 16 | 0 | 176 |
| exp17_reliability | round1 | 16 | 0 | 208 |
| exp18_matrix_only | estimator_v1 | 1 | 0 | 9 |
| exp18_matrix_only | round1 | 23 | 0 | 207 |
| exp19_model_library | frozen_holdout_v2 | 1 | 0 | 24 |
| exp19_model_library | grammar_v1 | 1 | 0 | 30 |
| exp19_model_library | np_transfer_v1 | 16 | 0 | 16 |


## 完整 tuning 网格（探索性，不按 TEST 选最终方案）

| model | repeat | channel | candidates | completed | mcc_min | mcc_max |
|---|---|---|---|---|---|---|
| C1 | 1 | logit | 88 | True | 0.7836 | 0.9687 |
| C1 | 1 | prob | 88 | True | 0.9686 | 0.9688 |
| C1 | 2 | logit | 88 | True | 0.8923 | 0.9693 |
| C1 | 2 | prob | 88 | True | 0.9685 | 0.9693 |
| C2 | 1 | logit | 88 | True | 0.8081 | 0.9712 |
| C2 | 1 | prob | 88 | True | 0.8304 | 0.9710 |
| C2 | 2 | logit | 88 | True | 0.7146 | 0.9705 |
| C2 | 2 | prob | 88 | True | 0.8611 | 0.9694 |
| C3 | 1 | logit | 88 | True | 0.8152 | 0.9624 |
| C3 | 1 | prob | 88 | True | 0.8740 | 0.9616 |
| C3 | 2 | logit | 88 | True | 0.7429 | 0.9646 |
| C3 | 2 | prob | 88 | True | 0.7180 | 0.9643 |
| CR3 | 1 | prob | 18 | False | 0.6652 | 0.8173 |

MCC 范围描述全部预设候选，不是无监督选择规则；完整逐项结果见对应 baseline CSV。

## 可用 checkpoint 的 beta / PCA / X 单位敏感性

| model | repeat | channel | candidates | completed | mcc_min | mcc_max |
|---|---|---|---|---|---|---|
| C1 | 1 | logit | 3 | True | 0.8859 | 0.9243 |
| C1 | 1 | prob | 3 | True | 0.9688 | 0.9688 |
| C1 | 2 | logit | 3 | True | 0.9266 | 0.9456 |
| C1 | 2 | prob | 3 | True | 0.9689 | 0.9690 |
| C2 | 1 | logit | 3 | True | 0.8904 | 0.9155 |
| C2 | 1 | prob | 3 | True | 0.9597 | 0.9654 |
| C2 | 2 | logit | 3 | True | 0.8715 | 0.8959 |
| C2 | 2 | prob | 3 | True | 0.9676 | 0.9682 |
| C3 | 1 | logit | 3 | True | 0.8570 | 0.9116 |
| C3 | 1 | prob | 3 | True | 0.9536 | 0.9598 |
| C3 | 2 | logit | 3 | True | 0.8162 | 0.8310 |
| C3 | 2 | prob | 3 | True | 0.8747 | 0.9619 |
| CR3 | 1 | logit | 5 | True | 0.6437 | 0.8310 |
| CR3 | 1 | prob | 5 | True | 0.6401 | 0.8170 |
| CR3 | 2 | logit | 3 | True | 0.6475 | 0.8088 |
| CR3 | 2 | prob | 3 | True | 0.7088 | 0.8118 |

MCC 范围描述全部预设候选，不是无监督选择规则；完整逐项结果见对应 baseline CSV。

## exp14_probability

CR3 全候选（所有方法按预设规则搜索后评价，未按 TEST MCC 选解）：

| tag | repeat | channel | evidence | method | mcc | leakage | max_leakage |
|---|---|---|---|---|---|---|---|
| full | 1 | prob | CR3_rep1_decoder | AUX_BUU | 0.7760 | 0.2019 | 0.5475 |
| full | 1 | prob | CR3_rep1_decoder | B__combo | 0.8057 | 0.1944 | 0.3622 |
| full | 1 | prob | CR3_rep1_decoder | p__combo | 0.8173 | 0.1646 | 0.3677 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | AUX_BUU | 0.8227 | 0.2514 | 0.5936 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | AUX_etaUU | 0.9041 | 0.1568 | 0.5902 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | AUX_pUU | 0.7342 | 0.3370 | 0.6957 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | A__UU | 0.8783 | 0.1377 | 0.6757 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | A__UX | 0.8236 | 0.2151 | 0.7232 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | A__combo | 0.8882 | 0.1673 | 0.6242 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | A_plus_0.25B | 0.7925 | 0.2977 | 0.5858 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | A_plus_0.5B | 0.7825 | 0.3057 | 0.5932 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | A_plus_2B | 0.7870 | 0.3157 | 0.6138 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | A_plus_4B | 0.7701 | 0.3272 | 0.5959 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | BUX_AUU | 0.8628 | 0.2463 | 0.5316 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | BUX_etaUU | 0.8415 | 0.2594 | 0.5513 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | BUX_pUU | 0.8019 | 0.2990 | 0.5807 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | B__UU | 0.7751 | 0.3302 | 0.5702 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | B__UX | 0.7909 | 0.3113 | 0.5867 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | B__combo | 0.7099 | 0.3599 | 0.6795 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | asinh | 0.7934 | 0.3035 | 0.5757 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | etaUX_AUU | 0.8988 | 0.1557 | 0.6087 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | etaUX_BUU | 0.8165 | 0.2591 | 0.5923 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | etaUX_pUU | 0.7323 | 0.3371 | 0.6958 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | eta__UU | 0.8892 | 0.1335 | 0.6510 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | eta__UX | 0.8409 | 0.2035 | 0.7171 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | eta__combo | 0.9127 | 0.1502 | 0.5677 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | gg__combo | 0.7389 | 0.3394 | 0.6023 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | gradient_and_context | 0.7323 | 0.3342 | 0.6010 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | p1gg__combo | 0.6952 | 0.3723 | 0.6751 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | pUX_AUU | 0.8928 | 0.2191 | 0.4757 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | pUX_BUU | 0.7522 | 0.3111 | 0.6902 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | pUX_etaUU | 0.7946 | 0.2870 | 0.6307 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | p__UU | 0.9011 | 0.2042 | 0.4339 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | p__UX | 0.7873 | 0.3016 | 0.6317 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | p__combo | 0.7585 | 0.3234 | 0.6309 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | softplus | 0.8485 | 0.2088 | 0.6666 |
| mechanism_v1 | 1 | prob | CR3_rep1_analytic | tanh | 0.7944 | 0.2939 | 0.6403 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | AUX_BUU | 0.7760 | 0.2019 | 0.5475 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | AUX_etaUU | 0.8268 | 0.1660 | 0.3845 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | AUX_pUU | 0.8017 | 0.1836 | 0.4835 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | A__UU | 0.8069 | 0.1878 | 0.4007 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | A__UX | 0.7263 | 0.2431 | 0.6314 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | A__combo | 0.8190 | 0.1714 | 0.3911 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | A_plus_0.25B | 0.8108 | 0.1812 | 0.3830 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | A_plus_0.5B | 0.8086 | 0.1829 | 0.3868 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | A_plus_2B | 0.7633 | 0.2273 | 0.4702 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | A_plus_4B | 0.7528 | 0.2288 | 0.4829 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | BUX_AUU | 0.7616 | 0.2157 | 0.5114 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | BUX_etaUU | 0.7525 | 0.2148 | 0.5443 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | BUX_pUU | 0.7384 | 0.2287 | 0.5688 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | B__UU | 0.7953 | 0.1900 | 0.4769 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | B__UX | 0.6501 | 0.2693 | 0.6257 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | B__combo | 0.8057 | 0.1944 | 0.3622 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | asinh | 0.8136 | 0.1697 | 0.3647 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | etaUX_AUU | 0.8242 | 0.1641 | 0.4012 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | etaUX_BUU | 0.7788 | 0.1985 | 0.5365 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | etaUX_pUU | 0.8091 | 0.1730 | 0.4632 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | eta__UU | 0.8207 | 0.1794 | 0.3859 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | eta__UX | 0.6947 | 0.2573 | 0.6475 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | eta__combo | 0.8308 | 0.1597 | 0.3938 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | gg__combo | 0.7852 | 0.2097 | 0.4105 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | gradient_and_context | 0.6414 | 0.2759 | 0.6043 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | p1gg__combo | 0.7358 | 0.2184 | 0.5456 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | pUX_AUU | 0.8129 | 0.1836 | 0.3699 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | pUX_BUU | 0.7955 | 0.1911 | 0.4004 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | pUX_etaUU | 0.8106 | 0.1844 | 0.3492 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | p__UU | 0.7589 | 0.2313 | 0.5232 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | p__UX | 0.7301 | 0.2415 | 0.6738 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | p__combo | 0.8173 | 0.1646 | 0.3677 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | softplus | 0.8284 | 0.1556 | 0.3909 |
| mechanism_v1 | 1 | prob | CR3_rep1_decoder | tanh | 0.6749 | 0.2855 | 0.5402 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | AUX_BUU | 0.8227 | 0.2514 | 0.5936 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | AUX_etaUU | 0.9041 | 0.1568 | 0.5902 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | AUX_pUU | 0.7342 | 0.3370 | 0.6957 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | A__UU | 0.8783 | 0.1377 | 0.6757 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | A__UX | 0.8236 | 0.2151 | 0.7232 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | A__combo | 0.8882 | 0.1673 | 0.6242 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | A_plus_0.25B | 0.7925 | 0.2977 | 0.5858 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | A_plus_0.5B | 0.7825 | 0.3057 | 0.5932 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | A_plus_2B | 0.7870 | 0.3157 | 0.6138 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | A_plus_4B | 0.7701 | 0.3272 | 0.5959 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | BUX_AUU | 0.8628 | 0.2463 | 0.5316 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | BUX_etaUU | 0.8415 | 0.2594 | 0.5513 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | BUX_pUU | 0.8019 | 0.2990 | 0.5807 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | B__UU | 0.7751 | 0.3302 | 0.5702 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | B__UX | 0.7909 | 0.3113 | 0.5867 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | B__combo | 0.7099 | 0.3599 | 0.6795 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | asinh | 0.7934 | 0.3035 | 0.5757 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | etaUX_AUU | 0.8988 | 0.1557 | 0.6087 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | etaUX_BUU | 0.8165 | 0.2591 | 0.5923 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | etaUX_pUU | 0.7323 | 0.3371 | 0.6958 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | eta__UU | 0.8892 | 0.1335 | 0.6510 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | eta__UX | 0.8409 | 0.2035 | 0.7171 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | eta__combo | 0.9127 | 0.1502 | 0.5677 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | gg__combo | 0.7389 | 0.3394 | 0.6023 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | gradient_and_context | 0.7323 | 0.3342 | 0.6010 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | p1gg__combo | 0.6952 | 0.3723 | 0.6751 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | pUX_AUU | 0.8928 | 0.2191 | 0.4757 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | pUX_BUU | 0.7522 | 0.3111 | 0.6902 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | pUX_etaUU | 0.7946 | 0.2870 | 0.6307 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | p__UU | 0.9011 | 0.2042 | 0.4339 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | p__UX | 0.7873 | 0.3016 | 0.6317 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | p__combo | 0.7585 | 0.3234 | 0.6309 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | softplus | 0.8485 | 0.2088 | 0.6666 |
| mechanism_v1 | 2 | prob | CR3_rep2_analytic | tanh | 0.7944 | 0.2939 | 0.6403 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | AUX_BUU | 0.7849 | 0.2044 | 0.5061 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | AUX_etaUU | 0.8095 | 0.1921 | 0.3644 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | AUX_pUU | 0.8053 | 0.1878 | 0.4316 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | A__UU | 0.8013 | 0.1888 | 0.3786 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | A__UX | 0.7722 | 0.1910 | 0.6046 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | A__combo | 0.8130 | 0.1878 | 0.3536 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | A_plus_0.25B | 0.8103 | 0.1887 | 0.3683 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | A_plus_0.5B | 0.8091 | 0.1875 | 0.3773 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | A_plus_2B | 0.7242 | 0.2594 | 0.4784 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | A_plus_4B | 0.7019 | 0.2545 | 0.6011 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | BUX_AUU | 0.7964 | 0.1917 | 0.3959 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | BUX_etaUU | 0.7860 | 0.1990 | 0.4184 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | BUX_pUU | 0.7683 | 0.2133 | 0.4832 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | B__UU | 0.7907 | 0.1960 | 0.4385 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | B__UX | 0.7016 | 0.2533 | 0.5749 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | B__combo | 0.7831 | 0.2234 | 0.3757 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | asinh | 0.8027 | 0.1925 | 0.4164 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | etaUX_AUU | 0.8134 | 0.1852 | 0.3642 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | etaUX_BUU | 0.7857 | 0.2016 | 0.4994 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | etaUX_pUU | 0.8060 | 0.1855 | 0.4279 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | eta__UU | 0.7948 | 0.1927 | 0.4124 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | eta__UX | 0.7243 | 0.2425 | 0.6165 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | eta__combo | 0.8101 | 0.1906 | 0.3757 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | gg__combo | 0.7187 | 0.2444 | 0.5755 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | gradient_and_context | 0.6697 | 0.2672 | 0.5562 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | p1gg__combo | 0.7658 | 0.2114 | 0.4978 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | pUX_AUU | 0.7981 | 0.1989 | 0.4053 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | pUX_BUU | 0.8001 | 0.1830 | 0.3931 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | pUX_etaUU | 0.7946 | 0.2007 | 0.4198 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | p__UU | 0.7863 | 0.1918 | 0.5127 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | p__UX | 0.7328 | 0.2419 | 0.6630 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | p__combo | 0.8122 | 0.1832 | 0.3781 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | softplus | 0.8156 | 0.1874 | 0.3407 |
| mechanism_v1 | 2 | prob | CR3_rep2_decoder | tanh | 0.6634 | 0.2829 | 0.5622 |


## exp15_finite_scale

CR3 全候选（所有方法按预设规则搜索后评价，未按 TEST MCC 选解）：

| tag | repeat | channel | evidence | method | mcc | leakage | max_leakage |
|---|---|---|---|---|---|---|---|
| matched_limit | 1 | logit | CR3_rep1_logit | ambient_Adam | 0.7730 | 0.2148 | 0.5185 |
| matched_limit | 1 | logit | CR3_rep1_logit | ambient_Givens | 0.7742 | 0.2135 | 0.5161 |
| matched_limit | 1 | logit | CR3_rep1_logit | pair_Adam | 0.7439 | 0.2434 | 0.5489 |
| matched_limit | 1 | logit | CR3_rep1_logit | pair_Givens | 0.7444 | 0.2428 | 0.5492 |
| matched_limit | 1 | prob | CR3_rep1_prob | ambient_Adam | 0.7640 | 0.2215 | 0.4714 |
| matched_limit | 1 | prob | CR3_rep1_prob | ambient_Givens | 0.7632 | 0.2226 | 0.4709 |
| matched_limit | 1 | prob | CR3_rep1_prob | pair_Adam | 0.7100 | 0.2832 | 0.5400 |
| matched_limit | 1 | prob | CR3_rep1_prob | pair_Givens | 0.7093 | 0.2835 | 0.5413 |
| matched_limit | 2 | logit | CR3_rep2_logit | ambient_Adam | 0.7018 | 0.2634 | 0.5380 |
| matched_limit | 2 | logit | CR3_rep2_logit | ambient_Givens | 0.7014 | 0.2638 | 0.5387 |
| matched_limit | 2 | logit | CR3_rep2_logit | pair_Adam | 0.6968 | 0.2777 | 0.5423 |
| matched_limit | 2 | logit | CR3_rep2_logit | pair_Givens | 0.6932 | 0.2811 | 0.5579 |
| matched_limit | 2 | prob | CR3_rep2_prob | ambient_Adam | 0.6839 | 0.2717 | 0.5866 |
| matched_limit | 2 | prob | CR3_rep2_prob | ambient_Givens | 0.6815 | 0.2729 | 0.5864 |
| matched_limit | 2 | prob | CR3_rep2_prob | pair_Adam | 0.7103 | 0.2630 | 0.5017 |
| matched_limit | 2 | prob | CR3_rep2_prob | pair_Givens | 0.7107 | 0.2628 | 0.5021 |
| round1 | 1 | logit | CR3_rep1_logit | h0.1_combo | 0.7444 | 0.2428 | 0.5492 |
| round1 | 1 | logit | CR3_rep1_logit | h0.25_combo | 0.7426 | 0.2445 | 0.5495 |
| round1 | 1 | logit | CR3_rep1_logit | h0.5_combo | 0.7439 | 0.2421 | 0.5489 |
| round1 | 1 | logit | CR3_rep1_logit | h1.0_combo | 0.7430 | 0.2405 | 0.5522 |
| round1 | 1 | logit | CR3_rep1_logit | multiscale_equal | 0.7439 | 0.2425 | 0.5502 |
| round1 | 1 | prob | CR3_rep1_prob | h0.1_combo | 0.7093 | 0.2835 | 0.5413 |
| round1 | 1 | prob | CR3_rep1_prob | h0.25_combo | 0.7096 | 0.2837 | 0.5374 |
| round1 | 1 | prob | CR3_rep1_prob | h0.5_combo | 0.7114 | 0.2825 | 0.5291 |
| round1 | 1 | prob | CR3_rep1_prob | h1.0_combo | 0.7136 | 0.2813 | 0.5083 |
| round1 | 1 | prob | CR3_rep1_prob | multiscale_equal | 0.7112 | 0.2831 | 0.5280 |


## exp16_local_context

CR3 全候选（所有方法按预设规则搜索后评价，未按 TEST MCC 选解）：

| tag | repeat | channel | evidence | method | mcc | leakage | max_leakage |
|---|---|---|---|---|---|---|---|
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | Vregion_plus_share | 0.9172 | 0.1489 | 0.5510 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | cluster_plus_offdiag | 0.8515 | 0.1626 | 0.7184 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | cluster_plus_share | 0.9070 | 0.1629 | 0.5712 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | energy_weighted_UX | 0.8479 | 0.2182 | 0.6403 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | global_baseline | 0.9127 | 0.1502 | 0.5677 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | gradient_context | 0.8298 | 0.2359 | 0.6720 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | knn_plus_share | 0.8933 | 0.1843 | 0.5787 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | pointwise_UX | 0.8061 | 0.2456 | 0.6260 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | pointwise_plus_share | 0.8916 | 0.1858 | 0.5057 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | quantile_plus_share | 0.9092 | 0.1566 | 0.5744 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_analytic | weighted_plus_share | 0.9209 | 0.1590 | 0.4964 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | Vregion_plus_share | 0.8319 | 0.1581 | 0.3988 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | cluster_plus_offdiag | 0.6287 | 0.3007 | 0.6077 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | cluster_plus_share | 0.8280 | 0.1537 | 0.3750 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | energy_weighted_UX | 0.6420 | 0.2836 | 0.6342 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | global_baseline | 0.8308 | 0.1597 | 0.3938 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | gradient_context | 0.7084 | 0.2453 | 0.6761 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | knn_plus_share | 0.8313 | 0.1553 | 0.3874 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | pointwise_UX | 0.6079 | 0.3311 | 0.6625 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | pointwise_plus_share | 0.8092 | 0.1736 | 0.4624 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | quantile_plus_share | 0.8318 | 0.1567 | 0.3841 |
| context_full_v2 | 1 | logit | CR3_rep1_logit_decoder | weighted_plus_share | 0.7984 | 0.1908 | 0.5118 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | Vregion_plus_share | 0.7571 | 0.3245 | 0.6374 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | cluster_plus_offdiag | 0.8719 | 0.1905 | 0.6438 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | cluster_plus_share | 0.7038 | 0.3493 | 0.6792 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | energy_weighted_UX | 0.7009 | 0.3580 | 0.6892 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | global_baseline | 0.7585 | 0.3234 | 0.6309 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | gradient_context | 0.8241 | 0.2572 | 0.6544 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | knn_plus_share | 0.7071 | 0.3475 | 0.6919 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | pointwise_UX | 0.6197 | 0.3841 | 0.6369 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | pointwise_plus_share | 0.7141 | 0.3453 | 0.6128 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | quantile_plus_share | 0.7175 | 0.3431 | 0.6802 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_analytic | weighted_plus_share | 0.6921 | 0.3549 | 0.7121 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | Vregion_plus_share | 0.8221 | 0.1622 | 0.3777 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | cluster_plus_offdiag | 0.5984 | 0.3129 | 0.7123 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | cluster_plus_share | 0.7182 | 0.2351 | 0.6506 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | energy_weighted_UX | 0.5954 | 0.3234 | 0.6954 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | global_baseline | 0.8173 | 0.1646 | 0.3677 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | gradient_context | 0.6848 | 0.2751 | 0.6299 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | knn_plus_share | 0.8060 | 0.1811 | 0.3696 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | pointwise_UX | 0.6054 | 0.3085 | 0.7038 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | pointwise_plus_share | 0.7749 | 0.2234 | 0.4258 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | quantile_plus_share | 0.7495 | 0.2199 | 0.5724 |
| context_full_v2 | 1 | prob | CR3_rep1_prob_decoder | weighted_plus_share | 0.7531 | 0.2243 | 0.5328 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | Vregion_plus_share | 0.9172 | 0.1489 | 0.5510 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | cluster_plus_offdiag | 0.8515 | 0.1626 | 0.7184 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | cluster_plus_share | 0.9070 | 0.1629 | 0.5712 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | energy_weighted_UX | 0.8479 | 0.2182 | 0.6403 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | global_baseline | 0.9127 | 0.1502 | 0.5677 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | gradient_context | 0.8298 | 0.2359 | 0.6720 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | knn_plus_share | 0.8933 | 0.1843 | 0.5787 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | pointwise_UX | 0.8061 | 0.2456 | 0.6260 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | pointwise_plus_share | 0.8916 | 0.1858 | 0.5057 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | quantile_plus_share | 0.9092 | 0.1566 | 0.5744 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_analytic | weighted_plus_share | 0.9209 | 0.1590 | 0.4964 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | Vregion_plus_share | 0.8081 | 0.1923 | 0.3811 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | cluster_plus_offdiag | 0.6247 | 0.3093 | 0.5068 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | cluster_plus_share | 0.8154 | 0.1826 | 0.3865 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | energy_weighted_UX | 0.6277 | 0.2924 | 0.6660 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | global_baseline | 0.8101 | 0.1906 | 0.3757 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | gradient_context | 0.7244 | 0.2305 | 0.6447 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | knn_plus_share | 0.8105 | 0.1897 | 0.3848 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | pointwise_UX | 0.6314 | 0.3043 | 0.6375 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | pointwise_plus_share | 0.7286 | 0.2240 | 0.5106 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | quantile_plus_share | 0.8115 | 0.1901 | 0.3776 |
| context_full_v2 | 2 | logit | CR3_rep2_logit_decoder | weighted_plus_share | 0.6359 | 0.2862 | 0.6679 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | Vregion_plus_share | 0.7571 | 0.3245 | 0.6374 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | cluster_plus_offdiag | 0.8719 | 0.1905 | 0.6438 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | cluster_plus_share | 0.7038 | 0.3493 | 0.6792 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | energy_weighted_UX | 0.7009 | 0.3580 | 0.6892 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | global_baseline | 0.7585 | 0.3234 | 0.6309 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | gradient_context | 0.8241 | 0.2572 | 0.6544 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | knn_plus_share | 0.7071 | 0.3475 | 0.6919 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | pointwise_UX | 0.6197 | 0.3841 | 0.6369 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | pointwise_plus_share | 0.7141 | 0.3453 | 0.6128 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | quantile_plus_share | 0.7175 | 0.3431 | 0.6802 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_analytic | weighted_plus_share | 0.6921 | 0.3549 | 0.7121 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | Vregion_plus_share | 0.8108 | 0.1847 | 0.3841 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | cluster_plus_offdiag | 0.5677 | 0.3289 | 0.6322 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | cluster_plus_share | 0.7102 | 0.2524 | 0.6200 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | energy_weighted_UX | 0.6423 | 0.2845 | 0.5999 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | global_baseline | 0.8122 | 0.1832 | 0.3781 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | gradient_context | 0.7135 | 0.2322 | 0.6241 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | knn_plus_share | 0.7892 | 0.2109 | 0.4019 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | pointwise_UX | 0.6386 | 0.2931 | 0.6274 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | pointwise_plus_share | 0.6690 | 0.3010 | 0.5359 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | quantile_plus_share | 0.7546 | 0.2317 | 0.5276 |
| context_full_v2 | 2 | prob | CR3_rep2_prob_decoder | weighted_plus_share | 0.7088 | 0.2617 | 0.6490 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | Vregion_plus_share | 0.7529 | 0.2517 | 0.4715 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | cluster_plus_offdiag | 0.6006 | 0.3100 | 0.6622 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | cluster_plus_share | 0.7459 | 0.2523 | 0.5269 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | energy_weighted_UX | 0.6799 | 0.2664 | 0.6004 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | global_baseline | 0.7505 | 0.2549 | 0.4526 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | gradient_context | 0.6474 | 0.2993 | 0.6435 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | knn_plus_share | 0.7540 | 0.2508 | 0.3954 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | pointwise_UX | 0.5802 | 0.3330 | 0.6815 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | pointwise_plus_share | 0.7366 | 0.2405 | 0.5876 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | quantile_plus_share | 0.7660 | 0.2455 | 0.3878 |
| local_v1 | 1 | prob | CR3_rep1_prob_decoder | weighted_plus_share | 0.7111 | 0.2636 | 0.6569 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | Vregion_plus_share | 0.7286 | 0.2497 | 0.6048 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | cluster_plus_offdiag | 0.6805 | 0.2685 | 0.5280 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | cluster_plus_share | 0.7450 | 0.2273 | 0.6273 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | energy_weighted_UX | 0.7044 | 0.2655 | 0.6127 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | global_baseline | 0.7265 | 0.2503 | 0.6117 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | gradient_context | 0.6389 | 0.3144 | 0.5897 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | knn_plus_share | 0.7207 | 0.2422 | 0.6434 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | pointwise_UX | 0.6184 | 0.2984 | 0.6533 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | pointwise_plus_share | 0.7451 | 0.2054 | 0.6272 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | quantile_plus_share | 0.7095 | 0.2512 | 0.6663 |
| round1 | 1 | logit | CR3_rep1_logit_decoder | weighted_plus_share | 0.7538 | 0.2287 | 0.5501 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | Vregion_plus_share | 0.7529 | 0.2517 | 0.4715 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | cluster_plus_offdiag | 0.6006 | 0.3100 | 0.6622 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | cluster_plus_share | 0.7459 | 0.2523 | 0.5269 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | energy_weighted_UX | 0.6799 | 0.2664 | 0.6004 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | global_baseline | 0.7505 | 0.2549 | 0.4526 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | gradient_context | 0.6474 | 0.2993 | 0.6435 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | knn_plus_share | 0.7540 | 0.2508 | 0.3954 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | pointwise_UX | 0.5802 | 0.3330 | 0.6815 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | pointwise_plus_share | 0.7366 | 0.2405 | 0.5876 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | quantile_plus_share | 0.7660 | 0.2455 | 0.3878 |
| round1 | 1 | prob | CR3_rep1_prob_decoder | weighted_plus_share | 0.7111 | 0.2636 | 0.6569 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | Vregion_plus_share | 0.7613 | 0.2164 | 0.4369 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | cluster_plus_offdiag | 0.7170 | 0.2534 | 0.4432 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | cluster_plus_share | 0.7212 | 0.2483 | 0.4713 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | energy_weighted_UX | 0.7286 | 0.2514 | 0.4973 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | global_baseline | 0.7611 | 0.2149 | 0.4381 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | gradient_context | 0.6642 | 0.3023 | 0.5036 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | knn_plus_share | 0.7605 | 0.2213 | 0.4170 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | pointwise_UX | 0.6357 | 0.2904 | 0.6666 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | pointwise_plus_share | 0.6971 | 0.2658 | 0.5163 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | quantile_plus_share | 0.7473 | 0.2305 | 0.4399 |
| round1 | 2 | logit | CR3_rep2_logit_decoder | weighted_plus_share | 0.7773 | 0.2224 | 0.5032 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | Vregion_plus_share | 0.7722 | 0.2313 | 0.4356 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | cluster_plus_offdiag | 0.6718 | 0.2930 | 0.4820 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | cluster_plus_share | 0.7114 | 0.2606 | 0.5472 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | energy_weighted_UX | 0.7069 | 0.2613 | 0.5601 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | global_baseline | 0.7714 | 0.2325 | 0.4291 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | gradient_context | 0.6848 | 0.2832 | 0.5762 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | knn_plus_share | 0.7482 | 0.2463 | 0.4536 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | pointwise_UX | 0.6678 | 0.2871 | 0.5569 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | pointwise_plus_share | 0.6243 | 0.3072 | 0.6425 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | quantile_plus_share | 0.6928 | 0.2673 | 0.6103 |
| round1 | 2 | prob | CR3_rep2_prob_decoder | weighted_plus_share | 0.7220 | 0.2429 | 0.6258 |


## exp17_reliability

CR3 全候选（所有方法按预设规则搜索后评价，未按 TEST MCC 选解）：

| tag | repeat | channel | evidence | method | mcc | leakage | max_leakage |
|---|---|---|---|---|---|---|---|
| round1 | 1 | logit | CR3_rep1_logit | equal | 0.8308 | 0.1597 | 0.3938 |
| round1 | 1 | logit | CR3_rep1_logit | subjects2_meanSD | 0.8310 | 0.1576 | 0.4000 |
| round1 | 1 | logit | CR3_rep1_logit | subjects2_rho | 0.8308 | 0.1597 | 0.3939 |
| round1 | 1 | logit | CR3_rep1_logit | subjects4_meanSD | 0.8279 | 0.1608 | 0.3972 |
| round1 | 1 | logit | CR3_rep1_logit | subjects4_rho | 0.8308 | 0.1597 | 0.3939 |
| round1 | 1 | logit | CR3_rep1_logit | subjects8_meanSD | 0.8295 | 0.1588 | 0.3935 |
| round1 | 1 | logit | CR3_rep1_logit | subjects8_rho | 0.8308 | 0.1597 | 0.3941 |
| round1 | 1 | logit | CR3_rep1_logit | trials2_meanSD | 0.8029 | 0.1800 | 0.4322 |
| round1 | 1 | logit | CR3_rep1_logit | trials2_rho | 0.8308 | 0.1597 | 0.3936 |
| round1 | 1 | logit | CR3_rep1_logit | trials4_meanSD | 0.8274 | 0.1494 | 0.4315 |
| round1 | 1 | logit | CR3_rep1_logit | trials4_rho | 0.8307 | 0.1597 | 0.3948 |
| round1 | 1 | logit | CR3_rep1_logit | trials8_meanSD | 0.8349 | 0.1453 | 0.4315 |
| round1 | 1 | logit | CR3_rep1_logit | trials8_rho | 0.8305 | 0.1597 | 0.3961 |
| round1 | 1 | prob | CR3_rep1_prob | equal | 0.8173 | 0.1646 | 0.3677 |
| round1 | 1 | prob | CR3_rep1_prob | subjects2_meanSD | 0.8176 | 0.1625 | 0.3732 |
| round1 | 1 | prob | CR3_rep1_prob | subjects2_rho | 0.8173 | 0.1645 | 0.3679 |
| round1 | 1 | prob | CR3_rep1_prob | subjects4_meanSD | 0.8150 | 0.1667 | 0.3735 |
| round1 | 1 | prob | CR3_rep1_prob | subjects4_rho | 0.8173 | 0.1645 | 0.3682 |
| round1 | 1 | prob | CR3_rep1_prob | subjects8_meanSD | 0.8126 | 0.1621 | 0.3669 |
| round1 | 1 | prob | CR3_rep1_prob | subjects8_rho | 0.8173 | 0.1645 | 0.3682 |
| round1 | 1 | prob | CR3_rep1_prob | trials2_meanSD | 0.7921 | 0.1928 | 0.4658 |
| round1 | 1 | prob | CR3_rep1_prob | trials2_rho | 0.8175 | 0.1641 | 0.3716 |
| round1 | 1 | prob | CR3_rep1_prob | trials4_meanSD | 0.7905 | 0.1893 | 0.4753 |
| round1 | 1 | prob | CR3_rep1_prob | trials4_rho | 0.8174 | 0.1643 | 0.3698 |
| round1 | 1 | prob | CR3_rep1_prob | trials8_meanSD | 0.8169 | 0.1567 | 0.3830 |
| round1 | 1 | prob | CR3_rep1_prob | trials8_rho | 0.8176 | 0.1634 | 0.3765 |
| round1 | 2 | logit | CR3_rep2_logit | equal | 0.8101 | 0.1906 | 0.3757 |
| round1 | 2 | logit | CR3_rep2_logit | subjects2_meanSD | 0.8093 | 0.1917 | 0.3770 |
| round1 | 2 | logit | CR3_rep2_logit | subjects2_rho | 0.8101 | 0.1906 | 0.3757 |
| round1 | 2 | logit | CR3_rep2_logit | subjects4_meanSD | 0.8113 | 0.1892 | 0.3717 |
| round1 | 2 | logit | CR3_rep2_logit | subjects4_rho | 0.8101 | 0.1906 | 0.3757 |
| round1 | 2 | logit | CR3_rep2_logit | subjects8_meanSD | 0.8112 | 0.1896 | 0.3726 |
| round1 | 2 | logit | CR3_rep2_logit | subjects8_rho | 0.8102 | 0.1906 | 0.3757 |
| round1 | 2 | logit | CR3_rep2_logit | trials2_meanSD | 0.8110 | 0.1874 | 0.3841 |
| round1 | 2 | logit | CR3_rep2_logit | trials2_rho | 0.8101 | 0.1907 | 0.3760 |
| round1 | 2 | logit | CR3_rep2_logit | trials4_meanSD | 0.8072 | 0.1911 | 0.3997 |
| round1 | 2 | logit | CR3_rep2_logit | trials4_rho | 0.8102 | 0.1906 | 0.3756 |
| round1 | 2 | logit | CR3_rep2_logit | trials8_meanSD | 0.8105 | 0.1884 | 0.3860 |
| round1 | 2 | logit | CR3_rep2_logit | trials8_rho | 0.8103 | 0.1904 | 0.3752 |
| round1 | 2 | prob | CR3_rep2_prob | equal | 0.8122 | 0.1832 | 0.3781 |
| round1 | 2 | prob | CR3_rep2_prob | subjects2_meanSD | 0.8107 | 0.1849 | 0.3823 |
| round1 | 2 | prob | CR3_rep2_prob | subjects2_rho | 0.8122 | 0.1832 | 0.3781 |
| round1 | 2 | prob | CR3_rep2_prob | subjects4_meanSD | 0.8125 | 0.1833 | 0.3783 |
| round1 | 2 | prob | CR3_rep2_prob | subjects4_rho | 0.8122 | 0.1832 | 0.3780 |
| round1 | 2 | prob | CR3_rep2_prob | subjects8_meanSD | 0.8114 | 0.1843 | 0.3826 |
| round1 | 2 | prob | CR3_rep2_prob | subjects8_rho | 0.8122 | 0.1832 | 0.3782 |
| round1 | 2 | prob | CR3_rep2_prob | trials2_meanSD | 0.8114 | 0.1842 | 0.3789 |
| round1 | 2 | prob | CR3_rep2_prob | trials2_rho | 0.8121 | 0.1833 | 0.3790 |
| round1 | 2 | prob | CR3_rep2_prob | trials4_meanSD | 0.8062 | 0.1865 | 0.3968 |
| round1 | 2 | prob | CR3_rep2_prob | trials4_rho | 0.8120 | 0.1838 | 0.3807 |
| round1 | 2 | prob | CR3_rep2_prob | trials8_meanSD | 0.8124 | 0.1833 | 0.3797 |
| round1 | 2 | prob | CR3_rep2_prob | trials8_rho | 0.8120 | 0.1839 | 0.3808 |


## exp18_matrix_only

CR3 全候选（所有方法按预设规则搜索后评价，未按 TEST MCC 选解）：

| tag | repeat | channel | evidence | method | mcc | leakage | max_leakage |
|---|---|---|---|---|---|---|---|
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | GP384 | 0.6437 | 0.2838 | 0.6046 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | ensemble_reliability | 0.7059 | 0.2704 | 0.6302 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | kernel_ridge | 0.7174 | 0.2528 | 0.6334 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | mlp31 | 0.6139 | 0.3223 | 0.7057 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | mlp32 | 0.7094 | 0.2490 | 0.6542 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | mlp33 | 0.7294 | 0.2534 | 0.4996 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | mlp_ensemble | 0.6542 | 0.3068 | 0.5772 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | polynomial | 0.6145 | 0.3031 | 0.6124 |
| estimator_v1 | 1 | prob | CR3_rep1_prob_nn | smooth_mlp | 0.6216 | 0.3140 | 0.6332 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | GP384 | 0.7741 | 0.2941 | 0.6547 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | ensemble_reliability | 0.7169 | 0.3578 | 0.6480 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | kernel_ridge | 0.7106 | 0.3394 | 0.6993 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | mlp31 | 0.6885 | 0.3606 | 0.7017 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | mlp32 | 0.7791 | 0.2841 | 0.6820 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | mlp33 | 0.6852 | 0.3614 | 0.6785 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | mlp_ensemble | 0.7744 | 0.2990 | 0.6564 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | polynomial | 0.7063 | 0.3545 | 0.7276 |
| round1 | 1 | logit | CR3_rep1_logit_analytic | smooth_mlp | 0.7339 | 0.3497 | 0.6946 |
| round1 | 1 | logit | CR3_rep1_logit_nn | GP384 | 0.6879 | 0.2459 | 0.5715 |
| round1 | 1 | logit | CR3_rep1_logit_nn | ensemble_reliability | 0.6248 | 0.3029 | 0.6153 |
| round1 | 1 | logit | CR3_rep1_logit_nn | kernel_ridge | 0.7086 | 0.2547 | 0.6534 |
| round1 | 1 | logit | CR3_rep1_logit_nn | mlp31 | 0.6104 | 0.3219 | 0.5887 |
| round1 | 1 | logit | CR3_rep1_logit_nn | mlp32 | 0.6486 | 0.2924 | 0.6631 |
| round1 | 1 | logit | CR3_rep1_logit_nn | mlp33 | 0.6809 | 0.2674 | 0.5831 |
| round1 | 1 | logit | CR3_rep1_logit_nn | mlp_ensemble | 0.6494 | 0.2986 | 0.5855 |
| round1 | 1 | logit | CR3_rep1_logit_nn | polynomial | 0.6954 | 0.2680 | 0.6126 |
| round1 | 1 | logit | CR3_rep1_logit_nn | smooth_mlp | 0.6756 | 0.2707 | 0.5834 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | GP384 | 0.8107 | 0.2947 | 0.6079 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | ensemble_reliability | 0.7250 | 0.3551 | 0.6168 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | kernel_ridge | 0.7075 | 0.3432 | 0.7061 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | mlp31 | 0.7331 | 0.3469 | 0.5916 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | mlp32 | 0.6693 | 0.3852 | 0.6888 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | mlp33 | 0.7539 | 0.3290 | 0.6257 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | mlp_ensemble | 0.7228 | 0.3515 | 0.5867 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | polynomial | 0.7262 | 0.3505 | 0.6896 |
| round1 | 1 | prob | CR3_rep1_prob_analytic | smooth_mlp | 0.7019 | 0.3565 | 0.6757 |
| round1 | 1 | prob | CR3_rep1_prob_choice | GP384 | 0.7349 | 0.2142 | 0.5600 |
| round1 | 1 | prob | CR3_rep1_prob_choice | ensemble_reliability | 0.7635 | 0.2105 | 0.5167 |
| round1 | 1 | prob | CR3_rep1_prob_choice | kernel_ridge | 0.7081 | 0.2548 | 0.5162 |
| round1 | 1 | prob | CR3_rep1_prob_choice | mlp31 | 0.7756 | 0.2162 | 0.4276 |
| round1 | 1 | prob | CR3_rep1_prob_choice | mlp32 | 0.6725 | 0.3070 | 0.5286 |
| round1 | 1 | prob | CR3_rep1_prob_choice | mlp33 | 0.7075 | 0.1954 | 0.6280 |
| round1 | 1 | prob | CR3_rep1_prob_choice | mlp_ensemble | 0.7562 | 0.2146 | 0.5338 |
| round1 | 1 | prob | CR3_rep1_prob_choice | polynomial | 0.6308 | 0.2990 | 0.6686 |
| round1 | 1 | prob | CR3_rep1_prob_choice | smooth_mlp | 0.6448 | 0.2894 | 0.5830 |
| round1 | 1 | prob | CR3_rep1_prob_nn | GP384 | 0.6437 | 0.2838 | 0.6046 |
| round1 | 1 | prob | CR3_rep1_prob_nn | ensemble_reliability | 0.7059 | 0.2704 | 0.6302 |
| round1 | 1 | prob | CR3_rep1_prob_nn | kernel_ridge | 0.7174 | 0.2528 | 0.6334 |
| round1 | 1 | prob | CR3_rep1_prob_nn | mlp31 | 0.6139 | 0.3223 | 0.7057 |
| round1 | 1 | prob | CR3_rep1_prob_nn | mlp32 | 0.7094 | 0.2490 | 0.6542 |
| round1 | 1 | prob | CR3_rep1_prob_nn | mlp33 | 0.7294 | 0.2534 | 0.4996 |
| round1 | 1 | prob | CR3_rep1_prob_nn | mlp_ensemble | 0.6542 | 0.3068 | 0.5772 |
| round1 | 1 | prob | CR3_rep1_prob_nn | polynomial | 0.6145 | 0.3031 | 0.6124 |
| round1 | 1 | prob | CR3_rep1_prob_nn | smooth_mlp | 0.6216 | 0.3140 | 0.6332 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | GP384 | 0.7741 | 0.2941 | 0.6547 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | ensemble_reliability | 0.7169 | 0.3578 | 0.6480 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | kernel_ridge | 0.7106 | 0.3394 | 0.6993 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | mlp31 | 0.6885 | 0.3606 | 0.7017 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | mlp32 | 0.7791 | 0.2841 | 0.6820 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | mlp33 | 0.6852 | 0.3614 | 0.6785 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | mlp_ensemble | 0.7744 | 0.2990 | 0.6564 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | polynomial | 0.7063 | 0.3545 | 0.7276 |
| round1 | 2 | logit | CR3_rep2_logit_analytic | smooth_mlp | 0.7339 | 0.3497 | 0.6946 |
| round1 | 2 | logit | CR3_rep2_logit_nn | GP384 | 0.6950 | 0.2558 | 0.5175 |
| round1 | 2 | logit | CR3_rep2_logit_nn | ensemble_reliability | 0.6578 | 0.2865 | 0.6955 |
| round1 | 2 | logit | CR3_rep2_logit_nn | kernel_ridge | 0.6167 | 0.3026 | 0.6609 |
| round1 | 2 | logit | CR3_rep2_logit_nn | mlp31 | 0.6720 | 0.2956 | 0.5995 |
| round1 | 2 | logit | CR3_rep2_logit_nn | mlp32 | 0.6108 | 0.3357 | 0.6071 |
| round1 | 2 | logit | CR3_rep2_logit_nn | mlp33 | 0.6261 | 0.3128 | 0.7257 |
| round1 | 2 | logit | CR3_rep2_logit_nn | mlp_ensemble | 0.6743 | 0.2806 | 0.6816 |
| round1 | 2 | logit | CR3_rep2_logit_nn | polynomial | 0.7892 | 0.1991 | 0.3909 |
| round1 | 2 | logit | CR3_rep2_logit_nn | smooth_mlp | 0.6798 | 0.2772 | 0.5893 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | GP384 | 0.8107 | 0.2947 | 0.6079 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | ensemble_reliability | 0.7250 | 0.3551 | 0.6168 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | kernel_ridge | 0.7075 | 0.3432 | 0.7061 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | mlp31 | 0.7331 | 0.3469 | 0.5916 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | mlp32 | 0.6693 | 0.3852 | 0.6888 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | mlp33 | 0.7539 | 0.3290 | 0.6257 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | mlp_ensemble | 0.7228 | 0.3515 | 0.5867 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | polynomial | 0.7262 | 0.3505 | 0.6896 |
| round1 | 2 | prob | CR3_rep2_prob_analytic | smooth_mlp | 0.7019 | 0.3565 | 0.6757 |
| round1 | 2 | prob | CR3_rep2_prob_choice | GP384 | 0.7420 | 0.2462 | 0.4981 |
| round1 | 2 | prob | CR3_rep2_prob_choice | ensemble_reliability | 0.6413 | 0.3136 | 0.5670 |
| round1 | 2 | prob | CR3_rep2_prob_choice | kernel_ridge | 0.7114 | 0.2649 | 0.5570 |
| round1 | 2 | prob | CR3_rep2_prob_choice | mlp31 | 0.5572 | 0.3684 | 0.5667 |
| round1 | 2 | prob | CR3_rep2_prob_choice | mlp32 | 0.7773 | 0.2048 | 0.4880 |
| round1 | 2 | prob | CR3_rep2_prob_choice | mlp33 | 0.6059 | 0.3330 | 0.5346 |
| round1 | 2 | prob | CR3_rep2_prob_choice | mlp_ensemble | 0.6280 | 0.3320 | 0.6260 |
| round1 | 2 | prob | CR3_rep2_prob_choice | polynomial | 0.5924 | 0.3466 | 0.5433 |
| round1 | 2 | prob | CR3_rep2_prob_choice | smooth_mlp | 0.6356 | 0.3188 | 0.5623 |
| round1 | 2 | prob | CR3_rep2_prob_nn | GP384 | 0.6800 | 0.2664 | 0.5658 |
| round1 | 2 | prob | CR3_rep2_prob_nn | ensemble_reliability | 0.6548 | 0.2882 | 0.6339 |
| round1 | 2 | prob | CR3_rep2_prob_nn | kernel_ridge | 0.6307 | 0.3132 | 0.6369 |
| round1 | 2 | prob | CR3_rep2_prob_nn | mlp31 | 0.5970 | 0.3495 | 0.5402 |
| round1 | 2 | prob | CR3_rep2_prob_nn | mlp32 | 0.7390 | 0.2236 | 0.5989 |
| round1 | 2 | prob | CR3_rep2_prob_nn | mlp33 | 0.6747 | 0.2826 | 0.5219 |
| round1 | 2 | prob | CR3_rep2_prob_nn | mlp_ensemble | 0.6270 | 0.3221 | 0.6563 |
| round1 | 2 | prob | CR3_rep2_prob_nn | polynomial | 0.6570 | 0.2701 | 0.6396 |
| round1 | 2 | prob | CR3_rep2_prob_nn | smooth_mlp | 0.5977 | 0.3158 | 0.5442 |


## exp19_model_library

批次 `frozen_holdout_v2`：测试函数族与训练/validation 分离；负对照单列，不能平均成成功率。
| split | family | method | mcc | leakage | axis_error | abstain |
|---|---|---|---|---|---|---|
| new_test | radial_two_blocks | EXP13 | 0.7710 | 0.2773 | 0.2301 | 1.0000 |
| new_test | radial_two_blocks | learned | 0.8052 | 0.2328 | 0.1952 | 1.0000 |
| new_test | rational | EXP13 | 0.9240 | 0.0926 | 0.0759 | 0.0000 |
| new_test | rational | learned | 0.9991 | 0.0187 | 0.0009 | 0.0000 |
| new_test | sine_product | EXP13 | 0.9997 | 0.0149 | 0.0003 | 0.0000 |
| new_test | sine_product | learned | 0.9905 | 0.0340 | 0.0095 | 0.0000 |
| new_test | softplus_mix | EXP13 | 0.7791 | 0.2173 | 0.2209 | 0.0000 |
| new_test | softplus_mix | learned | 0.9966 | 0.0298 | 0.0034 | 0.0000 |

批次 `grammar_v1`：测试函数族与训练/validation 分离；负对照单列，不能平均成成功率。
| split | family | method | mcc | leakage | axis_error | abstain |
|---|---|---|---|---|---|---|
| test | composite | EXP13 | 0.9192 | 0.1087 | 0.0806 | 0.0000 |
| test | composite | learned | 0.9925 | 0.0457 | 0.0076 | 0.0000 |
| test | difference_square | EXP13 | 0.8801 | 0.1359 | 0.1197 | 0.3333 |
| test | difference_square | learned | 0.8244 | 0.2227 | 0.1762 | 0.6667 |
| test | radial | EXP13 | 0.7340 | 0.3409 | 0.2670 | 1.0000 |
| test | radial | learned | 0.7522 | 0.3157 | 0.2491 | 1.0000 |
| test | units | EXP13 | 0.9974 | 0.0315 | 0.0026 | 0.0000 |
| test | units | learned | 0.9160 | 0.1442 | 0.0844 | 0.3333 |
| validation | cubic | EXP13 | 0.9927 | 0.0461 | 0.0073 | 0.0000 |
| validation | cubic | learned | 0.9847 | 0.0573 | 0.0153 | 0.0000 |

批次 `np_transfer_v1`：冻结 synthetic score 迁移到原 NP，未在 benchmark T 上重新训练。
| model | repeat | channel | method | mcc | leakage | max_leakage |
|---|---|---|---|---|---|---|
| C1 | 1.0000 | logit | frozen_meta_score | 0.9597 | 0.0533 | 0.2087 |
| C1 | 1.0000 | prob | frozen_meta_score | 0.9686 | 0.0228 | 0.0612 |
| C1 | 2.0000 | logit | frozen_meta_score | 0.9629 | 0.0509 | 0.1290 |
| C1 | 2.0000 | prob | frozen_meta_score | 0.9688 | 0.0208 | 0.0689 |
| C2 | 1.0000 | logit | frozen_meta_score | 0.7933 | 0.2139 | 0.6773 |
| C2 | 1.0000 | prob | frozen_meta_score | 0.8906 | 0.1459 | 0.5287 |
| C2 | 2.0000 | logit | frozen_meta_score | 0.7878 | 0.2258 | 0.7037 |
| C2 | 2.0000 | prob | frozen_meta_score | 0.9504 | 0.0843 | 0.2897 |
| C3 | 1.0000 | logit | frozen_meta_score | 0.8761 | 0.1185 | 0.4774 |
| C3 | 1.0000 | prob | frozen_meta_score | 0.9309 | 0.0905 | 0.2914 |
| C3 | 2.0000 | logit | frozen_meta_score | 0.8669 | 0.1223 | 0.4916 |
| C3 | 2.0000 | prob | frozen_meta_score | 0.8918 | 0.1125 | 0.4119 |
| CR3 | 1.0000 | logit | frozen_meta_score | 0.7317 | 0.2485 | 0.4135 |
| CR3 | 1.0000 | prob | frozen_meta_score | 0.6900 | 0.2634 | 0.4729 |
| CR3 | 2.0000 | logit | frozen_meta_score | 0.7669 | 0.2174 | 0.4181 |
| CR3 | 2.0000 | prob | frozen_meta_score | 0.6846 | 0.2843 | 0.5066 |


## 必须保留的限制与后续工作

- EXP14/17 正式字段预算为 300 TRAIN 被试和全部冻结刺激；EXP15 为 8×8 中心点的查询实验，EXP16 为 64×64（逐点 128 点），EXP18 拟合 64×64 真实试次。采样量、估计器与优化器差异不能混同于 loss 改善。
- EXP15 的 UX 是真实 bank 最近邻配对的方向响应，不是完整环境 UX Hessian；全尺度、全矩阵有限差分都在每个 R 重新查询。
- 可靠性共识不能证明正确；独立 surrogate 仍共享原 NP 表示，可能保留系统偏差。
- EXP18 按 validation 被试的 Y-MSE 选超参；post-fit teacher 导数只用于审计。choice 的 NP 导数参考不是真实条件均值的导数。
- EXP19 使用合成监督和一个小型固定 grammar；不宣称通用 learned simplicity。径向反例应拒绝唯一恢复；满响应 Jacobian 秩的反例仍可能偏好另一套轴。
- 已补充可用 CR3 beta checkpoint、保留全部维数的 PCA 和 X 单位敏感性；仍未完成新 S1 数据、NP repeat 3–5、更高维 PCA/残差协议、更大有限扰动预算、有限差分局部张量、新刺激设计 NP 重训与大规模 grammar。

完整逐候选机器可读结果：[all_metrics.csv](all_metrics.csv)。每个结果子目录含 config、rotations、diagnostics、全部平面与 U–T 热图；运行命令和失败栈在 `logs/`。
