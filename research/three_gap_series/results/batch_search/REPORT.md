# 连续批量搜索：当前检查点

已完成 9 批，累计 1313 个“单指标 × 模型 × inference × 优化设置”案例，对应 3939 次 gap 测量。跨批去除精确别名后有 277 个定义；阈值、汇总方式及尺度变体均明确列出，不把它们当作同样数量的独立思想。

研究仍未完成。每个候选在四模型用同一公式和推断规则；不同 inference 分支不能拼接。
所有案例同批测 G1/G2/G3，上游失败时下游结果仍保留作诊断。

| 批次 | 完成案例 | 四模型齐全的去重候选 | 四模型 G1 | 四模型 G1+G2 | 四模型全部通过 |
|---|---:|---:|---:|---:|---:|
| [b01_three_gap_batch](../b01_three_gap_batch/report/REPORT.md) | 385 | 64 | 8 | 0 | 0 |
| [b02_three_gap_batch](../b02_three_gap_batch/report/REPORT.md) | 124 | 20 | 10 | 1 | 0 |
| [b03_three_gap_batch](../b03_three_gap_batch/report/REPORT.md) | 100 | 25 | 1 | 0 | 0 |
| [b04_three_gap_batch](../b04_three_gap_batch/report/REPORT.md) | 132 | 33 | 17 | 6 | 0 |
| [b05_three_gap_batch](../b05_three_gap_batch/report/REPORT.md) | 84 | 21 | 3 | 0 | 0 |
| [b06_three_gap_batch](../b06_three_gap_batch/report/REPORT.md) | 116 | 29 | 12 | 6 | 0 |
| [b07_three_gap_batch](../b07_three_gap_batch/report/REPORT.md) | 124 | 31 | 1 | 0 | 0 |
| [b08_three_gap_batch](../b08_three_gap_batch/report/REPORT.md) | 148 | 37 | 25 | 9 | 0 |
| [b09_three_gap_batch](../b09_three_gap_batch/report/REPORT.md) | 100 | 25 | 12 | 9 | 0 |

表中四模型计数采用统一 Gaussian256、8 起点初筛；另行复核与替代推断也计入完成案例。G1 平面反例和不可辨识方向均如实列出；初筛成功仍需更多起点及新样本确认。

## 已确认的推进与失败

- 无向 UU 边与 UX 边的单一 RMS 软计数（阈值 .02）在四模型的 64 起点复核中通过 G1、G2，但 G3 失败，已单独保存。不能把这个结论说成“没有通用指标”，也不能称作完整解决。
- 纯 UU 非线性单项在 CR3 decoder 可达到约 .824 的平均相关，但 C1/C2/C3 的解析 logit UU 严格为零，故它单独不是通用解；还存在明显错配。
- 已比较 Gaussian、ANOVA、五阶 latent 多项式、latent Gaussian 等通用 WY 推断分支。当前 CR3 decoder 的 UX 导数相对误差仍接近 1，需要与目标函数偏差分开。
- 事后参考方向 loss 诊断只读已经锁定的旋转，绝不把监督 R 用作起点或供给无监督目标。
- 精确网络 Hessian 另记为诊断对照，不能冒充仅凭 WY 推断成功。

## 可复现性

全部代码和结果均在 `research/three_gap_series`。先推送代码，再拉取到本机 py11 执行。保留模型/数据哈希、固定数值场来源、每个候选的 R 锁定文件、完整平面曲线与多起点结果。
原始 NP、数据、Z 与其他实验目录未修改。TEST 已用于探索，最终候选必须冻结后再用独立新样本验证。

[指标定义与文献依据](../../BATCH_SEARCH.md) · [批次计数](batches.csv)
