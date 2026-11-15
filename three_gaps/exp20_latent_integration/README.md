# EXP20 — 大样本经验积分（20A）与 Gaussian latent 积分（20B）

冻结 EXP13（p 通道、UX_neff + UUd_share、Haar-200 seed 1、24 起点 seed 7、Adam 250×lr .05）不变，只改 Hessian 二阶矩的积分测度 / 样本数。
设计见 `readmes/rotation_research/07_latent_integration_and_multiscale_hessian.md`；实际结果见 `RESULTS.md`（只写已跑出的数字）。

| 文件 | 作用 |
|---|---|
| `x7core.py` | 公共核心：Setting（checkpoint/latent/X bank）、autograd 二阶矩、矩版 EXP13 损失（与冻结语义一致）、事后评价 |
| `verify_frozen.py` | Step 0：92 个归档文件 SHA256、冻结代码 diff、300 被试复现 |
| `stream_moments.py` | 全部被试逐被试矩（5000×500 约 43 秒），子集聚合不再重算 Hessian |
| `run_20a.py` | EXP20A：N→landscape/恢复；A-native / A-fixed；5 个种子 |
| `nodes.py`, `run_20b.py` | EXP20B：MC / 加扰 Sobol / Gauss–Hermite；P0–P6 测度 |
| `tests/` | 单元测试：`python three_gaps/exp20_latent_integration/tests/test_core.py`、`test_nodes.py` |
| `results/` | 小型 CSV / JSON；大缓存在 `outputs/three_gaps/exp20_wave07/`（git 忽略） |

指标：MCC（匹配后对角平均）、泄漏均值 / 最大（非对应相关）、与 Oracle 的逐轴最大角度（诊断）、与参考解的角度。
