# x6_plan06 — 按 `readmes/rotation_research/06_complementary_ideas_and_experiment_cards.md` 逐个实施

本文件夹是文档 06 计划的**实施区**。规则：

- **冻结基线不动**：`three_gaps/exp13_pure_simplicity/`（`UX_neff + UUd_share`，p 通道）及其依赖（`three_gaps/exp11_cr3_search/harness.py` 等）只被**只读导入**，不修改。
- 每个实验一个子目录（`e01_diagnosis/` …），独立入口、独立输出；小结果（CSV / PNG）放在子目录 `results/`，大文件在被 git 忽略的 `outputs/x6_plan06/`。
- 公共代码在 `common/x6lib.py`：缓存加载、冻结损失（含 Haar 标准化）、双精度评估、旋转工具。
- **阶段性结果写入 [`LOG.md`](LOG.md)**（按时间倒序追加，每条含日期、做了什么、数字、结论、限制），并及时提交推送。
- 无角色检测；成功只在 NN 输出的 logit / p（或观测 choice）上判定；T 与 Oracle 只用于事后评价与诊断。

## 实验进度

| 编号 | 名称 | 状态 |
|---|---|---|
| E1 | 诊断：损失错位还是优化失败 | 完成（见 LOG） |
| E2 | 交叉 NP 选超参 | 待做 |
| E3 | 稳健性与置信区间 | 待做 |
| E4 | ε-约束（词典序）组合 | 完成：不采纳（见 LOG） |
| E5 | 精度加权共识 | 待做 |
| E6 | 冻结 Z 上的 decoder 集成 | 待做 |
| E7–E12 | 稀疏代理、函数层面、观测 choice、新生成器、统一基准、理论 | 待做 |

## 数据与环境

- 缓存来自 `three_gaps/exp13_pure_simplicity/cache2.py`（`outputs/three_gaps/exp13/cache2_*.pt`，git 忽略）：四个模型 × 重复 1 / 2 × logit / p。
- 环境：conda `py11`（`D:\Programs\anaconda_new\envs\py11`），Windows。
