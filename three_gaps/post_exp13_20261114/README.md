# 文档 05 实施区：EXP13 冻结，EXP14–19 独立实验

这是对 [`05_next_steps_after_exp13.md`](../../readmes/rotation_research/05_next_steps_after_exp13.md) 的新增实施目录。历史 EXP13、common/harness、已有 `x6_plan06` 和仓库中其他未提交工作均不修改。**实现与首轮实测不等于整个长期路线已经验证成功。** 执行状态和实际数字见 [RESULTS.md](RESULTS.md)；配置、失败和所有候选均保留。

## 目录与独立入口

| 目录 | 实验 | 主要实施内容 |
|---|---|---|
| `baseline/` | 冻结与复现 | checkpoint、Z、数据、导数缓存、历史旋转与源码归档；原 float32 24×250 优化器复现；独立 tuning |
| `exp14_probability/` | EXP14 | η/A/B/p 单项、组合、4×4 UX/UU 交叉、ggᵀ、梯度与上下文、κ 与输出变换 |
| `exp15_finite_scale/` | EXP15 | 每个 R 重新查询的四尺度有限差分；固定可行 X 配对；支持域与查询成本 |
| `exp16_local_context/` | EXP16 | 总体、逐点、能量加权、X 聚类/分位数/近邻、V 区域、UU 非对角对照 |
| `exp17_reliability/` | EXP17 | 被试/试次 2/4/8 分组；ρ 权重、均值+SD、低谷排名与旋转一致性 |
| `exp18_matrix_only/` | EXP18 / 17B | 严格 W,Y 拟合；多项式、kernel、受限规模 GP、平滑 MLP、3 个固定 Z 的 surrogate ensemble |
| `exp19_model_library/` | EXP19 | 按公式族划分；旋转等变的结构特征 learned score；不可识别与错误坐标反例 |

## 运行

```cmd
cd /d D:\Projects\rotation\mod-rot-3
conda activate py11
python three_gaps/post_exp13_20261114/test_core.py
python three_gaps/post_exp13_20261114/test_surrogates.py
python -u three_gaps/post_exp13_20261114/baseline/run.py --all-configs
python -u three_gaps/post_exp13_20261114/campaign.py --experiments 14 --analytic --tag mechanism_v1
python -u three_gaps/post_exp13_20261114/campaign.py --experiments 15 17 16 18 19 --tag round1
python three_gaps/post_exp13_20261114/report.py
```

`campaign.py` 按任务串行执行，逐项保存日志和 `.done.json`，同一命令可续跑。每套实验也可直接执行 `run.py --help`。例如：

```cmd
python three_gaps/post_exp13_20261114/exp14_probability/run.py --model CR3 --repeat 2 --source decoder --subjects 300 --trials 500 --tag mechanism_v1
python three_gaps/post_exp13_20261114/exp15_finite_scale/run.py --model CR3 --repeat 1 --tag scale_check --subjects 8 --trials 8 --starts 24 --steps 250
python three_gaps/post_exp13_20261114/exp18_matrix_only/run.py --model CR3 --y-source choice --channel prob --tag choice_check
python three_gaps/post_exp13_20261114/exp19_model_library/run.py --tag grammar_v1
```

`--stage` 在公共 moment 实验中支持 `selftest/cache/scan/optimize/evaluate/report/all`；`cache` 生成字段/配置，`scan` 只扫描，`evaluate/report` 读取已保存的旋转，`all` 完成搜索与诊断。EXP15 是查询型独立流程，EXP19 是模型库独立流程，不借用这个阶段接口。

## 固定协议

- 冻结 EXP13：p、未标准化 X、200 Haar（seed 1）、24 起点（seed 7）、Adam 250 步、lr .05；原始 float32 入口复现，新增 moment 实验使用 float64 的向量化独立 Adam。复现表单独保存，不能以新实现代替历史核对。
- T 与 Oracle 不进入盲优化、可靠性权重或 surrogate 选择。搜索结果先保存，再加载 T 评价。取最低 loss，保存全部起点终点；不取最高 MCC。
- 所有已用过的四模型 TEST 均属于探索性重复评价，不能称新的独立测试。Rep2 也已在 EXP13 使用。最终还需要新 S1 seed、独立 NP repeat 3–5 和新的函数族。
- EXP14/17 的正式批次复用全部 300 TRAIN 被试和全部冻结刺激。EXP15 首轮仅 8×8 中心点，EXP16 64×64、逐点部分固定 128 点，EXP18 使用 64 TRAIN 被试×64 真实试次拟合；这些不冒充与全量基线统计预算相同。
- EXP15 每个 R 的有限差分都重新查询；优化采用固定共同随机 Givens 提案，24 起点×250 步。它的优化器和查询成本不同于基线 Adam，不能据此把差异全部归因于尺度。X 配对是支持域内方向响应，**不是**完整环境 UX Hessian。
- EXP18 fitting 模块仅接收 W,Y、validation W,Y；采集后训练和旋转不访问原 decoder。单独的 post-fit audit 才访问 teacher 导数作诊断。choice 不取 logit；GP 明确限制 384 个观测，kernel 用 384 个基函数。
- EXP19 learned score 的监督来自合成 T/Q，编码训练函数族先验。测试公式族不参与 score 训练/早停。径向模型不可恢复真轴；满 Jacobian 秩也不保证生成坐标是最简坐标。

## 产物与归档

小型 CSV/JSON/PNG、旋转、曲线和日志在各实验 `results/<tag>/`；大数据在被 git 忽略的 `outputs/three_gaps/post_exp13_20261114/`。包含 config/源文件 hash、metrics、全部终点 R、逐轴尺度、原始有符号 U–T 相关、所有平面（d=4 为 6，d=6 为 15）、独立 Haar 诊断及支持域/拟合质量。

冻结归档：`D:\Projects\rotation\mod-rot-3\outputs\three_gaps\post_exp13_20261114\frozen_exp13\manifest.json`，92 个历史产物约 0.69 GB，SHA256 逐文件核对。源码 zip 是实验开始时 HEAD 的 `three_gaps` 与 `np/src` 快照；manifest 同时记录历史参考提交 a515a45。复现时补存的全起点旋转、尺度、U–T 原始矩阵位于 `baseline/results/`。

尚未将以下长期拓展当成首轮已完成：额外 NP repeat 3–5 / 新 S1 数据训练、beta/PCA 全网格、64×64→128×128→全量有限差分、有限差分局部张量、原 NP 新刺激设计训练、splines、大规模自动 grammar 和直接预测 R 的另一条路线。已有代码和首轮结果仅按实际覆盖范围报告。
