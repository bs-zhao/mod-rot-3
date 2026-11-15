# 冻结、复现与独立稳健性检查

`run.py --all-configs` 先复制并校验历史产物，再使用原 float32 harness 复现 16 个设置。归档约 0.69 GB、92 个历史文件；源码快照是开始执行时的 bfda311。

Git 标签：[exp13-frozen-doc05-20261114](https://github.com/bs-zhao/mod-rot-3/tree/exp13-frozen-doc05-20261114) 固定在同一 bfda311，历史参考为 a515a45。标签不意味着大缓存上传 Git；本地完整归档路径与 SHA256 在 `results/archive_manifest.json`。复现补存的全部起点 R、轴尺度、匹配相关与原始有符号 U–T 矩阵在 `results/`。

`batch_tune.py` 在新命名空间运行 stat∈{neff,l1l2,quartimax,softL0}、curvature∈{share,sum}、w∈{0,.25,.5,1,2,4}、stdX∈{0,1}。w=0 的两个 curvature 定义相同，只运行一次，因此每设置 88 个独立候选，四模型×两重复×两通道共 1408 行。所有候选公开，不使用 TEST 选择新的冻结方法。

`robustness.py` 单独记录 full-d PCA、X 第一/第二列单位×100/×.01 及两者组合（48 个设置），加可用 CR3 beta=1e-5/1e-4 两通道（4 个设置）。PCA 在全部 TRAIN 拟合、保留全部 d 维；不冒充高维降维/残差协议。不同 beta NP 的坐标不能直接复用另一 NP 的 R。

独立 NP repeat 3–5 和新 S1 数据尚未训练；当前结果保留探索性标签。
