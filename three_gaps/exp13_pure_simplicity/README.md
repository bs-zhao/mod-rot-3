# EXP13 — 无角色检测的 W–Y 简单性旋转损失（当前最好版本）

**最好的无角色方法：`UX_neff + UUd_share`，NN decoder 的 p 通道。** 不含阶段 A / 增益轴 / 设计类别 / 噪声底扣除，T 只在事后评价。

## 方法（一句话）
对 NN decoder 的 p = sigmoid(logit) 做自动微分，得到二阶导 H_UX、H_UU；在正交旋转 Q∈SO(d) 下（旋转后逐轴 z-score）最小化
`L = UX_neff / s1 + UUd_share / s2`（s 为各项在 200 个 Haar 随机旋转上的标准差，等权，种子 1），24 个随机起点 + Adam（lr 0.05，250 步）取最低。

- `UX_neff`：对 e_UX[i,j] = E[(∂²p/∂u_i∂x_j)²] 在全部 (轴, X 列) 上归一成份额 q，取 exp(−Σ q log q)（有效活跃的 U–X 作用数，越小越稀疏）。
- `UUd_share`：Σᵢ e_UU[i,i] / Σᵢₖ e_UU[i,k]（自曲率占 U–U 二阶总能量的份额，越小说明沿各轴越线性）。

## 结果（NN decoder p 通道，TEST，MCC / 泄漏均值；括号为监督 Oracle）
| NP | CR3 | C1 | C2 | C3 |
|---|---|---|---|---|
| 重复 1 | .817 / .165 (.872 / .089) | .969 / .021 (.969 / .018) | .965 / .060 (.971 / .026) | .960 / .046 (.963 / .036) |
| 重复 2 | .812 / .183 (.869 / .105) | .969 / .020 (.970 / .015) | .968 / .043 (.971 / .027) | .962 / .048 (.965 / .035) |

logit 通道在 C1–C3 上不稳（重复 2 的 C3 掉到 .827），所以以 p 通道为准。CR3 仍有 .26–.37 的残余混合，未追平 Oracle。
未经带角色检测的流水线（`../exp11_cr3_search/`，CR3 .83–.86）对比：那条路用了“找增益轴”，不属于本目录的无角色版本。

## 复现
```cmd
conda activate py11
cd /d D:\Projects\rotation\mod-rot-3
python three_gaps/exp13_pure_simplicity/cache2.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --channel prob [--repeat 2]
:: C1 / C2: --beta 0.0005 --zdim 4 ;  C3: --beta 0.0005 --zdim 6   (各跑一次)
python three_gaps/exp13_pure_simplicity/combos.py CR3 prob 1 "UX_neff+UUd_share"
python three_gaps/exp13_pure_simplicity/heatmaps.py 1        :: U–T 热图（logit 和 p 各一张）
```

## 文件
| 路径 | 内容 |
|---|---|
| `losses2.py` | 二阶矩、旋转后能量、全部简单性损失 |
| `cache2.py` | 缓存 decoder 的 g、H_UU、H_UX |
| `combos.py` | Haar 标准化的等权组合与搜索（**最好方法的入口**） |
| `heatmaps.py` | U–T 热图 |
| `tune.py` | 超参扫描 / 稳健性（权重、统计量、曲率形式、X 标准化；**未跑完**） |
| `oracle_scan.py`、`cells_opt.py`、`floor_multiple_scan.py`、`plugin_cache.py` | Oracle z 扫描与噪声底倍数诊断 |
| `natural.py`、`weighted.py`、`screen.py` | 自然尺度 / z 加权 / 单损失筛选（权重试验均未成功） |
| `results/` | 热图 PNG、汇总 CSV、筛选 CSV、重复 2 日志、各脚本日志 |

详细说明：`../../readmes/rotation_research/04_role_free_simplicity_losses.md`。
