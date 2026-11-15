# 独立冻结核验与复现

四模型×两个repeat的全部8个p配置已完成独立复现，并通过事先声明的1e-4
容差。实际MCC最大误差<2e-7、非匹配相关最大误差<2e-6，见
results/reproduction_check.csv。92/92 归档与 live source hash 匹配，冻结源码无 diff。
300×500 decoder Hessian 重新计算，SX/T4 与历史差异 <1e-8；MCC 与泄漏复现误差 <1e-6。
完整比较见上级 RESULTS 和本目录 results。A0_strict 把历史缓存审核放入独立
子进程，优化进程本身不反序列化含有 T 的旧缓存。初始 A0 的 generic C_ref
字段有 metadata errata；实际旋转及其 npz 一直使用正确的历史300协方差。
没有修改历史代码或覆盖历史产物。
