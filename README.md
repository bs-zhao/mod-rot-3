# mod-rot-3

**New research plan:** [Orthogonal rotation, scale-invariant Hessian, and parameter-axis identifiability](readmes/ORTHOGONAL_ROTATION_AND_HESSIAN_PLAN.md) — detailed mathematical search space, six-angle SO(4)/full O(d), signed permutations, Z-score Hessian invariances, supervised-oracle limits, and G0–G4 validation roadmap.

**Research architecture:** [readmes/README.md](readmes/README.md) — 中文研究总览：C1/C2/C3/CR3 生成模型、NP/Z、Y 类型、三条 Hessian 计算路径、积分/平均、Hessian 准确性、简单性指标、旋转与验证分层。

**Ground-truth generators:** [readmes/TOY_GENERATIVE_MODELS.md](readmes/TOY_GENERATIVE_MODELS.md) — **本项目 C1/C2/C3/CR3** 的真实公式、参数先验、X 设计、logit/probability/choice 和 Hessian 结构。

**Hessian estimator methods:** [readmes/HESSIAN_FROM_ZXY_METHODS.md](readmes/HESSIAN_FROM_ZXY_METHODS.md) — 独立统计/数值算法维度，E01–E14 14 类方法与 15 项原始文献、目标量、局限及已有实现对照。

**Independent integration & matrix-statistics dimensions:** [Q: integration and averaging](readmes/HESSIAN_INTEGRATION_AND_AVERAGING.md) · [a: Hessian → numeric matrix metrics M01–M20](readmes/HESSIAN_TO_MATRIX_METRICS.md). Both are distinct from downstream scalar simplicity k.

**General Hessian operations:** [D × P: eliminate input-variable dependence independently from changing output shape](readmes/HESSIAN_FUNCTIONAL_OPERATIONS.md) — includes scalar/vector/smaller matrices, operation order, and rotation invariance.

The two independent parts of the project are:

- [np/](np/README.md): **seven manually executed stages** for C1/C2/C3/CR3
generation, precheck, NP training, prediction evaluation, full-trial Z,
PCA and parameter regression.
- [rotations/](rotations/README.md): **one folder per complete rotation
suite**, with its own run.py, all of its metrics and training/optimization
steps, configuration, tests, post-fit evaluation and output handling.

There is **no** command to execute all rotation suites. Each suite runs by
calling its own `run.py` directly. No central method registry, shared rotation
engine or batch launcher exists.

```cmd
cd /d D:\Projects\mod-rot-3
git pull --ff-only
pip install -e ".[dev]"
python rotations/xy_effect/run.py --model CR3 --condition beta0_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
```

All data, models, rotations and plots are saved to the root ignored
`outputs/` directory. Rotation optimization sees only frozen ZXY; theta
is used strictly afterward for held-out TEST diagnostics.
