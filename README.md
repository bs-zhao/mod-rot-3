# mod-rot-3

**Research architecture:** [readmes/README.md](readmes/README.md) — 中文研究总览：C1/C2/C3/CR3 生成模型、NP/Z、Y 类型、三条 Hessian 计算路径、积分/平均、Hessian 准确性、简单性指标、旋转与验证分层。

**Toy generator catalog:** [readmes/TOY_GENERATIVE_MODELS.md](readmes/TOY_GENERATIVE_MODELS.md) — T1–T11 的真实解析公式、参数与 X 定义域、全部六种连续输出变换、理论值域和 T9/T11 与 C2/CR3 的区别。

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
