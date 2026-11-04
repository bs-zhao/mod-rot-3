# mod-rot-3

Two independent modules:

- **[`np/`](np/README.md)** — all four original C1/C2/C3/CR3 binary-choice simulators, precheck, NP model/training, trained network evaluation, full-trial Z extraction, PCA and parameter Ridge recovery. **Seven separately runnable Python scripts. No `all` stage.**
- **[`rotations/`](rotations/README.md)** — self-contained `rotations/<method>/` subfolders for individual post-hoc rotation algorithms, each with its own implementation, configuration, tests and documentation.

The NN workflow is explicitly manual; run exactly one script for the stage you want:

```cmd
python np/s1_generate.py --model CR3 --subjects 5000
python np/s2_precheck.py --model CR3
python np/s3_train.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.8 --batch-size 128 --learning-rate 0.0005 --max-epochs 500 --patience 50
python np/s4_evaluate.py --model CR3 --beta 0 --latent-dim 8
python np/s5_extract_z.py --model CR3 --beta 0 --latent-dim 8
python np/s6_pca.py --model CR3 --beta 0 --latent-dim 8
python np/s7_regression.py --model CR3 --beta 0 --latent-dim 8
```

Commands 4–7 must reuse the same checkpoint condition flags (`--beta`, `--lambda-sparse`, `--latent-dim`, `--context-min`, `--context-max`, `--repeat`). Each script accepts `--help` and only performs its own stage.

The rotator continues reading the same frozen `outputs/choice/<MODEL>/s4_evaluate/<CONDITION>/repeatXX/s4_evaluate__latent.npz`; outputs are entirely excluded from Git by `.gitignore`.

Install with `pip install -e ".[dev]"` and run `pytest -q`. The frozen original generator and NP numerical implementation live under `np/src/modlearn_rot/`.