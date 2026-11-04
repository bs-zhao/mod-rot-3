# NP research: independently executed steps

Each experiment is its **own runnable Python file**, with actual stage logic in that file. No `np/run.py`, no `--stage` argument and no `all` orchestration.

| Order | Python entrypoint | What it does | Required previous output |
|---|---|---|---|
| 1 | `np/s1_generate.py` | Exact C1/C2/C3/CR3 simulator, theta/X sampling and binary choice Y | None; CR3 will prepare original design bank |
| 2 | `np/s2_precheck.py` | Original validity, identification and CR3 FOSD checks | S1 dataset |
| 3 | `np/s3_train.py` | Train original NP; choose beta, lambda, latent dimension, optimizer hyperparameters | S2 qualification |
| 4 | `np/s4_evaluate.py` | Saved NP validation/test accuracy, CE, KL and learning curves | S3 checkpoint |
| 5 | `np/s5_extract_z.py` | Encode every subject using **ALL observed trials**, save frozen posterior-mean Z | S3 checkpoint and S1 dataset |
| 6 | `np/s6_pca.py` | TRAIN-only PCA of Z; explained variance and correlations | S5 Z |
| 7 | `np/s7_regression.py` | Full-Z and top-k-PC Ridge regression to simulator theta; validation-selected alpha; held-out test | S5 Z and S6 PCA |

Run commands **one at a time** from repository root, e.g. for CR3 beta=0 L8:

```cmd
python np/s1_generate.py --model CR3 --subjects 5000
python np/s2_precheck.py --model CR3
python np/s3_train.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8 --context-min 0.25 --context-max 0.80 --batch-size 128 --learning-rate 0.0005 --weight-decay 0 --max-epochs 500 --patience 50 --repeat 1 --device auto
python np/s4_evaluate.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
python np/s5_extract_z.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
python np/s6_pca.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
python np/s7_regression.py --model CR3 --beta 0 --lambda-sparse 0 --latent-dim 8
```

Training-dependent steps must use matching `--beta`, `--lambda-sparse`, `--latent-dim`, `--context-min`, `--context-max`, and `--repeat` to locate the same checkpoint and Z.

**Code organization**: `np/src/modlearn_rot/` contains frozen upstream simulator/NP/training mathematics. `np/src/modrot3/pipeline.py` contains only shared paths/data adaptation/provenance functions; `np/src/modrot3/analysis.py` contains only shared numerical analysis utilities. Each step's actual implementation lives in its corresponding `np/sN_*.py` file.

All data, checkpoints, figures, and parameter recovery CSV/JSON files are under **root-level `outputs/choice/<MODEL>/`**, excluded by `.gitignore`. In particular the rotation platform consumes the unchanged `s4_evaluate__latent.npz` file from S5. Neither S6 nor S7 alters that file. True theta is never used for NP training, Z extraction, or fitting rotations.