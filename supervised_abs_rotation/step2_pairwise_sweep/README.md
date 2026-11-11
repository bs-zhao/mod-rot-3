# Step 2 — Pairwise rotations RELATIVE TO Step-1 optimal R*

Independent script: `supervised_abs_rotation/step2_pairwise_sweep/run.py`.

**Every additional rotation is explicitly relative to the optimized R\* from Step 1.** It is **not** a sweep starting from the identity or the original latent Z axes. The zero-degree control is R\* itself.

Using column-vector notation:

\[
U_0=R^*Z_{\rm std},\qquad
R_{ij,\phi}=G_{ij}(\phi)R^*,\qquad
U_{ij,\phi}=R_{ij,\phi}Z_{\rm std}.
\]

Only **one** output-coordinate pair \((i,j)\) is rotated at a time. We do not chain multiple angles or pair rotations. Each experiment starts from the SAME fixed R\*, and the original decoder-derived Y remains fixed.

## Grid

Every unordered axis pair is combined with these **additional** rotation angles:

\[
\phi\in\{15^\circ,30^\circ,45^\circ,60^\circ,75^\circ,90^\circ\}.
\]

An additional single baseline `baseline` has \(\phi=0^\circ\) and \(R=R^*\).

| Model | Latent dimensions | Distinct axis pairs | Extra rotations | Total cases |
|---|---:|---:|---:|---:|
| C1 | 4 | 6 | 36 | **37** |
| C2 | 4 | 6 | 36 | **37** |
| C3 | 6 | 15 | 90 | **91** |
| CR3 | 4 | 6 | 36 | **37** |

For example, `u1_u2_045` applies a 45° rotation to **rows 1 and 2 of the best R\***, yielding `R_case=G_12(45°)@R_best`.

## Data and output

Input folder (generated in Step 1):

```text
outputs/supervised_abs_rotation/<MODEL>/<CONDITION>/repeat01/abs_mse/stage1_data/
    Zstd.npy    (N, dz)
    R.npy       (dz, dz): Step-1 optimal R*
    W.npy       (M, dz+dx): [Zstd[subject_index], X_original]
    Y.npz       all M observed/decoder Y labels
    subject_index.npy
    trial_index.npy
    trial_counts.npy
    manifest.json
```

Output folder (Step 2 only):

```text
outputs/supervised_abs_rotation/<MODEL>/<CONDITION>/repeat01/abs_mse/step2_pairwise_sweep/
    index.json                # all cases, source data hashes, angle and orientation
    cases.csv                 # index of every rotation's W and R paths
    Zstd.npy                  # shared subject-level original standardized Z
    R_best.npy                # shared, fixed, Step-1 optimal R*
    Y.npz                     # shared once, NOT recomputed per rotation
    subject_index.npy         # shared original S1 row ordering
    trial_index.npy
    trial_counts.npy
    cases/
        baseline/
            R.npy            # exactly R_best
            U_subject.npy    # (N,dz) = Zstd @ R_best.T
            W.npy            # (M,dz+dx) = [U[subject_index], X_original]
            case.json
        u1_u2_015/
            R.npy            # G_12(15°) @ R_best
            U_subject.npy
            W.npy
            case.json
        u1_u2_030/
            ...
        ...
        u3_u4_090/           # 4D models only; 6D ends at u5_u6_090
```

All `W.npy` matrices retain **every original S1 trial** in **exactly the
same order** as shared `Y.npz`. W column order is `[U1,...,Ud,X1,...,Xp]`.
The X columns are copied unchanged from Step 1. The original choice,
frozen-decoder binary logit, softmax probability, and frozen-decoder predicted
choice remain exactly the same for every rotation.

`case.json` records the Givens plane, its angle, R composition and W shape.
`index.json` contains a list of every case and the source Step-1 SHA256
fingerprints to guard against mixing runs.

**Data size:** This deliberately stores a separate complete trial-level W
for every case, which can consume several GiB per model. The script prints a
disk-usage estimate and has `--dry-run` to estimate without saving. Full W
arrays are written in configurable row chunks as float32 NumPy NPY files.
Shared Y is copied **once**; it is not duplicated in each case. Case outputs
are staged and moved into place only after completion. Existing valid cases
can be reused without recomputation.

## Commands on Windows CMD

From the repository root:

```cmd
conda activate py11
cd /d D:\Projects\mod-rot-3
git pull origin main

python supervised_abs_rotation/step2_pairwise_sweep/run.py --model C1 --dry-run
python supervised_abs_rotation/step2_pairwise_sweep/run.py --model C1
python supervised_abs_rotation/step2_pairwise_sweep/run.py --model C2
python supervised_abs_rotation/step2_pairwise_sweep/run.py --model C3
python supervised_abs_rotation/step2_pairwise_sweep/run.py --model CR3
```

Step 1 `stage1_data/` must already exist with the matching condition, objective,
and repeat. If absent, run Step 1 first:

```cmd
python supervised_abs_rotation/run.py --model C1 --overwrite
```

You can run a single case (for validation), then generate the rest later:

```cmd
python supervised_abs_rotation/step2_pairwise_sweep/run.py --model CR3 --only-case u1_u2_045
python supervised_abs_rotation/step2_pairwise_sweep/run.py --model CR3
```

Optional controls: `--objective signed_mse`, `--beta`, `--latent-dim`,
`--lambda-sparse`, `--context-min`, `--context-max`, `--repeat`,
`--stage1-dir`, `--output`, `--chunk-rows`, `--overwrite`,
`--dry-run`, `--only-case`.

## Methodological note

Since the Y values are predictions from a **frozen decoder acting on ORIGINAL
latent Z**, rotating W does **not** change Y. This is intentional: the sweep
compares the same function in alternative coordinate systems and later
complexity measures can be computed on `W_case,Y` pairs. These are
in-sample data from all-observed-trial posterior means and **not independent
held-out predictions**.
