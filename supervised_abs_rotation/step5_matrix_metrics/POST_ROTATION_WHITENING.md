# ZCA whitening after the saved rotations

The relocated project is `D:\Projects\rotation\mod-rot-3`.
The result viewer is
`outputs/supervised_abs_rotation/retained_metric_rotation_gallery/whitened/index.html`.
The original gallery also links to this viewer.

The export snapshots every frozen `grid_search`, `best_available_search` and
`continuous_search` U from the 44-entry gallery. The initial snapshot has 365
rotations: 176 grid minima, 176 best-available minima and 13 completed continuous
searches. Missing continuous results remain visibly absent for those models.
The source R and U files and historical manifests are preserved.

For each U, using all 5000 subjects and population covariance (ddof=0):

```
mu = mean(U)
C = (U - mu).T @ (U - mu) / 5000
A = C^(-1/2)                  # symmetric positive square root inverse
U_white = (U - mu) @ A
```

ZCA keeps the original rotated coordinate orientation as far as possible;
it does not replace the axes with PCA eigenvectors. It is rotation equivariant:
whitening `Z @ R.T` equals `whiten(Z) @ R.T` for orthogonal R. No T is used
to fit whitening, select its parameters, choose signs or order coordinates.
T is used only for Pearson/Spearman correlations after the new U and A freeze.

The combined transform is `B = A @ R` and
`U_white = Zstd @ B.T - mu @ A`. B generally is not orthogonal. R is not
reoptimized, and the old rotation-metric loss is not assigned to whitened U.
If U is numerically rank deficient, the routine uses the declared positive
eigenspace pseudoinverse and reports its rank and covariance projector error;
it never silently adds an arbitrary ridge or claims full whitening.

Each candidate has four-model PNG/PDF figures for each available search kind.
Each model/search directory saves:

- `U_whitened.npy`, `U_mean.npy`, `ZCA_matrix.npy` / `.csv`;
- `combined_transform_B.npy`, `centering_offset.npy`;
- Pearson/Spearman U_white–T matrices as NPY/CSV;
- original U–T, U_white–U_white correlation and whitened covariance matrices;
- individual before/after PNG/PDF heatmaps;
- `whitening_freeze.json` and evaluation metadata.

`summary.csv` and `verification.json` summarize all outputs. Verification
checks source R/U hashes, the U=Zstd@R.T convention, whitening covariance,
the combined transform, independently recomputed Pearson correlations, and
original T dataset SHA/subject ordering. The old absolute dataset path is
rebased to the new repo root for read-only provenance verification; historical
files are not rewritten.

Run from the new repo:

```powershell
conda activate py11
python supervised_abs_rotation/step5_matrix_metrics/export_whitened_rotation_gallery.py
```
