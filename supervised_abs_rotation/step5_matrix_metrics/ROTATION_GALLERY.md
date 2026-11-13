# Retained criteria: angle curves, loss-selected rotations and T heatmaps

The user's request is to save both outputs for every retained criterion,
including approximate and displaced minima. The manifest is the complete
44-entry `write_approximate_shortlist.CANDIDATES`, without an added performance
veto. Each entry has CR3, C1, C2 and C3, using all 5000 subjects.

Local deliverable:
`outputs/supervised_abs_rotation/retained_metric_rotation_gallery/index.html`.
Each numbered criterion directory contains raw-loss versus angle PNG/PDF,
the numerical curves CSV, a grid-minimum correlation figure and a best-available
search correlation figure. Every plane is shown, with angles 0 to 90 degrees.
The y axis is the original metric, not a relative difference. The 0-degree
point is the saved case baseline, not raw Z.

Each model has `grid_search/` and `best_available_search/` with `R.npy`,
`U.npy`, Pearson/Spearman arrays and CSV, `search_freeze.json`, and
`evaluation.json`. The best-available search also records every candidate
frame's loss and origin. `source_coordinates/<model>/` saves Zstd and T.
The convention is **U = Zstd @ R.T**, equivalent to U_column = R @ Zstd_column.
The original supervised `R.npy` is never read; the coordinate conversion is
derived by registering original Zstd against saved W subject coordinates.

The finite search minimizes the exact frozen metric over saved cases and
additional scored stress frames. Matching completed continuous optimizers
are included only for the identical channel and criterion with verified
source/frame hashes. BU6 predates source-score hashes, so its exact objective
is replayed against the current frozen BU source before inclusion. Original
radial GP and AU tree searches are imported with their own immutable records.
This is a **best available searched result**, not a global optimum certificate.
The saved grid minimum is retained separately because it can look substantially
better aligned with T than a lower-loss off-grid rotation.

The new exact continuous runner additionally supports 23 entries (BU/BU3,
CB/CC/CD/CE/CF/CG/CH/CH2/CI/CJ/CK/BY). It uses the minimum W-grid frame,
anonymous identity, eight Haar proposals, two Powell starts with 48 evaluations
each, then one Givens pass at +/-0.1 radians. It works through all selected
criteria for CR3 before C1, C2, C3. Every result is saved immediately; the
complete batch refreshes the best-available gallery automatically. It can be
resumed and does not repeat frozen completed searches. Other entries retain
their finite candidate-pool and any matching prior continuous results; they
must not be described as newly continuously optimized by this runner.

T is loaded **after** R and U have been saved and hashed, only by a separate
evaluation function. The original dataset SHA and subject/trial indices are
verified. T does not enter the metric, proposal generation, search stopping
or selection of R. Heatmaps display the signed coordinates without matching
or reordering them using T. Only NN logit or NN p is used by this shortlist;
no analytic response is counted as success.

CI and BY were completed on the previously missing three datasets for this
request. CH and CH2 are the same quantile estimator with occupancy=20; CH2
provides the completed C1/C2/C3 runs for the older CH shortlist entries.
This exact alias is recorded in each result. Entries remain individually
numbered for continuity with the existing retained list.

Reproduce/export:

```powershell
conda activate py11
cd D:\Projects\mod-rot-3
python supervised_abs_rotation/step5_matrix_metrics/export_candidate_rotation_gallery.py
python supervised_abs_rotation/step5_matrix_metrics/export_saved_search_heatmaps.py
python supervised_abs_rotation/step5_matrix_metrics/verify_rotation_gallery.py
```

Continue new exact continuous searches:

```powershell
python supervised_abs_rotation/step5_matrix_metrics/search_retained_metric_rotations.py
```

`--numbers 1 2 29` selects specific shortlist entries. No argument runs all
23 supported entries. Source W/Y and historical experiment folders are not
overwritten. Newly fitted scratch models are removed only after the selected
R, U and loss have frozen; history and source hashes are retained.
