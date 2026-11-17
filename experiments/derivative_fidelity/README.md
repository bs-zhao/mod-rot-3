# Derivative Fidelity Benchmark (C1, C2, C3, CR3)

Independent, read-only comparison of derivatives of the true analytic generator
and the frozen Neural Process decoder. This experiment **does not optimize
latent rotations**, retrain NP, modify existing math/three_gaps suites, or
reinterpret fitted scores as evidence of unsupervised parameter recovery.

**Source code:** experiments/derivative_fidelity/  
**ALL output files:** outputs/derivative_fidelity/<TAG>/  
**Default branch:** main (no separate branch needed)

## Research questions

1. Does NN recover local first-, second-, and third-order derivative fields?
2. Which field summaries survive even when derivatives are inaccurate?
3. What is the difference between derivative *estimation* error (FD/GP/etc
   versus the same NN under autodiff) and derivative *learning* error (NN
   autodiff versus the coordinate-matched analytic proxy)?
4. Which input partitions give different answers: U only, U–X mixed,
   U-related joint, or W=(U,X) treated as a single vector?

## Inputs and two reference protocols

True generators live in three_gaps/common/analytic.py. A subject has theta=T,
with X varying by trial. The NP decoder receives frozen posterior mean Z
(extracted with all observed trials), not T. For the current benchmark
V is the existing mod-rot-3 transformed coordinate (default pca=0).

**G1 analytic_truth:** evaluate the actual analytic function h(Tstd,X) and
differentiate with respect to true standardized T. This is the genuine
generator-coordinate structural reference; its derivatives cannot be compared
coordinate-by-coordinate with NN derivatives in V.

**G2 analytic_mapped:** fit Tstd ≈ A V + b from TRAIN subjects only, and set
h_mapped(V,X)=h_true(A V+b,X). Its derivatives and NN derivatives use exactly
the same input V,X and can be compared directly. This is a *supervised
coordinate-matched analytic proxy*, not a claim that the unknown T(V) map is
exact. Both G1 and G2 are saved independently. The estimated mapping and
held-out recovery statistics are in case.json.

Every source supports logit eta and probability sigmoid(eta). The binary
choice itself is not differentiated. C1/C2/C3 logit have exact UU=UUU=UUX=0
in any fixed linear transform of true T, whereas UXX is only nonzero for
C2/C3. Probability can have nonzero UU and third derivatives due to sigmoid.

## Derivative blocks

| Order | U only | Mixed U/X | X only | Entire W |
|---|---|---|---|---|
| 1 | U | -- | X | W |
| 2 | UU | UX | XX | WW |
| 3 | UUU | UUX, UXX | XXX | WWW |

Output index ordering follows the block spelling (e.g. UUX -> d,d,p).
Mixed symmetric permutations occur once in saved canonical tensors. Full-W
tensor construction explicitly restores all permutations; quadratic and
cubic aggregate energies use multiplicity 2 and 3 respectively.

For CR3, X boundaries require special treatment: zero gains/losses and
probability clipping break standard differentiability in some X directions.
--domain interior (default) uses a strict, conservative smooth common X
subset for ALL blocks. --domain all includes the complete bank but X
derivatives on such boundaries must be considered implementation-specific,
not mathematical gold standards. U-only derivatives may be examined
separately on the complete bank with --blocks U,UU,UUU --domain all.

--standardized uses TRAIN native means/SDs for V and X. It applies chain
rules inside the function definition; without it, native coordinate units
are retained. Do not compare cross-order ratios in incompatible units.

## Metric inventory (A-F)

One shared metric implementation is used for analytic and every estimator.

* **A01-A12 elementwise**: signed mean, mean absolute, RMS, variance, std,
  |D| quantiles 50/90/95/99, max, near-zero occupancy, sign proportions,
  adjacent-grid sign changes, skew, excess kurtosis, between-subject and
  within-subject variability. Adjacent-grid sign changes are order-sensitive;
  use them as descriptive diagnostics, not spatially invariant quantities.
* **B01-B13 gradient**: mean norm, norm², covariance, Gram second moment,
  mean unit gradient, correlations, eigen/singular spectrum, effective/stable
  rank, condition, per-axis energy and sparsity, off-diagonal coupling,
  subject-first versus trial-first operators, approximate joint diagonalization.
* **C01-C20 Hessian**: signed and RMS matrices, Frobenius energy,
  spectral/nuclear norm, eigenvalue spectra, trace, positive/negative curvature,
  diagonal/off-diagonal energy and share, Geomin/entropy-like support via
  effective number, Gini, L1/L2, quartimax, tail energy and soft-L0,
  row/column entropies, matrix rank, variance, conditional/subject means,
  approximate joint diagonalization and directional curvature.
* **D01-D14 third-order**: energy and RMS, signed mean, repeated/distinct
  index energy, sparsity, mode unfolding spectra and effective ranks,
  optional CP approximation (requires tensorly), basis-direction cubic
  derivative, conditional energy, thresholded support and block energies.
* **E01-E14 joint/full-W**: U+UX, UU/UX and UX/XX energy ratios,
  group-wise and full-W order energies, full-W neff/Gini, energy profiles and
  higher-to-lower-order ratios. Full-W gradient, Hessian and third-order
  fields are reconstructed on the SAME limited evaluation subset to compute
  additional A/B/C/D full-W metrics. These subset results are labeled.
* **F01-F03 finite scale**: directional first-/second-/third-order Taylor
  reconstruction error with U-only, X-only and complete-W perturbations.
  F04-F06: controlled gradient/Hessian variation and U-X factorial contrasts
  are also recorded as finite-scale scalar output errors.

Not all possible theoretical indicators in the earlier proposal have a
well-defined finite-sample implementation or scalar numerical convention.
Unsupported indicators are identified explicitly; no arbitrary scores are
invented. E14 is a numerical consistency audit, not a sparsity objective.

Most summaries are in *_metrics.npz; reference results include full field
.npy arrays. A scalar entry is a 0-D NumPy array; array-valued metrics preserve
tensor coordinates. Full-W summary metrics have the prefix W./WW./WWW.
Metric-level NN-vs-G2 comparisons are saved in metric_comparisons.jsonl.

## Estimator inventory and capability

| ID | --methods name | Source | Capabilities |
|---|---|---|---|
| N01 | ad_mixed | NN graph | exact NN field (mixed jacrev/jacfwd) |
| N02 | ad_forward, ad_reverse | NN graph | independent AD formulations |
| N04 | fd2 | NN output values | central polynomial stencil |
| N05 | fd4, richardson | NN output values | higher-order stencil / extrapolation |
| N06 | hybrid | NN gradient/partial tensor | AD first, central difference last |
| N07 | gaussian_smoothing | NN output values + AD | antithetic U-only Gaussian smoothing |
| N08 | local_poly | NN output values | local overdetermined polynomial |
| N09 | spectral_chebyshev | NN output values | Chebyshev interpolation nodes |
| N10 | krr | NN value-only surrogate | RBF Nyström kernel ridge |
| N11 | gp | NN value-only surrogate | small exact Gaussian process kernel |
| N12 | mlp | NN value-only surrogate | independent smooth SiLU MLP |
| N13 | hvp_directional | NN graph | directional contractions only |
| N14 | stein_gaussian_u | NN values | Gaussian-weighted U-only derivative integrals |

Methods marked contraction / aggregate_only do NOT produce dense derivatives
and therefore cannot populate every entry-wise A-F metric. The manifest
records the correct capability instead of silently treating them as full fields.
N14 assumes a separately defined Gaussian probe measure and must not be
interpreted as the derivative under the empirical non-Gaussian study design.

GP/KRR/MLP surrogate hyperparameters are selected from **NN function-value
validation**, never from analytic derivative error. No oracle T is supplied to
these estimators. Limited full-W metric reconstruction uses an explicitly
recorded sample of at most 12 subjects x 32 X points; full grid blockwise energy
metrics use the entire saved field. The extrapolation error and exact runtime
will depend on the machine, design, and estimator.

## Execution

From repository root, in the py11 environment:

    conda activate py11
    python -m pytest experiments/derivative_fidelity/tests -q

A single small C1 run (no NP retraining):

    python -m experiments.derivative_fidelity.run --models C1 --repeats 1 --channels logit --subjects 8 --stimuli 16 --methods ad_mixed,fd2,hybrid --tag smoke01 --fail-fast

All four models, two trained NP repeats, both output channels:

    python -m experiments.derivative_fidelity.run --models C1 C2 C3 CR3 --repeats 1 2 --channels logit prob --subjects 300 --stimuli 500 --methods all --blocks all --tag full01 --standardized --taylor

Note: --methods all includes multiple expensive high-order finite difference,
local polynomial, and surrogate methods. At 300x500, some are astronomically
expensive (especially 3D tensor-product stencils). Use smaller subjects,
stimuli and --blocks selections for method calibration before requesting
full scale. Jobs report errors and saved status; a full run is not guaranteed
to complete within ordinary desktop resources. Use --storage metrics
(default) to delete bulky NN temporary fields after summarization; G1/G2
full fields are retained for reproducibility. Use --storage all to retain
NN field .npy files too.

Recommended staged runs:

    python -m experiments.derivative_fidelity.run --models C1 C2 C3 CR3 --repeats 1 --channels logit --subjects 16 --stimuli 32 --methods ad_mixed,ad_forward,ad_reverse --tag ad_pilot --fail-fast
    python -m experiments.derivative_fidelity.run --models C1 CR3 --repeats 1 --channels logit prob --subjects 32 --stimuli 64 --methods fd2,fd4,richardson,hybrid,local_poly,spectral_chebyshev --tag fd_pilot
    python -m experiments.derivative_fidelity.run --models C1 C2 C3 CR3 --repeats 1 2 --channels logit prob --subjects 64 --stimuli 128 --methods gaussian_smoothing,krr,gp,mlp,hvp_directional,stein_gaussian_u --tag surrogate_pilot
    python -m experiments.derivative_fidelity.run --stage report --tag ad_pilot

For existing runs, the same tag can resume completed blocks; use --force
to regenerate. Use separate tags for new numerical or dataset conditions.
If the reference cache already exists and source checkpoint changes, give
a new tag; the manifest records the checksum but an old cache is not
automatically reconciled.

New experiment output structure:

    outputs/derivative_fidelity/<tag>/<model>/repeatNN/<logit|prob>/<native|standardized>/<grid>/<domain>/
      case.json
      config.json
      domain.json
      analytic_truth/{U.npy,...,U_metrics.npz,...,joint_metrics.npz,fullw_metrics.npz}
      analytic_mapped/{U.npy,...,U_metrics.npz,...,joint_metrics.npz,fullw_metrics.npz}
      ad_mixed/{U_metrics.npz,...,field_comparisons.json,metric_comparisons.jsonl,run_manifest.json}
      fd2/...
      ...
    outputs/derivative_fidelity/<tag>/REPORT.md
    outputs/derivative_fidelity/<tag>/summary_field_accuracy.csv
    outputs/derivative_fidelity/<tag>/summary_metric_accuracy.csv
    outputs/derivative_fidelity/<tag>/summary_rmse.png
    outputs/derivative_fidelity/<tag>/all_run_manifests.json

## Statistical correctness notes

1. ANALYTIC MAPPED is a proxy, not the exact true function of V. G1 must
   remain separately available for statements about true theta derivatives.
2. NN AD64 is the numerical reference for *the NN's own derivative*. Relative
   NN-vs-generator errors require G2 (or an explicitly justified alternative).
3. Exact local derivatives can differ despite almost identical predicted
   output values. Finite differences and smoothing generally estimate a
   *different finite-scale or smoothed estimand*.
4. C1-C3 logit theoretical zero tensors have undefined Pearson/R² and relative
   error. Record absolute false energy instead; do not manufacture zeros.
5. Functional, matrix and tensor metrics are not interchangeable.
   Averaging-before-square differs from squaring-before-average; spectral
   norm after averaging differs from averaging spectral norms.
6. Bootstrapping should resample subjects, not individual trials. Full
   bootstrap CI and plot atlases are future additions; currently per-subject
   conditional statistics are saved to enable this.
7. The default output directory is .gitignored by the existing repository
   policy. Outputs are generated on the machine executing the pipeline;
   creating GitHub code does **not** execute CPU/GPU jobs or upload results.

## Status

Source code is committed; the README and tests are not evidence that the
full 4-model neural-network jobs have run. Unit tests do not require NP data,
but running against the frozen checkpoints requires the existing
outputs/choice/.../s3_train checkpoints and latent files on the local machine.
This suite is an experimental implementation. Check CI and perform the smoke
run before allocating substantial GPU time.
