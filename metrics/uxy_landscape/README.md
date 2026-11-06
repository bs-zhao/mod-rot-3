# UXY metric landscape — experiment 1

Independent experiment in `metrics/`, with no imports from rotation suites.
This is a **supervised descriptive diagnostic**, not a new unsupervised rotation
method. It addresses whether sample-computable UXY metrics distinguish a
TRAIN-linear reference from mixed coordinates and saved historical coordinates.

```powershell
python metrics/uxy_landscape/run.py
python metrics/uxy_landscape/run.py --workers 4
python metrics/uxy_landscape/plot_comparisons.py --run outputs/metrics/uxy_landscape/run_HASH
python metrics/uxy_landscape/run.py --models C1 CR3 --smoke --no-plots
python -m pytest metrics/uxy_landscape/test_metrics.py -q
```

## Reference and five responses

TRAIN OLS maps the original full 8-dimensional latent Z to TRAIN-standardized
true parameters. An affine offset is represented by TRAIN centering. A single
isotropic RMS scale is fixed before rotations. No individual rotated axis is
restandardized. The resulting A* is usually rectangular/oblique and supervised;
its actual held-out correlations are measured, never assumed perfect.

Each actual selected trial has five fixed responses:

1. Actual observed binary choice.
2. One Bernoulli choice sampled from the frozen decoder probability.
3. Frozen decoder logit difference.
4. Exact generating logit at actual X and the actual subject parameter.
5. Exact logit plus independent Gaussian noise; SD=.25 × SD of TRAIN calibration
   logits. All rotations reuse the same choice draws and noise.

True parameters are authorized only to construct/audit reference coordinates,
generate the two oracle response versions and give a separately labeled oracle-U
positive control. Neither metric module receives theta, a decoder, a generator,
or the original Z. Metric computation consumes only U/X/Y plus fixed numerical
settings and subject-fold metadata.

## Coordinates compared

- U* (zero angle), every pair plane from -90° to 90° in 5° steps.
- A fixed full-dimensional Givens-product joint path (all planes change).
- 16 fixed Haar orthogonal rotations.
- All compatible saved coordinates in the completed historical-transfer
  manifest, and saved complete-coordinate outputs under other rotation suites
  for the same condition/repeat/input hashes. No rotation suite is run.
- Standardized oracle theta, explicitly an unavailable supervised control.

Exact duplicate coordinates are cached but labels retained. Saved historical
coordinates need not belong to the same subspace or orthogonal family as U*;
they are comparison cases, not falsely described as pure rotations of U*.
Models run in four bounded worker processes by default, each with one numerical
thread. Case caches include fixed X/Y/subject hashes and input provenance is
checked on resume. No rotation fitting is launched.

## Metric families

The catalogue includes marginal Pearson/Spearman/Kendall, tie-corrected
Chatterjee xi, regularized RDC, biased squared dCor, normalized RBF-HSIC,
quantile-bin eta²/MI, an explicitly named adaptive-grid NMI proxy; linear partial
correlations, residualized-rank partial correlation, U-centered pdCor, CODEC,
continuous kNN CMI, coarsened-X CMI and signed interaction-information proxies.

It also includes grouped-CV spline-ridge GAM R², delete-variable partial and
semipartial CV R²; complete functional linear/quadratic ridge prediction controls;
quadratic-proxy module code/L1, pure/mixed Hessian fractions and normalized L1,
XU support/entropy/Geomin, gradient-energy concentration, profile effective rank;
U kurtosis/fourth moments and exact orthogonal invariance controls.

Raw per-axis statistics are retained. Axis RMS and axis entropy are two
aggregates of each estimator, **not two independent sources of evidence**.
Structural proxies likewise share one fitted function; their CV fidelity is
reported beside them. The catalogue JSON gives actual counts and all names.
The complete operational definitions are in [METRICS.md](METRICS.md).

### Estimation contracts

- 384 NP TEST subjects × 4 actual trials (fixed seed). Descriptive regressions
  use 3 folds split by subject, so repeated trials never cross fold boundaries.
- Nonparametric dependence estimators use one actual trial from each of 256
  different subjects; no repeated U rows masquerade as independent subjects.
- Low-order proxy: the tensor product of complete degree-2 U and X polynomial
  features, with orthonormal symmetric packing and ridge alpha=1. It is fit once
  per base coordinate system, then exactly transformed along reference scans.
  It is **not** an MLP derivative bank and is not claimed to be the true function.
- GAM uses additive cubic B-splines. No claim of arbitrary interaction capture.
  CV partial/semipartial R² may be negative. Unlabeled spline knot fitting uses
  the descriptive sample; this is not prospective prediction validation.
- Y centering/scaling is fixed on the full descriptive sample, before grouped
  coefficient fitting. Thus these CV diagnostics are not a fully nested
  preprocessing/prediction assessment; do not use them as fresh test estimates.
- Continuous kNN CMI is marked NA for binary Y, never computed by artificial
  label jitter. Deterministic outcomes and high-dimensional conditioning may
  saturate or destabilize conditional estimators.
- CODEC denominator zero is NA, not zero evidence. pdCor is not an equivalent
  characterization of conditional independence. No inferential p-values.
- Coarsened-X CMI controls only 3×3 quantile strata of two X PCs. It is not full
  multivariate CMI. Adaptive grid NMI is not MIC/MICe; signed interaction
  information is not PID synergy. Exact MIC is not implemented in this first
  experiment; it is not silently replaced under its name.
- Full Euclidean RBF prediction, complete polynomial prediction and complete
  derivative energies are negative controls, not axis identification objectives.
- S5 Z already contains observed choice. NP TEST outcomes have been examined in
  earlier research. This study is exploratory, not independent fresh validation.

## Outputs

Under ignored `outputs/metrics/uxy_landscape/run_<fingerprint>/`:

- Protocol/source hashes; per-model input provenance, A*, all chosen UXY rows,
  response draws, noise SD, folds and real reference recovery.
- `metric_values.csv`, raw `axis_metrics.npz`, coordinate quality and case lists.
- All plane and joint angle curves in vector PDFs for every response version;
  all-metric joint-path heatmaps and raw CSV values.
- Reference metric positions, min/max angles, availability and descriptive
  landscape agreement between oracle/decoder/noisy outcomes.
- Historical comparisons, including failed methods. No candidate is selected
  using parameter recovery; outcome-dependent rankings remain posthoc diagnostics.

The optional postprocessing command adds historical-coordinate atlases, color
keys for the plane scans, a cross-response landscape agreement heatmap, and
four-model/five-response example curve grids. It changes no metric values.

## Primary formula references

- [Chatterjee xi](https://arxiv.org/abs/1909.10140).
- [Azadkia–Chatterjee CODEC, rank/nearest-neighbor formulas](https://arxiv.org/html/1910.12327v6).
- [RDC](https://arxiv.org/abs/1304.7717).
- [Székely–Rizzo U-centering](https://mariarizzo.github.io/energy/reference/centering.html)
  and [partial distance correlation](https://arxiv.org/abs/1310.2926).
- [Frenzel–Pompe continuous CMI estimator, documented formula](https://jakobrunge.github.io/tigramite/_modules/tigramite/independence_tests/cmiknn.html).

Formulas are implemented independently; no third-party source bodies are copied.

## Completed experiment

[Results and limitations](RESULTS.md), [all metric definitions](METRICS.md).

- [Four-model / five-response module-code angle curves](figures/proxy_module_code__four_models_five_Y.png).
- [Four-model / five-response CODEC angle curves](figures/codec_given_otherU_X__axis_rms__four_models_five_Y.png).
- [Cross-response landscape agreement for all metrics](figures/response_landscape_agreement.png).
- Per-model vector PDFs for every metric/response and all historical cases are in `figures/<MODEL>/`.
- Compact evidence and reference-coordinate audits are in `results/`. Full scalar/axis arrays and portable UXY samples remain in the ignored run output, and in the user-facing export bundle.
