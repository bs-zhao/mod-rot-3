# functional_composition — complete independent ZXY rotation suite

This is **one integrated rotation experiment**, not 20 separately launched methods.  
It uses only frozen NP subject representations `Z`, observed stimuli `X`, binary responses `Y`, and fits one orthogonal matrix `R`. It uses no CR3/CPT formula or true `theta` when choosing the rotation. NP code and other rotation suites are untouched.

## What the optimization actually does

1. Read NP S5 all-observed-trials `Z` and S1 `X,Y`, confirm SHA256 provenance.
2. Fit PCA on TRAIN subjects; use the first K PCs and **whiten** them. Subsample FIT participants deterministically.
3. Fit a **stimulus response-profile proxy** from only the FIT ZXY: for each participant, ridge-project binary Y onto up to 18 data-derived polynomial X features (constant, linear, square, cross). Fit cubic polynomial `F(q)` to these coefficients. Evaluate fidelity on held-out FIT subjects.
4. Transform that **same learned function** under each tested orthogonal rotation by rotating its symmetric degree-1/2/3 coefficient tensors. No per-angle surrogate refitting or oracle derivatives.
5. Optimize a joint dimensionless loss using the following independently adjustable pieces:
   - `weight_operator`: 08 — soft codelength of monomial function modules. A simple `ab` needs one product module; a 45° rotation uses two squared-axis modules, which is generally more costly. Products are **not automatically penalized more than squares**.
   - `weight_fingerprint`: 11 — stimulus-effect profile effective rank (above rank one).
   - `weight_jointdiag`: 13 — joint gradient covariance diagonalization over data-derived X channels, only one auxiliary structural preference.
   - `weight_separation`: 07 — effective rank of pairwise interaction submodules; off by default because minimizing it *alone* prefers additive squares over simple products.
   - `weight_hypergraph`: 09 — edge/support codelength. Off by default for the same reason.
6. Save optimizer diagnostics, original-Z→U affine map, R, and **all-subject U** in the normal ignored `outputs/` directory. Only after saving rotation results, evaluate true parameters on TEST subjects.
7. Report all 20 **operationalized metrics**, with unambiguous distinctions between genuine estimators, response-profile proxies, and negative controls. Plot identity-centered and best-centered 2D rotation-plane loss curves. Compute 20 (restricted prequential predictive MDL) **after optimization**, since a subject-forward refit is too expensive to evaluate at every candidate angle.

### Twenty indicator names and validity status

| ID | Metric | How this suite measures it | Optimize by default? |
|---|---|---|---|
| 01 | EDF | Soft count of univariate polynomial modules; **not spline EDF** | No |
| 02 | Derivative total variation | Mean absolute pure axis curvature in learned proxy | No |
| 03 | Derivative sign changes | Changes across one-dimensional coordinate paths | No |
| 04 | Spectral entropy | Per-axis polynomial degree energy entropy | No |
| 05 | Functional ANOVA | Energy-weighted distinct-coordinate interaction order; **ANOVA proxy** | No |
| 06 | Friedman H | Pair-only polynomial energy proportion; **not the exact H-statistic** | No |
| 07 | Pairwise separation rank | Effective SVD rank of two-variable polynomial modules | Optional (0) |
| 08 | Operator description length | Soft count of monomial function modules | **Yes** |
| 09 | Hypergraph MDL | Soft count of distinct active interaction supports | Optional (0) |
| 10 | Tensor-train rank | Tensor matricization effective rank, **rotation-invariant control** | No |
| 11 | Effect fingerprint rank | Per-axis derivative-profile effective rank across subjects | **Yes** |
| 12 | Effect direction drift | Direction instability of stimulus-profile derivatives | No |
| 13 | Contextual joint diagonalization | Joint off-diagonal gradient covariance over X effect channels | **Yes** |
| 14 | U–X mixed derivative | Finite-difference X derivative of response-profile proxy, soft axis/channel support | No |
| 15 | X context gates | X-only learned quantile contexts, axis×context support | No |
| 16 | Polynomial chaos | Rotated polynomial soft sparsity; **not orthogonal PCE** | No |
| 17 | Conditional MI | Heldout axis-ablation response-profile MSE; **not MI** | No |
| 18 | HSIC | RBF-HSIC of an axis versus a linearly residualized response profile; **not conditional HSIC** | No |
| 19 | PID synergy | Positive bound from discretized pair MI conditional on X bin; **not exact PID** | No |
| 20 | Prequential MDL | Forward-subject Bernoulli log-loss plus code nats for sparse polynomial readout; **restricted MDL proxy** | After fit |

Importantly, 07/08/09/16 share polynomial coefficient statistics and are **not** independent evidence. 10 is intentionally a negative control. 17–19 are diagnostic approximations only. These distinctions are also recorded in `functional__20_metrics.json`.

## Usage on your already trained CR3 β=0.0005, latent_dim=8

From the repository root (Windows CMD):

```cmd
python rotations/functional_composition/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
```

Optional heavier run with more starts and angle samples:

```cmd
python rotations/functional_composition/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --set starts=8 --set maxiter=300 --set sweep_steps=73
```

For β=0.0001, change only the condition to `beta0p0001_lambda0_ctx0p25to0p8_L8`. To rotate all 8 dimensions, specify `--representation full --rotation-dim 8` (much more expensive and potentially poorly conditioned when eigenvalues are tiny).

The entire suite has its own `run.py`, `experiment.py`, `surrogate.py`, `metrics.py`, `mdl.py`, `optimizer.py`, `diagnostics.py`, `config.json`, tests and this README. No central rotation dispatch or batch run is created.

## Output

`outputs/choice/<MODEL>/rotations/functional_composition/<CONDITION>/repeat01/run_<fingerprint>/`

- `rotation__matrix.npz` — SO(K) rotation R, original-Z→U map A, affine offset and PCA information.
- `rotation__U.npz` — all subjects' U, subject splits.
- `rotation__fit.json`, `rotation__complete.json` — full run configuration, source SHA, surrogate fidelity and optimizer metrics.
- `rotation__test_evaluation.json`, `rotation__test_corr2.png` — **TEST-only theta diagnostics** (after fit), comparable to other suites.
- `rotation__method_arrays.npz` — learned degree 0–3 polynomial tensors and FIT response profiles.
- `functional__20_metrics.json` — 20 per-rotation diagnostics, **identity vs best** (includes slow MDL).
- `functional__plane_curves.csv`, `functional__plane_curves.png` — angle curves for selected U–U planes.
- `functional__metric_changes.png` — change in each metric from identity.

### Scientific limitations

- The response-profile function is **not the exact logit generating function**. The full ZXY→Y map is only approximated by a finite stimulus basis and a cubic-in-Z polynomial. Complex CR3 nonlinearities can be missed. Inspect `heldout_profile_r2` and binary cross entropy against the constant baseline; do not interpret fitted curvature when fidelity is poor.
- S5 encodes **all of a participant's observed Y** into their Z. Held-out subject evaluation is not the same as predicting an unobserved trial from an unobserved response set.
- Surrogate binary profiles are least-squares projections; clipping predicted probabilities for validation CE is a numerical approximation, not a Bernoulli maximum-likelihood fit.
- Maximizing simplicity **cannot uniquely identify theta in general**. It can favor stable but incorrect axes. The toy test `ab` versus rotated squares only establishes one of many desirable structural properties.
- Trial X values are drawn from the observed design; finite differences in X are extrapolation/interpolation assumptions.
- Weight changes use no true theta automatically. Tune on analytic toys or independent discovery datasets, then evaluate CR3 parameters only at the end.
## V2: calibrated module selection and integrated ablation comparison

**V2 supersedes the descriptions of 08 and 20 in the older table above.**

- **08:** noise-calibrated operator code. Two disjoint FIT-subject half-fits
  estimate the uncertainty of each rotated polynomial term. An uncertainty
  threshold and fixed relative floor determine soft module activation.
  A simple product does not receive an arbitrary penalty over a square.
  The half-fit scale is heuristic, not a calibrated z-test or universal MDL.
- **20:** nested nonlinear predictive code. In each forward FIT-subject
  prefix, select degree-2/3 polynomial modules using inner-validation
  Bernoulli log-loss plus structural code; refit on the prefix and evaluate
  future FIT subjects. A full 4D linear subspace alone cannot win by default.
  Module shortlists are ranked using **inner-FIT only** responses, not future.
- **08/11/13 ablations:** three independent optimization objectives run
  **inside this one suite**, yielding separate R/U, held-out TEST correlation
  heatmaps, and angle-loss curves. No other rotation package is invoked.
- **MDL20 candidate selector:** ranks identity, joint, 08-only, 11-only,
  and 13-only based on FIT-subject predictive code. Saves the winner as
  an additional candidate, without replacing the joint rotation.
  True theta is not used to rank candidates.

New configurable keys: operator_mode, operator_relative_floor,
operator_noise_z, operator_temperature, ablation_enabled, ablation_starts,
ablation_maxiter, ablation_sweep_steps, ablation_sweep_pairs,
mdl_compare_candidates, mdl_candidate_pool, mdl_max_extra_modules,
mdl_inner_validation_frac, mdl_min_delta_nats, mdl_selection_trials.

### Run this single package

```cmd
python rotations/functional_composition/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4
```

For a quick fit without the additional 08/11/13 runs or predictive selector:

```cmd
python rotations/functional_composition/run.py --model CR3 --condition beta0p0005_lambda0_ctx0p25to0p8_L8 --rotation-dim 4 --set starts=1 --set maxiter=20 --set ablation_enabled=false --set mdl_compare_candidates=false --set sweep_steps=9
```

### New outputs in the **same suite** run directory

- functional__ablation_fit.json: 08/11/13 fit information, nested MDL20
  candidate scores, nonlinear modules and held-out FIT-subject predictions.
- functional__candidate_rotations.npz: candidate matrices for identity,
  joint, 08-only, 11-only, 13-only and MDL-selected coordinates.
- functional__ablation_comparison.csv and functional__ablation_test.json:
  FIT predictive MDL vs independently audited TEST theta recovery.
- functional__ablation_plane_curves.csv/png: separate loss-vs-angle curves
  for each of the three single-objective ablations.
- ablation__<candidate>/rotation__matrix.npz, rotation__U.npz,
  rotation__test_evaluation.json, rotation__test_corr2.png: independent
  saved transformation and TEST diagnostics for each candidate.
- rotation__test_truth.npz: TEST-only audit copy, created strictly after
  the joint R/U are saved. FIT objectives never receive this array.

**Scientific boundary:** a finite-degree ZXY response-profile surrogate
is not the true CR3 generating function. Selecting a candidate using FIT
predictive-MDL can overfit the candidate library, so TEST theta should only
be used for final evaluation, not for changing weights. S5 Z contains
each subject's already-observed Y and is not an unseen-trial forecast.
