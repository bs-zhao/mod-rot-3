# 10 — Cross-model analytic Hessian simplicity atlas (C1/C2/C3/CR3)

Standalone experiment with **Gap 1 analytic screening**, followed by **Gap 2 formula-generated numeric WY and Gap 3 frozen-NP-generated numeric WY simultaneously**. No NP retraining or neural differentiation. See [the implementation and run instructions for Gap 2 + Gap 3](README_GAPS23.md). The comparison uses a supervised TRAIN-only Z-to-T benchmark mapping, not a claim of unsupervised axis recovery.

Research sources: [history Hessian complexity](../../readmes/history/README_HESSIAN_COMPLEXITY_SPARSITY_METRICS_20261109.md), [history rotation and sparsity](../../readmes/history/README_SPARSITY_SIMPLICITY_AND_FUNCTIONAL_ROTATION_20261109.md), [rotation research overview](../../readmes/rotation_research/01_project_overview.md), and [Hessian simplicity proposed experiment](../../readmes/rotation_research/02_hessian_simplicity_experiment.md).

## One command on Windows CMD, from repository root

~~~cmd
conda activate py11
git pull origin main
python math/10_multimodel_hessian_simplicity/run_pipeline.py --models C1 C2 C3 CR3
~~~

Alternatively run **run_all.cmd** from repository root. It now runs verified Gap 1 followed by **paired Gap 2 and Gap 3**. The original run.py remains the stand-alone Gap1 oracle entrypoint; the pipeline automatically reuses valid saved Gap1 data. The requested spelling **CR1** is not a model in this repository. A command containing CR1 is a **visibly warned alias for CR3**. It does NOT test an independent CR1. The actual four generators in np/ are C1, C2, C3 and CR3; C3 has **6 latent theory parameters**, not 4.

**Precondition**: original simulation files must already exist at outputs/choice/MODEL/s1_generate/s1_generate__dataset.npz; CR3 also needs the exact saved 500-trial X bank. If missing, run the existing S1 separately (which does not overwrite an existing dataset):

~~~cmd
python np/s1_generate.py --model C1
python np/s1_generate.py --model C2
python np/s1_generate.py --model C3
python np/s1_generate.py --model CR3
~~~

The oracle experiment will never silently replace real simulation data with synthetic data. A synthetic debug flag is provided for C1–C3 only; debug outputs are not scientific data.

By default function-value contrasts are enabled; use --no-functional to benchmark Hessian-only speed, or adjust --function-anchors / --function-step. Higher-order third derivatives use centered finite differences of analytically exact second derivatives, with fourth-order optional.

More thorough run, including fourth derivatives and 5-degree pairwise grid:

~~~cmd
python math/10_multimodel_hessian_simplicity/run.py --models C1 C2 C3 CR3 --subjects 60 --trials-per-subject 20 --angles 0:90:5 --multiway 16 --fourth
~~~

Self-tests:

~~~cmd
python -m unittest discover -s math/10_multimodel_hessian_simplicity -p "test_*.py" -v
~~~

Minimal smoke test not requiring original C1 data:

~~~cmd
python math/10_multimodel_hessian_simplicity/run.py --models C1 --synthetic-debug --subjects 4 --trials-per-subject 3 --angles 0,45,90 --planes 12 --multiway 0 --plots 0 --no-higher
~~~

## Coordinates, target, derivative scope

Models are copied from the exact toy-generative formulas: C1 4x3, C2 4x4, C3 6x4, CR3 4x6 (T dimensions by X dimensions). We compute continuous logit and probability Hessians. **A single Bernoulli choice has no classical analytic Hessian; the conditional expected choice is probability**, so the two distinct analytic channels are logit and probability. This is not 3 independent analytic Y outputs.

Define standardized true parameters v_i=(T_i-mean(T_i))/sd_T_i and original X (whose derivative unit is multiplied by sd_X). Given orthogonal R and U=R v, exact chain rule on the same physical T,X points is:

~~~text
H_UU = R diag(sd_T) H_TT diag(sd_T) R^T
H_UX = R diag(sd_T) H_TX diag(sd_X)
~~~

**Critical**: rotate each local Hessian FIRST; only THEN apply nonlinear statistics (RMS, Mean Absolute, Q99, etc.). Rotating an already aggregated RMS matrix is mathematically wrong. Original data and labels never change with angle.

We evaluate all d(d-1)/2 pairwise planes, 0–90 degrees in 10-degree increments by default. We also evaluate a fixed number of randomly sampled embedded SO(3), SO(4) and SO(d) rotations. This is a **random multi-axis stress test**, NOT exhaustive proof of global minimality.

CR3 X includes zero gain/loss amounts; X derivatives can be undefined at zero (for alpha<1). To compare ZZ and ZX on the same set, we restrict to differentiable X rows and report valid_x_fraction in metadata. The resulting expectation is explicitly **conditional on differentiable rows**, not a full unconditional average over all 500 X.

## Actual implemented metrics

The metric registry is computed by metrics.py, not written as claims about future implementation. Metric categories include:

| Level | Computations |
|---|---|
| 14 Hessian field aggregators | Signed mean, Mean Absolute, RMS, Mean Square, variance, SD, median, Q90, Q99, max, sign consistency, positive part, negative part, trimmed absolute mean |
| ZZ structure | diagonal and off-diagonal L1/L2/energy, normalized offdiag ratios, per-row off-diagonal sparsity |
| ZX structure | global L1/L2, per-U row support, per-X column support |
| Best k-term sparsity | Tail-L1, Tail-L2, squared Tail-L2, relative variants, next-largest ignored effect for k=1..4 (where valid) |
| Support-count approximations | hard threshold, rational/exponential soft-L0, capped L1/capped-square, log-sum, Lp p=.25/.5/.75, adaptive penalized k |
| Relative concentration | Hoyer, entropy, Rényi-2, effective support, participation ratio, Gini, top-1 energy |
| Local-first versus matrix-first | average pointwise row Tail-L2/soft-L0 versus Tail-L2 of aggregated RMS/mean |
| Spectral/matrix | row Gram E[HH^T], Gram offdiag, spectral absolute, covariance of unique Hessian elements, Gram effective rank |
| Higher order | Third derivatives via finite differences OF EXACT ANALYTIC SECOND derivatives; self/other-U and ZX-U modulation; optional fourth own-axis |
| Function-value contrasts (separate from Hessian) | Default controlled 3-point own-axis second differences; 4-corner UU/UX mixed non-additivity; row-wise Tail-L1/L2 and soft support; **NOT** population Sobol |
| Sanity diagnostics | Exact orthogonal invariance of local Frobenius squared norm, flatness, cancellation and sign persistence |

Each recorded metric is labelled loss (candidate lower-is-simpler) or diagnostic (not a legitimate blind optimization target). Do not mistake invariant metrics for successful axis recovery.

### Important Tail-L2 preference test

With k=2 fixed a priori:

~~~text
A = [2, 2, 0.01, 0.001] -> Tail-L2 = 0.0100498756
B = [1, 0.5, 0.01, 0.01] -> Tail-L2 = 0.0141421356
~~~

A has LESS residual complexity than B. Total L1/L2 and Hoyer need not rank the same way. Change k or absolute/relative threshold and the preference may reverse; report sensitivity and avoid tuning k from the true T answer.

## Results

All new artifacts are written to **outputs/math/10_multimodel_hessian_simplicity/**, leaving older experiments untouched.

~~~text
manifest.json
all_models_summary.csv
cross_model_ranking.csv
C1/
  metadata.json
  metric_catalog.csv
  curves.csv
  screens.csv
  ranking_by_model_channel.csv
  multiway_random_rotations.csv
  multiway_comparison.csv
  plots/*.png
C2/ ...
C3/ ...
CR3/ ...
~~~

- curves.csv: wide rows, model × channel × plane × angle, measured scores.
- cross_model_ranking.csv: per-metric coverage and worst/mean successful-plane fraction across **all four models and both Y channels**. Missing metrics are flagged as partial coverage, not silently treated as successes.
- screens.csv: actual minimum angle, 0/90 endpoint values, flatness flag, minimum direction.
- ranking_by_model_channel.csv: both **strict** (exact 0°/90°) and **near endpoint** (default within 10°, configurable via --endpoint-tolerance) fractions of planes with a **nonflat** minimum. Flat curves are failures, not successes.
- multiway_comparison.csv: how many random SO(3)/SO(4)/SO(d) candidates beat the identity. The top-level cross_model_ranking.csv also counts these sampled counterexamples and records metric coverage; this is not an exhaustive SO(d) search.
- metadata.json: source hash, original units, singular-domain coverage, derivative method and essential provenance.

**The standalone Gap1 outputs do not establish NN recovery.** The paired numeric-WY Gap2+3 results are saved in a separate gaps23 directory; see [README_GAPS23.md](README_GAPS23.md). In particular: C1–C3 logit H_TT is exactly zero, so ZZ-logit metrics are flat and cannot distinguish parameter axes. Even if a complexity loss prefers a particular basis, that does not prove the basis is the true psychological coordinate system.

**Now implemented:** Gap 2 and Gap 3 run jointly after Gap 1, use the same four numeric Y channels and matched estimators, and report source-specific errors and landscape agreement. Remaining: richer estimators and actual unsupervised rotation optimization. The fitted decoder does not become an oracle just because its Y prediction is accurate. Coordinate maps from Z to T are supervised references only.

Relevant literature: Li (1992) principal Hessian directions; Sobol & Kucherenko (2009) DGSM; Roustant et al. (2014) crossed DGSM; Hoyer (2004), Hurley & Rickard (2009) sparsity; Tibshirani (1996) Lasso; Candès, Wakin & Boyd (2008) reweighted L1; DeVore (1998) best k-term approximation; Sobol (2001) functional ANOVA; Friedman & Popescu (2008) interaction; Apley & Zhu (2020) ALE. Full citations/DOIs are in the referenced history files. These citations do not claim those authors proved Hessian rotation axis recovery.
