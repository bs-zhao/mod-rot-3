# Step 5 — WY-only matrix metrics, wave A

**Status: implementation and numerical tests only. Real C1/C2/C3/CR3
experiments have NOT been run by the authoring cloud session. That session
cannot access the Windows machine or its local case matrices.**

This is the next experimental batch, not a successful universal rotation
method. Earlier global/local first-gradient concentration experiments failed
on CR3. This batch estimates *second-order interaction structure* using two
different numerical function estimators, and reports the disagreement rather
than changing metric signs or choosing a separate rule for each model.

## Data boundary

Allowed inputs are actual `cases/*/W.npy`, common `Y.npz`, and metadata needed
to identify U/X columns and original subject/trial rows. The scorer reads no
theta, behavioral formula, neural-network weights, `Zstd.npy`, `R_best.npy`,
case R, or relative rotation matrix. `read_audit.json` enumerates the files
opened through the input loader. This is an application audit, not an OS
sandbox.

Case order is shuffled before fitting. The first anonymous W establishes a
reference coordinate system. Every other orthogonal map is estimated from
the U columns of the two W matrices by Procrustes registration. It is not
obtained from saved R or angle labels. U repetition and X alignment are
checked over every row of every W.

Case names and angles enter endpoint evaluation only after `blind_scores.csv`
is saved and its SHA256 recorded. Baseline is a supervised reference for
evaluation; it is never a scoring input. Repeated research across these same
four datasets remains exploratory, regardless of case anonymization.

## Numerical estimators

Every estimator streams through **all rows of all 5,000 subjects**. No
behavioral equation is assumed. The polynomial and kernel families are
explicit approximation choices, not representations guaranteed to match an
unknown true function.

| Estimator | Fixed default settings | Purpose |
| --- | --- | --- |
| Polynomial kernel basis | Complete degrees 2 and 3 | Two resolutions of a generic polynomial sieve |
| Nystrom RBF kernel ridge | 128 centers; lengthscales 0.5, 1, 2 | Smooth nonparametric approximation with a different bias |

Polynomial monomials carry square-root multinomial weights. Within each
degree this yields an isotropic polynomial kernel. U has one shared scalar
scale; it is **not** restandardized independently after each rotation. The
isotropic RBF kernel is also rotation equivariant. Consequently the same
fitted function and its derivative tensors can be transformed exactly to
each case without retraining separately at every angle.

The RBF estimator is a low-rank kernel-ridge approximation. It is related to
a GP posterior mean, but it is not a full Bayesian GP and does not provide
posterior uncertainty.

For each estimator:

1. Split subjects 75%/25%, identically across cases.
2. Learn coordinate scaling and RBF centers using training subjects only.
3. Choose ridge from `1e-6, 1e-4, 1e-2` using mean validation normalized MSE
   across the requested Y channels. Baseline identity/rank is not involved.
4. Refit coefficients using **all subjects and all trials**.
5. Evaluate derivatives at observed W rows, by default eight rows per
   subject, with equal total weight per subject.

Validation normalized MSE <= 0.10 is a fixed *prediction adequacy screen*.
It is not proof of derivative accuracy. A candidate that passes the screen
can still have incorrectly estimated Hessians. The outputs explicitly keep
`derivative_accuracy_verified=false`.

The default channels are NN logit and probability. Choices can also be
specified, but their irreducible noise can prevent the same adequacy screen
from passing; no model-specific relaxation is made.

## Metrics and controls

Write the estimated Hessian in standardized coordinate units as H. Average
squares over empirical observed locations with equal subject weights:

`E_UU[i,j] = mean(H[u_i,u_j]^2)` and
`E_UX[i,k] = mean(H[u_i,x_k]^2)`.

For an energy vector e, `tailK_l2 = sqrt(sum(e outside top K)/sum(e))`.
It measures the size of the remaining L2 tail, rather than merely whether
an entry exceeds a threshold. In particular it distinguishes
`[2,.011,.001]` from `[2,.001,.001]`.

| Family | What is concentrated? |
| --- | --- |
| `UU_all_*` | Unique UU entries, with twice the off-diagonal squared energy to preserve Frobenius multiplicity |
| `UU_edges_*` | Off-diagonal UU interaction edges only |
| `UX_entries_*` | Entries of the U-by-X interaction matrix |
| `UX_column_tail*_l2` | U-axis effect concentration separately for each X coordinate |
| `U_curvature_*` | Per-axis diagonal curvature energy |
| `UU_interaction_fraction` | Relative strength of off-diagonal UU effects; deliberately distinct from sparsity |
| `U_global_sign_inconsistency` | Global gradient sign cancellation, invariant to axis sign flips |
| `control_*` | Full UU/UX Hessian and U-gradient squared Frobenius norms; rotation invariant |

Each concentration family includes tail 1/2/4 and normalized entropy. These
are correlated diagnostics, not independent tests. An off-diagonal matrix
with a single active edge has tail1=0 just like a zero matrix: this is why
edge sparsity alone is not a sufficient recovery criterion. Likewise, a
globally consistent gradient sign is not proof of monotonicity conditional
on every other coordinate. Stimulus-dependent signs can be legitimate.

Global orthogonally invariant smoothness measures cannot choose a rotation.
They are controls, not candidate loss functions. Lower means simpler for
every non-control score in every model; no per-model sign inversion is used.

## Evaluation and outputs

Flat curves fail. A plane passes only if both 0 and 90 degrees agree within
a fixed numerical tolerance, and both have strictly lower loss than **every**
15/30/45/60/75 degree interior case. C1/C2/CR3 have 6 planes and 37 cases;
C3 has 15 planes and 91 cases.

The four-model comparison requires all four completed 5,000-subject datasets,
the same estimator/channel/metric, exactly matching experimental settings,
adequate prediction, baseline rank 1, and every plane passing. Missing models
cannot become successes. `all_four_pass` is a per-estimator screen;
`cross_family_pass` additionally requires the SAME channel/metric to pass
all four datasets under at least one polynomial AND one RBF setting. The
CLI candidate count uses this stronger cross-family screen. A full pass here is a candidate on this testbed,
not a proof of identifiability or recovery at arbitrary rotations.

Outputs include `protocol.json`, `fit_quality.csv`, per-estimator fit JSON,
full derivative fields at evaluation rows, `blind_scores.csv`, `reveal.json`,
`metric_summary.csv`, and `read_audit.json`. Existing nonempty output folders
are refused to preserve earlier failures. Use a new `--run-name` for a new
batch. This implementation does not yet produce confidence intervals or
arbitrary full-dimensional rotation optimization; those are subsequent
validation requirements if a common metric survives.

## Running in Windows

From the project root in `py11`:

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run.py --model C1
python supervised_abs_rotation/step5_matrix_metrics/run.py --model C2
python supervised_abs_rotation/step5_matrix_metrics/run.py --model C3
python supervised_abs_rotation/step5_matrix_metrics/run.py --model CR3
python supervised_abs_rotation/step5_matrix_metrics/run.py --compare
```

Or run `supervised_abs_rotation\step5_matrix_metrics\run_wave.cmd` to perform
this one package's four-model batch sequentially, stopping on errors. This
does not run unrelated rotation packages.

Results live under each original `abs_mse` folder:
`step5_matrix_metrics/waveA_seed381/`. The final comparison CSV goes in
`outputs/supervised_abs_rotation/step5_waveA_seed381_four_models.csv`.

`--step2-dir` accepts a custom input directory and `--output` a new output
directory. `--channels`, `--estimators`, `--centers`, `--per-subject`, `--chunk`,
and `--seed` are explicit. For a valid four-model comparison apply the same
settings and run name to all four models. `--chunk` controls memory only;
`--per-subject` changes derivative evaluation density, not fitting sample size.

Requirements are existing NumPy and SciPy; direct script execution needs no
`pip install -e .`. Tests use pytest.

## Subsequent branches, not implemented claims

If wave A fails, retain it and investigate function decomposition rather
than repeating first-gradient concentration. Three distinct branches are:

1. Centered ANOVA kernels / additive GPs: compare single-axis and interaction
   component complexity at matched prediction error. Use generic kernels,
   never task-specific utility forms or stimulus flags.
2. Dependence-aware functional ANOVA: rotations generally change input
   dependence. Classical independent-input Sobol pick-freeze can create
   unsupported coordinate combinations. Do not call it a valid empirical
   Sobol decomposition without a joint-distribution/support argument.
3. Finite-scale local effect contrasts / ALE: assess interactions and
   conditional monotonicity with support and step-size diagnostics, and
   compare to point derivatives. Prediction adequacy alone is insufficient.

No ranking-based weights or composite metric are tuned separately for C1,
C2, C3, or CR3. A batch failure is saved and reported as a failure.

Primary methodological sources:

- Durrande et al., *ANOVA kernels and RKHS of zero mean functions for
  model-based sensitivity analysis*: https://arxiv.org/abs/1106.3571
- Duvenaud, Nickisch & Rasmussen, *Additive Gaussian Processes*:
  https://arxiv.org/abs/1112.4394
- Chastaing, Gamboa & Prieur, *Generalized Hoeffding-Sobol Decomposition for
  Dependent Variables*: https://arxiv.org/abs/1112.1788
- Apley & Zhu, *Visualizing the Effects of Predictor Variables in Black Box
  Supervised Learning Models*: https://doi.org/10.1111/rssb.12377

These sources motivate future branches. Wave A does not implement their
ANOVA/ALE algorithms and should not be described as doing so.
