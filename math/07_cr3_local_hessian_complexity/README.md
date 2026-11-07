# 07 · CR3 local ZZ / ZX Hessian complexity atlas

**Independent experiment folder** extending 06's six-plane \(A^*\)-centered
rotation study. Previous 01–06 code, saved NN checkpoints, original Z/X/Y,
and the old 06 mean Hessian rotation files are not modified.

## Why not the old mean Hessian?

A mean Hessian can be zero for a highly nonlinear function. For example,

\[
f(z)=z^2\implies H_{zz}=2,\qquad
f(z)=z^3\implies H_{zz}=6z.
\]

For symmetric sampled z, the *mean* cubic Hessian vanishes despite its
curvature varying with z. Moreover, \(f(z,X)=z^2g(X)\) has \(H_{zz}=2g(X)\):
the Hessian changes over X, but not over its own z. Conflating those
forms of variation mischaracterizes polynomial degree.

The new experiment retains *pointwise* second-, third-, and fourth-order
derivatives on a common held-out subject × valid stimulus grid, plus
controlled single-coordinate and pairwise sweeps. All scores are
computed for **nine source/Y combinations**:

| Hessian/function source | Choice | Logit | Probability |
|---|---|---|---|
| Exact CR3 generator | \(H[\mathbb E({\rm Choice})]=H[p]\) | \(H[\eta]\) | \(H[p]\) |
| **Fit numeric Y from formula** | sampled Bernoulli choice | formula logit | formula probability |
| **Fit numeric Y from frozen NN** | hard NN choice | NN logit | NN probability |

The actual NN never receives an autograd call: all derivatives in
the latter two rows are from the **same fitted polynomial surrogate**.
For the analytic source, the **second derivative is CR3-exact**;
third/fourth derivatives are symmetric finite differences of the exact
Hessian, checked with a half-sized difference step and reported to
\`analytic_fd_convergence.csv\`. They are not closed-form analytic
third/fourth derivatives. The regression and prior map are exactly
the 06 step-04 source mapping \(T_{\rm hat}=A^*Z+b\).

## Orientation and coordinate conventions

For each of the six parameter-coordinate pairs and angle
\(\theta=0°,5°,\dots,90°\):

\[
A(\theta)=R_{ij}(\theta)A^*,\qquad
U(\theta)=A(\theta)Z+b.
\]

Original Z, X, and Y are fixed. The data-only model
\(\hat f(U,X)=\phi_U(U)^\top B\phi_X(X)\) is **refitted for each angle**,
using cached fixed-X projections from 06. For analytic CR3, local
Z, ZZ, ZX, ZZZ, ZZX, ZXX, ZZZZ, ZZXX and ZXXX tensors are
transformed exactly as coordinate tensors by \(R\). The underlying
CR3 generating formula never changes as a function of the original T.

## Metric inventory: all families implemented

\`local_metrics.py\`, \`controlled_sweeps.py\`,
\`pairwise_sweeps.py\`, and the multiscale sparsity package
\`../06_cr3_hessian_rotation_landscapes/sparsity.py\` **implement**
the families below. The actual numeric registry, automatically
generated after a run as \`metric_catalog.csv\`, lists every
individual loss or diagnostic and its direction. There are
hundreds of **explicitly computed** indicators, not merely names.

| Family | Example implemented names | Mathematical interpretation |
|---|---|---|
| Local ZZ/ZX magnitude | \`zz_diagonal_magnitude_rms_loss\`, \`zx_magnitude_q90_abs_loss\` | RMS, L1, P90, P99, max |
| Mean versus RMS curvature | \`zz_cancellation_ratio_diagnostic\`, \`zx_local_rms_over_mean_diagnostic\` | Local curvature hidden by mean cancellation |
| ZZ diagonal vs off-diagonal | \`zz_offdiag_total_energy_fraction_loss\`, \`zz_offdiag_over_diag_energy_loss\` | Number and strength of interactions among Z coordinates |
| ZZ self-nonlinearity | \`zz_self_z_third_rms_loss\`, \`zz_self_z_fourth_rms_loss\` | \(\partial_{z_i}H_{ii}\), \(\partial^2_{z_i}H_{ii}\) |
| ZZ dependence on other Z | \`zz_other_z_third_rms_loss\`, \`zz_three_distinct_z_third_rms_loss\` | \(\partial_{z_j}H_{ii}\), higher interaction order |
| ZZ dependence on X | \`zz_diagonal_X_modulation_third_rms_loss\`, \`zz_diagonal_X_twice_fourth_rms_loss\` | \(\partial_{x_k}H_{ii}\), \(\partial^2_{x_k}H_{ii}\) |
| ZX dependence on Z | \`zx_own_Z_modulation_third_rms_loss\`, \`zx_other_Z_modulation_third_rms_loss\` | Change of cross-Hessian with its own/other latent parameters |
| ZX dependence on X | \`zx_own_X_modulation_third_rms_loss\`, \`zx_other_X_modulation_third_rms_loss\` | Stimulation-dependent interaction curvature |
| Normalized higher-order ratios | \`zz_self_z_higher_order_ratio_loss\` and \`zx_own_X_higher_order_ratio_loss\` | Input-SD-standardized third derivative relative to Hessian RMS |
| Four-derivative complexity | \`zz_Z_pair_fourth_rms_loss\`, \`zz_Z_three_one_fourth_rms_loss\` | Quartic complexity and mixed fourth-order effects |
| ANOVA of local Hessian | \`zz_diagonal_curvature_subject_X_interaction_variance_loss\` | Between-subject, between-X and residual variations on Cartesian evaluation grid |
| Mean/RMS/mean-abs sparsity | \`zx_rms_zx_row_mean_tail2sq_k2_loss\`, \`zz_mean_flat_logsum_t0p005_loss\` | Interactions visible even if signed mean cancels |
| Pointwise ZX row supports | \`zx_local_row_mean_capped_l2_t0p005_loss\`, \`zx_local_row_max_softl0_t0p005_loss\` | Penalize small extra X effects for each Z dimension |
| Trimmed residual and adaptive supports | \`tail1/tail2/tail2sq_k1..k4\`, \`penalized_support_t0p005_loss\` | Don't pre-assume that only one or two X effects may remain |
| Nonconvex/effective support | \`lp025/lp05/lp075\`, Hoyer, Shannon/Rényi, Gini, soft-L0, log-sum | Different notions of effect concentration |
| Self-curvature variation (controlled) | \`sweep_mean_self_curvature_variance_normalized_loss\`, \`sweep_mean_self_curvature_tv_normalized_loss\` | Change of \(H_{ii}\) along **only \(U_i\)** at fixed \(U_{-i},X\) |
| Quadratic/cubic polynomial approximation | \`sweep_mean_quadratic_approx_relative_mse_loss\`, \`sweep_mean_cubic_approx_relative_mse_loss\` | Compare polynomial degree d approximation of response along a controlled Z axis |
| Fourier/DCT smoothness | \`sweep_mean_high_frequency_dct_ratio_diagnostic\` | Relative high-frequency energy; not an exact polynomial degree test |
| Single-Z × X nonadditivity | \`sweep_mean_self_Z_X_additivity_gap_loss\` | Counterfactual response not explained by U_i-only + X-only components |
| Pairwise ZZ nonadditivity | \`zz_pair_01_additivity_gap_loss\`, \`zz_pairwise_additivity_mean_loss\` | 2D functional-ANOVA interaction of any pair of the four Z directions |
| DGSM-like derivative energy | \`zz_mixed_DGSM_like_loss\`, \`zx_mixed_DGSM_like_loss\` | Average squared mixed derivatives, not an asserted Sobol bound |
| In-distribution reliability | \`sweep_z0_OOD95_fraction_diagnostic\` | Fraction of counterfactual Z grids outside training covariance-based 95th-percentile support |

Existing step-06 angles and multiscale/rowwise support metrics are
incorporated automatically. The full numeric metric list is generated
from code rather than manually hardcoded in documentation.

### Scale and input-distribution discipline

- Compute derivatives in **U** and fixed-X physical coordinates,
  then multiply each derivative by the product of the corresponding
  original-input standard deviations (including repeated derivative
  indices). This makes third/fourth order scores comparable across
  heterogeneous parameter and X units.
- Report separately the raw (unstandardized) scores where relevant.
- Only use X rows with **positive gain/loss amounts** and **interior
  probabilities** for local differentiability; this X subset is selected
  ONCE, then reused at every angle and for every source.
- Subjects are a fixed held-out subset; X stimuli are a fixed random
  subset of eligible stimuli. Subject × X forms a Cartesian evaluation
  grid, not a random context split.
- Controlled U_i perturbations and 2D grids can be out of the joint
  support of correlated U coordinates; the OOD95/Mahalanobis diagnostics
  must be inspected before treating such scores as scientifically robust.

### Success criteria

The indicator is intended as a candidate **simple-axis** loss. For
each of six rotation planes and each Y response:

1. The analytic reference curve, fit-to-formula-Y curve, and
   fit-to-frozen-NN-Y curve should correlate strongly.
2. **All three curves should attain a MINIMUM at 0° or 90°** (within
   selected angular tolerance), not merely an extremum or maximum.
3. Flat curves do not qualify. A single successful plane does not
   establish six-plane or global SO(4) recovery.

The screen writes \`strict_screen_all.csv\`,
\`strict_screen_groups.csv\`, and \`strict_screen_ranking.csv\` using
the same **three-way** rules as experiment 06.

A crucial alternative explanation is data-only **inference error**
from the Hessian estimator. Each derivative family is compared by:

\[
K_{\rm NNfit}-K_{\rm analytic}=
(K_{\rm formulaYfit}-K_{\rm analytic})+
(K_{\rm NNfit}-K_{\rm formulaYfit}).
\]

\`three_way_local_derivative_errors.csv\` reports all three norms,
their correlations and the cross-term (error cancellation).
A large formulaYfit-analytic difference must not be attributed to NN.
Choice is additionally confounded by stochastic Bernoulli versus
deterministic threshold labels.

## Run (Windows CMD, repository root)

\`\`\`cmd
conda activate py11
git pull origin main
python -m pytest math/07_cr3_local_hessian_complexity/test_local.py -q
\`\`\`

**Cheap end-to-end validation** (one plane, 3 angles):

\`\`\`cmd
python math/07_cr3_local_hessian_complexity/run.py --planes 01 --angles 0,45,90 --eval-subjects 12 --eval-stimuli 8 --anchors 3 --no-pairwise --no-fd-convergence
\`\`\`

**Full requested experiment:**

\`\`\`cmd
python math/07_cr3_local_hessian_complexity/run.py --planes all --angles 0:90:5
\`\`\`

To reduce CPU cost while retaining all ZZ/ZX third-order indicators,
add \`--no-fourth\` or \`--no-pairwise\`; to skip expensive analytic
finite-difference convergence verification, add
\`--no-fd-convergence\`. These omit the named optional metric
families, rather than silently labeling them as evaluated.

The script accepts \`--source "D:\path\to\repeat01(2).zip"\` to
read original step04 source directly if your saved step04 directory
has been moved.

**Output root**:

\`\`\`text
outputs/math/07_cr3_local_hessian_complexity/
  beta5em05_lambda0_ctx0p25to0p8_L4/repeat01/
    monomial_u4_x4_lambu1e-06_lambx1e-06/
      summary.json
      metric_catalog.csv
      all_local_metrics.csv
      curves__01_alpha_lambda.csv
      ... 5 other plane CSVs
      analytic_fd_convergence.csv
      three_way_local_derivative_errors.csv
      heldout_Y_rmse.csv
      strict_screen_all.csv
      strict_screen_groups.csv
      strict_screen_ranking.csv
      PLOT_INDEX.md
      local_curves/*.png
\`\`\`

No previous output directory is overwritten, and the original
step-04/06 experiments remain unmodified.

## Mathematical context

These metric families correspond to several complementary ideas, not
one unique definition of "simplicity":

- **Trend filtering / higher-order total variation** of derivatives,
  e.g. Ryan Tibshirani, *Adaptive Piecewise Polynomial Estimation via
  Trend Filtering* (2014), arXiv:1304.2986.
- **Derivative-based sensitivity (DGSM)** and analytic derivatives of
  polynomial surrogate functions, Sudret & Mai (2015), Reliability
  Engineering & System Safety 134, 241–250.
- **Functional-ANOVA style interaction / partial dependence gap**,
  Friedman–Popescu's H-statistic family. The present controlled-grid
  implementation is a related *diagnostic*, not an unbiased population
  Sobol or H-statistic.
- **Soft sparse support** via rowwise capped, quasi-norm, and log-sum
  losses at multiple relative thresholds.

No theorem guarantees the original CR3 parameter basis minimizes all
of these. Many are exactly invariant under signed axis permutations,
but finite-grid regressions can deviate from this symmetry. All
claims of recovery must be checked empirically across six planes and
across both data-only control paths.
