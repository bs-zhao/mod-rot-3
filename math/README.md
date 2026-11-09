# Math experiments

Mathematical and statistical experiments are isolated by subdirectory.
Each experiment contains its own code, tests, documentation, and outputs,
and must not import Python modules from other math experiment folders.
There is no aggregate "run everything" script.

## Experiments

- 01_cr3_hessian_expectation/: compute and compare full 10x10 Hessian
  expectations of CR3's analytic choice model against estimates obtained
  only from true T, fixed X, and three forms of Y (choice/logit/probability).
  The experiment includes sample averages and prior quadrature.
  See 01_cr3_hessian_expectation/README.md.


- 02_cr3_hessian_estimators/: independent benchmark of Nyström Gaussian KRR,
  penalized tensor-product B-splines, and weighted second-order Stein
  estimation, evaluated against CR3's analytic 4x4 parameter Hessian.
  All fixed X stimuli are averaged before fitting to study the *mean*
  theta-theta Hessian; the Stein estimand is explicitly weighted.
  See 02_cr3_hessian_estimators/README.md.

- 03_cr3_hessian_six_y/: fit an invertible TRAIN-only affine Z-to-T map,
  invert true T back to decoder Z, create three additional NN-derived Y
  channels (hard choice, logit, probability), and use the identical
  standalone original 10x10 **T-X-Y polynomial Hessian estimator** for
  all six Y. The NN produces Y ONLY; it is never differentiated.
  See 03_cr3_hessian_six_y/README.md.

- 04_cr3_zx_hessian_threeway/: fit an affine A from the original all-trial
  encoder Z to true T, evaluate CR3 Y at T_hat=A Z+b and frozen decoder Y
  at actual Z, then infer two full 10x10 Hessians from the SAME Z,X
  matrix estimator. Compare with analytic CR3 H at T_hat pulled back
  by J=diag(A,I6). No NN derivatives are computed.
  See 04_cr3_zx_hessian_threeway/README.md.

Numbered math source directories follow experiment chronology. Previously generated
output directories retain their original unnumbered names.

- 05_cr3_zx_hessian_optimization/: extend step 04 with independent
  Z/X ridge and polynomial-degree grids, monomial versus Chebyshev basis,
  a full-ZX factorized Gaussian RBF estimator, and two additional ZZ-only
  methods (Z-adaptive cubic B-splines and KRR).
  Reuses EXACT step-04 Z, X, six Y and analytic H; models are selected by
  validation Y RMSE, never by the oracle Hessian. Reports full-matrix and
  diagonal/off-diagonal pattern recovery metrics.
  See 05_cr3_zx_hessian_optimization/README.md.

- 06_cr3_hessian_rotation_landscapes/: fit the original affine A_star mapping
  Z to T and rotate every pair of its four output dimensions over 0–90°.
  For each of six planes and every angle compute nine mean Hessians:
  3 sources (exact analytic CR3, numeric analytic-Y regression, frozen
  numeric NN-Y regression) × 3 Y types (Choice/Logit/Probability).
  Save UU/UX/XX and diagonal/off-diagonal structure/similarity curves,
  rotation-equivariance diagnostics, all 10×10 Hessians, and 3×3 dashboards.
  Original Z/X/Y are frozen and each rotated U=AZ+b is refitted.
  See 06_cr3_hessian_rotation_landscapes/README.md.

- 07_cr3_local_hessian_complexity/: extends the six-plane nine-source 06
  design to LOCAL second/third/fourth derivatives, not just mean H.
  Implements self-vs-other-Z curvature complexity, ZZ/X modulation,
  ZX per-row and per-trial multiscale sparsity, finite-difference
  roughness, controlled one-axis quadratic/cubic approximation errors,
  ZZ pairwise and ZX functional additivity-gap metrics, support/OOD
  diagnostics, and strict 0°/90° minimum screening. Distinguishes
  analytic-Y inference error from the additional frozen-NN-Y gap.
  See 07_cr3_local_hessian_complexity/README.md.

- 08_cr3_functional_sparsity/: independent CR3 oracle rotation benchmark on
  all six true-parameter planes. Compare exact logit/probability ZZ/ZX
  Hessian-RMS support sparsity and self-curvature against derivative-free
  3x3 function-value factorial interaction gaps; save six-plane angle
  curves, 4x6 heatmaps, flatness and endpoint-minimum audit. No NP/NN,
  no invalid independent-input Sobol on the fixed designed X bank.
  See 08_cr3_functional_sparsity/README.md.

- 09_cr3_three_gap_reaudit/: independently re-audit historical math/07
  winner metrics in analytic (Gap 1), numeric formula-Y (Gap 2) and
  frozen decoder-Y (Gap 3) conditions, without rerunning models. Preserve
  the original 10-degree endpoint-tolerance gate and also report exact
  endpoint minima. Includes six-plane loss plots, candidate ranking and
  explicit failure diagnostics. See 09_cr3_three_gap_reaudit/README.md.


- 10_multimodel_hessian_simplicity/: **independent, one-command cross-model analytic-oracle screen**
  for C1/C2/C3/CR3.  Uses original S1 subjects/X, exact logit/probability TT/TX
  Hessians, local-to-matrix aggregators, rowwise and columnwise Tail-L1/L2,
  soft support, entropy/Hoyer/Gini, Gram, third/fourth-order curvature and
  pairwise plus randomized SO(3)/SO(4)/SO(d) rotations.  Reports flatness,
  strict endpoint minima, invariance and candidate rankings.  **This is
  Gap 1 analytic only**: not yet formula-WY/NN-WY, and not a claim of
  unsupervised recovery.  The repository has CR3, not an independent CR1.
  See 10_multimodel_hessian_simplicity/README.md.

Read-only input datasets are stored under outputs/choice/CR3 and all
experiment results under outputs/math/<experiment_name>. Existing NP and
rotation pipelines are left unchanged.
