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

Read-only input datasets are stored under outputs/choice/CR3 and all
experiment results under outputs/math/<experiment_name>. Existing NP and
rotation pipelines are left unchanged.
