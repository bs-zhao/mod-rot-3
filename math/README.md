# Math experiments

Mathematical and statistical experiments are isolated by subdirectory.
Each experiment contains its own code, tests, documentation, and outputs,
and must not import Python modules from other math experiment folders.
There is no aggregate "run everything" script.

## Experiments

- cr3_hessian_expectation/: compute and compare full 10x10 Hessian
  expectations of CR3's analytic choice model against estimates obtained
  only from true T, fixed X, and three forms of Y (choice/logit/probability).
  The experiment includes sample averages and prior quadrature.
  See cr3_hessian_expectation/README.md.


- cr3_hessian_estimators/: independent benchmark of Nyström Gaussian KRR,
  penalized tensor-product B-splines, and weighted second-order Stein
  estimation, evaluated against CR3's analytic 4x4 parameter Hessian.
  All fixed X stimuli are averaged before fitting to study the *mean*
  theta-theta Hessian; the Stein estimand is explicitly weighted.
  See cr3_hessian_estimators/README.md.

- cr3_hessian_six_y/: fit an invertible TRAIN-only affine Z-to-T map,
  invert true T back to decoder Z, create three additional NN-derived Y
  channels (hard choice, logit, probability), and use the identical
  standalone original 10x10 **T-X-Y polynomial Hessian estimator** for
  all six Y. The NN produces Y ONLY; it is never differentiated.
  See cr3_hessian_six_y/README.md.

Read-only input datasets are stored under outputs/choice/CR3 and all
experiment results under outputs/math/<experiment_name>. Existing NP and
rotation pipelines are left unchanged.
