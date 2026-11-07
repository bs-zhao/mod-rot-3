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

Read-only input datasets are stored under outputs/choice/CR3 and all
experiment results under outputs/math/<experiment_name>. Existing NP and
rotation pipelines are left unchanged.
