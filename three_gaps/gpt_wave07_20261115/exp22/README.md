# EXP22 — independent signed Hessian smoothing

```powershell
python three_gaps/gpt_wave07_20261115/exp22/run.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --stage pilot --resume
```

64x64 common centres, sigma0/.05/.1/.25/.5/1, nested 8/16/32/64/128
antithetic inner QMC nodes. Kernels are fixed in physical V with TRAIN covariance.
Signed mean H is squared only after the inner integral; square-then-mean is
kept as a separate control. Both use the frozen original loss and search budget.
Quadrature selection uses moment/landscape convergence, not recovery metrics.
`--stage full --inner-nodes N` uses A0 300xfull bank after convergence audit.
`--stage joint_train` / `joint_gaussian` use predeclared sigma .25 at full TRAIN /
1024 P1 QMC nodes. Their numerical sufficiency must be reported separately.
Full Gaussian quadrature convergence is not implied by this joint pilot budget.
Status: [RESULTS](../RESULTS.md).
