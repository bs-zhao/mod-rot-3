# EXP20B — independent continuous latent integration

```powershell
python three_gaps/gpt_wave07_20261115/exp20B/run.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --resume
```

P1 is the full multivariate TRAIN mean/covariance, physical nodes fixed before
any rotation. MC5 seeds, RQMC4 scrambles with nested prefixes, physicists GH
normalization. CR3 budgets 256–8192, GH3/5/7/9; C3 GH3/5 only to avoid enormous
tensor grids. P0 and physical N(0,I) P3 share full-TRAIN C_ref. Native P3 and
fixed historical Haar scales are sensitivity controls. Convergence is <1%
relative SX and T4 and Spearman>.99; no T-based budget or measure selection.
Failing the budget cap is reported as unresolved, never silently converged.
Virtual nodes query the frozen decoder; this is not strict WY-only evidence.
Status: [RESULTS](../RESULTS.md).
