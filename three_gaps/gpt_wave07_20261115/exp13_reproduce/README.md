# Fresh frozen EXP13 reproduction

```powershell
python three_gaps/gpt_wave07_20261115/exp13_reproduce/run.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --run-name A0_strict --resume
```

Repeat2 uses the same command with --repeat 2. For C1/C2 use beta .0005,
zdim4; C3 beta .0005, zdim6. Historical source/cache/archive is read-only.
Fresh derivatives and independent per-subject moments are computed first.
The historical-cache numerical audit runs in a separate subprocess because
the old cache format also contains truth. The optimizer never deserializes
that old cache. HistoricalLoss preserves frozen arithmetic and sequential
24-start Adam250; new directories retain all starts and rotations.
CR3 uses300×500 actual bank points; C1–C3 use300×400 antithetic EXP13 points.
See [actual results](RESULTS.md) and results/reproduction_check.csv.
