# EXP23 — three aggregation orders

```powershell
python three_gaps/gpt_wave07_20261115/exp23/run.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --outer pilot --inner-nodes 128 --resume
```

Uses this independent wave's own EXP22 fields, never previous-wave fields.
Equal weights on sigma .1/.25/.5. A averages signed fields before forming
moments, including cross terms; B averages moment energies; C averages each
individually Haar-standardized loss. All three perform a new blind multistart
search; rotations are never averaged. `--outer full` consumes own full-bank fields.
Status: [RESULTS](../RESULTS.md).
