# EXP21 — observed resolution and finite halfwidth

```powershell
python three_gaps/gpt_wave07_20261115/exp21/run.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --resume
```

64x64 shared centres. Matched 8-start/120-step baselines, original float32 and
separate float64 decoder copy; checkpoint itself never modified. U halfwidths
.05/.1/.25/.5/1 SD; X halfwidths .5/1/2/4 times median positive observed gaps.
All mixed differences divide by original X units. Every R changes query points;
there is no tensor rotation shortcut. Support audits do not remove query points.
Ambient, U-finite/X-autograd hybrid, and chord reconstruction are distinct.
Chord full UX reconstruction is refused if neighbourhoods are rank deficient
or have condition >1e5. A refusal is saved as a failure case, not fake evidence.
This expensive pilot is not equivalent to the full frozen 24x250/300x500 baseline.
Status: [RESULTS](../RESULTS.md).
