# EXP20A — independent streaming

From repository root in Anaconda py11:

```powershell
python three_gaps/gpt_wave07_20261115/exp20A/run.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --resume
```

Repeat 2 changes only `--repeat 2`; logit control adds `--channel logit`.
C1/C2 use beta .0005 / zdim4; C3 beta .0005 / zdim6. All use pca0.
This recomputes each TRAIN subject's moments; no TEST subjects in the measure.
Five nested seed sequences, sizes 100/300/500/1000/2000/full; native and fixed
full-TRAIN axis scaling, original A0 and fixed historical Haar SD controls.
The same 1000 Haar rotations compare landscapes; float64 accumulations.
Evaluate separately after COMPLETE, using `../evaluate.py` and the same flags.
Status and independent findings: [RESULTS](../RESULTS.md).
