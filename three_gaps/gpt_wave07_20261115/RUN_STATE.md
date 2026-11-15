# Live independent execution state

User clarified: disregard other people's EXP20–23 runs; start afresh from EXP13.
User requires matched AND nonmatched correlations, worst axes, rotation angles.
Repository is D:/Projects/rotation/mod-rot-3; Anaconda py11 python is
D:/Programs/anaconda_new/envs/py11/python.exe. Other dirty files are pre-existing;
only this gpt directory may be staged. Main publishing explicitly authorized.

Published initial source/results: main commit 95c3d31. New source/diagnostic
improvements remain to be committed. No historical files modified.

Completed independently:
- archive audit 92/92, fresh EXP13 CR3 rep1/2, p;
- EXP20A CR3 rep1/2, 3500 TRAIN, 5 nested seeds, all sizes/native/fixed;
- EXP20B CR3 rep1, MC5 / QMC4 / GH3,5,7,9 / P0,P1,P3;
- mathematical, FD toy, actual decoder limit/precision, blind access tests.

EXP20B both repeats and rep1 supplemental provenance/controls are now complete.
Strict process-separated A0_strict CR3 rep1/2 is also complete.
Active: exec session 77355 runs `run_queue.py 22pilot_cr3`, currently repeat1.
Logs: outputs/three_gaps/gpt_wave07_20261115/logs/22pilot_cr3_prob_default_0.log.
Wait for this to finish before starting another substantial GPU job.

Next (all authorized, continue working):
1. DONE: Rerun EXP20B rep1 with --resume: code added exact node/weight files, support
   diagnostics, MC N300, P3 native covariance, historical SD controls, QQ.
   Existing own per-node caches are reused; no other wave's caches are touched.
2. DONE: Strict process-separated EXP13 audit rerun using --run-name A0_strict,
   CR3 rep1/2. Initial A0 numerics were blind, but its audit deserialized an old
   cache in the same process; the improved version isolates that audit. Old A0
   generic config C_ref metadata is corrected in config_errata.json; actual
   rotations.npz always contained the correct native covariance.
3. RUNNING: EXP22 CR3 pilot 64x64 (same centre seed21 as EXP21), sigma grid, inner8/16/32/64/128. Review moment/landscape
   convergence, signed vs square-first, sigma0, validation CE/Brier/delta p.
   If cap fails, retain unresolved; optional larger-inner separate tag follows
   T-free convergence only. Do not select sigma by MCC.
4. EXP21 CR3 matched pilot (original float32; separate float64 reference), all
   preset U halfwidths and observed X gap widths. 8x120 matched budget. Save
   chord reconstruction rejection if k12 directions rank-deficient (expected).
5. EXP23 A/B/C using this wave's own sigma .1/.25/.5 signed fields, equal weights;
   joint experiments only with explicit matched settings and recorded coverage.
6. Four-model confirmation: fresh baseline C1/C2/C3, EXP20A, P0/P1/P3 via
   exp20B/confirm.py, appropriate smoothing confirmation. Logit mechanism
   controls for CR3 EXP13/20A are also pending. No NP retraining in this wave.
7. Separate evaluate.py after R COMPLETE, report.py, inspect figures, update
   RESULTS per experiment, record failures/remaining coverage, final archive
   protection check, commit/push ONLY this gpt directory to main.

Current findings (own runs, exploratory): all-TRAIN CR3 .814098/.812209 MCC;
offmean .163545/.180653; offmax .400645/.369824; max Oracle-axis angle
27.277/29.699 deg. Frozen .817297/.812199; offmax .367730/.378094.
Rep1 P1 .817434, offmean .155591, offmax .401751, angle27.227deg.
Rep1 P3 .824461 but offmax .426510, angle25.425deg. Not cleaner recovery.
No promising candidate has been frozen. Do not claim unrun experiments done.
