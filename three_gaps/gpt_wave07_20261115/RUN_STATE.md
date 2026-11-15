# Live independent execution state

User clarified: disregard other people's EXP20–23 runs; start afresh from EXP13.
User requires matched AND nonmatched correlations, worst axes, rotation angles.
Repository is D:/Projects/rotation/mod-rot-3; Anaconda py11 python is
D:/Programs/anaconda_new/envs/py11/python.exe. Other dirty files are pre-existing;
only this gpt directory may be staged. Main publishing explicitly authorized.

Published source/results: main commits95c3d31 and0bcc7853. New source/diagnostic
improvements remain to be committed. No historical files modified.

Completed independently:
- archive audit 92/92, fresh EXP13 CR3 rep1/2, p;
- EXP20A CR3 rep1/2, 3500 TRAIN, 5 nested seeds, all sizes/native/fixed;
- EXP20B CR3 rep1, MC5 / QMC4 / GH3,5,7,9 / P0,P1,P3;
- mathematical, FD toy, actual decoder limit/precision, blind access tests.

EXP20B both repeats and rep1 supplemental provenance/controls are now complete.
Strict process-separated A0_strict CR3 rep1/2 is also complete.
128 and512-node EXP22 CR3 pilots are complete. At512 signed sigma.5/1 still
failed the1% moment criterion, so were NOT declared converged. 1024 with grouped
physical-node queries is now running; actual CPU/GPU comparison passed <1e-6.
1024 verification is COMPLETE in BOTH repeats; cap_status has NO unresolved scales.
Full300x500 sigma.1 matched smoothing is COMPLETE for both repeats and passed
the32->64 inner convergence check. Scalar FD rep1 h=.05/.1/.25 completed at
~255s/config. Own scalar FD worker was intentionally stopped for throughput
revision; old rest status/logs retained, reason in continuation_failure.json.
Vectorized identical-stencil queries passed actual64x64 CUDA energy (<6e-7)
and gradient (<3e-6) agreement tests and CPU float64 checks; about5-6x faster.
Active: continue_run.py remaining has now been launched (see latest session ID).
Current active exec session is55259. FD CR3 both repeats, ABC and joint128
pilots are COMPLETE. All8 probability frozen baselines checked against history
and passed. C1–C3 EXP20A computations are complete; Gaussian confirmation now runs.
Logs: outputs/three_gaps/gpt_wave07_20261115/logs/continuation/remaining_000.log etc.
FD now runs under fd_vectorized tag with --engine vectorized, same8x120 budget.
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
3. RUNNING: EXP22 CR3 pilot 64x64 (same centre seed21 as EXP21), sigma grid, now up to1024. Review moment/landscape
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

The remaining authorized sequence is concretely implemented in continue_run.py.
Session13246 has finished and cap_status was audited; now RUNNING
`python three_gaps/gpt_wave07_20261115/continue_run.py remaining`.
It runs full300x500 sigma.1 matched smoothing; finite64x64 all preset stencils;
ABC from verified1024; matched joint empirical/Gaussian128x64 pilots at inner512;
fresh C1/C2/C3 baseline/20A/P0P1P3-QMC confirmation and fixed sigma.1/.25 pilots;
CR3 logit controls; separate evaluation of all tags. Logs/commands/statuses in
outputs/.../logs/continuation. Stop/fix failures then resume segments.
The joint128 pilot does NOT claim full J4/J5 or converged outer Gaussian integral.
Actual ambient/hybrid CUDA forward/backward smoke tests passed (tests/fd_gpu_audit.json).
Convergence audits use LAST adjacent budgets, not a lucky earlier passing prefix.

Current findings (own runs, exploratory): all-TRAIN CR3 .814098/.812209 MCC;
offmean .163545/.180653; offmax .400645/.369824; max Oracle-axis angle
27.277/29.699 deg. Frozen .817297/.812199; offmax .367730/.378094.
Rep1 P1 .817434, offmean .155591, offmax .401751, angle27.227deg.
Rep1 P3 .824461 but offmax .426510, angle25.425deg. Not cleaner recovery.
No promising candidate has been frozen. Do not claim unrun experiments done.
