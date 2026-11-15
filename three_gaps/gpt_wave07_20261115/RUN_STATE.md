# Final independent execution state — 2026-11-15

The authorized first wave is complete; no experiment process remains active.
Repository: D:/Projects/rotation/mod-rot-3. Interpreter:
D:/Programs/anaconda_new/envs/py11/python.exe. GPU jobs ran sequentially.
Only this gpt folder was edited/staged by this task; other work is preserved.

Completed from freshly recomputed EXP13 fields:
- Frozen archive/live SHA audit92/92; no code diff against frozen tag.
- All8 probability baselines independently reproduced within1e-4; actual MCC
  error<2e-7, nonmatching-correlation error<2e-6. Separate-process cache audit.
- EXP20A all4 models x2 repeats,3500 TRAIN, five nested seeds/all sizes,
  fixed/native covariance and historical Haar-SD controls.
- EXP20B CR3 both repeats MC5/QMC4/GH3,5,7,9; P0/P1/P3/tails/support.
  C1–C3 both repeats P0/P1/P3 QMC confirmation; all final caps passed.
- EXP22 CR3 sigma sweep64x64, final1024 passed; earlier failed128/512 retained.
  Full300x500 sigma0/.1 passed32→64; axis and Haar scale controls retained.
- Four-model smoothing: C1r1/r2,C2r2 final128; C2r1,C3r1/r2 extended to512
  using numerical criteria only; all final caps passed.
- EXP21 CR3 both repeats,9 preset configurations, matched64x64/8x120,
  original float32 plus float64 reference, complete-stencil support proxies.
  CR3 k12 chord full-UX reconstruction formally rejected for rank deficiency.
- EXP23 both repeats A/B/C; empirical/Gaussian128x64 joint pilots passed
  inner256→512. These are not full J4/J5 or certified outer Gaussian integrals.
- CR3 logit EXP13/20A controls; separate post-hoc evaluation and all reports.

Final integrity:1563 completed configurations including numerical budgets and
controls,37080 start endpoints. Every selected R is lowest loss, completion
hashes match, signed/matched matrices reproduce all reported correlation stats.
One intentionally interrupted scalar-FD partial candidate remains incomplete;
its failure is retained and the equivalent vectorized full sweep completed.

Final user criterion: frozen EXP13 probability baseline; max nonmatched
correlation must fall by absolute0.03. Added after exploration, separately
recorded in user_criterion.json. One individual CR3 subset passes this necessary
condition but lowers MCC and fails repeat2. No method passes both repeats.
No new method frozen; EXP13 remains protected.

Unrun: full300x500 FD, full-TRAIN/converged-P1 smoothing J4/J5, reverse hybrid,
full chord recovery, X smoothing, extra weights/normalizations, new S1/NP3–5,
unseen families. Current TEST comparisons are exploratory. See RESULTS.md.

Large caches/logs: outputs/three_gaps/gpt_wave07_20261115. Exact commands and
statuses: reports/execution_manifest.json and local logs. Reproduce:
REPRODUCE.md. Audits: tests/, reports/final_integrity.json. Failures and metadata
errata are retained. Final tables/figures/runtime/per-axis metrics: reports/.
