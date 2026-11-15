# Reproduce this independent wave

Use the local original checkpoints, latents and datasets recorded by absolute
path and SHA256 in each config.json. Historical EXP13 archive is needed only
for read-only auditing; no historical EXP20–23 output is needed. Large caches
belong under outputs/three_gaps/gpt_wave07_20261115 and are deliberately absent
from GitHub. Model beta/d/repeat/channel, all source Python hashes, covariance,
seeds and exact subject/X IDs are retained in each result config.

```powershell
conda activate py11
Set-Location D:/Projects/rotation/mod-rot-3
python three_gaps/gpt_wave07_20261115/tests/test_math.py
python three_gaps/gpt_wave07_20261115/tests/test_stencils.py
python three_gaps/gpt_wave07_20261115/tests/test_blind.py
python three_gaps/gpt_wave07_20261115/run_queue.py baseline_cr3 --run-name A0_strict
python three_gaps/gpt_wave07_20261115/run_queue.py 20a_cr3
python three_gaps/gpt_wave07_20261115/run_queue.py 20b_cr3
python three_gaps/gpt_wave07_20261115/run_queue.py 22pilot_cr3
python three_gaps/gpt_wave07_20261115/run_queue.py 22refine_cr3
python three_gaps/gpt_wave07_20261115/run_queue.py 22verify_cr3
python three_gaps/gpt_wave07_20261115/run_queue.py 22fullsmall_cr3
python three_gaps/gpt_wave07_20261115/continue_run.py remaining
python three_gaps/gpt_wave07_20261115/exp22/convergence_audit.py
python three_gaps/gpt_wave07_20261115/confirm_convergence.py
python three_gaps/gpt_wave07_20261115/report.py
python three_gaps/gpt_wave07_20261115/summarize.py
python three_gaps/gpt_wave07_20261115/final_audit.py
```

Run GPU stages sequentially. `--resume` reuses only this wave's completed
matching caches and never changes sealed rotations. To recompute rather than
resume, use a fresh copied gpt directory and the same protocol; update its
directory name in commands. Do not delete or replace historical outputs.
Some reporting and sensitivity controls were separate CPU jobs. Their actual
commands/statuses are in reports/execution_manifest.json and local logs;
per-experiment READMEs give direct entry points. For example, for each repeat:

```powershell
python three_gaps/gpt_wave07_20261115/exp22/historical_axis_scale.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --device cpu --resume
python three_gaps/gpt_wave07_20261115/exp21/audit_support.py --model CR3 --beta .00005 --zdim 4 --pca 0 --repeat 1 --tag fd_vectorized --device cpu
```

`evaluate.py` reads truth only after SHA-verifying every completed R. The optional
`--experiment exp23` narrows reporting to that folder, never chooses a candidate.
Re-run final reporting after additional evaluations to rebuild coverage tables.
Use final `*_cap_status.json` for convergence certification: an earlier passing
prefix in `first_passing_budget` is not the final cap decision. Changing the
kernel, budgets, decoder dtype or centre set requires a separate tag.

Old scalar-FD partial results and failed convergence budgets remain visible.
The equivalent vectorized stencil and grouped physical-node implementation
were verified against scalar queries before use; audit JSONs are in tests/.
runtime.csv distinguishes candidate timing and whole subprocess wall time;
those overlapping scopes must not be added. GPU allocation is PyTorch's
reported process peak where present; whole-process RAM peaks were not captured.
