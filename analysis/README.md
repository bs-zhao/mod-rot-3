# Independent analyses

Each subdirectory is independently runnable and stores its outputs in its
own directory under `outputs/analysis/`. Existing S1 data, S3 checkpoints,
S5 latent extraction and rotation experiments are not modified.

- `cr3_np_logit_vs_true/`: Freeze an S3 CR3 Neural Process, infer a
  subject-specific posterior mean z from **all 500 observed trials**, decode
  all 500 original stimuli, and compare both scalar binary log-odds and
  choice probabilities against the exact CR3 simulator. Produces 2.5 million
  paired predictions and per-subject/per-trial diagnostics.

See the individual subdirectory README for commands, sources and limitations.
