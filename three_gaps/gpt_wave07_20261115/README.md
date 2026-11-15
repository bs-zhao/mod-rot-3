# Independent GPT doc07 experiments — 2026-11-15

This is a fresh execution from frozen EXP13, as explicitly requested. No
EXP20–23 moments, rotations or numerical results from the prior wave are reused.
Original checkpoint / latent / dataset files and EXP13 archive are read-only.
Each experiment has its own entry point, results and local output namespace.

Protocol is fixed before evaluation: probability, UX_neff + UUd_share, raw X,
Haar 200 seed1, 24 Haar starts seed7, Adam250 lr .05, float32 search and decoder,
float64 moment accumulation. Final rotations are selected by lowest loss.
`evaluate.py` is a separate process and the only code that reads theta / Oracle.
Historical TEST has already been inspected; all current comparisons exploratory.

Execution order: audit/reproduce EXP13 → EXP20A (CR3 two repeats) → EXP20B →
EXP22 → EXP21 → EXP23 → same frozen rules on C1/C2/C3. Large files stay under
`outputs/three_gaps/gpt_wave07_20261115/`; small results and source go to main.
Failed runs retain traceback and partial files; COMPLETE is written atomically.
An existing completed run requires `--resume`; configuration hashes must match.

The research loss is unchanged. Gaussian sampling is only an integration
measure, never ICA. Higher-dimensional PCA without a residual protocol fails.

Status: implementation in progress; no new numerical result claimed yet.
