# Local execution handoff: persistent WY-only rotation research

Run this work in the user's real Windows checkout:
`D:\Projects\mod-rot-3`, Anaconda environment `py11`.

The user authorizes code edits, GitHub commits/pushes, local runs, evaluation,
literature investigation, and repeated redesign. Do not stop at generating
commands or ask again for this authorization. Preserve user changes; use
`main` and no force push. No external messages are authorized.

Objective: identify one dataset-independent simplicity criterion using only
the actual case matrices and shared Y, applicable to C1, C2, C3 and CR3. True
theta, generators, decoder weights, task-specific flags and saved R are
excluded from scoring. Baseline/angles are evaluation labels only. All
5,000 subjects may be used for fitting and eventual optimization. C3 has six
latent dimensions, 91 cases and 15 planes; the other models have four
dimensions, 37 cases and six planes.

Latest user steering (2026-11-11): prediction-quality thresholds must not
be rigid vetoes. In particular, a .01 heldout relative-MSE cutoff must not
discard a structurally interesting candidate before further datasets are
examined. Report approximation error continuously and judge it together
with resolution, resampling and blind-optimization evidence. Preserve the
same score across datasets and distinguish provisional grid evidence from
a validated universal metric. This does not waive numerical-integrity or
matrix-only input checks.

Further user steering (2026-11-12): keep approximate results when the
optimum is slightly displaced, including the user's example of a point
around 15. Record both rotation position in degrees and relative score
error in percent, together with the affected plane and stability. Do not
silently reject a candidate solely because its optimum is not exactly at
baseline. Exact ranks remain useful diagnostics alongside continuous gaps.

First inspect local status and existing outputs. Read this package's README,
then test and run wave A on the four real datasets. Default source paths are
resolved by `matrix_io.default_step2`. Never regenerate Y from a known
formula or neural network to fill a missing matrix. Missing local case data
are a blocker that must be reported accurately.

The wave-A authoring session had only cloud Linux access. It did not execute
the user's Windows data. Treat its synthetic numerical tests only as code
verification, not as evidence that any candidate succeeds on the real models.

After each real batch:

1. Inspect approximation adequacy, rotation-invariant controls, endpoint
   equality and all individual plane curves, including worst-case planes.
2. Save full scores and failed candidates. Summarize actual numerical results
   in a new research log, then commit code/docs. Do not push bulky local data.
3. Require identical metric definition, direction and settings in all four
   datasets. No selected subsets of planes, per-model signs, hand-tuned
   mixture weights or model-specific function forms.
4. If no common candidate survives, investigate the distinct branches in the
   README: centered ANOVA/additive kernels, dependence-aware variance
   decomposition, and support-aware finite-scale/ALE effects. Prioritize
   sparsity of UU and UX interactions (including normalized tail-L2) over
   repeatedly measuring total gradient concentration. Include monotonicity
   and smoothness only with precise definitions and invariance controls.
5. Refit and evaluate each new batch on all four models. Preserve a frozen
   protocol per batch. Literature claims must use primary sources.
6. If a common candidate survives, test subject resampling, approximation
   resolution, random multi-plane rotations, and unsupervised rotation
   optimization from random starts. Scoring and optimization may not read
   evaluation labels. All-subject fitting does not waive these stability
   checks. A pairwise sweep win alone is not the final target.

Do not promise that an unknown universal metric must exist. Continue until
there is a well-supported common result or a concrete access/resource/
identifiability blocker; report that distinction. During long runs keep the
user informed of actual findings, not just elapsed process activity.
