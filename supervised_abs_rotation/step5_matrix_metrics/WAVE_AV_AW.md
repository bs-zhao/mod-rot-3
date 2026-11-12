# Tree resolution, blind rotations and paired subject refits

Prediction quality is reported continuously following the user's instruction;
there is no fixed-MSE veto. The same frozen AU logit/worst-X/10%-relative-L2
criterion is evaluated on all 5000 subjects in every dataset.

| Model | Strict planes | Baseline rank | Baseline leaves | Best grid leaves | Grid gain over baseline | Worst heldout relative MSE |
| --- | --- | --- | --- | --- | --- | --- |
| CR3 | 6/6 | 1 | 739 | 739 | 0% | .06621 |
| C1 | 6/6 | 1 | 25 | 25 | 0% | .01508 |
| C2 | 5/6 | 2 | 252 | 248 | 1.59% | .04376 |
| C3 | 11/15 | 5 | 1324 | 1311 | .98% | .11778 |

C3 is run despite the small C2 deviation. All 0/90 endpoints agree exactly.
`summarize_tree_evidence.py` writes every plane and the continuous comparison
with original score hashes. The small grid deviations remain uncertainty
to investigate, rather than successes relabeled with a new tolerance.

AV uses saved distortion paths at common relative-L2 budgets
.05/.06/.07/.08/.09/.10/.12/.15/.20. The original 10% per-X counts reproduce
exactly. No single channel/pooling/budget combination passes all strict
four-model grids. No trees are refitted or per-model settings selected.

## AW: broader rotation evidence on CR3

The fast scorer first reproduces every W-only coordinate class and per-X
count, including discarded function-mode energy. All actual W registrations
are checked again. There are 64 random SO(4) rotations, two blind Powell
runs with 80 evaluations each, and 24 paired subject bootstrap refits of
all original classes. Each draw has 5000 subjects with replacement. Scoring
reads no evaluation labels; blind scores freeze before label-based assessment.

All random rotations score above baseline: minimum 761, median 978.5,
versus 739. Blind optimization nevertheless reaches 527, 28.69% lower.
The optimized frames have maximum matched-axis errors about 52.3/52.8
degrees. Grid and unoptimized-random-rotation preferences do not establish
a global simplicity minimum.

Only 11/24 draws preserve all strict grid-interior advantages. The weakest
case, U2/U4 at 75 degrees, is above baseline in 14/24 draws, with empirical
relative-margin 5th/50th/95th percentiles -.03347/.00469/.03146.

AW2 freezes the optima and exactly replays the previous RNG state and
subject draws, refitting trees at the fixed rotations. Both remain below
baseline in all 24 draws. The better frame's relative-margin empirical
5th/50th/95th percentiles are -.26863/-.24786/-.22113; the other gives
-.19764/-.18344/-.16202. These are exploratory bootstrap diagnostics, not
independent new-subject confirmation. The counterexample persists under
these refits and is much larger than the small original grid deviations.

This remains useful function-description evidence, not a validated universal
criterion. Four stability tests check explicit/fast risk equivalence,
resampled tail recentering, SO rotations and a matrix-only complete pipeline.
