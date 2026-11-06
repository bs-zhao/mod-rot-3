# First recoverability checkpoint: C1 and CR3

Source frozen runs: `run_70c97b3aadcb41f2` (C1) and
`run_fdc1bc1dfbec091b` (CR3), 750 previously inspected NP TEST participants
per model. Four identity PCA coordinates, no additional standardization of
the coordinates before orthogonal fitting. This is a supervised diagnostic,
not a new functional rotation method or an independent replication.

| Quantity, mean parameter r² | C1 | CR3 |
|---|---:|---:|
| Identity, existing frozen coordinates | 0.701049 | 0.484174 |
| Empirical unrestricted linear upper bound, same sample | 0.943677 | 0.763118 |
| Supervised orthogonal numerical reference, same sample | 0.943318 | 0.760604 |
| Supervised orthogonal reference, held-out half, mean | 0.942681 | 0.758389 |
| Held-out half range over six evaluations | 0.941450–0.943611 | 0.754017–0.765916 |
| Supervised unrestricted linear reference, held-out half, mean | 0.942872 | 0.757587 |

Each half-split reference is fitted using parameter labels from one half of
these 750 participants and evaluated on the other half. Three permutations
(seeds 31, 47, 73) each run in both directions. The parameter-axis assignment
is fixed during fitting; evaluation does not rematch the supervised axes.
These six evaluations overlap, so the range is descriptive, not a confidence
interval or six independent experiments. Sixteen orthogonal optimizer starts
are used per fit. The numerical reference is not a certified global optimum.

The full-sample linear bound is computed separately for each parameter as the
squared correlation between its OLS prediction and observed value. It is an
exact upper bound for any linear one-axis predictor **on this empirical
sample**, including axes constrained to an orthogonal rotation. It is not a
population bound or a guarantee for independently trained NP representations.

| Empirical per-parameter linear bound | C1 | CR3 |
|---|---:|---:|
| b1 / alpha | 0.947451 | 0.749097 |
| b2 / lambda_loss | 0.939978 | 0.839738 |
| b3 / gamma | 0.944193 | 0.714897 |
| b4 / tau | 0.943084 | 0.748740 |

The largest absolute off-diagonal entry of the normalized linear-direction
Gram matrix is 0.0303 (C1) and 0.1086 (CR3). The close orthogonal and unrestricted
references corroborate that orthogonality is not currently the main obstacle.

## Consequences for the next iteration

1. C1's successful JointDiag13 result (~0.943) already approaches the empirical
   linear bound in this frozen representation. Further gains there are limited.
2. CR3 retains parameter information compatible with an orthogonal rotation:
   supervised held-out recovery ~0.758 is substantially above identity ~0.484.
   The present subspace geometry alone does not explain failure of the tested
   functional objectives.
3. This cannot distinguish decoder derivative distortion from a poorly chosen
   simplicity objective, and does not establish unsupervised identifiability.
   These require decoder/simulator structure comparisons and fresh NP repeats.
4. Do not tune an unsupervised loss against these supervised directions and
   describe the result as held-out validation. Use fresh generated datasets
   and NP seeds for any eventual generality claim.

Validation: two synthetic tests pass (known orthogonal mixture recovery and
invariance of the unrestricted linear bound to an invertible basis change).
Both uploaded runs were executed with the standalone script on Linux; this
checkpoint does not represent a run in the user's Windows `py11` environment.
