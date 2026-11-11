# Waves H/I: effective support, rather than dominant-entry concentration

The user clarified the primary sparsity requirement with:
`[2,2,0.01,0.01]` is sparser than `[1,0.5,0.1,0.1]`.
Their normalized tail-2 L2 values are approximately 0.005 and 0.1255.
Shannon entropy or top-1 concentration can reverse that ordering because
they favor one dominant coefficient over two equally important ones.
They are therefore not sufficient primary sparsity evidence here.

## Frozen primary score definitions

Let p be squared amplitudes divided by total squared amplitude.

- Effective support: `(sum(p**(r/2)))**(1/(1-r/2))`, for r=0.5 and 1.
  For K equal active amplitudes, this equals K. It is unchanged by
  multiplying the whole vector by any nonzero constant.
- Soft support count: `sum(p/(p+epsilon**2))`, with fixed relative L2
  levels epsilon=0.01/0.03/0.10.
- Fractional support count: sort p, retain 1-epsilon² cumulative energy,
  and interpolate the last retained coefficient's contribution. This is
  a continuous counterpart of the integer count at a fixed L2 error.
- Tail-2 L2 is retained for comparison. It is constant zero when a vector
  has at most two entries and cannot identify those rotations.

No mixture weights or model-specific signs are selected from rankings.
The exact user example and overall scale invariance are regression tests.
All settings remain exploratory correlated diagnostics, not independent
statistical tests or proof of a uniquely correct sparsity definition.

## Wave H: derivative-support profiles

`run_tail_support.py --batch`, protocol `waveH_support_seed387`.
Reuse Wave C's accurate numerical fields. Compare UU entries, UU edges,
diagonal curvature, and effects of observed stimulus contrasts. Three
pooling rules are frozen: original empirical pooling, equal stimulus
contexts, and equal observation points. Context/point normalization uses
invariant Frobenius norms before aggregation, reducing magnitude effects.

Fourth-moment sufficient statistics reproduce exactly the pooled squared
Hessians under every rotation. Tests compare them to directly rotating
the entire fields. Stimulus contrasts are not called partial X derivatives.

CR3 had zero full-plane candidates. The best plane count was 3/6, rank
7/37. C1/C2/C3 were not launched. Source predictions were adequate.

## Wave I: ANOVA component support

`run_anova_support.py --batch`, protocol `waveI_anova_support_seed388`.
Apply the same primary support definitions to Wave F's frozen component
energies, using fresh anonymous IDs before revealing source case names.
There is no refit or ranking-based adjustment of the component energies.
Preserve the original numerical and support adequacy screens.

CR3 again had zero full-plane candidates. The best passing-plane count
was 3/6, rank 8/37, for rbf2 at scale 0.2. Other models were not launched.

Both batches require a CR3 candidate to survive at least one polynomial
and one RBF estimator with the same channel, metric and settings before
launching C1/C2/C3. No response-intensity reduction alone is accepted as
an effective-support result.
