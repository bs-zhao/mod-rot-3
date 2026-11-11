# Further approximation and UX-complexity screens

## V2: increase X approximation resolution

`waveV2_product_resolution_seed401` increased the generic X dictionary to
complete degree 4 plus 128 RBF centers, with the same U kernels and fit
rules. Raw held-out-subject NMSE improved to .00708441/.0104085 for U-RBF1
and .00597715/.00957792 for U-RBF2 (logit/probability). All 2.5 million CR3
rows entered the final fit audit and interaction descriptor. However, its
X-probe interpolation CV error increased to .0322256/.0546479. Higher
dictionary resolution is therefore not automatically better interpolation.
No adequate CR3 full-plane criterion survived. Other models were skipped.

## W: joint product-reference ANOVA

`waveW_joint_anova_auditfixed_seed402` decomposed orthogonal joint U/X
polynomial coefficients by their complete variable-support subsets. It
scored UX components, pure pair UX components, UU components, and marginal
hyperedge co-occurrence graphs. The reference is isotropic Gaussian U
times observed univariate X marginals, not the dependent empirical joint
law. This distinction matters because CR3 X columns are correlated.

The exact common intersection narrowed as follows:

| Evaluated dataset | Passing settings in that dataset | Common survivors |
| --- | ---: | ---: |
| CR3 | 12 | 12 |
| C1 | 792 | 8 |
| C2 | 310 | 7 |
| C3 | 54 | 0 |

All seven settings surviving the first three were poly4/logit pure-pair
UX supports. C3 rejected every one. The strict all-four comparison found
zero candidates. An initial audit writer could not serialize NumPy integer
subset labels; its partial output is marked FAILED_AUDIT_SERIALIZATION.
The type conversion was fixed, tested through the complete pipeline, and
the identical scientific definitions rerun in a fresh directory.

## X: observed-X local contrasts

CR3's six recorded X columns have 15/15/21/15/15/21 unique levels and
correlations up to approximately .47 in absolute value. To avoid treating
X interpolation derivatives as uniquely observed effects, X fits local
linear contrasts of the high-quality U-kernel gradients on the recorded
X-neighbor graph. It deconvolves simultaneous changes in X coordinates,
unlike the earlier unstructured gradient-difference descriptor Q.

Neighborhood sizes 16/32/64, a relative SVD threshold .001, and raw/point/
column balancing were fixed before scoring. The 16-neighbor design was
full rank at only 84.8% of nodes and is inadequate under the 95% rule.
The 32/64-neighbor designs were full rank at every node, but still found
no adequate full-plane CR3 criterion; the best valid count was 3/6.
Other datasets were skipped. Affine recovery on correlated designs and
orthogonal equivariance are verified numerically.

Primary background: [Cleveland and Devlin, local fitting](https://doi.org/10.1080/01621459.1988.10478639).
The observed contrast descriptor is custom and is not presented as their
published axis-recovery algorithm.

## Z: approximate first, then measure UX sparsity

Following the user's emphasis on UX, `waveZ_UX_complexity_seed405` uses
polynomial3/5 and RBF1/2 approximations of responses at recorded X nodes.
These functions use all 5,000 subjects and all original W/Y rows. No
behavioral formula or original decoder is used.

For each Xj, five-fold X crossfitting estimates what cannot be predicted
after omitting that column. The complementary affine X span is unpenalized,
so constants and all other linear X effects are annihilated exactly.
Nonlinear features use degrees 2/4 plus 32 RBF centers. Ridge selection
uses training-U subjects only. All source X nodes receive held-out-X
residuals; X columns are never assumed mutually independent.

Apply U first-order and total-effect ANOVA within explicit local isotropic
Gaussian references to those residual-response functions. Resulting Ui-Xj
energies measure predictive coupling, then normalized tails/effective
supports describe sparsity. They are not causal effects or empirical
dependent-input Sobol indices. Pure X functions vanish under U centering,
and pure U functions vanish under the X residual map. RBF moments are
analytic and polynomial moments/coordinate transports are exact.

Eight observed-U anchors, scales .2/.5, pooled/equal-anchor aggregation,
and a 90% neighborhood coverage rule were fixed prospectively. CR3 found
zero adequate full-plane candidates; some settings reached 5/6 and rank
2/37. Other models were skipped. Full failed curves and numerical controls
are retained. Further scale balancing of each X/reference and aggregation
of local supports, rather than only pooled energy, remain separate checks.

## Experimental infrastructure without a real-data claim

`cp_regression.py` and `run_cp_regression.py` provide separated-rank budget
regression, with independent batched fits, W-only sign/permutation
canonicalization, empirical marginal feature conditioning, and all-row
accounting. Only numerical/small synthetic pipeline tests have run at this
checkpoint. No real-data CP success or failure is claimed. This can later
test sparse functional representations of a fitted UX coupling function;
it is auxiliary to the requested primary UX interaction sparsity.
