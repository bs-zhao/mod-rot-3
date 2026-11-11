# Higher-budget separated UX function compression

AF (degree-four joint polynomial teacher) and AG (U-RBF2 teacher) replace
the under-resolved AB/AC Adam pilots with exact empirical-grid alternating
least squares (ALS). Both completed CR3 with zero grid survivors, as
documented below; the higher-budget AI/AJ extension is now running.

The function is first fitted to all original WY rows. Its whole-U/whole-X
empirical means are subtracted. A rank-r sum of products of learned
univariate functions then approximates this UX coupling. The factor basis
is 1,z,z2,z3,z4 plus five Gaussian knots, centered/whitened per marginal.
There is a free scalar intercept, not free arbitrary U-only/X-only terms.
This measures functional-expression complexity, not just graph-edge count.

The prescribed budgets are ranks 1,2,4,8,16; three starts and forty sweeps
per rank, all in float64. Each linear subproblem uses ridge 1e-8 relative to
its Gram scale. W-only signed/permutation classes share identical scores
by construction; classes/channels/restarts otherwise have separate fits.
The best state is retained independently for each fit.

For CR3 the numerically verified common recorded-X grid is complete:
all 5000 subjects x 500 recorded X rows. Grid sufficient statistics factor
only between the whole U and X blocks. They average products over observed
joint U vectors and observed joint X vectors, preserving all dependence
within either block. A synthetic test with correlated coordinates compares
the contracted loss to explicit full-grid predictions and checks each ALS
linear update decreases actual squared error. Real runs also audit direct
full-grid loss for one fitted model at each checkpoint.

For unshared designs, the same procedure uses the source bank's observed
joint-X nodes as a product-reference grid with all 5000 U subjects. The
original-row source-fit errors and the changed compression reference are
recorded explicitly. It must not be reported as scoring every recorded
U-X pair in that case. Polynomial and RBF shared/unshared pipelines are
all covered by W/Y-only fixtures with forbidden rotation files present.

The classic CP/ALS framework is reviewed by Kolda and Bader:
https://www.kolda.net/publication/koba09/ . Our block-empirical function
regression uses its alternating-linear-subproblem idea; it does not treat
observed dependent scalar coordinates as an independent tensor-product
quadrature rule. ALS does not guarantee the global best separated rank.
Restarts, rank/iteration sensitivity and blind rotation optimization remain
required before a preliminary grid result can support a universal claim.

Validation before launch: seven ALS tests passed, including all four
polynomial/kernel x shared/unshared pipelines. The complete matrix-metrics
test suite then passed 131 tests. No test uses a scientific success ranking
as its expected result.

## Actual CR3 outcome and capacity extension

AF/AG completed with no preliminary full-plane criteria. The mean relative
squared errors across all independent rank-16 fits are .0312556 and
.0789873. Endpoints are identical by W-only signed/permutation grouping.
Source fits use all 2.5 million recorded CR3 rows. This is an improvement
over the rank-four pilots, but the remaining errors and local optimization
preclude a conclusive rejection of converged separated compression.

AI/AJ prospectively extend both teachers to ranks 32 and 64, forty sweeps
and three starts. They verify every saved feature-geometry array before
transferring rank-16 checkpoints, preserve all original low-rank losses,
and hash the reused artifacts. Shared-grid pipeline tests check exact loss
preservation. Both extensions are currently running CR3 first.

Only budgets >=2 and required counts at 1%,3%,10% relative L2 error qualify
as primary sparse-support scores. Rank-one residual and 30% counts remain
diagnostics: the user's [2,2,.01,.01] versus [1,.5,.1,.1] example demonstrates
that dominance and effective support need not agree. The scoped AF2/AG2
runs retain identical numerical-score SHA256 values and still have zero
primary passes. This scope rule was recorded before AI/AJ completed.
