# Higher-budget separated UX function compression

AF (degree-four joint polynomial teacher) and AG (U-RBF2 teacher) replace
the under-resolved AB/AC Adam pilots with exact empirical-grid alternating
least squares (ALS). Both are running CR3 first. No result is asserted here.

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
