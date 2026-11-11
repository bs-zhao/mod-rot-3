# All-observed-point support and unrestricted X-profile compression

AK and AL completed CR3 without a full-plane criterion.
Source fitting and scoring use all 5000 subjects. No task equation, saved
rotation, original neural weights or true theta is used.

## AK: normalize local UX support before aggregation

T and AA integrate squared derivatives over every person's trials before
evaluating support. AK evaluates support at every recorded W point first,
then pools the support scores. This resolves a different question: whether
individual observed conditions have sparse UX effects even if their
average covariance is dense.

Joint degree-four and degree-five approximations are fitted to all original
WY rows. Each point's UX column is independently normalized before squared
energies, so arbitrary positive changes in its magnitude do not change
its support. Primary scores are effective support and small relative-tail
counts (1%,3%,10%); the only fixed-entry tail budget is two U axes per X.
Trials are pooled within each person by mean, q25, median or q75; X columns
by mean or worst; people by mean or median. These choices are prescribed
before scores are revealed and stay identical across all four datasets.

Every original trial enters these distributions, without choosing eight
or thirty-two contexts. Unequal trial counts still give equal person
weight. Per-person score tables are retained for paired bootstrap if a
criterion survives. Prediction adequacy does not prove derivative accuracy;
degree/estimator stability and blind rotations remain necessary.

Tests check exact polynomial derivatives against an independent derivative
evaluation, all-recorded-row coverage, unequal person counts, the user's
support example, and point/column rescaling from 1e-200 to 1e200.

## AL: few U-polynomial terms with free X profiles

AL first approximates the response by the verified degree-five U-response
bank or RBF2 response-bank mean. Each source's UX coupling is double-centered
over the whole observed U and X blocks. Omitted-X contrasts use the frozen
Z maps, preserving X's recorded joint design. Source-output compression
checks <=1e-7 relative squared reconstruction error for each X task.

Complete U-polynomial dictionaries of degree five/seven then approximate
these functions. A selected U monomial receives a completely free vector
of responses over observed X, rather than an assumed X polynomial or
task-specific function. This differs from M/AD's scalar full-W polynomial
terms. Greedy orthogonal least squares adds U monomials using the exact
all-subject prediction gain for each X contrast. Complexity is the
remaining relative L2 error at budgets 2..256, or fractional term counts
needed for 1%,3%,10% relative L2 error. X tasks are normalized independently.

Sufficient-statistic updates are checked against direct selected-span OLS
at multiple budgets. The full U-polynomial projection must be adequate for
every X task and invariant under U rotation. Otherwise the result is an
approximation/numerical failure, not evidence against sparse support.
Tests compare arbitrary correlated multiresponse regression to direct
least squares and verify the user's example with extreme dictionary units.

Sparse approximation/OLS context: Tropp and Wright,
https://tropp.caltech.edu/papers/TW10-Computational-Methods-preprint.pdf .
The algorithms here are explicit exploratory group-function compressions;
no theorem of universal rotation recovery is asserted.

AL's best adequate settings pass 3/6 planes and rank baseline 4/37.
Polynomial-source degree-five reconstruction error is below 6e-12;
RBF-source degree-seven projection error is <=.000230 (logit). Thus the
tested sparse-term criterion fails even when the numerical projection is
well resolved. C1/C2/C3 were skipped. Exact failed curves and direct-OLS
audits are retained. Input-marginal coordinate variants follow in
WAVE_AN_AO_AP.md; AL's original runner is retained in its result folder.
