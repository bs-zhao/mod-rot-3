# Finer optima, stronger searches, and a more accurate learned global link

Experiments completed on the user's Windows machine, 2026-11-12,
America/New_York. Nearby minima and continuous score differences are kept.
No MSE=.01 or exact-zero-angle veto is applied.

## BN: a one-degree local audit of the same fixed GP statistic

The BB2 radial GP is fitted once using all 5000 subjects. Let H_j be its
pooled gradient covariance for normalized conditional X-response function j,
divided by its trace. The original pooled UX support is exactly
`max_j sum_i p_ij/(p_ij + eps^2)`, with `p_ij = diag(Q.T H_j Q)` and eps=.03.
BN independently reproduces all four frozen primary baseline scores before
evaluating every coordinate plane from -45 to +45 degrees in one-degree
steps. This is a labelled local audit, not another blind search.

| Model | Best one-plane angle | Relative score change | Maximum possible improvement from the optimistic matrix bound |
|---|---:|---:|---:|
| CR3 | +25 degrees | -.02968% | 4.5635% |
| C1 | -2 degrees | -2.1748% | 6.1119% |
| C2 | +2 degrees | -1.0775% | 2.2990% |
| C3 | -2 degrees | -.5357% | 2.2004% |

These small angular optima were missed by the coarse 15-degree grid and are
retained. Score differences and angular differences are separate quantities.

The bound is an exact statement about these **frozen GP matrix statistics**,
not about the unknown original function. For each PSD H_j, the diagonal
in any orthogonal basis is a doubly stochastic average of its eigenvalues.
Since t/(t+eps^2) is concave, the sum on that diagonal is at least its sum
on the eigenvalues. Taking the maximum over X gives an optimistic global
lower bound. It need not be jointly attainable when the matrices have no
common eigenbasis. Thus a small score gap does not certify axis recovery.

## BO/BP/BP2: improve optimization without changing the criterion

BO proposes the anonymous W coordinate frame, eigenbases of the normalized
X matrices and their mean, and 128 random frames. The 12 lowest-scoring
proposals initialize Powell searches with up to 2000 evaluations each.
The 1%, 3% and 10% tail policies are frozen in advance and unchanged across
all four models. The labelled reference frame is accessed only after all
searches for a model have frozen their scores and frames.

BP refines the frozen blind frames by exact Givens line searches over the
complete 90-degree fundamental interval: a .25-degree grid followed by
bounded scalar refinement. A **separate provided-case arm** starts from the
best four anonymous W-only case registrations. It reads no original saved R
and does not use case labels, but it is explicitly distinguished from the
random/eigenbasis search because the supplied case pool contains reference
axes among its candidates.

BP's fixed axis-order sweep can enter a narrow basin before reaching another
more beneficial pair. BP2 instead evaluates all pairs and chooses the largest
available decrease at each step, for up to 60 steps. The exact objective and
source function remain unchanged. Tests compare fast line evaluations with
actual rotated matrices and check monotone improvement and the first-step
best-line guarantee.

For the same 3% primary, BP2's best **blind-frame refinement** gives:

| Model | Relative score change | Maximum matched axis angle |
|---|---:|---:|
| CR3 | -2.9241% | 53.25 degrees |
| C1 | -2.6649% | 2.08 degrees |
| C2 | -2.1772% | 9.79 degrees |
| C3 | -1.9201% | 15.85 degrees |

This retains a useful approximate signal in C1/C2/C3 and a harder CR3
coordinate discrepancy. The provided-case arm is also saved, with score
changes -2.3553%/ -2.3163%/ -2.0950%/ -1.8558% and maximum matched angles
47.95/2.59/9.77/15.80 degrees. Neither finite search proves global optimality.
The coarse four-grid approximate results remain recorded alongside these
stronger audits; they are not retroactively erased.

## BQ: refine the global warped-GP approximation using all original rows

BQ replaces BE's 256-center joint radial dictionary with a complete joint-W
degree-four polynomial GP dictionary. Multinomial feature weights scaled
by dimension^(degree/2) give the kernel sum_k (z dot z'/D)^k, k=0..4.
This is a numerical approximation family, not a supplied response equation.
The common U scaling makes its kernel invariant under U rotations. No
baseline geometry is used to learn the global output map.

Every original WY row enters streamed sufficient statistics. The single
global 33-knot strictly monotone map per output channel is learned with the
density Jacobian, using the unchanged BE optimization policy. Identity and
learned maps both refit all 5000 subjects. Subject-heldout errors are measured
on original Y rows; validation is conditional on knot locations frozen
from all observed Y. The dictionary factory and feature metadata are saved.

Original-Y heldout relative MSE, identity -> learned:

| Model | logit_nn | p_nn |
|---|---:|---:|
| CR3 | .008909 -> .008939 | .016416 -> .009525 |
| C1 | .0000799 -> .0000721 | .006889 -> .000158 |
| C2 | .001380 -> .001360 | .023333 -> .001247 |
| C3 | .003005 -> .002962 | .042557 -> .002450 |

These are substantial source-resolution improvements over BE, particularly
for p_nn. They do not by themselves establish better rotation preferences.

## BR: unchanged conditional kernel support on the transformed actual Y

The learned BQ map is applied to actual Y **before** generic X-profile
preparation. Predicted GP outputs do not replace observations. The same
32-center conditional Ui feature moments then give 12/220/103/19 strict
primary combinations in CR3/C1/C2/C3, with no identical strict four-grid
combination. All nearby minima and the frozen CR3 counterexample remain
in the continuous reports. Better source approximation is not treated as
automatic structural success, and larger errors are not automatic rejection.

The full code suite passes 237 tests. Full source artifacts, optimization traces, frame
hashes, numerical replay checks and unsuccessful variants remain local;
only code and documentation are pushed.
