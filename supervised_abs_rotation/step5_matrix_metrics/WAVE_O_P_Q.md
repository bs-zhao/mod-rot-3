# Follow-up screens O/P/Q

## O: pure-coupling and homogeneous-degree descriptors

Run `run_joint_separation.py --batch --resolved --seed 394 --run-name
waveO_resolved_seed394`. This removes product-reference U-only/X-only
components before scoring mixed-block separation spectra. Separately it
scores U degrees 1 through the chosen total degree, U-axis-support groups,
and equal-weighted X-multiindex profiles, using the same fixed support
definitions and Gaussian scales 0.5/1/2 in every dataset.

CR3 completed with zero full-plane candidates. Several scores reached
5/6 and rank 2/37. No sixth plane was dropped. The other datasets were
skipped. All previous N scores remain frozen at their original revision.

## P: input-only identifiability diagnostic

Run `run_input_independence.py --batch`. Kozachenko–Leonenko Euclidean
kNN marginal entropy sums (neighbors 3/10/30/100) check whether input
independence alone recovers the supervised axes. All 5,000 U observations
are used. Joint entropy is a rotation-invariant constant subtracted to
form an exploratory total-correlation estimate. No Y enters this score;
it cannot establish the requested W-to-Y response simplicity criterion.

CR3 failed: neighbor counts 3/10/30/100 gave ranks 16/3/5/5 and passed
2/4/3/3 planes. The remaining datasets were skipped. There is no evident
successful input-ICA shortcut in this screen.

Primary background: [Hyvarinen's ICA tutorial](https://cis.legacy.ics.tkk.fi/aapo/papers/IJCNN99_tutorialweb/)
and [Kozachenko–Leonenko](https://www.mathnet.ru/eng/ppi797).

## Q: support before pooling

Run `run_local_support.py --batch`. Frozen Wave C polynomial3/5 and RBF1/2
response-bank fields provide UU Hessians and U gradients at all 5,000
subjects and 64 observed X probes. X contrasts connect each probe to its
nearest observed X neighbor; these contrasts are explicitly not X partial
derivatives. No new response is generated from a decoder or known equation.

Prior H screens pooled effect energy before computing support, even when
fields were point-balanced. Q instead computes the support of each local
energy vector first, then averages or takes its median. It also scores
subject-wise and context-wise pooled energy before the final aggregation.
These are different descriptors when locally sparse patterns vary by person.

The vectorized definitions match individual scale-free support calculations.
A switching-pattern test has local support 1 but global pooled support 4.
Signed permutations, response rescaling and invariant field norms are checked.
The complete source field files are hashed into the new read audit.

CR3 completed with zero full-plane candidates; the best passing-plane count
was 3/6. All four estimators and both response channels were retained in
the failed output. The next R screen computes actual mixed U/X derivatives
from full joint-W projections; Q's observed-X contrasts are not substituted
for those derivatives.
