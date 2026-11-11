# Wave T: exact all-trial per-person derivative support

Run `run_all_trial_support.py --batch`. Uses each frozen joint polynomial
degree 2/3/4 fit, with its original all-subject/all-row WY boundary, and
streams through **every original W row** to compute derivative second
moments separately for every subject. No favorable stimulus subset is used.

For UU, symmetric Hessians use the orthonormal basis of diagonal entries
and square-root-two upper-triangle entries. Per-person covariance in this
basis transports exactly under U rotations. UX covariances are stored for
each X column; gradient covariance provides another control. Variable trial
counts are supported. Aggregation is equal across subjects after all of
each person's recorded rows have contributed to their local moment.

The scores retain effective support, relative-tail term counts and tail-2
L2, then take subject means/medians. UX-column medians use the same
definition as R/S, with all trials replacing sampled contexts. A separate
UU+UX total sums support inside each subject before aggregation; it has a
distinct metric name from R's sum of separate family medians.

Tests compare exact moments and scores with brute-force rotated derivatives,
including unequal trial counts. A complete synthetic pipeline verifies
that all original rows are included and forbidden inputs remain unopened.

CR3 completed as `waveT_all_trials_seed399`, using all 5,000 people and
2,500,000 trial rows. **Zero full-plane candidates** survived. The best
count was 5/6, with baseline rank 4 or worse; no near-pass was accepted.
C1/C2/C3 were skipped. This confirms the sampled-context R grid survivor
cannot be treated as a reliable all-context success.

Next examine local output-link-invariant UX geometry and point/column
scale balancing, still using every recorded trial. These are distinct,
prospective descriptors; T and S failures remain frozen and visible.
