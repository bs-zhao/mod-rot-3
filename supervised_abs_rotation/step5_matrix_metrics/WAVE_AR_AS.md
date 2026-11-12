# Direct matrix GP relevance and conditional monotone UX pieces

Both experiments completed CR3 on 2026-11-11 (America/New_York). Neither
found a qualified six-plane criterion; C1/C2/C3 were therefore skipped.
No universal criterion has been validated.

## AR: remove the U-function teacher

`waveAR_matrix_GP_ARD_UX_seed416` takes the actual response matrix on the
numerically verified common X grid. Every one of the 2.5 million original
Y entries and corresponding W rows is checked. There is no polynomial or
RBF U-function approximation before fitting the GP. Pure-U and pure-X means
are removed. Each cross-fitted X-column-omission target is normalized,
compressed by direct SVD with a 1e-6 squared-energy budget, and fitted by
the same GP policy as AQ: 128 training-subject centers, three isotropic
starts, 30 steps on 3750 subjects and 60 refit steps on all 5000.

The fidelity check includes the exact discarded SVD energy. Worst selected
GP heldout relative MSE is .00383126/.00376755 (logit/probability), and
full-refit MSE is .00269837/.00272513. These pass the declared .01 guard.
There are zero full-plane candidates; the best support scores reach 5/6
planes, baseline rank 2. Optimizer/center-count convergence remains
unverified, so this is a negative result for this finite GP protocol.

For unshared X designs, the general loader declares its generic-X-profile
approximation and audits original-row prediction error. It never claims
those interpolated profiles are actual shared response observations.

## AS: finite conditional slices

`waveAS_conditional_UX_pieces_seed418` uses both verified poly5 and RBF2
source functions fitted from all WY rows. It removes the pure-U component
and applies the frozen X-omission operators. At every one of the 5000 U
anchors, one Ui varies while other U coordinates stay fixed. Intervals
are the 10th--90th percentiles of 64 nearest neighbors in the other U
coordinates. Every source X context and 17 slice nodes are evaluated.

For each Ui/Xj/context slice, effect energy is squared range divided by
squared interval width. Relative soft activation counts use 1%, 3%, 10%
thresholds. A second complexity counts active monotone pieces by multiplying
activation by total variation/range. Both signs of a monotone effect have
the same cost. Each X column is normalized before pooling; function scale
does not set the support cost. X mean/worst and subject mean/median are
declared aggregation variants, not dataset-specific choices.

Observed-support probe coverage is at least .996388 in every W-only
coordinate class. Source validation NMSE is .007148/.007475 for poly5 and
.001097/.001077 for RBF2. Despite these guards, there are zero qualified
CR3 criteria; the best result is 4/6 planes, baseline rank 6. Per-person
cost tables, probe coverage and all failures remain in the local output.

Nine tests cover the actual matrix table, discarded-energy accounting,
declared unshared-X fallback, finite polynomial/RBF evaluation, support
scale invariance, conditional monotonicity and full WY-only pipelines.
They verify implementations, not success on the real datasets.
