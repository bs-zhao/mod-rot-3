# Conditional UX function rule complexity

AU uses the same W-only conditional X feature witnesses as AT, followed by
axis-aligned multioutput regression trees of the direct matrix-response
functions. It removes pure-U and pure-X means and normalizes every X task.
Output-mode SVD drops at most 1e-6 relative squared energy, and that exact
discarded energy is included in every reported prediction error.

One tree is trained on 3750 subjects and evaluated on the other 1250.
A separate tree uses all 5000 subjects to measure description complexity.
The complexity is the number of leaves needed to attain 1%, 3% or 10%
relative L2 error on the observed function table. Counts are aggregated
by mean or worst X column. This is rule complexity of UX functions, not
a claimed causal interaction graph or an optimal search over every tree.
All best-first expansions depend on training risk; validation does not
select splits. The complete per-person distortion paths are retained.

CR3 completes on 2026-11-11 (America/New_York). Logit, worst-X tree leaves
at 10% relative L2 error passes all six planes, baseline rank 1. The fixed
full-tree worst heldout relative MSE is .0662075 (logit) and .0823308
(probability). These are empirical approximation diagnostics.

The original runner used a <=.01 heldout-MSE eligibility gate and therefore
reported zero *adequate* candidates, despite that six-plane grid pass.
The user explicitly instructed that these standards should not be rigid.
Accordingly AU2 continues the frozen grid candidate to C1/C2/C3, recording
prediction errors without a fixed cutoff. It does not change CR3 scores,
metric signs, thresholds, aggregation or failed planes. Approximation
resolution, subject resampling and blind rotations remain subsequent
evidence for judging reliability rather than a premature declaration of
success. See `continue_tree_function_wave.py`.

Four tests verify all intermediate distortion values by explicit query
predictions, the user's sparsity example (4 versus 16 leaves at 3% error),
output-basis invariance/discarded-energy accounting, and the complete
actual-matrix-only pipeline. CART reference:
[scikit-learn decision trees](https://scikit-learn.org/stable/modules/tree.html).
