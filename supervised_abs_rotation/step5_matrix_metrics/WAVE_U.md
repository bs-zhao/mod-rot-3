# Wave U: output-link-invariant local UX geometry

Run `run_link_ux.py --batch`. All original rows contribute to every
person's covariance. Degrees 2/3/4 and both response channels use the
same fixed definitions; the CR3-first intersection gate is retained.

For gradient g and mixed Hessian HUX, subtract its scalar projection onto
the normalized gradient outer product gU*gX. Alternatively estimate that
scalar using the entire U-by-W Hessian block, or project each UX row into
the X-gradient tangent space. Raw UX is retained as a control family.
Each field is normalized locally either across the whole UX matrix or
separately within each X column, before all-trial per-person covariance.
The same support/relative-tail scores are then averaged or median-aggregated.

The chain rule for y=h(f) gives HUX(y)=h' HUX(f)+h'' gU(f)gX(f)^T.
The profiled/tangent fields remove the second term and local normalization
removes the magnitude of h'. This is a local geometric invariance for
regular transformations, not an assumed formula for h or a constructed
global response calibration. It differs from E's UU-only profiling.

Tests cover arbitrary point-dependent first/second link derivatives,
orthogonal equivariance, affine scales from 1e-120 to 1e120, exact rank-one
null residuals, and a complete all-trial WY-only pipeline. Residuals below
1e-12 relative local norm are treated as numerical nulls and counted in
the audit; no absolute response-strength threshold selects interactions.

CR3 completed as `waveU_link_ux_seed400`. **No full-plane candidate**
survived. Best count was 5/6, with tangent/column variants reaching
rank 2/37. Other models were skipped. The invariance tests verify the
mathematics, not success of the metric on these data.

Geometric background: [Riviere et al., differential geometry notes](https://pure.manchester.ac.uk/ws/portalfiles/portal/295567946/FULL_TEXT.PDF).
The scalar UX profiling diagnostic here is custom and is not attributed
to that source as a published axis-recovery algorithm.

Next fit a generic U-kernel times X-dictionary response representation,
so actual UX derivatives can be tested with a non-polynomial U estimator
and raw held-out-subject prediction checks. No decoder/generator is used.
