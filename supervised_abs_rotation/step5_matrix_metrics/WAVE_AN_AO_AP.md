# Monotone input conventions and conditionally balanced output geometry

These are prospective extensions after AL/AM failed CR3. The same rule
and score direction apply to all datasets, using only W/Y-derived values.

## AN/AP: empirical marginal coordinate conventions

AL compresses observed coupling values with generic U-polynomial terms and
free observed-X profiles. AN/AP replace each U coordinate by a deterministic
empirical marginal coordinate before building that dictionary: midpoint
ranks mapped to a standard-normal reference (AN) or a bounded uniform
reference with variance one (AP). The true response remains the same
fitted WY function on all 5000 subjects. No Gaussian joint distribution,
independent inputs, or generating response equation is assumed.

This defines complexity in ordinal U coordinates and removes arbitrary
strictly monotone changes in individual input units. It is a different
feature convention, not proof that a true physical parameter is Gaussian
or uniform. W-only signed/permutation classes fix numerical endpoint
symmetry. Polynomial degrees five/seven and relative-tail/count rules are
unchanged. Full projection errors are recorded for every case; the
nonlinear feature spaces are intentionally not rotation equivariant.

Marginal-transform context: Liu, Lafferty and Wasserman (2009),
https://www.jmlr.org/papers/v10/liu09a.html . We borrow the marginal-rank
idea, not that paper's Gaussian graphical-model assumptions or algorithm.

AN completed with no criterion (best 3/6 planes, baseline rank 8/37).
All projection guards pass; worst degree-seven relative MSE is <=.000294
for polynomial sources and <=.000790 for RBF sources. AP's bounded-rank
convention is now running CR3 first. Both conventions were listed before
AP was launched; no case labels select an input transformation.

## AO: balance X columns while estimating one global output link

AM conditions the mixed-derivative correction on predicted output, making
it globally integrable. AO keeps that construction but gives X columns
equal conditional scale before estimating k(y). A common 64-bin reference
estimates D_j(y)=E[||gU*gX_j||^2 | f=y]. Regression uses weighted sums

    n = sum_j <H_UXj, gU*gX_j> / D_j(f)
    d = sum_j ||gU*gX_j||^2 / D_j(f).

The candidate correction is k(y)=-E[n|y]/E[d|y]. At the population level
this equals the mean of the individual X-column conditional ratios.
Every candidate bin resolution uses the SAME reference weights, so its
subject-heldout validation objective is comparable. Identity plus
16/64/256 bins are prescribed; only the validation-selected procedure is
primary. We refit both reference weights and k on all 5000 subjects.

Multiplying an entire derivative-X column by an arbitrary positive scalar
scales its D_j by the scalar squared; the correction and resolution choice
are unchanged. Tests verify this from 1e-40 to 1e40. The population
output-link chain rule still holds. Finite conditional bins are an
approximation, and no exact nonlinear-output finite-sample invariance is
claimed.

Here k is piecewise constant in output. Its integral is piecewise linear,
so h' is continuous and positive; the Hessian chain rule is used away from
bin boundaries with a consistent one-sided convention at ties. Exact
integration and derivative-range guards are retained. Corrected columns
are normalized at each observed W point before all-trial moment scores.

AO reuses complete AM derivative caches after checking immutable source-fit
manifests, array shapes and hashes. It writes its own frozen scores and
retains all source failures. It is currently running CR3 first.

Validation: 152 tests pass, including monotone marginal-unit invariance,
signed/permuted endpoint symmetry, X-column scale invariance, exact
piecewise-link integration and complete W/Y-only pipelines.
